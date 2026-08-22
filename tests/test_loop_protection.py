"""
验证三层防线：重复调用短路、observation 渲染保真、错误信息保真。

背景是三次真实事故（同一个成因的三种表现）：
  1. 问教授是谁——工具把 Canvas 返回的 teachers 字段丢弃，模型以为
     参数传错，同参数重试 15 次烧掉 46 万 token
  2. 问最新公告——Canvas 隐藏的 14 天默认窗口返回空，工具渲染成
     光秃秃的标题，模型分不清"没有"和"没干活"
  3. 问 quizzes——未启用经典测验的课返回 404，工具把状态码吞成
     "Resource not found"，模型读作"课程不存在"而重试

分层原则（与 description/prompt 的分工）：
  - L2（本文件测的）只保证事实不丢失：条数永远可见、错误带方法+URL+
    状态码。不做任何 per-case 解读——解读知识在 description（L1），
    推断是模型的事
  - L4 短路是机制兜底：同一次 run 内完全相同的 (工具, 参数) 第二次
    直接掐掉，不打 API

纯本地，不打网络、不需要 API key。

用法：
    python -m tests.test_loop_protection
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ.setdefault("CANVAS_ACCESS_TOKEN", "dummy-token-for-test")
os.environ.setdefault("CANVAS_URL", "https://example.invalid")

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, condition))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")
    if not condition and detail:
        print(f"        {detail}")


# ---------------------------------------------------------------- 事故 1 回归
SAMPLE_COURSE = {
    "id": 155084,
    "name": "Fall_2025.CSE.5100.01 - Deep Reinforcement Learning",
    "course_code": "Fall 2025.CSE.5100.01",
    "workflow_state": "available",
    "enrollments": [{"computed_current_score": 101.08, "computed_current_grade": "A"}],
    "teachers": [{"id": 1, "display_name": "Chongjie Zhang"}],
    "syllabus_body": "<p>Grading: <b>40%</b> homework</p>",
}


def test_include_rendering():
    from src.tools.canvas_tools import CanvasListCourses

    tool = CanvasListCourses()

    async def render(course):
        async def fake(endpoint, params=None, max_pages=20):
            return [course]
        tool._fetch_all_pages = fake
        return (await tool.forward(include="teachers")).output

    out = asyncio.run(render(SAMPLE_COURSE))
    check("include=teachers 时教师姓名被渲染（事故 1 回归）",
          "Chongjie Zhang" in out, out[:200])
    check("分数仍然渲染", "101.08" in out, out[:200])
    check("syllabus 去 HTML 后渲染", "40%" in out and "<b>" not in out, out[:200])
    bare = dict(SAMPLE_COURSE, teachers=[], syllabus_body=None)
    out2 = asyncio.run(render(bare))
    check("无 teachers 数据时不输出空字段", "teachers:" not in out2, out2[:200])


# ------------------------------------------------- 事故 2 回归:空结果不哑巴
def test_empty_render_sweep():
    """遍历所有列表工具:空结果必须输出"共 0 条",不允许裸标题。

    这是 render_list 收敛的验收——新工具只要走 render_list 就自动合规,
    不需要任何人记得写空结果处理。
    """
    from src.tools import canvas_tools as ct

    async def empty_fetch(endpoint, params=None, max_pages=20):
        return []

    async def empty_req(method, endpoint, params=None, data=None, _return_link=False):
        # grades 先调 users/self 拿用户 id,再查 enrollments——
        # 只有列表端点该为空,身份端点要给个合法对象
        return {"id": 1} if endpoint == "users/self" else []

    cases = [
        (ct.CanvasGetModules, {"course_id": "1"}),
        (ct.CanvasGetModuleItems, {"course_id": "1", "module_id": "2"}),
        (ct.CanvasGetFiles, {"course_id": "1"}),
        (ct.CanvasGetDiscussions, {"course_id": "1"}),
        (ct.CanvasGetGrades, {"course_id": "1"}),
        (ct.CanvasGetPages, {"course_id": "1"}),
        (ct.CanvasGetQuizzes, {"course_id": "1"}),
        (ct.CanvasGetTodoItems, {}),
        (ct.CanvasGetUpcomingEvents, {}),
        (ct.CanvasGetCalendarEvents, {}),
        (ct.CanvasGetGroups, {}),
        (ct.CanvasGetFolders, {"course_id": "1"}),
        (ct.CanvasGetFolderFiles, {"folder_id": "1"}),
        (ct.CanvasGetAnnouncements, {}),
        (ct.CanvasSearchFiles, {"course_id": "1", "search_term": "x"}),
    ]
    bad = []
    for cls, kwargs in cases:
        t = cls()
        t._fetch_all_pages = empty_fetch
        t._make_request = empty_req
        out = asyncio.run(t.forward(**kwargs)).output or ""
        if "共 0 条" not in out:
            bad.append(f"{cls.name}: {out[:60]!r}")
    check(f"15 个列表工具空结果全部输出'共 0 条'（事故 2 回归）",
          not bad, "; ".join(bad))


def test_announcements_window():
    """公告默认窗口是一年,不是 Canvas 隐藏的 14 天;窗口写进输出"""
    from src.tools.canvas_tools import CanvasGetAnnouncements

    captured = {}

    async def fake(endpoint, params=None, max_pages=20):
        captured.update(params or {})
        return []

    t = CanvasGetAnnouncements()
    t._fetch_all_pages = fake
    out = asyncio.run(t.forward()).output
    check("公告默认窗口一年而非 14 天", captured.get("start_date", "9999") < "2026-01-01",
          str(captured))
    check("空结果输出里带查询窗口（模型能据此决定是否扩大）",
          "共 0 条" in out and captured.get("start_date", "") in out, out[:120])


def test_list_sorting():
    """问"最新"时第一条就该是答案:quizzes/公告 从新到旧"""
    from src.tools.canvas_tools import CanvasGetAnnouncements, CanvasGetQuizzes

    async def quizzes(endpoint, params=None, max_pages=20):
        return [
            {"id": 1, "title": "old", "due_at": "2025-09-01T00:00:00Z"},
            {"id": 2, "title": "new", "due_at": "2025-12-01T00:00:00Z"},
        ]

    q = CanvasGetQuizzes()
    q._fetch_all_pages = quizzes
    out = asyncio.run(q.forward(course_id="1")).output
    check("quizzes 从新到旧且带总数",
          "共 2 条" in out and out.index("new") < out.index("old"), out[:150])

    async def anns(endpoint, params=None, max_pages=20):
        return [
            {"title": "old", "posted_at": "2025-09-01T00:00:00Z", "message": "m"},
            {"title": "new", "posted_at": "2025-12-16T00:00:00Z", "message": "m"},
        ]

    a = CanvasGetAnnouncements()
    a._fetch_all_pages = anns
    out2 = asyncio.run(a.forward()).output
    check("公告从新到旧且带总数",
          "共 2 条" in out2 and out2.index("new") < out2.index("old"), out2[:150])


# ------------------------------------------------- 事故 3 回归:错误保真
def test_error_fidelity():
    """404 必须带方法+URL+状态码原样到达模型——预训练知识挂在'HTTP 404'
    这三个字符上,description 里的合同挂在端点路径上,两者都不能丢。
    不做 per-case 翻译:解读是模型的事。"""
    from src.tools.canvas_tools import CanvasGetQuizzes

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"errors":[{"message":"The specified resource does not exist."}]}'
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    t = CanvasGetQuizzes()
    t.base_url = f"http://127.0.0.1:{port}/api/v1"
    t.max_retries = 0
    r = asyncio.run(t.forward(course_id="155084"))
    server.shutdown()

    err = str(r.error)
    check("404 错误带状态码（'HTTP 404'字样,激活预训练知识）", "HTTP 404" in err, err[:120])
    check("404 错误带方法与完整 URL（description 合同的锚点）",
          "GET" in err and "/courses/155084/quizzes" in err, err[:120])
    check("不做 per-case 翻译（无'未启用'等解读字样）", "未启用" not in err, err[:120])


# ------------------------------------------------- L4 短路机制
def test_short_circuit():
    import inspect

    from src.agent.general_agent import general_agent as ga
    from src.base.async_multistep_agent import AsyncMultiStepAgent

    run_src = inspect.getsource(AsyncMultiStepAgent.run)
    check("run() 开始时清空调用签名（跨轮不误伤）",
          "_call_signatures" in run_src and "clear()" in run_src)

    mod_src = inspect.getsource(ga)
    guard = mod_src.index("if signature in self._call_signatures")
    call = mod_src.index("await self.execute_tool_call(tool_name, tool_arguments)")
    check("短路判定在真正打 API 之前", guard < call)
    check("短路返回明确指令而非空串",
          "[repeated call]" in mod_src and "final_answer_tool using what you have" in mod_src)

    sigs = set()

    def would_short_circuit(name, args):
        sig = (name, json.dumps(args, sort_keys=True, default=str))
        if sig in sigs:
            return True
        sigs.add(sig)
        return False

    a = {"enrollment_state": "active"}
    check("首次调用不短路", not would_short_circuit("canvas_list_courses", a))
    check("相同参数第二次被短路", would_short_circuit("canvas_list_courses", a))
    check("不同参数不被短路（合理重试放行）",
          not would_short_circuit("canvas_list_courses",
                                  {"enrollment_state": "active", "include": "teachers"}))
    check("参数顺序不同但内容相同仍被短路（键已排序）",
          would_short_circuit("canvas_list_courses",
                              {"include": "teachers", "enrollment_state": "active"}))
    check("同参数不同工具不被短路", not would_short_circuit("canvas_get_grades", a))


def main() -> int:
    print("\n短路 + 渲染保真 + 错误保真 验证\n")
    test_include_rendering()
    test_empty_render_sweep()
    test_announcements_window()
    test_list_sorting()
    test_error_fidelity()
    test_short_circuit()
    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} 通过\n")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
