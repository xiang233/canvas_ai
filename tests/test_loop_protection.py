"""
验证重复工具调用的短路机制，以及 include 字段的渲染完整性。

回归的是一次真实故障：问"深度强化学习课的教授是谁"，agent 连续 15 次
调用 canvas_list_courses(include="teachers")，每次返回完全一样，烧掉
46 万 token 直到撞上步数上限，最后答"材料里没有教授姓名"。

两个成因，各修一处：
  1. 工具在 schema 里声明支持 include=teachers，却只渲染 id/name/
     course_code/分数，把 teachers 整个丢弃。模型看不到自己要的字段，
     以为参数传错了，于是反复重试。
  2. prompt 里"不要重复相同调用"只是约定，没有机制兜底。

纯本地，不打网络、不需要 API key。

用法：
    python -m tests.test_loop_protection
"""

import json
import os

os.environ.setdefault("CANVAS_ACCESS_TOKEN", "dummy-token-for-test")
os.environ.setdefault("CANVAS_URL", "https://example.invalid")

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, condition))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")
    if not condition and detail:
        print(f"        {detail}")


SAMPLE_COURSE = {
    "id": 155084,
    "name": "Fall_2025.CSE.5100.01 - Deep Reinforcement Learning",
    "course_code": "Fall 2025.CSE.5100.01",
    "workflow_state": "available",
    "enrollments": [{"computed_current_score": 101.08, "computed_current_grade": "A"}],
    "teachers": [{"id": 1, "display_name": "Chongjie Zhang"}],
    "syllabus_body": "<p>Grading: <b>40%</b> homework</p>",
}


def main() -> int:
    print("\n重复调用短路 + include 渲染验证\n")

    # --- 1. include 渲染完整性 ---
    import asyncio

    from src.tools.canvas_tools import CanvasListCourses

    tool = CanvasListCourses()

    async def render(course):
        # 只跑格式化逻辑，不打网络
        async def fake(endpoint, params=None, max_pages=20):
            return [course]
        tool._fetch_all_pages = fake
        return (await tool.forward(include="teachers")).output

    out = asyncio.run(render(SAMPLE_COURSE))
    check("include=teachers 时教师姓名被渲染出来（回归死循环成因）",
          "Chongjie Zhang" in out, out[:200])
    check("分数仍然渲染（原有行为不回退）", "101.08" in out, out[:200])
    check("syllabus 去掉 HTML 标签后渲染",
          "40%" in out and "<b>" not in out, out[:200])

    # 没有 teachers 数据时不应该凭空出现字段
    bare = dict(SAMPLE_COURSE, teachers=[], syllabus_body=None)
    out2 = asyncio.run(render(bare))
    check("无 teachers 数据时不输出空字段", "teachers:" not in out2, out2[:200])

    # --- 2. 短路机制 ---
    # 短路状态必须是"本次 run 内"的：跨轮重复提问合理，只有同一次推理里
    # 反复打同一调用才是死循环。这里断言 run() 会清空它
    import inspect

    from src.base.async_multistep_agent import AsyncMultiStepAgent

    run_src = inspect.getsource(AsyncMultiStepAgent.run)
    check("run() 开始时清空调用签名（跨轮不误伤）",
          "_call_signatures" in run_src and "clear()" in run_src)

    from src.agent.general_agent import general_agent as ga

    mod_src = inspect.getsource(ga)
    # 短路判定必须在 execute_tool_call 之前，否则 API 还是会被打
    guard = mod_src.index("if signature in self._call_signatures")
    call = mod_src.index("await self.execute_tool_call(tool_name, tool_arguments)")
    check("短路判定在真正打 API 之前", guard < call)
    check("短路返回的提示明确指向 final_answer_tool（不是空字符串）",
          "[repeated call]" in mod_src and "final_answer_tool using what you have" in mod_src)

    # 用签名集合直接验证判定逻辑（与 process_single_tool_call 内一致）
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
    check("不同参数不被短路（合理重试要放行）",
          not would_short_circuit("canvas_list_courses",
                                  {"enrollment_state": "active", "include": "teachers"}))
    check("参数顺序不同但内容相同仍被短路（键已排序）",
          would_short_circuit("canvas_list_courses",
                              {"include": "teachers", "enrollment_state": "active"}))
    check("同参数不同工具不被短路",
          not would_short_circuit("canvas_get_grades", a))

    # --- 3. 公告工具：默认窗口与空结果渲染 ---
    # 同一类病的另一处：Canvas /announcements 不传日期只返回最近 14 天，
    # 结课后的课用默认值永远查不到；且空结果渲染成光秃秃的标题会诱发重试
    from src.tools.canvas_tools import CanvasGetAnnouncements

    ann = CanvasGetAnnouncements()
    captured = {}

    async def fake_fetch(endpoint, params=None, max_pages=20):
        captured.update(params or {})
        return captured.pop("_ret", [])

    ann._fetch_all_pages = fake_fetch

    out = asyncio.run(ann.forward(context_codes="course_1")).output
    check("公告默认窗口是一年而非 Canvas 的 14 天",
          "start_date" in captured and captured["start_date"] < "2026-01-01",
          str(captured))
    check("空结果明确说明范围并提示扩大（不是光秃秃的标题）",
          "没有公告" in out and "start_date" in out, out[:120])

    async def fake_fetch2(endpoint, params=None, max_pages=20):
        return [
            {"title": "old", "posted_at": "2025-09-01T00:00:00Z", "message": "m"},
            {"title": "new", "posted_at": "2025-12-16T00:00:00Z", "message": "m"},
        ]

    ann._fetch_all_pages = fake_fetch2
    out2 = asyncio.run(ann.forward()).output
    check("公告按时间从新到旧排列且带总数",
          "共 2 条" in out2 and out2.index("new") < out2.index("old"), out2[:150])

    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} 通过\n")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
