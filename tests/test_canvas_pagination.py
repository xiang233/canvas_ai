"""
验证 Canvas API 的翻页行为。

起一个本地 HTTP server 返回带 Link 头的分页响应，走真实的
_fetch_all_pages 代码路径，不 mock 内部实现。不需要 Canvas token，
也不会打真实 Canvas。

回归的是这个 bug：列表端点只传 per_page，60 个作业的课程只回 50 个，
调用方看不出还有下一页。

用法：
    python -m tests.test_canvas_pagination        # 从仓库根目录运行
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# 基类构造时要求有 token，这里给个假的即可（请求打到本地 server）
os.environ.setdefault("CANVAS_ACCESS_TOKEN", "dummy-token-for-test")

STATE = {"total_pages": 1, "requests": [], "port": 0, "fail_page": None}


class _Handler(BaseHTTPRequestHandler):
    """按 page 参数返回分页数据，最后一页不带 rel="next" """

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        page = int(query.get("page", ["1"])[0])
        STATE["requests"].append(self.path)

        if STATE["fail_page"] is not None and page == STATE["fail_page"]:
            body = b'{"errors":["boom"]}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 每页 2 条，条目 id 连续，便于断言总数与顺序
        items = [{"id": (page - 1) * 2 + i, "name": f"item-{(page - 1) * 2 + i}"}
                 for i in range(1, 3)]
        body = json.dumps(items).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if page < STATE["total_pages"]:
            nxt = f'http://127.0.0.1:{STATE["port"]}/api/v1/courses?page={page + 1}'
            # 真实 Canvas 会同时给 current/next/last，这里一并带上，
            # 确保解析器是按 rel="next" 取而不是取第一个链接
            self.send_header(
                "Link",
                f'<{nxt}&rel=current>; rel="current", <{nxt}>; rel="next"',
            )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # 静音访问日志


async def _case(tool, name, total_pages, expect_count, fail_page=None):
    STATE["total_pages"] = total_pages
    STATE["requests"] = []
    STATE["fail_page"] = fail_page

    result = await tool._fetch_all_pages("courses", params={"per_page": 2})

    is_list = isinstance(result, list)
    count = len(result) if is_list else -1
    passed = is_list and count == expect_count
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"        条目数={count} (期望 {expect_count})  请求数={len(STATE['requests'])}")
    if not passed:
        print(f"        result={result}")
    return passed


async def _case_first_page_error(tool):
    """首页就失败时要如实返回错误，不能假装是空列表"""
    STATE["total_pages"] = 3
    STATE["requests"] = []
    STATE["fail_page"] = 1

    result = await tool._fetch_all_pages("courses", params={"per_page": 2})
    passed = isinstance(result, dict) and "error" in result
    print(f"  {'PASS' if passed else 'FAIL'}  首页失败返回 error 而非空列表")
    if not passed:
        print(f"        result={result}")
    return passed


async def _case_max_pages(tool):
    """超过 max_pages 时要留下 _truncated 标记，不能静默截断"""
    STATE["total_pages"] = 99
    STATE["requests"] = []
    STATE["fail_page"] = None

    result = await tool._fetch_all_pages("courses", params={"per_page": 2}, max_pages=3)
    truncated = isinstance(result, list) and any(
        isinstance(x, dict) and "_truncated" in x for x in result
    )
    print(f"  {'PASS' if truncated else 'FAIL'}  超上限时带 _truncated 标记")
    if not truncated:
        print(f"        result={result}")
    return truncated


async def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    STATE["port"] = port
    os.environ["CANVAS_URL"] = f"http://127.0.0.1:{port}"

    from src.tools.canvas_tools import CanvasListCourses

    tool = CanvasListCourses()
    # 基类会把 http:// 重写成 https://，测试里指回本地明文服务
    tool.base_url = f"http://127.0.0.1:{port}/api/v1"
    # 重试等待会拖慢测试，这里只验证翻页
    tool.max_retries = 0

    print("\n翻页行为验证\n")
    results = [
        await _case(tool, "单页（无 next）返回全部", total_pages=1, expect_count=2),
        await _case(tool, "三页全部跟完（回归静默截断）", total_pages=3, expect_count=6),
        await _case_first_page_error(tool),
        await _case_max_pages(tool),
    ]

    server.shutdown()
    passed = sum(1 for r in results if r)
    print(f"\n{passed}/{len(results)} 通过\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
