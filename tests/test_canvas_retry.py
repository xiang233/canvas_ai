"""
验证 Canvas API 的重试/退避行为。

起一个本地 HTTP server 按脚本返回状态码，走真实的 _make_request 代码路径，
不 mock 内部实现。不需要 Canvas token，也不会打真实 Canvas。

用法：
    python -m tests.test_canvas_retry        # 从仓库根目录运行
"""

import asyncio
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# 基类构造时要求有 token，这里给个假的即可（请求打到本地 server）
os.environ.setdefault("CANVAS_ACCESS_TOKEN", "dummy-token-for-test")

STATE = {"hits": 0, "script": []}


class _Handler(BaseHTTPRequestHandler):
    """按 STATE["script"] 依次返回状态码，用完后返回 200"""

    def do_GET(self):
        i = STATE["hits"]
        STATE["hits"] += 1
        status = STATE["script"][i] if i < len(STATE["script"]) else 200
        body = b'{"ok": true}' if status == 200 else b'{"errors":["boom"]}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # 静音访问日志


async def _run_case(tool, name, script, expect_ok, expect_hits):
    STATE["hits"] = 0
    STATE["script"] = script

    t0 = time.time()
    result = await tool._make_request("GET", "courses")
    elapsed = time.time() - t0

    got_ok = "error" not in result
    hits = STATE["hits"]
    passed = (got_ok == expect_ok) and (hits == expect_hits)

    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"        请求次数={hits} (期望 {expect_hits})  成功={got_ok} (期望 {expect_ok})  耗时={elapsed:.2f}s")
    if not passed:
        print(f"        result={result}")
    return passed


async def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    os.environ["CANVAS_URL"] = f"http://127.0.0.1:{port}"

    from src.tools.canvas_tools import CanvasListCourses

    tool = CanvasListCourses()
    # 基类会把 http:// 重写成 https://，测试里指回本地明文服务
    tool.base_url = f"http://127.0.0.1:{port}/api/v1"

    print(f"max_retries={tool.max_retries}  timeout={tool.timeout}\n")

    results = [
        # 正常路径不应产生额外请求
        await _run_case(tool, "200 一次成功，不重试", [200], True, 1),
        # 瞬时故障应重试并最终成功
        await _run_case(tool, "429 两次后成功（应重试）", [429, 429, 200], True, 3),
        await _run_case(tool, "503 一次后成功（应重试）", [503, 200], True, 2),
        # 持续失败应在 max_retries 后放弃，共 1 + max_retries 次
        await _run_case(tool, "持续 500 → 耗尽重试后放弃", [500] * 10, False, 1 + tool.max_retries),
        # 客户端错误重试无意义，必须立即返回
        await _run_case(tool, "403 鉴权失败 → 不重试", [403], False, 1),
        await _run_case(tool, "404 → 不重试", [404], False, 1),
    ]

    server.shutdown()
    print()
    print("ALL PASS" if all(results) else "SOME FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
