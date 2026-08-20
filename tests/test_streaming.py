"""
验证流式事件层与两个传输的行为。

核心主张：事件抽象在 agent_stream 里，SSE 和 WebSocket 只是传输适配，
所以同一次对话在两个协议上产出的事件序列必须完全一致。

另外断言 ReAct 循环里"流什么、不流什么"的分类：推理文本逐 token 流，
工具调用推状态，最终答案单个事件——中间步骤逐字流出去没有意义。

桩 agent，不打网络、不需要 API key。

用法：
    python -m tests.test_streaming        # 从仓库根目录运行
"""

import asyncio
import json
import os

os.environ.setdefault("CANVAS_ACCESS_TOKEN", "dummy-token-for-test")
os.environ.setdefault("CANVAS_URL", "https://example.invalid")

from src.memory import ActionStep, FinalAnswerStep
from src.memory.memory import Timing, ToolCall
from src.models.base import ChatMessageStreamDelta

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, condition))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")
    if not condition and detail:
        print(f"        {detail}")


class StubStreamAgent:
    """模拟一次两步的 ReAct 运行：推理增量 -> 工具调用 -> 最终答案"""

    def __init__(self, explode=False):
        self.explode = explode
        self.last_reset = None

    async def run(self, query, stream=False, reset=True):
        self.last_reset = reset
        assert stream, "流式路径必须用 stream=True"

        async def gen():
            yield ChatMessageStreamDelta(content="Let me ")
            yield ChatMessageStreamDelta(content="check.")
            t = Timing(start_time=0.0)
            t.end_time = 1.5
            yield ActionStep(
                step_number=1, timing=t,
                tool_calls=[ToolCall(name="canvas_list_courses", arguments={}, id="c1")],
                observations="19 courses",
            )
            if self.explode:
                raise RuntimeError("boom")
            yield FinalAnswerStep(output="You have 19 courses.")

        return gen()


async def collect(agent, query="q", reset=True):
    from src.agent_stream import stream_events
    return [e async for e in stream_events(agent, query, reset=reset)]


async def main_async():
    # 1. 事件分类：推理逐 token，工具推状态，答案单事件
    events = await collect(StubStreamAgent())
    types = [e["type"] for e in events]
    check("事件序列分类正确",
          types == ["token", "token", "tool_call", "step", "answer",
                    "answer_chunk", "done"],
          f"实际 {types}")

    tokens = [e["text"] for e in events if e["type"] == "token"]
    check("推理文本逐 token 流出", tokens == ["Let me ", "check."], f"实际 {tokens}")
    check("工具调用推的是名字不是参数片段",
          [e for e in events if e["type"] == "tool_call"][0]["tool"] == "canvas_list_courses")
    check("最终答案是单个 answer 事件",
          [e for e in events if e["type"] == "answer"][0]["text"] == "You have 19 courses.")
    check("step 事件带工具序列与耗时",
          [e for e in events if e["type"] == "step"][0]["duration_s"] == 1.5)

    # answer_chunk 是渲染节奏，拼接必须无损还原 answer
    full = [e for e in events if e["type"] == "answer"][0]["text"]
    joined = "".join(e["text"] for e in events if e["type"] == "answer_chunk")
    check("answer_chunk 拼接无损还原 answer", joined == full, f"{joined!r} != {full!r}")

    # 2. 出错也要收尾：error 之后仍然有 done，否则客户端永远挂着
    err_events = await collect(StubStreamAgent(explode=True))
    err_types = [e["type"] for e in err_events]
    check("流中途出错时发 error 并仍然收尾 done",
          "error" in err_types and err_types[-1] == "done", f"实际 {err_types}")

    # 3. reset 语义透传
    a = StubStreamAgent()
    await collect(a, reset=False)
    check("reset 透传给 agent.run", a.last_reset is False)


def main() -> int:
    print("\n流式事件与传输验证\n")
    asyncio.run(main_async())

    # 4. 两个传输产出同一套事件序列
    from fastapi.testclient import TestClient
    import api_server as api

    api.sessions.ready = True
    api.sessions._build_agent = lambda: StubStreamAgent()

    with TestClient(api.app) as client:
        # SSE
        with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as r:
            check("SSE 响应 content-type 正确",
                  r.headers["content-type"].startswith("text/event-stream"),
                  r.headers.get("content-type"))
            raw = "".join(chunk for chunk in r.iter_text())
        sse_types = [line[len("event: "):] for line in raw.splitlines()
                     if line.startswith("event: ")]
        sse_payloads = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                        if line.startswith("data: ")]
        check("SSE 事件序列完整",
              sse_types == ["session", "token", "token", "tool_call", "step",
                            "answer", "answer_chunk", "done"],
              f"实际 {sse_types}")
        check("SSE 首帧给出 session_id 供 follow-up 使用",
              "session_id" in sse_payloads[0])

        # WebSocket
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hi"})
            ws_events = []
            while True:
                ev = ws.receive_json()
                ws_events.append(ev)
                if ev["type"] == "done":
                    break
        ws_types = [e["type"] for e in ws_events]
        check("WebSocket 事件序列与 SSE 一致（传输可替换）",
              ws_types == sse_types, f"WS={ws_types}")

    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} 通过\n")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
