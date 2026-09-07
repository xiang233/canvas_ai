"""
把 agent 的流式产出翻译成协议无关的事件。

为什么需要这一层：ReAct 循环里吐出来的东西不是同一种性质的内容。
模型的推理文本是逐 token 产生的、适合流给用户看；工具调用和 observation
是给 agent 看的中间状态，逐字流出去没有意义，但也不能静默——用户盯着
空白屏幕等三十秒的体验很差，所以要推进度事件。

一个诚实的限制，实测确认过：这个 agent 每一步的模型输出都是工具调用，
不是自由文本。同一个模型同一个问题，带 tools_to_call_from 时 content
增量为 0、tool_call 增量为 8；不带工具时 content 增量为 72。也就是说
ReAct + 强制工具调用的形态下，"逐 token 的推理文本"实际并不存在，
最终答案是作为 final_answer_tool 的参数一次性到达的。

所以这里流的是**步骤级进度**而不是 token 级推理：工具调用一发生就推
事件，用户在 4 秒和 6 秒看到"正在查课程"、"正在作答"，而不是干等
8 秒。answer 事件到达后再按词切块推 answer_chunk，让前端可以做打字机
效果——但那是渲染节奏，不是模型生成节奏，两者不能混为一谈。

若模型某步真的产出了自由文本（例如关掉工具约束），token 事件会照常
逐个发出，这条路径是通的。

事件类型：
    token         模型自由文本增量（本 agent 形态下通常不出现，见上）
    tool_call     开始调用某个工具
    step          一步结束，带上工具序列与耗时
    answer        最终答案（完整文本，一次性）
    answer_chunk  答案分块，供前端做打字机渲染
    error         出错
    done          流结束

这一层不认识 SSE 也不认识 WebSocket：api_server 里两个传输各自套壳。
"""

import inspect
from typing import Any, AsyncGenerator, Dict

from src.memory import ActionStep, FinalAnswerStep, PlanningStep
from src.models.base import ChatMessageStreamDelta


def _tool_names(step: ActionStep) -> list:
    return [tc.name for tc in (step.tool_calls or [])]


async def stream_events(agent: Any, query: str, reset: bool = True) -> AsyncGenerator[Dict[str, Any], None]:
    """跑一次 agent，产出协议无关的事件字典。

    reset 语义与 agent.run 一致：会话首轮清空记忆，后续保留上下文。
    """
    seen_tools: set = set()
    try:
        # run(stream=True) 直接返回异步生成器，不是 coroutine：不能 await。
        # 但桩 agent 若把 run 写成 async def 就需要 await —— 两种都支持，
        # 否则测试用的桩和真实 agent 会走出不同的行为
        stream = agent.run(query, stream=True, reset=reset)
        if inspect.isawaitable(stream):
            stream = await stream
        async for item in stream:
            if isinstance(item, ChatMessageStreamDelta):
                # 推理文本增量。tool_calls 增量不外发：参数拼到一半的
                # JSON 片段对前端没用，等 ActionStep 里拿完整的工具名
                if item.content:
                    yield {"type": "token", "text": item.content}

            elif isinstance(item, ActionStep):
                for name in _tool_names(item):
                    if name not in seen_tools:
                        seen_tools.add(name)
                        yield {"type": "tool_call", "tool": name}
                if item.error:
                    yield {"type": "error", "message": str(item.error)}
                yield {
                    "type": "step",
                    "step": item.step_number,
                    "tools": _tool_names(item),
                    "duration_s": round(item.timing.duration or 0.0, 2)
                    if item.timing else None,
                }

            elif isinstance(item, PlanningStep):
                yield {"type": "planning"}

            elif isinstance(item, FinalAnswerStep):
                text = _as_text(item.output)
                yield {"type": "answer", "text": text}
                # 分块推送供前端渲染。按空白切分保留原文，客户端直接拼接
                # 即可还原——这是渲染节奏，模型早已把整段答案一次性给出
                for chunk in _chunks(text):
                    yield {"type": "answer_chunk", "text": chunk}

    except Exception as exc:  # 流已经开始后出错，只能作为事件发出去
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        yield {"type": "done"}


def _chunks(text: str, size: int = 4):
    """按词切块。size 是每块的词数，不是字符数——切在词边界上，
    客户端拼接后与原文完全一致"""
    words = text.split(" ")
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])
        yield chunk if i + size >= len(words) else chunk + " "


def _as_text(output: Any) -> str:
    if output is None:
        return ""
    text = getattr(output, "output", None)
    return str(text if text is not None else output)
