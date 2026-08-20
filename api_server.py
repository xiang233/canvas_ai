"""
Canvas Agent HTTP/WebSocket API。

与 ws_server.py 的区别是会话模型：这里每个 session 拥有自己的 agent 实例，
所以多个用户可以并发使用而互不干扰，且每个会话有独立的多轮上下文。
ws_server.py 用的是进程级单例 + 单连接锁（串行化），保留不动。

设计取自黑客松期间队友 (deyu) 的 FastAPI 原型，修掉了其中几处问题：
  - WebSocket 端点原本走的是全局 agent，与 SessionManager 自相矛盾
  - 会话只增不减，每个会话持有一个 agent 实例，是内存泄漏
  - /api/login 只校验用户名不校验任何凭证，是“看着像鉴权”的陷阱
  - allow_origins=["*"] 配 allow_credentials=True 是浏览器直接拒绝的组合

用法：
    python api_server.py                  # 默认 127.0.0.1:8000
    ALLOWED_ORIGINS=https://example.com python api_server.py

交互式 API 文档在 /docs。
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from configs.canvas_agent_config import agent_config
from src.agent_stream import stream_events
from src.models import model_manager
from src.registry import AGENT
from src.tools import ToolResult

# 空闲多久回收会话。每个会话持有一个 agent 实例，不回收就是内存泄漏
SESSION_TTL_SECONDS = int(os.getenv("API_SESSION_TTL", "1800"))
MAX_SESSIONS = int(os.getenv("API_MAX_SESSIONS", "50"))


@dataclass
class Session:
    session_id: str
    agent: Any
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    message_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        self.last_active = time.time()


class SessionManager:
    """每个会话一个 agent 实例。

    这是与 ws_server 单例模型的关键差别：并发用户互不干扰，每个会话
    有独立的多轮上下文（首条消息 reset，之后保留，靠
    AGENT_MEMORY_KEEP_RECENT 压缩旧步骤控制 prompt 增长）。
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self.ready = False

    def initialize(self) -> None:
        model_manager.init_models(use_local_proxy=False)
        self.ready = True

    def _build_agent(self) -> Any:
        return AGENT.build(dict(
            type=agent_config["type"],
            config=agent_config,
            model=model_manager.registed_models[agent_config["model_id"]],
            # 已过 read_only_tools() 白名单过滤，写工具进不来
            tools=agent_config["tools"],
            max_steps=agent_config["max_steps"],
            planning_interval=agent_config.get("planning_interval"),
            name=agent_config.get("name"),
            description=agent_config.get("description"),
        ))

    def _evict_expired(self) -> int:
        now = time.time()
        stale = [sid for sid, s in self._sessions.items()
                 if now - s.last_active > SESSION_TTL_SECONDS]
        for sid in stale:
            self._sessions.pop(sid, None)
        return len(stale)

    async def get_or_create(self, session_id: Optional[str] = None) -> Session:
        if not self.ready:
            raise HTTPException(503, "Agent backend not initialized.")

        async with self._lock:
            self._evict_expired()

            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            if len(self._sessions) >= MAX_SESSIONS:
                # 满了就淘汰最久未活动的，而不是拒绝服务
                oldest = min(self._sessions.values(), key=lambda s: s.last_active)
                self._sessions.pop(oldest.session_id, None)

            new_id = session_id or str(uuid.uuid4())
            session = Session(session_id=new_id, agent=self._build_agent())
            self._sessions[new_id] = session
            return session

    async def ask(self, session: Session, query: str) -> str:
        """在会话自己的 agent 上执行一次查询。

        同一会话内串行（每会话一把锁）：并发请求在同一个 agent 的记忆上
        交错读写会互相踩。不同会话之间完全并行。
        """
        async with session.lock:
            # 首条消息清空记忆，之后保留，这样 follow-up 能引用上文
            reset = session.message_count == 0
            result = await session.agent.run(query, reset=reset)
            session.message_count += 1
            session.touch()

        if isinstance(result, ToolResult):
            return str(result.output) if result.output is not None else ""
        return str(result)

    async def ask_stream(self, session: Session, query: str):
        """流式版的 ask。

        锁必须覆盖整个生成器的生命周期，不能只包住启动：客户端读到
        一半时另一个请求进来，会在同一份记忆上交错读写。

        finally 里无条件推进 message_count，即使客户端中途断开——
        那一轮的记忆已经写进 agent 了，不认账的话下一轮会带着
        reset=True 把它抹掉，或者基于残缺上下文继续。
        """
        async with session.lock:
            reset = session.message_count == 0
            try:
                async for event in stream_events(session.agent, query, reset=reset):
                    yield event
            finally:
                session.message_count += 1
                session.touch()

    def info(self, session_id: str) -> Optional[Dict[str, Any]]:
        s = self._sessions.get(session_id)
        if not s:
            return None
        return {
            "session_id": s.session_id,
            "created_at": s.created_at,
            "last_active": s.last_active,
            "message_count": s.message_count,
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [self.info(sid) for sid in list(self._sessions)]

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    @property
    def active_count(self) -> int:
        return len(self._sessions)


sessions = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [k for k in ("CANVAS_URL", "CANVAS_ACCESS_TOKEN") if not os.getenv(k)]
    if missing:
        print(f"[api_server] 缺少环境变量 {missing}，/api/chat 将返回 503")
    else:
        try:
            sessions.initialize()
        except Exception as exc:
            # 模型配置不全时降级而不是崩：/api/health 会如实报 degraded，
            # 聊天端点返回 503。CI 里没有 .env，起服务不该失败
            print(f"[api_server] 模型初始化失败，降级运行: {type(exc).__name__}: {exc}")
    yield


app = FastAPI(
    title="Canvas Student Agent API",
    description="Per-session agents over a read-only Canvas tool set.",
    version="1.0.0",
    lifespan=lifespan,
)

# 默认只放行本地前端。用 ALLOWED_ORIGINS 显式配置部署域名。
# 不用 "*" —— 配上 allow_credentials 时浏览器会直接拒绝该组合
_origins = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_count: int


class ToolInfo(BaseModel):
    name: str
    description: str
    side_effect: str


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok" if sessions.ready else "degraded",
        "agent_ready": sessions.ready,
        "active_sessions": sessions.active_count,
        "session_ttl_seconds": SESSION_TTL_SECONDS,
    }


@app.get("/api/tools", response_model=List[ToolInfo])
async def list_tools() -> List[ToolInfo]:
    """当前暴露的工具。全部只读 —— 写工具在构建时就被白名单过滤掉了。"""
    return [
        ToolInfo(
            name=t.name,
            description=t.description,
            side_effect=getattr(t, "side_effect", "read"),
        )
        for t in agent_config["tools"]
    ]


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session = await sessions.get_or_create(request.session_id)
    try:
        answer = await sessions.ask(session, request.message)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    return ChatResponse(
        response=answer,
        session_id=session.session_id,
        message_count=session.message_count,
    )


def _sse_frame(event: Dict[str, Any]) -> str:
    """SSE 帧：event: 行给客户端按类型分发，data: 行是 JSON 负载"""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events 流式回答。

    浏览器端用 EventSource 或 fetch + ReadableStream 消费；
    事件类型见 src/agent_stream.py。
    """
    session = await sessions.get_or_create(request.session_id)

    async def body():
        # 先把 session_id 发出去，客户端后续 follow-up 要带上它
        yield _sse_frame({"type": "session", "session_id": session.session_id})
        async for event in sessions.ask_stream(session, request.message):
            yield _sse_frame(event)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx 之类的反代默认会缓冲，缓冲了就不是流式了
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/session/{session_id}")
async def session_info(session_id: str) -> Dict[str, Any]:
    info = sessions.info(session_id)
    if not info:
        raise HTTPException(404, "Unknown session.")
    return info


@app.get("/api/sessions")
async def list_sessions() -> List[Dict[str, Any]]:
    return sessions.list_sessions()


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> Dict[str, bool]:
    return {"deleted": await sessions.delete(session_id)}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """流式聊天。走的是和 REST 相同的会话模型，不是另一个全局 agent。

    每条消息只发给它自己的连接 —— 不做广播，否则一个用户的成绩会
    发给所有在线连接。
    """
    await websocket.accept()
    session: Optional[Session] = None
    try:
        while True:
            payload = await websocket.receive_json()
            query = (payload or {}).get("message", "")
            if not isinstance(query, str) or not query.strip():
                await websocket.send_json({"type": "error", "message": "message 不能为空"})
                continue

            if session is None or payload.get("session_id") not in (None, session.session_id):
                session = await sessions.get_or_create(payload.get("session_id"))

            await websocket.send_json({"type": "session", "session_id": session.session_id})
            # 与 SSE 完全相同的事件序列，只是换了传输。
            # 事件的抽象在 agent_stream 里，协议适配在这里
            try:
                async for event in sessions.ask_stream(session, query):
                    await websocket.send_json(event)
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
    )
