"""
验证 api_server 的会话模型。

回归的是这个架构缺陷：ws_server 用进程级单例 agent，所有连接共享同一份
记忆，只能靠单连接锁串行化。api_server 改成每会话一个 agent 实例，
断言的就是「隔离」和「多轮」这两件事同时成立。

用桩 agent 替换真实构建，纯本地、不打网络、不需要 API key。

用法：
    python -m tests.test_api_sessions        # 从仓库根目录运行
"""

import asyncio
import os

os.environ.setdefault("CANVAS_ACCESS_TOKEN", "dummy-token-for-test")
os.environ.setdefault("CANVAS_URL", "https://example.invalid")

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, condition))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")
    if not condition and detail:
        print(f"        {detail}")


class StubAgent:
    """记录自己收到的 query 和 reset 参数，用于断言隔离与多轮"""

    def __init__(self):
        self.seen = []

    async def run(self, query, reset=True):
        self.seen.append((query, reset))
        await asyncio.sleep(0.01)  # 让并发有机会交错
        return f"answer:{query}"


async def main_async(api):
    api.sessions.ready = True
    api.sessions._build_agent = lambda: StubAgent()

    # 1. 不同会话拿到不同的 agent 实例
    a = await api.sessions.get_or_create()
    b = await api.sessions.get_or_create()
    check("不同会话持有不同 agent 实例（隔离）", a.agent is not b.agent,
          f"a={id(a.agent)} b={id(b.agent)}")
    check("会话 id 不同", a.session_id != b.session_id)

    # 2. 同一 session_id 复用同一个 agent
    a2 = await api.sessions.get_or_create(a.session_id)
    check("同一 session_id 复用同一 agent", a2.agent is a.agent)

    # 3. 会话内多轮：首条 reset=True，后续 reset=False
    await api.sessions.ask(a, "q1")
    await api.sessions.ask(a, "q2")
    await api.sessions.ask(a, "q3")
    resets = [r for _, r in a.agent.seen]
    check("首条 reset=True，后续 reset=False（多轮上下文）",
          resets == [True, False, False], f"实际 {resets}")

    # 4. 一个会话的对话不出现在另一个会话的 agent 里
    await api.sessions.ask(b, "other-user-question")
    a_queries = [q for q, _ in a.agent.seen]
    check("会话之间对话不串", "other-user-question" not in a_queries,
          f"a 看到 {a_queries}")

    # 5. 并发：两个会话同时提问互不阻塞、互不污染
    c = await api.sessions.get_or_create()
    d = await api.sessions.get_or_create()
    await asyncio.gather(
        api.sessions.ask(c, "c-q"),
        api.sessions.ask(d, "d-q"),
    )
    check("并发会话各自只看到自己的问题",
          [q for q, _ in c.agent.seen] == ["c-q"]
          and [q for q, _ in d.agent.seen] == ["d-q"])

    # 6. TTL 回收：过期会话被清掉，agent 实例得以释放
    api.SESSION_TTL_SECONDS  # noqa: B018  (读一下确保常量存在)
    old = await api.sessions.get_or_create()
    old.last_active -= api.SESSION_TTL_SECONDS + 10
    before = api.sessions.active_count
    await api.sessions.get_or_create()  # 任何一次创建都会触发回收
    check("空闲超时的会话被回收（不是内存泄漏）",
          api.sessions.info(old.session_id) is None,
          f"回收前 {before} 个，现在 {api.sessions.active_count} 个")

    # 7. 上限保护：不超过 MAX_SESSIONS
    for _ in range(api.MAX_SESSIONS + 5):
        await api.sessions.get_or_create()
    check(f"活动会话数不超过上限 {api.MAX_SESSIONS}",
          api.sessions.active_count <= api.MAX_SESSIONS,
          f"实际 {api.sessions.active_count}")


def main() -> int:
    import api_server as api

    print("\napi_server 会话模型验证\n")
    asyncio.run(main_async(api))

    # 8. 工具端点只暴露只读工具
    from configs.canvas_agent_config import agent_config
    non_read = [t.name for t in agent_config["tools"]
                if getattr(t, "side_effect", "read") != "read"]
    check("API 暴露的工具全部只读", not non_read, f"漏进来: {non_read}")

    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} 通过\n")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
