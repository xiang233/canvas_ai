"""
冒烟测试：确认模型注册表按 LLM_PROVIDER 正确装配，并真实打通一次 chat 调用。

用法：
    python test_model_registry.py            # 只列出注册的模型别名
    python test_model_registry.py --call     # 额外发一次真实请求验证 endpoint/key 可用
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv(verbose=False)

from src.logger import logger
from src.models import model_manager


def main() -> int:
    provider = os.getenv("LLM_PROVIDER", "auto")
    target = os.getenv("AGENT_MODEL_ID", "gpt-4.1-mini")

    print(f"LLM_PROVIDER   = {provider}")
    print(f"AGENT_MODEL_ID = {target}")
    print(f"AZURE_ENDPOINT = {os.getenv('AZURE_OPENAI_ENDPOINT', '(unset)')}")
    print(f"AZURE_DEPLOYS  = {os.getenv('AZURE_OPENAI_DEPLOYMENTS', '(unset)')}")
    print("-" * 60)

    logger.init_logger("workdir/test_model_registry/log.txt")
    model_manager.init_models()

    aliases = model_manager.list_models()
    print(f"已注册 {len(aliases)} 个别名:")
    for alias in aliases:
        print(f"  - {alias}")
    print("-" * 60)

    if target not in aliases:
        print(f"[FAIL] AGENT_MODEL_ID='{target}' 不在注册表里。")
        print("       Azure 用户请确认 AZURE_OPENAI_DEPLOYMENTS 里包含这个 deployment name。")
        return 1

    print(f"[OK] '{target}' 已注册 -> {model_manager.get_model(target)}")

    if "--call" in sys.argv:
        print("-" * 60)
        print("发一次真实请求...")

        async def _ping():
            from src.models.base import ChatMessage, MessageRole

            model = model_manager.get_model(target)
            resp = await model(
                [ChatMessage(role=MessageRole.USER, content="Reply with exactly: pong")]
            )
            return resp

        try:
            resp = asyncio.run(_ping())
            print(f"[OK] 模型返回: {resp.content!r}")
            if getattr(resp, "token_usage", None):
                print(f"     token_usage: {resp.token_usage}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] 调用失败: {type(exc).__name__}: {exc}")
            return 1

    print("-" * 60)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
