"""
验证 system prompt 与实际配置的一致性。

这类失效在本仓库真实发生过三次，且现有测试全部拦不住（它们测 Python
逻辑，不测 prompt 文本）：
  1. prompt 的 few-shot 示例引用了不存在的工具
     （document_qa / image_generator / python_interpreter / search）
  2. final answer 工具命名不一致（final_answer vs final_answer_tool）
  3. README 声称的工具数与配置实际启用数不符（26 vs 23）

纯本地、无网络、无 API key。

用法：
    python -m tests.test_prompt_consistency        # 从仓库根目录运行
"""

import os
import re
from pathlib import Path

# 工具基类构造时要求有 token，给假的即可（不会发任何请求）
os.environ.setdefault("CANVAS_ACCESS_TOKEN", "dummy-token-for-test")

REPO = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO / "src/agent/general_agent/prompts/general_agent.yaml"

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, condition))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")
    if not condition and detail:
        print(f"        {detail}")


def main() -> int:
    import yaml

    from configs.canvas_agent_config import agent_config

    prompts = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    system_prompt = prompts["system_prompt"]
    enabled = {t.name for t in agent_config["tools"]}
    allowed = enabled | {"final_answer_tool"}

    print("\nprompt 一致性验证\n")

    # 1. JSON 示例引用的工具必须真实启用
    referenced = set(re.findall(r'"name":\s*"([a-z_]+)"', system_prompt))
    phantom = referenced - allowed
    check(
        "prompt 示例只引用已启用的工具",
        not phantom,
        f"幻影工具: {sorted(phantom)}；已启用: {sorted(enabled)}",
    )

    # 2. 被关闭的写工具不得出现在 prompt 里
    write_tools = {"canvas_submit_assignment", "canvas_post_discussion"}
    leaked = {w for w in write_tools if w in system_prompt}
    check("prompt 不引用被关闭的写工具", not leaked, f"出现了: {sorted(leaked)}")

    # 3. final answer 命名统一：不允许裸 "name": "final_answer"
    bare = re.findall(r'"name":\s*"final_answer"', system_prompt)
    check(
        'final answer 统一叫 final_answer_tool（无裸 "final_answer"）',
        not bare,
        f"出现 {len(bare)} 处旧名",
    )

    # 4. README 声称的工具数与配置一致
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"(\d+) read-only", readme)
    claimed = int(m.group(1)) if m else None
    check(
        f"README 工具数与配置一致（配置启用 {len(enabled)} 个）",
        claimed == len(enabled),
        f"README 声称 {claimed}",
    )

    # 5. unused_fork_examples 是惰性键：存在于 yaml，但不被任何代码读取
    has_key = "unused_fork_examples" in prompts
    readers = []
    for py in REPO.glob("src/**/*.py"):
        if "unused_fork_examples" in py.read_text(encoding="utf-8", errors="ignore"):
            readers.append(str(py.relative_to(REPO)))
    check(
        "unused_fork_examples 存在且无代码读取（保持惰性）",
        has_key and not readers,
        f"存在: {has_key}；读取者: {readers}",
    )

    # 6. 只读是机制不是约定：构建出的工具全部 side_effect == "read"
    non_read = [t.name for t in agent_config["tools"]
                if getattr(t, "side_effect", "read") != "read"]
    check("构建出的 agent 工具全部只读", not non_read, f"漏进来的: {non_read}")

    # 7. 过滤器真的在拦：模拟一个被"取消注释"的写工具，必须被丢弃
    from configs.canvas_agent_config import read_only_tools

    class FakeWriteTool:
        name = "canvas_submit_assignment"
        side_effect = "write"

    survived = read_only_tools([FakeWriteTool()])
    check("写工具经过构建过滤后被丢弃（取消注释也进不来）", survived == [],
          f"过滤后仍存活: {[t.name for t in survived]}")

    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} 通过\n")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
