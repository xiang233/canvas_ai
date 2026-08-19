# src/mcp/ — fork 遗留，未接入运行时

这个模块来自上游 fork（smolagents 派生），包含 51 个 GAIA 基准工具
（含 `python_interpreter` / `cpp_interpreter` 等）。当前 Canvas agent
的运行时不 import 它（仓库内唯一的引用是模块自己内部的），保留在这里
是作为未来可能的扩展方向，不是活跃代码。

如果要接入，注意两点：

1. `server.py` 用 `exec()` 执行注册表脚本，没有沙箱。接入前先解决
   隔离问题，否则等于给了 agent 任意代码执行。
2. 接入后把 `src/agent/general_agent/prompts/general_agent.yaml` 里
   `unused_fork_examples` 键下的对应示例改回 system_prompt——那些
   few-shot 示例（document_qa / python_interpreter / search）就是为
   这类工具写的，当前因为工具不存在而被移出了活跃 prompt。
