"""
Canvas MCP Server：把本项目的只读 Canvas 工具暴露为一个 MCP server，
供 Claude Desktop / Claude Code 等 MCP 客户端直接调用。

工具集与 agent 完全同源：同一份 configs/canvas_agent_config.py、同一个
read_only_tools() 副作用白名单 —— 写工具在协议边界同样进不来，
tests/test_prompt_consistency.py 有断言守着这一点。

用法：
    python mcp_server.py            # stdio，给 Claude Desktop 用

Claude Desktop 配置（claude_desktop_config.json）：
    {
      "mcpServers": {
        "canvas": {
          "command": "python",
          "args": ["/绝对路径/mcp_server.py"],
          "env": {"CANVAS_URL": "...", "CANVAS_ACCESS_TOKEN": "..."}
        }
      }
    }

依赖 mcp>=2.0（在 requirements.txt 里，不进 requirements-ci.txt：
CI 的一致性测试在 mcp 未安装时对相关断言做 SKIP）。
"""

import inspect
import os

from dotenv import load_dotenv

load_dotenv()

from mcp.server.mcpserver import MCPServer

from configs.canvas_agent_config import agent_config


def make_handler(tool):
    """把 AsyncTool.forward 包成 MCP 工具处理函数。

    schema 由 forward 的类型注解推导（SDK 行为），所以这里必须透传
    绑定方法的签名。ToolResult 拆包：error 抛异常（MCP 侧显示为
    工具错误），正常时只返回 output 文本。
    """

    async def handler(**kwargs):
        result = await tool.forward(**kwargs)
        if getattr(result, "error", None):
            raise RuntimeError(result.error)
        return result.output

    handler.__name__ = tool.name
    # 剥掉 -> ToolResult 返回注解：handler 拆包后返回的是 output 本体，
    # 留着注解会触发 SDK 的 structured-output 校验并报类型不符
    sig = inspect.signature(tool.forward)
    handler.__signature__ = sig.replace(return_annotation=inspect.Signature.empty)
    return handler


def build_server() -> MCPServer:
    server = MCPServer(
        name="canvas-student",
        description="Read-only Canvas LMS access for a student account",
        instructions=(
            "Query the student's Canvas: courses, grades, assignments, files, "
            "and semantic search over uploaded course materials "
            "(vector_store_list then vector_store_search for content questions)."
        ),
    )
    # agent_config["tools"] 已经过 read_only_tools() 白名单过滤，
    # 这里不重复过滤 —— 单一事实来源，CI 断言盯的也是这一份
    for tool in agent_config["tools"]:
        server.add_tool(
            make_handler(tool),
            name=tool.name,
            description=tool.description,
        )
    return server


if __name__ == "__main__":
    missing = [k for k in ("CANVAS_URL", "CANVAS_ACCESS_TOKEN") if not os.getenv(k)]
    if missing:
        raise SystemExit(f"缺少环境变量: {missing}")
    build_server().run(transport="stdio")
