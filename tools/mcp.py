from langchain_mcp_adapters.client import MultiServerMCPClient

class MCPClientManager:
    def __init__(self):
        self.client = MultiServerMCPClient(
            {
                "financial-datasets": {
                    "command": "uv",
                    "args": [
                        "--directory",
                        "C://Users//admin//Documents//campusX//langGraph//mcp-server",
                        "run",
                        "server.py"
                    ],
                    "transport": "stdio",
                }
            }
        )

    async def load_tools(self):
        print("Loading tools...")
        tools = await self.client.get_tools()
        print(f"Loaded {len(tools)} tools.")
        return tools

# Usage in another file:
# from tools.mcp import MCPClientManager
# mcp_manager = MCPClientManager()
# tools = asyncio.run(mcp_manager.load_tools())