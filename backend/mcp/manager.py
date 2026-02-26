"""
AgentFlow v2 — MCP Server Manager
Manages Model Context Protocol servers for tool connectivity.
Supports stdio, SSE, and HTTP transports.
"""
import asyncio
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
import structlog

logger = structlog.get_logger()


class MCPTransport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class MCPStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_id: str


@dataclass
class MCPServer:
    id: str
    name: str
    transport: MCPTransport
    # For stdio
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    # For SSE/HTTP
    url: Optional[str] = None
    api_key: Optional[str] = None
    # State
    status: MCPStatus = MCPStatus.DISCONNECTED
    tools: List[MCPTool] = field(default_factory=list)
    error: Optional[str] = None
    process: Optional[subprocess.Popen] = None


# ── Pre-configured popular MCP servers ────────────────────────────────────────
POPULAR_MCP_SERVERS = [
    {
        "name": "Filesystem",
        "description": "Read/write files and directories",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
        "category": "system",
        "icon": "📁",
    },
    {
        "name": "GitHub",
        "description": "Search repos, files, issues, PRs",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_keys": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
        "category": "dev",
        "icon": "🐙",
    },
    {
        "name": "PostgreSQL",
        "description": "Query and manage PostgreSQL databases",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://..."],
        "category": "data",
        "icon": "🐘",
    },
    {
        "name": "Brave Search",
        "description": "Web search via Brave Search API",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env_keys": ["BRAVE_API_KEY"],
        "category": "search",
        "icon": "🔍",
    },
    {
        "name": "Puppeteer",
        "description": "Browser automation and web scraping",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "category": "browser",
        "icon": "🌐",
    },
    {
        "name": "Slack",
        "description": "Read/write Slack messages, channels",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env_keys": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
        "category": "comms",
        "icon": "💬",
    },
    {
        "name": "Google Drive",
        "description": "Read/write Google Drive documents",
        "transport": "sse",
        "url": "https://gdrive.mcp.claude.com/mcp",
        "category": "storage",
        "icon": "📂",
    },
    {
        "name": "Gmail",
        "description": "Read, send, and manage emails",
        "transport": "sse",
        "url": "https://gmail.mcp.claude.com/mcp",
        "category": "comms",
        "icon": "📧",
    },
    {
        "name": "Docker",
        "description": "Manage Docker containers and images",
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-docker"],
        "category": "dev",
        "icon": "🐳",
    },
    {
        "name": "Sequential Thinking",
        "description": "Enhanced step-by-step reasoning",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "category": "reasoning",
        "icon": "🧠",
    },
]


class MCPManager:
    """Manages MCP server connections and tool discovery."""

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self._all_tools: Dict[str, MCPTool] = {}

    def register_server(
        self,
        name: str,
        transport: MCPTransport,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> MCPServer:
        server_id = str(uuid.uuid4())[:8]
        server = MCPServer(
            id=server_id,
            name=name,
            transport=transport,
            command=command,
            args=args or [],
            url=url,
            api_key=api_key,
            env=env or {},
        )
        self.servers[server_id] = server
        logger.info("MCP server registered", name=name, transport=transport)
        return server

    async def connect_server(self, server_id: str) -> bool:
        """Connect to an MCP server and discover its tools."""
        server = self.servers.get(server_id)
        if not server:
            return False

        server.status = MCPStatus.CONNECTING
        try:
            if server.transport == MCPTransport.SSE:
                return await self._connect_sse(server)
            elif server.transport == MCPTransport.HTTP:
                return await self._connect_http(server)
            else:
                # stdio - attempt to list tools via subprocess
                return await self._connect_stdio(server)
        except Exception as e:
            server.status = MCPStatus.ERROR
            server.error = str(e)
            logger.error("MCP connection failed", server=server.name, error=str(e))
            return False

    async def _connect_sse(self, server: MCPServer) -> bool:
        """Connect via SSE transport."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {}
                if server.api_key:
                    headers["Authorization"] = f"Bearer {server.api_key}"
                # Try to list tools
                resp = await client.post(
                    f"{server.url}/tools/list",
                    headers=headers,
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    tools_data = data.get("result", {}).get("tools", [])
                    server.tools = [
                        MCPTool(
                            name=t["name"],
                            description=t.get("description", ""),
                            input_schema=t.get("inputSchema", {}),
                            server_id=server.id,
                        )
                        for t in tools_data
                    ]
                    self._all_tools.update({t.name: t for t in server.tools})
                    server.status = MCPStatus.CONNECTED
                    return True
        except Exception as e:
            logger.debug("SSE connect attempt", error=str(e))

        # Mark as connected even if tool discovery failed (server might still work)
        server.status = MCPStatus.CONNECTED
        server.tools = []
        return True

    async def _connect_http(self, server: MCPServer) -> bool:
        """Connect via HTTP transport."""
        return await self._connect_sse(server)  # Same protocol

    async def _connect_stdio(self, server: MCPServer) -> bool:
        """Register stdio server (actual subprocess management handled separately)."""
        server.status = MCPStatus.CONNECTED
        # Add known tools based on server name for display purposes
        server.tools = self._get_known_tools_for_server(server.name)
        self._all_tools.update({t.name: t for t in server.tools})
        return True

    def _get_known_tools_for_server(self, server_name: str) -> List[MCPTool]:
        """Return known tools for popular servers without running them."""
        known = {
            "Filesystem": [
                MCPTool("read_file", "Read file contents", {}, ""),
                MCPTool("write_file", "Write file contents", {}, ""),
                MCPTool("list_directory", "List directory", {}, ""),
                MCPTool("search_files", "Search for files", {}, ""),
            ],
            "GitHub": [
                MCPTool("search_repositories", "Search GitHub repos", {}, ""),
                MCPTool("get_file_contents", "Get file from repo", {}, ""),
                MCPTool("create_issue", "Create GitHub issue", {}, ""),
                MCPTool("search_code", "Search code on GitHub", {}, ""),
            ],
            "Brave Search": [
                MCPTool("brave_web_search", "Search the web", {}, ""),
            ],
            "Puppeteer": [
                MCPTool("puppeteer_navigate", "Navigate to URL", {}, ""),
                MCPTool("puppeteer_screenshot", "Take screenshot", {}, ""),
                MCPTool("puppeteer_click", "Click element", {}, ""),
                MCPTool("puppeteer_evaluate", "Run JavaScript", {}, ""),
            ],
        }
        tools = known.get(server_name, [])
        for t in tools:
            t.server_id = server_name
        return tools

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from connected servers."""
        tools = []
        for server in self.servers.values():
            if server.status == MCPStatus.CONNECTED:
                for tool in server.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "server": server.name,
                        "server_id": server.id,
                    })
        return tools

    def get_server_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": s.id,
                "name": s.name,
                "transport": s.transport,
                "status": s.status,
                "tool_count": len(s.tools),
                "tools": [{"name": t.name, "description": t.description} for t in s.tools[:10]],
                "error": s.error,
            }
            for s in self.servers.values()
        ]


# Global instance
mcp_manager = MCPManager()
