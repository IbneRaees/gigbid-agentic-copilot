"""Unified Async MCP Client Manager using `langchain-mcp-adapters`.

Connects LangGraph agent nodes to GigBid's three internal MCP servers
(`portfolio`, `marketplace`, `workspace`) plus the optional external GitHub MCP
server (when `GITHUB_PERSONAL_ACCESS_TOKEN` is configured for repo ingestion).

Features:
  - Propagates `Authorization: Bearer ...`, `X-Trace-Id`, and `X-User-Id` headers
    across all Streamable HTTP MCP sessions for end-to-end observability & auth.
  - Exposes LangChain `BaseTool` instances filtered by server or `readOnlyHint`
    so analytical nodes strictly follow least-privilege tool allowlists.
  - Clean async lifecycle management (`async with GigBidMCPClientManager(...)`).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


@dataclass(slots=True)
class MCPServerEndpoint:
    """Connection descriptor for an internal or external MCP server."""

    name: str
    transport: str
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    def to_adapter_config(self) -> dict[str, Any]:
        if self.transport == "stdio":
            cfg: dict[str, Any] = {
                "transport": "stdio",
                "command": self.command,
                "args": self.args,
            }
            if self.env:
                cfg["env"] = self.env
            return cfg

        return {
            "transport": "streamable_http",
            "url": self.url,
            "headers": dict(self.headers),
        }


class GigBidMCPClientManager:
    """Lifecycle-managed multi-server MCP client for LangGraph nodes and RAG ingestion."""

    def __init__(
        self,
        *,
        portfolio_url: str | None = None,
        marketplace_url: str | None = None,
        workspace_url: str | None = None,
        auth_token: str | None = None,
        github_token: str | None = None,
        include_github: bool = True,
        trace_id: str | None = None,
        user_id: str = "persona_aiml_01",
    ) -> None:
        self.portfolio_url = portfolio_url or os.getenv(
            "PORTFOLIO_MCP_URL", "http://localhost:8091/mcp"
        )
        self.marketplace_url = marketplace_url or os.getenv(
            "MARKETPLACE_MCP_URL", "http://localhost:8092/mcp"
        )
        self.workspace_url = workspace_url or os.getenv(
            "WORKSPACE_MCP_URL", "http://localhost:8093/mcp"
        )
        self.auth_token = (
            auth_token
            or os.getenv("MCP_SERVER_TOKEN")
            or os.getenv("GIGSIM_API_TOKEN", "gigsim-dev-token-2026")
        )
        self.github_token = (
            github_token
            if github_token is not None
            else os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        ).strip()
        self.include_github = include_github and bool(self.github_token)
        self.trace_id = trace_id or f"trace-{uuid.uuid4().hex[:12]}"
        self.user_id = user_id
        self._client: MultiServerMCPClient | None = None

    def build_server_connections(
        self,
        *,
        trace_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Build `MultiServerMCPClient` connection map with propagated auth and trace headers."""
        active_trace = trace_id or self.trace_id
        active_user = user_id or self.user_id
        common_headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "X-Trace-Id": active_trace,
            "X-User-Id": active_user,
        }

        endpoints: list[MCPServerEndpoint] = [
            MCPServerEndpoint(
                name="portfolio",
                transport="streamable_http",
                url=self.portfolio_url,
                headers=common_headers,
            ),
            MCPServerEndpoint(
                name="marketplace",
                transport="streamable_http",
                url=self.marketplace_url,
                headers=common_headers,
            ),
            MCPServerEndpoint(
                name="workspace",
                transport="streamable_http",
                url=self.workspace_url,
                headers=common_headers,
            ),
        ]

        if self.include_github:
            endpoints.append(
                MCPServerEndpoint(
                    name="github",
                    transport="stdio",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-github"],
                    env={"GITHUB_PERSONAL_ACCESS_TOKEN": self.github_token},
                )
            )

        return {ep.name: ep.to_adapter_config() for ep in endpoints}

    def _get_or_create_client(
        self,
        *,
        trace_id: str | None = None,
        user_id: str | None = None,
        server_filter: set[str] | None = None,
    ) -> MultiServerMCPClient:
        connections = self.build_server_connections(trace_id=trace_id, user_id=user_id)
        if server_filter is not None:
            connections = {k: v for k, v in connections.items() if k in server_filter}
        client = MultiServerMCPClient(connections)
        self._client = client
        return client

    async def __aenter__(self) -> GigBidMCPClientManager:
        self._get_or_create_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Cleanly release any underlying client references."""
        if self._client is not None:
            aclose = getattr(self._client, "aclose", None)
            if callable(aclose):
                await aclose()
            self._client = None

    async def get_all_tools(
        self,
        *,
        trace_id: str | None = None,
        user_id: str | None = None,
    ) -> list[BaseTool]:
        """Load LangChain-compatible `BaseTool` instances across all configured MCP servers."""
        client = self._get_or_create_client(trace_id=trace_id, user_id=user_id)
        return await client.get_tools()

    async def get_tools_by_server(
        self,
        server_name: str,
        *,
        trace_id: str | None = None,
        user_id: str | None = None,
    ) -> list[BaseTool]:
        """Load LangChain tools scoped to a single MCP server (`portfolio`, `marketplace`, `workspace`, `github`)."""
        client = self._get_or_create_client(
            trace_id=trace_id,
            user_id=user_id,
            server_filter={server_name},
        )
        return await client.get_tools()

    async def get_read_only_tools(
        self,
        *,
        server_name: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
    ) -> list[BaseTool]:
        """Return only tools marked `readOnlyHint=True` (least-privilege allowlist for analysis nodes)."""
        tools = (
            await self.get_tools_by_server(server_name, trace_id=trace_id, user_id=user_id)
            if server_name
            else await self.get_all_tools(trace_id=trace_id, user_id=user_id)
        )
        read_only: list[BaseTool] = []
        for tool in tools:
            meta = getattr(tool, "metadata", None) or {}
            annotations = meta.get("annotations") if isinstance(meta, dict) else None
            if isinstance(annotations, dict):
                is_ro = annotations.get("readOnlyHint") or annotations.get("read_only_hint")
            else:
                is_ro = getattr(annotations, "readOnlyHint", None) or getattr(
                    annotations, "read_only_hint", None
                )
            if is_ro is True:
                read_only.append(tool)
        return read_only

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        trace_id: str | None = None,
        user_id: str | None = None,
    ) -> Any:
        """Invoke a named MCP tool on a target server with trace propagation."""
        tools = await self.get_tools_by_server(server_name, trace_id=trace_id, user_id=user_id)
        target = next((t for t in tools if t.name == tool_name), None)
        if target is None:
            available = [t.name for t in tools]
            raise KeyError(
                f"Tool '{tool_name}' not found on MCP server '{server_name}'. "
                f"Available tools: {available}"
            )
        raw_output = await target.ainvoke(arguments)
        if isinstance(raw_output, str):
            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                return raw_output
        return raw_output


__all__ = ["MCPServerEndpoint", "GigBidMCPClientManager"]
