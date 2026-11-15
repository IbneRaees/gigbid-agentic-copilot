"""GigBid Model Context Protocol (MCP) Server Suite.

Exposes three dedicated MCP servers over Streamable HTTP / SSE:
  - `mcp_servers.portfolio.server` (Port 8091, strictly read-only)
  - `mcp_servers.marketplace.server` (Port 8092, GigSim & Upwork Live adapters)
  - `mcp_servers.workspace.server` (Port 8093, Calendar & Email drafts)
"""

__all__ = ["security"]