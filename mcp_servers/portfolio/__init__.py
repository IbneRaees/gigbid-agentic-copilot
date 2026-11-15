"""Strictly Read-Only Portfolio MCP Server package (`port 8091`)."""

from mcp_servers.portfolio.server import build_portfolio_mcp_server

__all__ = ["build_portfolio_mcp_server"]
