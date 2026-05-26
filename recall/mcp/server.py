"""Recall MCP server entry point (stub).

Exposes a minimal FastMCP server over stdio so MCP-aware clients (Claude Code,
Cursor, etc.) can connect during early development. Real tool surface lands in
later phases.
"""

from fastmcp import FastMCP

mcp: FastMCP = FastMCP("recall")


@mcp.tool
def ping() -> str:
    """Health check tool. Returns 'pong'."""
    return "pong"


def main() -> None:
    """Entry point for the ``recall-mcp`` console script."""
    mcp.run()
