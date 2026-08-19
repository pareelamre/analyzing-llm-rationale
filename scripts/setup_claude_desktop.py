#!/usr/bin/env python3
"""Automated Claude Desktop & Cursor MCP Configurator for Foresea.

Installs or updates the Foresea Model Context Protocol (MCP) server configuration
in Claude Desktop (`claude_desktop_config.json`) and outputs Cursor configuration.

Usage:
    python scripts/setup_claude_desktop.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass



def get_claude_config_path() -> Path:
    """Determine the platform-specific Claude Desktop config path."""
    system = platform.system()
    if system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "Claude" / "claude_desktop_config.json"
        return Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    else:  # Linux / other
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def get_foresea_mcp_config() -> Dict[str, Any]:
    """Return the Foresea MCP server block for Claude Desktop."""
    python_exe = sys.executable
    return {
        "command": python_exe,
        "args": ["-m", "analyzing_llm_rationale.mcp_server", "--transport", "stdio"],
        "env": {
            "FORESEA_API_URL": "https://foresea.ink",
            "PYTHONIOENCODING": "utf-8",
        },
    }


def update_claude_config(dry_run: bool = False) -> bool:
    path = get_claude_config_path()
    print(f"🔍 Target Claude Desktop config: {path}")

    config: Dict[str, Any] = {"mcpServers": {}}
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            if "mcpServers" not in config or not isinstance(config["mcpServers"], dict):
                config["mcpServers"] = {}
        except Exception as e:
            print(f"⚠️ Could not parse existing config ({e}), creating fresh backup.")

    config["mcpServers"]["foresea"] = get_foresea_mcp_config()
    formatted = json.dumps(config, indent=2)

    if dry_run:
        print("\n[DRY RUN] Would write following config to Claude Desktop:")
        print(formatted)
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(formatted, encoding="utf-8")
    print(f"✅ Successfully registered Foresea MCP in {path}")
    print("👉 Restart Claude Desktop to use Foresea prediction market tools.")
    return True


def print_cursor_config() -> None:
    print("\n" + "=" * 60)
    print("🤖 CURSOR AI MCP CONFIGURATION")
    print("=" * 60)
    print("To use Foresea in Cursor AI:")
    print("1. Open Cursor Settings -> Features -> MCP")
    print("2. Add New MCP Server:")
    print("   Name: foresea")
    print("   Type: command (or sse)")
    print(f"   Command: {sys.executable} -m analyzing_llm_rationale.mcp_server --transport stdio")
    print("   (Or Remote SSE URL: https://foresea.ink/mcp/)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure Foresea MCP for Claude Desktop & Cursor")
    parser.add_argument("--dry-run", action="store_true", help="Print config without writing to disk")
    args = parser.parse_args()

    update_claude_config(dry_run=args.dry_run)
    print_cursor_config()


if __name__ == "__main__":
    main()
