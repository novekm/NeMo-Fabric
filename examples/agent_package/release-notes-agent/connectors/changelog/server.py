# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The changelog connector: a small MCP server started beside the agent.

It serves one tool, list_changes, over stdio. It uses the Python standard library only, so
nothing is installed. It reads changes.json from the folder it is started in. mcp.json in this
folder says how to start it, and the connector's folder is the working directory.
"""

import json
import sys
from pathlib import Path

PROTOCOL_VERSION = "2025-06-18"
TOOL = {
    "name": "list_changes",
    "description": "List the changes since the last release, each with its kind and its text.",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
}


def answer(method: str, params: dict) -> dict:
    """Return the result of one request."""
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "changelog", "version": "1.0.0"},
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [TOOL]}
    if method == "tools/call" and params.get("name") == TOOL["name"]:
        changes = Path("changes.json").read_text(encoding="utf-8")
        return {"content": [{"type": "text", "text": changes}], "isError": False}
    raise LookupError(method)


def main() -> None:
    """Read one JSON-RPC message per line and answer each request."""
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        if "id" not in message:
            continue  # a notification takes no answer
        reply = {"jsonrpc": "2.0", "id": message["id"]}
        try:
            reply["result"] = answer(message.get("method", ""), message.get("params") or {})
        except LookupError:
            reply["error"] = {"code": -32601, "message": "Method not found"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
