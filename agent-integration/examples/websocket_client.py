#!/usr/bin/env python3
"""Minimal WebSocket JSON-RPC client for ACloudViewer.

Connects to the JSON-RPC plugin running inside ACloudViewer GUI and
demonstrates: ping, method listing, file loading, scene querying,
and build-info introspection (CPU vs CUDA).

Usage:
    python websocket_client.py                      # ping + list methods
    python websocket_client.py /path/to/scene.ply   # load and list scene
"""

import json
import sys

from websockets.sync.client import connect

URL = "ws://localhost:6001"
_id = 0


def call(ws, method: str, params: dict | None = None) -> dict:
    global _id
    _id += 1
    request = {
        "jsonrpc": "2.0",
        "id": _id,
        "method": method,
        "params": params or {},
    }
    ws.send(json.dumps(request))
    response = json.loads(ws.recv())
    if "error" in response:
        print(f"ERROR: {response['error']}", file=sys.stderr)
    return response.get("result")


def main():
    with connect(URL) as ws:
        # Ping
        pong = call(ws, "ping")
        print(f"Ping: {pong}")

        # List methods
        methods = call(ws, "methods.list")
        print(f"\nAvailable methods ({len(methods)}):")
        for m in methods:
            print(f"  {m['method']:30s} {m['description']}")

        # Load file if provided
        if len(sys.argv) > 1:
            filename = sys.argv[1]
            print(f"\nLoading: {filename}")
            result = call(ws, "open", {"filename": filename, "silent": True})
            print(f"Loaded: {json.dumps(result, indent=2)}")

            # List scene
            entities = call(ws, "scene.list", {"recursive": True})
            print(f"\nScene entities ({len(entities)}):")
            print(json.dumps(entities, indent=2))

            # Show details of first entity
            if entities:
                info = call(ws, "scene.info", {"entity_id": entities[0]["id"]})
                print(f"\nEntity info:")
                print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
