# qJSonRPCPlugin — JSON-RPC automation (WebSocket)

## Introduction

**qJSonRPCPlugin** exposes ACloudViewer to **remote automation**: a **WebSocket** server (default port **6001**) speaks **JSON-RPC 2.0** so external tools can open files, drive the database tree, run processing, capture views, and more. Typical use includes AI agent integration, scripted QA, and custom operator panels.

**Headless note:** There are **no** extra `-SILENT` flags for this plugin; control is **JSON-RPC** while the application process is running with the plugin loaded.

## Usage

1. Build and enable the plugin, then start ACloudViewer.
2. Connect a WebSocket client to `ws://127.0.0.1:6001` (or your configured host/port).
3. Send JSON-RPC 2.0 requests with methods such as `app.getVersion`, scene queries, and processing calls—see the API reference below.

Minimal Python example:

```python
import asyncio, websockets, json

async def main():
    async with websockets.connect("ws://127.0.0.1:6001") as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "app.getVersion", "id": 1}))
        print(await ws.recv())

asyncio.run(main())
```

Agent integration (MCP, harness) reuses the same API; see `agent-integration/README.md` at the repository root.

## ACloudViewer CLI

**None** — automation is exclusively via **JSON-RPC over WebSocket** while the app is running.

## Build

```bash
-DPLUGIN_STANDARD_QJSONRPC=ON
```

Requires **Qt WebSockets** (and Qt network stack) in your Qt build.

## Dependencies

- **Qt** (including **Qt WebSockets**).

## References

- Method catalog and parameters: **[JSON-RPC-API.md](../../../../agent-integration/docs/JSON-RPC-API.md)** (`agent-integration/docs/JSON-RPC-API.md`).
- Agent integration overview: `agent-integration/README.md`.
