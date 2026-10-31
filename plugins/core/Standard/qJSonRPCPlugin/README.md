# qJSonRPCPlugin — JSON-RPC automation (WebSocket)

## Introduction

**qJSonRPCPlugin** exposes ACloudViewer to **local automation**: a **WebSocket** server (loopback, default port **6001**) speaks **JSON-RPC 2.0** so external tools can open files, drive the database tree, run processing, capture views, and more. Typical use includes AI agent integration, scripted QA, and custom operator panels.

**Headless note:** There are **no** extra `-SILENT` flags for this plugin; control is **JSON-RPC** while the application process is running with the plugin loaded.

## Usage

1. Build and enable the plugin, then start ACloudViewer.
2. Connect a WebSocket client to `ws://127.0.0.1:6001`. Set `ACV_RPC_PORT` before launch to listen on another port (e.g. a second, offscreen instance); a port that cannot be bound is logged as a warning.
3. Send JSON-RPC 2.0 requests such as `ping`, `methods.list` (the registry), scene queries, and processing calls—see the API reference below.

The server has no authentication, and its methods read/write arbitrary paths and start child programs (`process.run_cli`, `colmap.run`). It therefore listens on `127.0.0.1` only and refuses handshakes that carry an `Origin` header, which every browser sends (a web page could otherwise reach loopback). Python `websockets` sends none; `websocket-client` needs `suppress_origin=True`.

Minimal Python example:

```python
import asyncio, websockets, json

async def main():
    async with websockets.connect("ws://127.0.0.1:6001") as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1}))
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

## Source layout

One class (`JsonRPCPlugin`) with its handlers split by method family. Handler bodies do not call each other across families; they share only the entity lookups on the class and the helpers below.

| File | Contents |
|---|---|
| `src/jsonrpcserver.cpp` | WebSocket server, JSON-RPC framing; dispatches on the GUI thread |
| `src/JsonRPCPlugin.cpp` | Plugin lifecycle, `ACV_RPC_PORT`, method registry (`reg(...)`), dispatch, request logging, entity lookups |
| `src/JsonRPCCommon.{h,cpp}` | `entityToJson()`, `D(...)` error-data builder |
| `src/JsonRPCChildProcess.{h,cpp}` | `headlessChildEnvironment()`, `childTimeout()`, `runChildProcess()` (starts a child and returns a deferred reply; nothing waits on the GUI thread), `abortChildProcesses()` |
| `src/JsonRPCMethodsScene.cpp` | `open`, `export`, `file.convert`, `scene.*`, `clear`, `transform.apply`, `entity.*` |
| `src/JsonRPCMethodsUi.cpp` | `ui.listActions`, `ui.triggerAction`, `qt.invokeMethod` |
| `src/JsonRPCMethodsView.cpp` | `view.*` |
| `src/JsonRPCMethodsCloud.cpp` | `cloud.*` colorization, filtering, sampling, analysis |
| `src/JsonRPCMethodsCloudScalarFields.cpp` | `cloud.*` scalar-field methods |
| `src/JsonRPCMethodsCloudNormals.cpp` | `cloud.*` normal methods |
| `src/JsonRPCMethodsMesh.cpp` | `mesh.*` |
| `src/JsonRPCMethodsProcess.cpp` | `colmap.*`, `process.run_cli`, `process.csf/m3c2/ransac` |
| `src/JsonRPCMethodsAI.cpp` | `aicore.batch` (whitelist `kCommands`), `yolo.track`, `sam3d.generate`, `mapanything.reconstruct` |

A new method: `reg()` in `registerMethods()`, the declaration in `include/JsonRPCPlugin.h` under its family's section, the body in that family's file. A new family gets a new `JsonRPCMethods<Family>.cpp` listed in `src/CMakeLists.txt`. The agent-integration L1 tests check that every declared handler is defined exactly once in a built file and that no file other than `JsonRPCChildProcess.cpp` waits on a `QProcess`.

## Dependencies

- **Qt** (including **Qt WebSockets**).

## References

- Method catalog and parameters: **[JSON-RPC-API.md](../../../../agent-integration/docs/JSON-RPC-API.md)** (`agent-integration/docs/JSON-RPC-API.md`).
- Agent integration overview: `agent-integration/README.md`.
