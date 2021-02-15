#!/usr/bin/env python3

import asyncio
import websockets

async def hello():
    uri = "ws://localhost:6001"
    async with websockets.connect(uri) as websocket:
        await websocket.send('{"jsonrpc": "2.0", "method": "clear", "id": 5}')
        result = await websocket.recv()
        print("result: ", result)

asyncio.get_event_loop().run_until_complete(hello())



