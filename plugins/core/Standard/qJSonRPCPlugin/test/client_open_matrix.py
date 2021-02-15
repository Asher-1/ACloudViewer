#!/usr/bin/env python3

import asyncio
import websockets

async def hello():
    uri = "ws://localhost:6001"
    async with websockets.connect(uri) as websocket:
        await websocket.send('{"jsonrpc": "2.0", "method": "open", \
                "params": { \
                    "filename": "/home/adib/Dokumente/teapot.ply", \
                    "filter":"PLY mesh (*.ply)", \
                    "silent":true, \
                    "transformation": [-0.5732937009507673, 0.8193193174792813, 0.007084582014166903, -161.35002666963268, \
                        -0.8193308300123949, -0.5733179603522832, 0.001873784980341508, 320.67951255557966, \
                        0.005596946702519333, -0.004730387192182964, 0.9999731500865392, -230.60935194531334, \
                        0.0, 0.0, 0.0, 1.0]}, \
                "id": 4}')
        result = await websocket.recv()
        print("result: ", result)

asyncio.get_event_loop().run_until_complete(hello())



