JSON RPC plugin for CloudCompare
================================

This plugin executes commands/actions via RPC from a different proccess/computer
The plugin opens a TCP-server port when activated.
The RPC communication is done via Websocket http JSON transfers.

Currently implemented commands:
- open: opens a file from filesystem
- clear: clears all clouds


howto build
-----------

1. clone CloudCompare

```
mkdir cc
cd cc
git clone --recursive https://github.com/CloudCompare/CloudCompare.git
```

2. add this plugin as submodule in plugins

```
cd CloudCompare/plugins/core/Standard
git submodule add git@gitlab.com:theadib/JsonRPCPlugin.git
```
3. patch CMakeLists.txt

```
nano CMakeLists.txt
```
then add the plugin as indicated below

```
 set( submod_plugins
                ${CMAKE_CURRENT_SOURCE_DIR}/qColorimetricSegmenter
                ${CMAKE_CURRENT_SOURCE_DIR}/qMasonry
                ${CMAKE_CURRENT_SOURCE_DIR}/qMPlane
+               ${CMAKE_CURRENT_SOURCE_DIR}/JsonRPCPlugin
 )

```

3. build CloudCompare

```
cd cc/CloudCompare
mkdir build
cd build
cmake-gui ..
make -j
sudo make install
```

howto use
---------

Below a python script to open a file from filesystem:

```
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
```

interface
---------

```
- open - opens a file from filesystem
  filename: string, filename on the filesystem
  filter: string, optional, filter identificator, if not given then CloudCompare identifies fileformat based on extension
  silent: bool, optional, if true, then filterdialog is suppressed
  transformation: list of float (16values), optional, after import this transformation matrix is applied to the object,
- clear - clear all objects
```


contact
-------

Remarks, wishes, etc. are welcome
thAdib theAdib@gmail.com

