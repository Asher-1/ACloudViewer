#!/usr/bin/env python3
"""SelectionTools region-mapping fitter (agent-test integration).

Diagnoses the "rubber-band frames region A, highlights land in region B"
defect class by fitting the ACTUAL region->rendering mapping of the
hardware-selection chain against the on-screen ground truth, entirely
through the JSON-RPC surface (no screen forensics needed):

  1. view.screenshot  -> locate probe entities (green mesh) in the app's own
     offscreen render (same camera as the interactive viewport).
  2. view.selectRegion scan lines across those clusters -> the regions the
     selection chain ACTUALLY hits.
  3. Compare the two: a pure translation / mirror / scale factor falls out
     of the data, which pinpoints the layer at fault (rubber-band region
     production vs. selector region interpretation vs. marker coordinates).

Usage:  python3 selection_region_fit.py [host] [port]
Requires: ACloudViewer running with the qJSonRPCPlugin (port 6001) and a
scene whose entities are saturated green (e.g. /tmp/sel_probe.obj).
"""
import json
import sys
import uuid

from PIL import Image
from websocket import create_connection


def connect(host, port):
    # The server refuses handshakes carrying Origin (browser pages);
    # websocket-client sends one unless suppressed.
    return create_connection(f"ws://{host}:{port}", timeout=60,
                             suppress_origin=True)


def call(ws, method, params=None, timeout=45.0):
    req = {"jsonrpc": "2.0", "id": str(uuid.uuid4())[:8],
           "method": method, "params": params or {}}
    ws.send(json.dumps(req))
    ws.settimeout(timeout)
    while True:
        msg = json.loads(ws.recv())
        if isinstance(msg, dict) and msg.get("id") == req["id"]:
            return msg.get("result")


def green_clusters(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    cols, rows = set(), set()
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            r, g, b = px[x, y]
            if g > 150 and r < 120 and b < 120:
                cols.add(x)
                rows.add(y)
    return w, h, sorted(cols), sorted(rows)


def runs_of(xs, gap):
    out = []
    for x in xs:
        if out and x - out[-1][1] <= gap:
            out[-1][1] = x
        else:
            out.append([x, x])
    return out


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 6001
    ws = connect(host, port)

    shot = "/tmp/selection_fit_offscreen.png"
    call(ws, "view.screenshot", {"filename": shot}, 90)
    w, h, gx, gy = green_clusters(shot)
    if not gx:
        print("VERDICT: no saturated-green probe entities found in the "
              "offscreen render - load a green probe scene first.")
        return
    print(f"offscreen render {w}x{h}; "
          f"green x {gx[0]}..{gx[-1]}, y {gy[0]}..{gy[-1]}")

    def hit(x, y):
        r = call(ws, "view.selectRegion",
                 {"x1": x - 2, "y1": y - 2, "x2": x + 2, "y2": y + 2}, 30)
        return isinstance(r, dict) and r.get("count", 0) > 0

    # Horizontal scan lines through every green row-band, vertical ones
    # through every green column-band.
    h_bands = runs_of(gy, 12)
    v_bands = runs_of(gx, 12)
    fit = {}
    for lo, hi in h_bands[:4]:
        ymid = (lo + hi) // 2
        xs = [x for x in range(0, w, 20) if hit(x, ymid)]
        fit[f"H(y={ymid})"] = runs_of(xs, 40)
    for lo, hi in v_bands[:4]:
        xmid = (lo + hi) // 2
        ys = [y for y in range(0, h, 20) if hit(xmid, y)]
        fit[f"V(x={xmid})"] = runs_of(ys, 40)

    for k, v in fit.items():
        print(f"{k}: {v}")

    # Verdict heuristics: compare green cluster bounds with hit runs.
    print("\n-- interpretation --")
    print("green x-bands:", runs_of(gx, 12))
    print("green y-bands:", runs_of(gy, 12))
    print("If hit runs align with the green bands: region layer is HEALTHY; "
          "the defect lives downstream (marker coordinates / entity "
          "transform). If hits are mirrored (y' ~ H - y) or scaled "
          "(x' ~ s*x), the selector region interpretation is at fault and "
          "the factor above IS the fix constant.")


if __name__ == "__main__":
    main()
