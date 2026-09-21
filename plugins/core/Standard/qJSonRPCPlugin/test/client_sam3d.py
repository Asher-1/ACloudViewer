#!/usr/bin/env python3
"""SAM 3D batch generation over the qJSonRPCPlugin WebSocket API.

Usage:
    python client_sam3d.py <image> [image2 ...] [--out DIR] [--dtype q4_k]
                           [--device cuda] [--steps 25] [--seed 42]
                           [--port 6001]

Requires the JSON-RPC plugin action to be enabled in the GUI (Plugins menu).
Models are read from ~/cloudViewer_data/extract/sam3d_models; use the
"Download models" button in the qSAM3D dialog or the validation runner to
fetch them first.
"""

import argparse
import json
import sys

try:
    from websocket import create_connection
except ImportError:
    sys.exit("pip install websocket-client")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="input image paths")
    parser.add_argument("--out", default="/tmp/sam3d_out")
    parser.add_argument("--masks", nargs="*", default=[],
                        help="ordered mask paths (empty string = no mask)")
    parser.add_argument("--dtype", default="q4_k",
                        choices=["f16", "q8_0", "q4_k"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mesh", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rmbg", type=int, default=1, choices=[0, 1])
    parser.add_argument("--timeout-ms", type=int, default=3600000)
    parser.add_argument("--port", type=int, default=6001)
    args = parser.parse_args()

    params = {
        "images": args.images,
        "masks": args.masks,
        "output_dir": args.out,
        "device": args.device,
        "dtype": args.dtype,
        "steps": args.steps,
        "seed": args.seed,
        "mesh": args.mesh,
        "rmbg": args.rmbg,
        "timeout_ms": args.timeout_ms,
    }
    ws = create_connection(f"ws://127.0.0.1:{args.port}", timeout=args.timeout_ms / 1000)
    ws.send(json.dumps({"jsonrpc": "2.0", "id": 1,
                        "method": "sam3d.generate", "params": params}))
    while True:
        response = json.loads(ws.recv())
        if response.get("id") == 1:
            break
    ws.close()

    if "error" in response:
        print("FAILED:", json.dumps(response["error"], indent=2))
        sys.exit(1)

    result = response["result"]
    print(f"backend: {result.get('backend')}  "
          f"ok: {result.get('ok_count')}  failed: {result.get('failed_count')}  "
          f"elapsed: {result.get('elapsed_ms', 0) / 1000:.1f}s")
    for item in result.get("items", []):
        print(f"  [{item.get('status')}] {item.get('image')} -> {item.get('ply')} "
              f"({item.get('gaussians')} gaussians, {item.get('e2e_ms', 0):.0f} ms)")


if __name__ == "__main__":
    main()
