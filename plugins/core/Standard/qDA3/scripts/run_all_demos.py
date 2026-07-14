#!/usr/bin/env python3
"""Comprehensive benchmark: all models x all images x all inference modes."""
import subprocess, time, os, json, re, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

CLI = "build/examples/cli/da3-cli"
SAMPLES = sorted([f"assets/samples/{f}" for f in os.listdir("assets/samples") if f.endswith(".jpg")])
MODELS = [
    "models/depth-anything-small-f32.gguf",
    "models/depth-anything-base-f32.gguf",
    "models/depth-anything-base-f16.gguf",
    "models/depth-anything-base-q8_0.gguf",
    "models/depth-anything-base-q4_k.gguf",
    "models/depth-anything-large-f32.gguf",
    "models/depth-anything-giant-f32.gguf",
]
NESTED_AV = "models/depth-anything-nested-anyview.gguf"
NESTED_M = "models/depth-anything-nested-metric.gguf"
OUTDIR = "examples/demos"

def mname(path):
    return os.path.basename(path).replace(".gguf", "")

def iname(path):
    return os.path.basename(path).replace(".jpg", "")

def get_gpu_mem():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip().split("\n")[0])
    except:
        return -1

def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
    except:
        pass
    return -1

def run_cmd(args, timeout=120):
    gpu_before = get_gpu_mem()
    mem_before = get_rss_mb()
    t0 = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        output = r.stdout + r.stderr
        exit_code = r.returncode
    except subprocess.TimeoutExpired:
        elapsed = timeout
        output = "TIMEOUT"
        exit_code = -1
    gpu_after = get_gpu_mem()
    mem_after = get_rss_mb()
    return {
        "time_s": round(elapsed, 3),
        "exit_code": exit_code,
        "output": output,
        "gpu_before_MiB": gpu_before,
        "gpu_after_MiB": gpu_after,
        "rss_before_MB": mem_before,
        "rss_after_MB": mem_after,
    }

def parse_depth_info(output):
    m = re.search(r"depth (\d+)x(\d+) min=([\d.]+) max=([\d.]+)", output)
    if m:
        return {"w": int(m.group(1)), "h": int(m.group(2)),
                "min": float(m.group(3)), "max": float(m.group(4))}
    return None

# =============================================================================
print("=" * 70)
print(" PHASE 1: Depth inference (all models x all images)")
print("=" * 70)

results_depth = []
for model in MODELS:
    mn = mname(model)
    os.makedirs(f"{OUTDIR}/depth/{mn}", exist_ok=True)
    for img in SAMPLES:
        inp = iname(img)
        pfm = f"{OUTDIR}/depth/{mn}/{inp}.pfm"
        png = f"{OUTDIR}/depth/{mn}/{inp}.png"

        r = run_cmd([CLI, "depth", "--model", model, "--input", img, "--pfm", pfm, "--png", png])
        di = parse_depth_info(r["output"])
        entry = {"model": mn, "image": inp, "time_s": r["time_s"],
                 "gpu_MiB": r["gpu_after_MiB"], "depth_info": di,
                 "status": "ok" if r["exit_code"] == 0 else "fail"}
        results_depth.append(entry)
        status = f"min={di['min']:.4f} max={di['max']:.4f}" if di else "FAILED"
        print(f"  [{mn}] {inp}: {r['time_s']:.3f}s | {status} | GPU:{r['gpu_after_MiB']}MiB")

# =============================================================================
print("\n" + "=" * 70)
print(" PHASE 2: Pose estimation")
print("=" * 70)

results_pose = []
for model in MODELS:
    mn = mname(model)
    os.makedirs(f"{OUTDIR}/pose/{mn}", exist_ok=True)
    for img in SAMPLES:
        inp = iname(img)
        pose_json = f"{OUTDIR}/pose/{mn}/{inp}_pose.json"

        r = run_cmd([CLI, "depth", "--model", model, "--input", img, "--pose", pose_json])
        has_pose = os.path.exists(pose_json) and os.path.getsize(pose_json) > 10
        entry = {"model": mn, "image": inp, "time_s": r["time_s"],
                 "status": "ok" if has_pose else "not_supported"}
        results_pose.append(entry)
        print(f"  [{mn}] {inp} pose: {r['time_s']:.3f}s {'OK' if has_pose else 'N/A'}")

# =============================================================================
print("\n" + "=" * 70)
print(" PHASE 3: Nested metric depth")
print("=" * 70)

results_nested = []
os.makedirs(f"{OUTDIR}/nested", exist_ok=True)
for img in SAMPLES:
    inp = iname(img)
    pfm = f"{OUTDIR}/nested/{inp}_metric.pfm"
    png = f"{OUTDIR}/nested/{inp}_metric.png"

    r = run_cmd([CLI, "depth", "--model", NESTED_AV, "--metric-model", NESTED_M,
                 "--input", img, "--pfm", pfm, "--png", png], timeout=180)
    di = parse_depth_info(r["output"])
    status = "ok" if r["exit_code"] == 0 and di else "fail"
    entry = {"image": inp, "time_s": r["time_s"], "status": status, "depth_info": di,
             "error": r["output"].split("\n")[-2] if status == "fail" else None}
    results_nested.append(entry)
    if di:
        print(f"  nested {inp}: {r['time_s']:.3f}s | min={di['min']:.4f} max={di['max']:.4f}")
    else:
        print(f"  nested {inp}: {r['time_s']:.3f}s | FAILED ({entry['error'][:60] if entry['error'] else ''})")

# =============================================================================
print("\n" + "=" * 70)
print(" PHASE 4: Multi-view depth + pose")
print("=" * 70)

results_mv = []
os.makedirs(f"{OUTDIR}/multiview", exist_ok=True)
mv_pairs = [("canyon.jpg", "mountains.jpg"), ("desk.jpg", "street.jpg")]
for model in [MODELS[1], MODELS[-1]]:  # base-f32, giant-f32
    mn = mname(model)
    os.makedirs(f"{OUTDIR}/multiview/{mn}", exist_ok=True)
    for a, b in mv_pairs:
        pname = f"{iname(f'assets/samples/{a}')}_{iname(f'assets/samples/{b}')}"
        prefix = f"{OUTDIR}/multiview/{mn}/{pname}"
        r = run_cmd([CLI, "depth", "--model", model,
                     "--input", f"assets/samples/{a}", "--input", f"assets/samples/{b}",
                     "--out-prefix", prefix], timeout=180)
        out_files = [f for f in os.listdir(f"{OUTDIR}/multiview/{mn}") if pname in f]
        entry = {"model": mn, "pair": pname, "time_s": r["time_s"],
                 "status": "ok" if r["exit_code"] == 0 else "fail",
                 "output_files": out_files}
        results_mv.append(entry)
        print(f"  [{mn}] {pname}: {r['time_s']:.3f}s | {len(out_files)} files | "
              f"{'OK' if r['exit_code']==0 else 'FAIL'}")

# =============================================================================
print("\n" + "=" * 70)
print(" PHASE 5: 3D Export (glb + colmap + reconstruct/ply)")
print("=" * 70)

results_3d = []
os.makedirs(f"{OUTDIR}/3d_export", exist_ok=True)
for model in [MODELS[1], MODELS[-1]]:  # base-f32, giant-f32
    mn = mname(model)
    os.makedirs(f"{OUTDIR}/3d_export/{mn}", exist_ok=True)

    # glb + colmap
    glb = f"{OUTDIR}/3d_export/{mn}/scene.glb"
    colmap_dir = f"{OUTDIR}/3d_export/{mn}/colmap/"
    r1 = run_cmd([CLI, "depth", "--model", model, "--input", "assets/samples/street.jpg",
                  "--glb", glb, "--colmap", colmap_dir], timeout=180)
    glb_ok = os.path.exists(glb) and os.path.getsize(glb) > 100
    colmap_ok = os.path.isdir(colmap_dir) and len(os.listdir(colmap_dir)) > 0

    # reconstruct -> ply
    ply = f"{OUTDIR}/3d_export/{mn}/cloud.ply"
    r2 = run_cmd([CLI, "reconstruct", "--model", model, "--input", "assets/samples/street.jpg",
                  "--ply", ply], timeout=300)
    ply_ok = os.path.exists(ply) and os.path.getsize(ply) > 100

    entry = {"model": mn,
             "glb_time_s": r1["time_s"], "glb_ok": glb_ok, "colmap_ok": colmap_ok,
             "reconstruct_time_s": r2["time_s"], "ply_ok": ply_ok,
             "glb_size_KB": os.path.getsize(glb) // 1024 if glb_ok else 0,
             "ply_size_KB": os.path.getsize(ply) // 1024 if ply_ok else 0}
    results_3d.append(entry)
    print(f"  [{mn}] glb+colmap: {r1['time_s']:.3f}s | glb={'OK' if glb_ok else 'FAIL'} "
          f"colmap={'OK' if colmap_ok else 'FAIL'}")
    print(f"  [{mn}] reconstruct: {r2['time_s']:.3f}s | ply={'OK' if ply_ok else 'FAIL'} "
          f"({entry['ply_size_KB']}KB)")

# =============================================================================
print("\n" + "=" * 70)
print(" PHASE 6: Model Info")
print("=" * 70)

results_info = []
os.makedirs(f"{OUTDIR}/info", exist_ok=True)
for model in MODELS + [NESTED_AV, NESTED_M]:
    mn = mname(model)
    info_file = f"{OUTDIR}/info/{mn}.txt"
    r = run_cmd([CLI, "info", "--model", model])
    with open(info_file, "w") as f:
        f.write(r["output"])
    lines = r["output"].count("\n")
    entry = {"model": mn, "info_lines": lines, "status": "ok" if r["exit_code"] == 0 else "fail"}
    results_info.append(entry)
    print(f"  {mn}: {lines} lines")

# =============================================================================
print("\n" + "=" * 70)
print(" Writing report")
print("=" * 70)

report = {
    "depth_inference": results_depth,
    "pose_estimation": results_pose,
    "nested_metric": results_nested,
    "multiview": results_mv,
    "3d_export": results_3d,
    "model_info": results_info,
}

report_path = f"{OUTDIR}/benchmark_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"  JSON report: {report_path}")

# Generate markdown summary
md_path = f"{OUTDIR}/BENCHMARK.md"
with open(md_path, "w") as f:
    f.write("# Depth Anything .cpp - Comprehensive Benchmark Report\n\n")
    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"GPU: RTX 3060 12GB | CUDA 12.4\n\n")

    # Depth table
    f.write("## 1. Depth Inference (GPU mode)\n\n")
    f.write("| Model | canyon | desk | mountains | street | Avg Time |\n")
    f.write("|-------|--------|------|-----------|--------|----------|\n")
    for mn in [mname(m) for m in MODELS]:
        entries = [e for e in results_depth if e["model"] == mn]
        times = [e["time_s"] for e in entries]
        cells = []
        for e in entries:
            if e["depth_info"]:
                cells.append(f"{e['time_s']:.2f}s")
            else:
                cells.append("FAIL")
        avg = sum(times) / len(times) if times else 0
        f.write(f"| {mn} | {' | '.join(cells)} | {avg:.2f}s |\n")

    # Depth range comparison
    f.write("\n### Depth Range Comparison (min/max per image)\n\n")
    for img_name in [iname(s) for s in SAMPLES]:
        f.write(f"\n**{img_name}.jpg**\n\n")
        f.write("| Model | min | max | range |\n")
        f.write("|-------|-----|-----|-------|\n")
        for e in results_depth:
            if e["image"] == img_name and e["depth_info"]:
                di = e["depth_info"]
                f.write(f"| {e['model']} | {di['min']:.4f} | {di['max']:.4f} | {di['max']-di['min']:.4f} |\n")

    # Pose table
    f.write("\n## 2. Pose Estimation\n\n")
    f.write("| Model | canyon | desk | mountains | street |\n")
    f.write("|-------|--------|------|-----------|--------|\n")
    for mn in [mname(m) for m in MODELS]:
        entries = [e for e in results_pose if e["model"] == mn]
        cells = [f"{e['time_s']:.2f}s" if e["status"] == "ok" else "N/A" for e in entries]
        f.write(f"| {mn} | {' | '.join(cells)} |\n")

    # Nested
    f.write("\n## 3. Nested Metric Depth\n\n")
    f.write("| Image | Time | Status | min | max |\n")
    f.write("|-------|------|--------|-----|-----|\n")
    for e in results_nested:
        di = e.get("depth_info")
        if di:
            f.write(f"| {e['image']} | {e['time_s']:.2f}s | OK | {di['min']:.4f} | {di['max']:.4f} |\n")
        else:
            f.write(f"| {e['image']} | {e['time_s']:.2f}s | FAIL | - | - |\n")

    # Multi-view
    f.write("\n## 4. Multi-view Depth + Pose\n\n")
    f.write("| Model | Pair | Time | Files | Status |\n")
    f.write("|-------|------|------|-------|--------|\n")
    for e in results_mv:
        f.write(f"| {e['model']} | {e['pair']} | {e['time_s']:.2f}s | {len(e['output_files'])} | {e['status']} |\n")

    # 3D export
    f.write("\n## 5. 3D Export\n\n")
    f.write("| Model | GLB Time | GLB | COLMAP | Reconstruct Time | PLY | PLY Size |\n")
    f.write("|-------|----------|-----|--------|------------------|-----|----------|\n")
    for e in results_3d:
        f.write(f"| {e['model']} | {e['glb_time_s']:.2f}s | "
                f"{'OK' if e['glb_ok'] else 'FAIL'} | {'OK' if e['colmap_ok'] else 'FAIL'} | "
                f"{e['reconstruct_time_s']:.2f}s | {'OK' if e['ply_ok'] else 'FAIL'} | "
                f"{e['ply_size_KB']}KB |\n")

    # Model info summary
    f.write("\n## 6. Model Metadata\n\n")
    f.write("| Model | Info Lines |\n")
    f.write("|-------|------------|\n")
    for e in results_info:
        f.write(f"| {e['model']} | {e['info_lines']} |\n")

print(f"  Markdown report: {md_path}")
print("\nALL DONE!")
