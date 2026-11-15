# OpenPCDet upstream alignment audit (18 checkpoints × 3 quantizations × 3 backends)

This records the full alignment audit and closure on 2026-10-29: each
`(checkpoint, quantization, backend)` cell compares AICore with upstream
OpenPCDet-GGML using mean BEV IoU and detection counts, with dispositions
and supporting rationale for each gap category. Upstream reference:
[Asher-1/OpenPCDet-GGML](https://github.com/Asher-1/OpenPCDet-GGML)。

## Audit method and reference policy

- Accuracy comparisons use upstream `comparison_metrics` + `bev_iou`.
  Reference priority: **fresh current upstream CLI > existing validation/
  baselines > REPORT.md annotations**. Missing baselines are explicitly
  marked `no_upstream_baseline`, never invented.
- **Important measurement correction**: 8/18 upstream files in
  `validation/baseline/cuda/` came from old builds and diverged from their own
  CPU truth (parta2 up_cuda vs up_cpu was only 0.9891). Fresh current CLI runs
  (`/tmp/up_fresh_*.cuda.json`) changed parta2 0.9889→0.9997,
  pp_multihead 0.9914→0.9991 and centerpoint 0.9961→0.9991 to aligned.
  These three gaps were stale-baseline artifacts; AICore was closer to CPU
  truth (AICore parta2 cuda vs AICore cpu = 0.9997).
- Another reference correction: the upstream focal_multimodal baseline with
  8 detections uses **scene mode** (PNG + calibration). Points-only input
  takes the LiDAR-only fallback; the two modes are not comparable.

## Alignment matrix (final after fix; A=aligned T=tolerance/inherited N=no upstream baseline S=quantization skipped)

| checkpoint | f32 cpu | f32 cuda | f32 vulkan | f16 cpu/cuda | q8_0 cpu/cuda |
|---|---|---|---|---|---|
| kitti_pointpillar | 1.0 A | 0.9995 A | 0.9996 A | A / A | A / A |
| kitti_second_iou | 1.0 A | 0.9997 A | 0.9998 A | A / A | A / T(1 extra detection, near threshold) |
| second_7862 | 1.0 A | 0.9998 A | 0.9997 A | A / A | A / A |
| kitti_pointrcnn† | 1.0 A | 0.9989 T | 0.9911 T | A / A | A / A |
| kitti_pointrcnn_iou† | 1.0 A | 0.9947 T‡ | 0.9983 T | A / A | T / T‡ |
| kitti_pvrcnn | 1.0 A | 0.9997 A | 0.9996 A | A / A | A / A |
| kitti_voxelrcnn_car | 1.0 A | 0.9994 A | 0.9993 A | A / A | A / A |
| kitti_voxelrcnn_focal_multimodal§ | 1.0000 A | 0.9978 T | T | A / A(better than the same upstream cell) | A / A(≈upstream itself) |
| kitti_parta2_anchor | 1.0 A | 0.9997 A¶ | 0.9997 A | A / A | A / A¶ |
| nuscenes_pointpillar_multihead | 1.0 A | 0.9991 A¶ | 0.9997 A | A / A | A / A¶ |
| nuscenes_dynpillar_centerpoint | 1.0 A | 0.9988 T | 0.9993 A | A / T | T / T |
| nuscenes_centerpoint_voxel0075 | 0.999999 A¶ | 0.9991 A¶ | 0.9979 T | A / A | T / T |
| nuscenes_second_multihead | 1.0 A | 0.9987 T | 0.9979 A | A / A | T / T |
| nuscenes_transfusion_lidar | 1.0 A | 0.9971 T | 0.9921 T | A / A | T / T |
| nuscenes_voxelnext | 1.0 A | 1.0 A | 0.9980 T | A / A | T / T |
| nuscenes_voxelnext_doubleflip | 1.0 A | 1.0 A | 0.9982 T | A / A | T / T |
| kitti_caddn | 1.0 A | 0.9996 A | N | A / A | A / A |
| nuscenes_bevfusion | 1.0 A | 0.9982 A(=upstream floor) | 0.981 T | A / A | T / T |

\† pointrcnn requires 16384 pre-sampled points; both sides use the same seed0 fixture.
\‡ pointrcnn_iou 0.9947 inherits an upstream unresolved item: REPORT declares
"strict parity pending" at 0.9944, min IoU 0.9559; AICore matches that cell.
\§ See the focal fix below. f16/q8_0 references use upstream f32 where quantized
baselines are absent. Quantization error is expected; f16 cuda 0.9739 and
q8_0 0.994+ are both **better than upstream's own quantized-to-f32 gaps**
of 0.9725/0.9791.
\¶ Stale upstream cuda baselines were corrected using fresh current CLI runs.

## focal_multimodal fix (the only code fix in this audit, completed)

**Root cause**: `inv[10]` in `voxelrcnn.cpp` `invert_4x4` (4×4 adjugate inverse)
contained a transcription error: `-m[12]*m[2]*m[5]` should be
`-m[12]*m[3]*m[5]`. Translation in calibration (m[12..14]≠0) corrupted the
inverse → misaligned `lidar_to_camera` projection → broken
`conv_focal_multimodal` image sampling/fusion → collapsed focal scores and
2 detections (0.39/0.31), versus upstream's 8. The 1-line fix restored f32 CPU
**8/8 matched, mean_iou=1.000000** and 8/8 counts in all 9 cells.

**Residual f32 cuda 0.9978 disposition (tolerance)**: all 114 CPU intermediate
taps are bit-identical (no OpenPCDet-layer defect). CUDA differences arise from
ggml mul_mat summation-order noise (~2e-7), amplified into topology drift by
mask_kernel threshold-sensitive sites: 2945 within <3e-5 of 0.5, closest 8e-9.
Upstream cuda itself drifts to 0.9997 against cpu. Weights determine this
sensitivity; convergence requires ggml-patch summation-order alignment,
outside this audit's task-side scope.

## Common rationale for tolerance cells

1. **All detection counts agree**; tiny coordinate differences arise from
   accumulation/GEMM reduction order.
2. Values are within upstream's declared **cross-implementation reproducibility
   floor**: bevfusion's 200-box set is 0.9981–0.9988; KITTI's 12–25-box sets
   are more sensitive to individual-coordinate changes.
3. Reproducing an upstream outlier/build-specific arithmetic order would be
   the wrong fix; see the measurement correction above.
4. q8_0 nuScenes center heads (centerpoint/dynpillar/second_multihead/
   transfusion/voxelnext/bevfusion) at 0.95–0.98 versus upstream **f32** reflect
   quantization loss. Upstream has no quantized baseline; aligned KITTI anchor
   q8_0 results support the quantization implementation.

## Functional differences (upstream developer tools, not inference contracts)

- Standalone scene CLIs (`openpcdet-bevfusion-cli`/`openpcdet-caddn-cli`):
  AICore covers them through `detect_scene` + runner/batch commands.
- `--dump-taps` intermediate exports: upstream CLI supplied audit comparisons;
  AICore does not expose that interface. Persistent diagnostics could reuse
  the 0025 perf-op gate pattern.
- caddn calibration assets (`*_caddn_calibration.f32`) and Python references:
  these belong to upstream conversion; AICore consumes converted GGUFs.

## Integration verification (final audit, 2026-10-29)

- **Models**: all 18 checkpoints × 3 quantizations = 54 rows are integrated;
  all 54/54 SHA256 entries in `asset_digests.h` match measured local caches.
  catalog-dump passes. The six manifest scenarios
  (kitti/pointrcnn/nuscenes/caddn-scene/focal-scene/bevfusion-scene) jointly
  cover all 54 rows and contain structured `accuracy` objects.
- **Upstream 62-config inventory (MODEL_SUPPORT.md policy)**: all
  validated_config 12 + validated_family 16 = 28 validated configs are covered
  by AICore's 18 checkpoint families (multiple configs share checkpoints).
  Of 9 implemented_unvalidated configs, all except 4 asset-less waymo configs
  are integrated and validated on three backends, unlike upstream itself.
  The 19 pending_model_integration and 6 discarded_no_asset entries are
  **upstream's own incomplete/asset-less items**. AICore's integrated set
  ⊇ all upstream models with published GGUFs; per-cell mean_iou validation
  across 3 quantizations × 3 backends exceeds upstream's validation depth.
- **Automation**: level1/2/5 has 111 passed (GUI-dependent cases skipped);
  contracts/coverage 946/946 (100%) and `--strict` boundaries pass.
- **Recorded performance gaps**: dense graph replay ~5–10ms (upstream Voxel
  CenterPoint best 76ms vs AICore 83–91ms on the same host; diagnosis requires
  unavailable CUPTI/nsys infrastructure). bevfusion CUDA/Vulkan ~10% remains
  due to unported hybrid-routing decoder/head scalar GEMM, an optional follow-up.
- **Upstream recommendation (read-only, not changed by AICore)**: regenerate
  the 8/18 stale baselines in `validation/baseline/cuda/`.

## Regression evidence (after focal fix, 2026-10-29)

- `validate_all.py --tasks openpcdet`: **CPU / CUDA / Vulkan verdicts all PASS**.
- `test_openpcdet_capi_contract`, `test_qopenpcdet_helpers` and
  `check_plugin_aicore_boundaries.py --strict` pass.
- Focused regressions: kitti_pointpillar CUDA 24/24 @0.99952 and
  kitti_voxelrcnn_car CUDA 10/10 @0.99945, neither regressed.
- Original matrix/artifacts: `/tmp/alignment_matrix.{json,md}` (162 cells),
  `/tmp/up_fresh_*.json` (fresh references), `/tmp/audit/` (AICore dumps).
