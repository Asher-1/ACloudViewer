// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// The ONE scene-reconstruction driver shared by the GUI worker and the
// headless -MAP_ANYTHING_RECONSTRUCT command: chunk planning, per-chunk
// typed inference, dense shared-frame Sim3 alignment, per-architecture
// cloud assembly and the official scene-wide filter stack. Qt-free; it
// consumes only the public AICore C ABI and MapAnythingGeometry, so the two
// front ends cannot drift apart (they used to: the GUI streamed chunks while
// the headless path forced one pass).
//
// Frames: chunk 0's model frame is the global frame. Each later chunk is
// mapped into it by a robust Horn Sim3 fitted on DENSE per-pixel world
// points of the frames it shares with its predecessor, against those
// frames' already-aligned points (so the fit IS the chunk->global map; no
// chain composition error).

#pragma once

#include <aicore/image_view.h>
#include <aicore/map_anything_capi.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "MapAnythingGeometry.h"

namespace MapAnythingReconstruction {

struct Options {
    /** Views per forward. >= n: one pass (official semantics, best quality);
     *  otherwise a sliding window. Memory grows with the pass size (the
     *  vggt-omega family runs global attention over every view token); a
     *  pass the device cannot hold is halved automatically (see
     *  Result::framesPerPass), identically for every front end. */
    int maxFramesPerPass = 64;
    /** Frames shared by consecutive chunks (Sim3 anchors), clamped to
     *  [1, maxFramesPerPass - 1]. */
    int overlap = 2;
    /** Assemble the world-frame cloud (off: cameras/hook only). */
    bool buildCloud = true;
    /** Official depth_edge rtol (0 = off). */
    float depthEdgeRtol = 0.03f;
};

/** World-frame cloud, one entry per point, in view order. */
struct Cloud {
    std::vector<float> xyz;     // 3 per point
    std::vector<uint8_t> rgb;   // 3 per point
    std::vector<float> conf;    // raw model confidence (empty: no conf head)
    std::vector<int32_t> view;  // global view index
    /** Percentile population: every finite raw confidence inside each
     *  view's valid rect, taken BEFORE mask/edge/finite-point removal —
     *  the population the official percentile is defined over. Views are
     *  emitted in ascending order, so view g owns
     *  [populationOffset[g], populationOffset[g+1]). */
    std::vector<float> confPopulation;
    std::vector<size_t> populationOffset;  // views + 1
    /** Streamed scenes: the same population of each view's LATER covering
     *  chunks (shared frames emit no points there). upstream
     *  e2e_full_scene.py takes the scene-wide floor percentile over every
     *  chunk's confidences, shared frames once per covering chunk. */
    std::vector<float> overlapConfPopulation;
    size_t size() const { return view.size(); }
};

struct Chunk {
    int start = 0;
    int count = 0;
    MapAnythingGeometry::Sim3 toGlobal;
    size_t correspondences = 0;  // dense pixel pairs used by the fit
    float residualMedian = 0.0f;
    double e2eMs = 0.0;
    /** count*3 camera centers of ALL the chunk's views (shared ones
     *  included) in the chunk's own model frame, before toGlobal; NaN for
     *  unposed views. The per-chunk reference anchor fits these. */
    std::vector<float> centers;
};

struct Result {
    std::string architecture;
    std::string device;
    int height = 0;
    int width = 0;
    int views = 0;
    int poseStride = 0;
    Cloud cloud;
    /** views*16 row-major camera-to-world in the global frame; NaN rows
     *  when the architecture publishes no pose (dust3r). */
    std::vector<float> c2w;
    /** views focal px over the dense grid; NaN when unknown (only the
     *  vggt-family pose_enc carries intrinsics). */
    std::vector<float> focalPx;
    std::vector<Chunk> chunks;
    /** Pass size asked for (clamped to the view count) and the one that
     *  ran. They differ only when the device refused the requested pass
     *  (aicore_map_anything_run_images -3): the driver then halves it until
     *  the first chunk is admitted; memoryNotes holds each refusal. */
    int requestedFramesPerPass = 0;
    int framesPerPass = 0;
    std::vector<std::string> memoryNotes;
    bool hasMetricScale = false;  // mapanything: points already metric
    float metricScale = 1.0f;
    std::vector<float> textEmbedding;
    double e2eMs = 0.0;
};

struct ChunkView {
    aicore_map_anything_ctx* ctx = nullptr;
    int chunkIndex = 0;
    int chunkCount = 0;
    int start = 0;
    int count = 0;
};

/** Called once per global view, by the first chunk that covers it, while
 *  that chunk's result is live in ctx (rasters, progress). */
using ViewHook =
        std::function<void(const ChunkView&, int chunkViewIdx, int globalIdx)>;
/** Called before each forward; false cancels the run. */
using ChunkGate = std::function<bool(int chunkIndex, int chunkCount)>;

enum Status { kOk = 0, kError = 1, kCancelled = 2 };

/** Run the whole scene. \p views are borrowed for the call. On kOk the
 *  ctx still holds the LAST chunk's typed result (the only chunk for a
 *  single pass — the raw-parity export relies on this). */
int run(aicore_map_anything_ctx* ctx,
        const aicore_image_view* views,
        int n,
        const Options& options,
        Result* out,
        std::string* error,
        const ViewHook& hook = {},
        const ChunkGate& gate = {});

struct FilterOptions {
    /** mapanything: per-view percentile (e.g. 3 = drop the lowest 3% of
     *  each view); 1+exp families: normalized (c-1)/c gate; pi3/pi3x:
     *  sigmoid gate. */
    float minConf = 0.0f;
    /** 1+exp families: scene-wide floor max(2, p<confPercentile>). */
    float confPercentile = 20.0f;
    /** Conf-priority voxel merge; applied only in a metric frame. */
    float voxelSize = 0.02f;
};

struct FilterReport {
    bool ok = true;
    std::string rule;  // "percentile/view" | "floor+normalized" | "sigmoid" |
                       // "none"
    float confFloor = 0.0f;  // scene floor; NaN when per-view or no floor
    size_t input = 0;
    size_t kept = 0;  // after the confidence rule
    bool voxelApplied = false;
    /** Ascending indices into the (compacted) cloud of the voxel
     *  survivors; meaningful only when voxelApplied. */
    std::vector<size_t> fused;
};

/** Official scene-wide filter. Compacts \p cloud in place to the
 *  confidence-kept points (the per-view population), then computes the
 *  fused voxel subset when \p metricFrame. Unknown architecture -> ok=false
 *  (fail closed rather than guess a confidence scale). */
FilterReport filter_scene(Cloud& cloud,
                          const std::string& architecture,
                          const FilterOptions& options,
                          bool metricFrame);

/** Move every global-frame output by \p s (cloud, cameras, chunk maps). */
void transform_result(Result& result, const MapAnythingGeometry::Sim3& s);

struct ReferenceAnchor {
    MapAnythingGeometry::Sim3 sim3;  // global (model) frame -> reference frame
    int pairs = 0;                   // posed views with a reference center
    float residualMedian = 0.0f;     // |ref - sim3(center)|, reference units
    /** RMS spread of the reference centers across their main axis
     *  (sqrt of the middle covariance eigenvalue), reference units. */
    float crossSpread = 0.0f;
    /** Streamed scene anchored chunk by chunk: sim3 is chunk 0's fit,
     *  residualMedian is over every view's final center, crossSpread the
     *  smallest chunk's, and chunks[i] each chunk's own fit. */
    bool perChunk = false;
    struct ChunkFit {
        MapAnythingGeometry::Sim3 sim3;  // chunk model frame -> reference
        int pairs = 0;
        float residualMedian = 0.0f;
        float crossSpread = 0.0f;
    };
    std::vector<ChunkFit> chunks;
    /** Streamed scene anchored as a whole: why one chunk could not be
     *  anchored on its own (empty for a single pass). */
    std::string perChunkRefused;
};

/** Reference-frame anchor: robust Horn Sim3 from posed camera centers
 *  (c2w translation) to their reference centers. The benchmark
 *  scene_meta.json gives upstream e2e_full_scene.py --align gt (the frame
 *  of its published e2e PLYs); a COLMAP model_aligner file gives the
 *  user's GPS / surveyed frame. \p refCenters holds views*3 floats, NaN
 *  where a view has no reference.
 *  - Single pass: one Sim3 for the whole result.
 *  - Streamed (> 1 chunk): like --align gt, every chunk is fitted
 *    INDEPENDENTLY from all its views' centers in its own model frame and
 *    its views (cloud points and cameras, each from its first covering
 *    chunk) move by that fit, so the chain's cross-chunk scale/pose drift
 *    does not accumulate. Every chunk must pass the checks below on its
 *    own; otherwise the chained scene is anchored as a whole and
 *    perChunkRefused says why.
 *  Refused (false + \p error, \p result untouched) on fewer than 3 usable
 *  pairs, a degenerate fit, or (nearly) collinear centers whose spread
 *  across their main axis does not exceed the median fit residual (the
 *  roll about that axis would be noise). Without it the result stays in
 *  the model's official frame and unit (chunk 0's frame; see README
 *  "Output frame"). */
bool anchor_to_reference(Result& result,
                         const std::vector<float>& refCenters,
                         ReferenceAnchor* anchor,
                         std::string* error = nullptr);

/** Binary little-endian PLY (xyz + uchar rgb), all points or the given
 *  ascending indices. xyz is float, or double holding xyz + \p offset[3]
 *  when \p offset is given (reference frames with large coordinates). */
bool write_ply(const std::string& path,
               const Cloud& cloud,
               const std::vector<size_t>* indices,
               const double* offset = nullptr);

}  // namespace MapAnythingReconstruction
