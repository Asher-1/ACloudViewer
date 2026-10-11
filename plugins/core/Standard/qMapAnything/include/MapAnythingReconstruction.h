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
    size_t size() const { return view.size(); }
};

struct Chunk {
    int start = 0;
    int count = 0;
    MapAnythingGeometry::Sim3 toGlobal;
    size_t correspondences = 0;  // dense pixel pairs used by the fit
    float residualMedian = 0.0f;
    double e2eMs = 0.0;
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

/** Binary little-endian PLY (float xyz + uchar rgb), all points or the
 *  given ascending indices. */
bool write_ply(const std::string& path,
               const Cloud& cloud,
               const std::vector<size_t>* indices);

}  // namespace MapAnythingReconstruction
