// Graph-builder registry: one IGraphBuilder implementation per model
// architecture, routed by the GGUF general.architecture key.
//
// Adding an architecture (touch points — see FEATURE_PARITY_AUDIT.md for
// the full audit and AGENTS.md §2 for the ten-step checklist):
//   1. implement IGraphBuilder (usually by extending VGGTRuntime::Impl or
//      Pi3Impl to reuse the shared encoder/rope/decoder/DPT components);
//   2. register it with RegisterBuilder("<arch>", ...);
//   3. have the converter write general.architecture="<arch>" plus that
//      branch's key space (parse_meta in gguf_loader.cpp is dispatched per
//      architecture — a missing key must fail the builder, not fall back
//      to another architecture's defaults);
//   4. dispatch the output contract in cli.cpp — three families exist:
//      vggt (pose_enc+depth+depth_conf[+text_embedding]), pi3 (c2w 4x4 +
//      local_points/conf/depth/points[+mask/scale]), dust3r (no pose).
// Everything else (gate/bench/eval infra) is architecture-agnostic.
#pragma once

#include "map_anything_backend.hpp"
#include "map_anything_common.hpp"
#include "map_anything_gguf_loader.hpp"
#include "map_anything_graph.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace map_anything {

class IGraphBuilder {
public:
    virtual ~IGraphBuilder() = default;
    // Load-time: stage weights, build the compute graph(s), upload static
    // inputs. S/H/W come from the request (images or --bin); opts carries
    // the debug switches (dump_dir/dot_path — see RuntimeOptions).
    virtual bool init(const GGUFModel& model, Backend be, int s, int h,
                      int w, const RuntimeOptions& opts) = 0;
    // One inference: imgs is (S,3,H,W) C-order f32 in [0,1].
    virtual bool run(const float* imgs, VGGTOutputs& out) = 0;
};

// Registry of architecture-keyed factories. Keys are the normalized
// general.architecture values.
using BuilderFactory = std::function<std::unique_ptr<IGraphBuilder>()>;

inline std::map<std::string, BuilderFactory>& builder_registry() {
    static std::map<std::string, BuilderFactory> registry;
    return registry;
}

// RAII registration helper (one static instance per builder translation
// unit).
struct RegisterBuilder {
    RegisterBuilder(const std::string& arch, BuilderFactory factory) {
        builder_registry()[arch] = std::move(factory);
    }
};

// Looks up the registry; returns nullptr (after logging the registered
// architectures) when the key is unknown.
std::unique_ptr<IGraphBuilder> create_builder(const std::string& arch);

}  // namespace map_anything
