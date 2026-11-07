// Persistent AnyCalib inference session: GGUF load, static graph build,
// backend-routed execution. Task-private; the public surface is the C ABI
// in aicore/anycalib_capi.h.
#pragma once

#include <string>
#include <vector>

namespace aicore::anycalib {

struct SessionOptions {
    std::string device = "auto";
    int num_threads = 0;
};

/** Host-side run result. rays/tangent are row-major over the 322x322 grid:
 *  rays[3*i..] unit camera-frame ray (x right, y down, z forward) and
 *  tangent[tw*i..] the FoV tangent coordinates, matching the exported ONNX
 *  outputs "rays" [1, 322*322, 3] and "tangent_coords" [1, 322*322, 2]. */
struct RunResult {
    std::vector<float> rays;
    std::vector<float> tangent;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double e2e_ms = 0.0;
};

class Session {
public:
    Session() = default;
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    bool Load(const std::string& gguf_path, const SessionOptions& options);
    /** image_rgb_chw: 3*322*322 RGB floats in [0, 1], CHW. */
    bool Run(const float* image_rgb_chw, RunResult& out);

    bool ready() const { return loaded_; }
    const std::string& error() const { return error_; }

private:
    void Release();

    struct Impl;
    Impl* impl_ = nullptr;
    bool loaded_ = false;
    std::string error_;
};

}  // namespace aicore::anycalib
