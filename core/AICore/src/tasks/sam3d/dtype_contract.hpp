// Runtime evidence for the dtypes actually used by an end-to-end ggml graph.
//
// Model file names are not sufficient evidence: graph builders can insert
// casts, retain F32 accumulators, or route a quantized projection through a
// separate graph.  This recorder observes the built public ggml graph without
// changing its execution or the ggml source tree.
#pragma once

#include <string>

struct ggml_cgraph;

namespace sam3d {

struct DtypeContractMetadata {
    std::string backend;
    std::string base_dtype;
    std::string ss_dtype;
    std::string ss_decoder_dtype;
    std::string slat_dtype;
    std::string gs_dtype;
    std::string mesh_dtype;
};

// One recording may be active in a process.  An empty output path disables
// observation entirely, so production inference retains its normal overhead.
class ScopedDtypeContractRecording {
  public:
    ScopedDtypeContractRecording(const std::string& output_path,
                                 const DtypeContractMetadata& metadata,
                                 std::string& error);
    ~ScopedDtypeContractRecording();

    ScopedDtypeContractRecording(const ScopedDtypeContractRecording&) = delete;
    ScopedDtypeContractRecording& operator=(const ScopedDtypeContractRecording&) = delete;

    bool enabled() const { return enabled_; }
    bool complete(std::string& error);

  private:
    bool enabled_ = false;
    bool closed_ = false;
};

// Safe no-op unless ScopedDtypeContractRecording is active.  Call after a
// graph is fully expanded and before backend allocation so every graph node is
// present while no execution state has been mutated.
void record_dtype_contract_graph(const char* stage, ggml_cgraph* graph);

}  // namespace sam3d
