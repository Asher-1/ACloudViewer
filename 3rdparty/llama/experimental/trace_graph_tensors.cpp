// Offline LD_PRELOAD diagnostic. Compile separately against each runtime's
// exact ggml headers. This is never linked into AICore or a backend module.
//
// Set ACV_GGML_TRACE_DIRECTORY to an existing private output directory and
// disable backend fusion for a useful node-by-node comparison. Marking graph
// outputs changes allocation lifetimes: callers must independently verify
// that the final public result still equals an untraced run.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct Capture {
    int index;
    ggml_tensor * tensor;
};

struct OperandCapture {
    int node;
    int slot;
    ggml_tensor * tensor;
};

thread_local std::unordered_map<ggml_cgraph *, std::vector<Capture>> captures;
thread_local std::unordered_map<ggml_cgraph *, std::vector<OperandCapture>> operands;
thread_local bool dumping = false;
thread_local unsigned graph_number = 0;

template <typename Function>
Function original(const char * name) {
    auto function = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
    if (!function) {
        std::fprintf(stderr, "trace_graph_tensors: missing %s: %s\n", name, dlerror());
        std::abort();
    }
    return function;
}

void prepare(ggml_cgraph * graph) {
    if (!std::getenv("ACV_GGML_TRACE_DIRECTORY") || dumping) return;
    // Bound extra live tensor storage, independent of graph/model size.
    constexpr size_t tensor_limit = 4u * 1024u * 1024u;
    constexpr size_t graph_limit = 256u * 1024u * 1024u;
    size_t bytes = 0;
    auto & selected = captures[graph];
    selected.clear();
    auto & inputs = operands[graph];
    inputs.clear();
    const char * requested_node = std::getenv("ACV_GGML_TRACE_MATMUL_NODE");
    const int matmul_node = requested_node ? std::atoi(requested_node) : -1;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * tensor = ggml_graph_node(graph, i);
        if (tensor->op == GGML_OP_NONE || tensor->view_src ||
            (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) ||
            !ggml_is_contiguous(tensor)) continue;
        const size_t count = ggml_nbytes(tensor);
        if (!count || count > tensor_limit || count > graph_limit - bytes) continue;
        ggml_set_output(tensor);
        selected.push_back({i, tensor});
        bytes += count;
        if (i == matmul_node && tensor->op == GGML_OP_MUL_MAT) {
            // Views can alias an intermediate allocation. Preserve the owning
            // tensor too, otherwise a post-compute input dump may read reuse.
            for (int slot = -1; slot < 2; ++slot) {
                ggml_tensor * input = slot < 0 ? tensor : tensor->src[slot];
                if (!input || ggml_nbytes(input) > 64u * 1024u * 1024u) std::abort();
                for (ggml_tensor * owner = input; owner; owner = owner->view_src) {
                    ggml_set_output(owner);
                }
                inputs.push_back({i, slot, input});
            }
        }
    }
}

void capture(ggml_cgraph * graph, ggml_status status) {
    const char * directory = std::getenv("ACV_GGML_TRACE_DIRECTORY");
    auto found = captures.find(graph);
    if (!directory || dumping || found == captures.end() ||
        status != GGML_STATUS_SUCCESS) return;
    dumping = true;
    const unsigned serial = graph_number++;
    const std::string prefix = std::string(directory) + "/graph-" + std::to_string(serial);
    FILE * metadata = std::fopen((prefix + ".tsv").c_str(), "wx");
    if (!metadata) std::abort();
    std::fprintf(metadata, "index\tname\top\ttype\tne0\tne1\tne2\tne3\tfile\n");
    for (const Capture & entry : found->second) {
        ggml_tensor * tensor = entry.tensor;
        if (!tensor->buffer || !tensor->data) std::abort();
        const size_t count = ggml_nbytes(tensor);
        std::vector<unsigned char> data(count);
        ggml_backend_tensor_get(tensor, data.data(), 0, count);
        const std::string filename = "graph-" + std::to_string(serial) + "-node-" +
                                     std::to_string(entry.index) + ".bin";
        FILE * output = std::fopen((std::string(directory) + "/" + filename).c_str(), "wx");
        if (!output || std::fwrite(data.data(), 1, count, output) != count) std::abort();
        std::fclose(output);
        // Tensor names are arbitrary strings; keep the TSV one record per line.
        std::string name(tensor->name);
        for (char & value : name) {
            if (value == '\t' || value == '\r' || value == '\n') value = ' ';
        }
        std::fprintf(metadata, "%d\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%s\n",
                     entry.index, name.c_str(), ggml_op_name(tensor->op),
                     ggml_type_name(tensor->type),
                     static_cast<long long>(tensor->ne[0]),
                     static_cast<long long>(tensor->ne[1]),
                     static_cast<long long>(tensor->ne[2]),
                     static_cast<long long>(tensor->ne[3]), filename.c_str());
    }
    std::fclose(metadata);
    auto input_found = operands.find(graph);
    if (input_found != operands.end() && !input_found->second.empty()) {
        FILE * info = std::fopen((prefix + "-operands.tsv").c_str(), "wx");
        if (!info) std::abort();
        std::fprintf(info, "node\tslot\tname\top\ttype\tne0\tne1\tne2\tne3\tnb0\tnb1\tnb2\tnb3\top_params_hex\tfile\n");
        for (const OperandCapture & entry : input_found->second) {
            ggml_tensor * tensor = entry.tensor;
            if (!tensor->buffer || !tensor->data) std::abort();
            const size_t count = ggml_nbytes(tensor);
            std::vector<unsigned char> data(count);
            ggml_backend_tensor_get(tensor, data.data(), 0, count);
            const std::string filename = "graph-" + std::to_string(serial) + "-node-" +
                                         std::to_string(entry.node) + "-operand-" +
                                         std::to_string(entry.slot) + ".bin";
            FILE * output = std::fopen((std::string(directory) + "/" + filename).c_str(), "wx");
            if (!output || std::fwrite(data.data(), 1, count, output) != count) std::abort();
            std::fclose(output);
            std::string name(tensor->name);
            for (char & value : name) {
                if (value == '\t' || value == '\r' || value == '\n') value = ' ';
            }
            std::fprintf(info, "%d\t%d\t%s\t%s\t%s", entry.node, entry.slot, name.c_str(),
                         ggml_op_name(tensor->op), ggml_type_name(tensor->type));
            for (int i = 0; i < 4; ++i) std::fprintf(info, "\t%lld", static_cast<long long>(tensor->ne[i]));
            for (int i = 0; i < 4; ++i) std::fprintf(info, "\t%zu", tensor->nb[i]);
            std::fputc('\t', info);
            for (size_t i = 0; i < sizeof(tensor->op_params); ++i) {
                std::fprintf(info, "%02x", reinterpret_cast<unsigned char *>(tensor->op_params)[i]);
            }
            std::fprintf(info, "\t%s\n", filename.c_str());
        }
        std::fclose(info);
        operands.erase(input_found);
    }
    captures.erase(found);
    dumping = false;
}
}  // namespace

extern "C" bool ggml_gallocr_alloc_graph(ggml_gallocr_t allocator, ggml_cgraph * graph) {
    const auto function = original<decltype(&ggml_gallocr_alloc_graph)>("ggml_gallocr_alloc_graph");
    prepare(graph);
    return function(allocator, graph);
}

extern "C" bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t scheduler, ggml_cgraph * graph) {
    const auto function = original<decltype(&ggml_backend_sched_alloc_graph)>("ggml_backend_sched_alloc_graph");
    prepare(graph);
    return function(scheduler, graph);
}

extern "C" ggml_status ggml_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    const auto function = original<decltype(&ggml_backend_graph_compute)>("ggml_backend_graph_compute");
    const ggml_status status = function(backend, graph);
    capture(graph, status);
    return status;
}

extern "C" ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t scheduler,
                                                       ggml_cgraph * graph) {
    const auto function = original<decltype(&ggml_backend_sched_graph_compute)>("ggml_backend_sched_graph_compute");
    const ggml_status status = function(scheduler, graph);
    capture(graph, status);
    return status;
}
