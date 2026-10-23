#!/usr/bin/env python3
"""Compare actual optimizer orders and allocation dependencies without a GPU.

Extract both implementations, their real fusion helpers and pattern constants.
Only the Vulkan device shell is stubbed; ggml's graph/fusion APIs stay real.
This is a host algorithm check, not model or GPU performance acceptance.
"""

import argparse
from pathlib import Path
import re
import subprocess


def function(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def implementation(root, namespace):
    directory = root / "src/ggml-vulkan"
    source = (directory / "ggml-vulkan.cpp").read_text()
    patterns = re.findall(
        r"static constexpr std::initializer_list<ggml_op>\s+\w+\s*\{[^}]+\};",
        (directory / "ggml-vulkan-types.h").read_text())
    helpers = [function(source, signature) for signature in (
        "int ggml_vk_unary_mul_op_index(",
        "static bool ggml_vk_can_fuse_unary_mul(",
        "void ggml_vk_graph_optimize(")]
    return "namespace " + namespace + " {\n" + "\n".join(patterns + helpers) + "\n}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    program = r'''
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <random>
#include <set>
#include <utility>
#include <vector>
#define VK_LOG_DEBUG(x) ((void) 0)
struct TestDevice { bool disable_graph_optimize = false; };
struct ggml_backend_vk_context { TestDevice * device; };
IMPLEMENTATIONS
using Dependencies = std::vector<std::pair<ggml_tensor *, ggml_tensor *>>;
static void allocation(void * data, ggml_tensor * tensor, ggml_tensor * until) {
    static_cast<Dependencies *>(data)->emplace_back(tensor, until);
}
int main() {
    std::mt19937 random(0xAC10F05E);
    const ggml_op ops[] = { GGML_OP_NONE, GGML_OP_VIEW, GGML_OP_PERMUTE,
        GGML_OP_ADD, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_RMS_NORM,
        GGML_OP_UNARY, GGML_OP_GET_ROWS, GGML_OP_CONT, GGML_OP_CPY,
        GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_ARGSORT, GGML_OP_TOP_K,
        GGML_OP_SCALE, GGML_OP_SSM_CONV, GGML_OP_ROPE, GGML_OP_SET_ROWS };
    TestDevice device;
    ggml_backend_vk_context vk{&device};
    ggml_backend backend{};
    backend.context = &vk;
    int qsa_graphs = 0;
    size_t allocation_calls = 0;
    for (int scenario = 0; scenario < 2000; ++scenario) {
        ggml_init_params init{8*1024*1024, nullptr, true};
        ggml_context * context = ggml_init(init);
        ggml_cgraph * graph = ggml_new_graph_custom(context, 256, false);
        const int count = scenario == 0 ? 0 : 16 + random() % 112;
        std::vector<ggml_tensor> tensors(count + 1);
        auto & weight = tensors.back();
        for (auto & tensor : tensors) {
            tensor.type = GGML_TYPE_F32;
            tensor.flags = GGML_TENSOR_FLAG_COMPUTE;
            size_t stride = sizeof(float);
            for (int d = 0; d < 4; ++d) {
                tensor.ne[d] = d < 2 ? 4 : 1;
                tensor.nb[d] = stride;
                stride *= tensor.ne[d];
            }
            for (auto & src : tensor.src) { src = &weight; }
        }
        weight.op = GGML_OP_NONE;
        for (auto & src : weight.src) { src = nullptr; }
        for (int i = 0; i < count; ++i) {
            auto & tensor = tensors[i];
            tensor.op = ops[random() % (sizeof(ops)/sizeof(ops[0]))];
            if (i && random() % 3 == 0) { tensor.view_src = &tensors[random() % i]; }
            for (auto & src : tensor.src) {
                if (i && random() % 4 == 0) { src = &tensors[random() % i]; }
            }
            const ggml_unary_op unary = GGML_UNARY_OP_SILU;
            std::memcpy(tensor.op_params, &unary, sizeof(unary));
            graph->nodes[i] = &tensor;
            ggml_hash_insert(&graph->visited_hash_set, &tensor);
        }
        graph->n_nodes = count;
        if (count && scenario % 3 == 0) {
            const int start = random() % (count - 7 + 1);
            int offset = 0;
            for (auto op : original::topk_qsa_pattern) {
                tensors[start + offset++].op = op;
            }
            ++qsa_graphs;
        }
        if (count && scenario % 5 == 0) {
            const int start = count - original::hc_post_gate_pattern.size();
            int offset = 0;
            for (auto op : original::hc_post_gate_pattern) {
                tensors[start + offset++].op = op;
            }
        }
        for (auto & tensor : tensors) {
            for (auto * src : tensor.src) {
                if (src && src != &weight) {
                    ++graph->use_counts[ggml_hash_find(&graph->visited_hash_set, src)];
                }
            }
        }
        const std::vector<ggml_tensor *> initial(graph->nodes, graph->nodes + count);
        Dependencies before, after;
        ggml_backend_graph_optimize_params params{allocation, &before};
        original::ggml_vk_graph_optimize(&backend, graph, &params);
        const std::vector<ggml_tensor *> order(graph->nodes, graph->nodes + count);
        std::copy(initial.begin(), initial.end(), graph->nodes);
        params.user_data = &after;
        modified::ggml_vk_graph_optimize(&backend, graph, &params);
        if (!std::equal(order.begin(), order.end(), graph->nodes) || before != after) {
            std::fprintf(stderr, "optimizer mismatch scenario=%d\n", scenario);
            return 1;
        }
        allocation_calls += before.size();
        ggml_free(context);
    }
    std::printf("{\"graphs\":2000,\"injected_qsa_graphs\":%d,"
                "\"allocation_calls\":%zu,\"mismatches\":0}\n", qsa_graphs, allocation_calls);
}
'''
    program = program.replace("IMPLEMENTATIONS", implementation(args.before, "original") +
                              implementation(args.after, "modified"))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cpp = args.work_dir / "check_optimizer.cpp"
    cpp.write_text(program)
    binary = args.work_dir / "check_optimizer"
    library_dir = args.library_dir.resolve()
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra",
                    "-I" + str(args.after / "include"),
                    "-I" + str(args.after / "src"), str(cpp),
                    "-L" + str(library_dir), "-Wl,-rpath," + str(library_dir),
                    "-lggml-base", "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
