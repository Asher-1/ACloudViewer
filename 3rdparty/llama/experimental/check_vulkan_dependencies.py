#!/usr/bin/env python3
"""Check the actual old/new optimizer dependency predicates on generated graphs.

This CPU-only check compiles the predicates extracted from the specified source
trees. It covers sparse source slots, direct edges, empty operations, shared
storage and views of views. Model graph-order/output tests remain necessary.
"""

import argparse
from pathlib import Path
import subprocess


def predicate(source):
    start = source.index("    auto const &is_empty =", source.index(
        "void ggml_vk_graph_optimize("))
    end = source.index("    std::vector<ggml_tensor *> new_order;", start)
    return source[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    relative = "src/ggml-vulkan/ggml-vulkan.cpp"
    before = predicate((args.before / relative).read_text())
    after = predicate((args.after / relative).read_text())
    program = r'''
#include "ggml.h"
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
struct TestGraph { int n_nodes; ggml_tensor ** nodes; };
static bool original(const ggml_tensor * dst, const ggml_tensor * src) {
BEFORE
    return is_src_of(dst, src);
}
static uint64_t check(TestGraph * graph) {
AFTER
    uint64_t comparisons = 0;
    for (int d = 0; d < graph->n_nodes; ++d) {
        for (int s = 0; s < graph->n_nodes; ++s) {
            if (original(graph->nodes[d], graph->nodes[s]) != is_src_of(d, s)) {
                std::fprintf(stderr, "dependency mismatch dst=%d src=%d\n", d, s);
                std::exit(1);
            }
            ++comparisons;
        }
    }
    return comparisons;
}
int main() {
    std::mt19937 random(0xAC105EED);
    const ggml_op ops[] = { GGML_OP_NONE, GGML_OP_RESHAPE, GGML_OP_TRANSPOSE,
        GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_ADD, GGML_OP_MUL_MAT,
        GGML_OP_SET_ROWS, GGML_OP_CPY, GGML_OP_RMS_NORM };
    uint64_t comparisons = 0;
    for (int scenario = 0; scenario < 1000; ++scenario) {
        const int count = 1 + random() % 96;
        std::vector<ggml_tensor> tensors(count);
        std::vector<ggml_tensor *> nodes(count);
        for (int i = 0; i < count; ++i) {
            nodes[i] = &tensors[i];
            tensors[i].op = ops[random() % (sizeof(ops) / sizeof(ops[0]))];
            // Earlier storage owners allow nested views without cycles.
            if (i && random() % 3 == 0) {
                tensors[i].view_src = &tensors[random() % i];
            }
            // Populate every slot independently; source arrays may have holes.
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (i && random() % 5 == 0) {
                    tensors[i].src[s] = &tensors[random() % i];
                }
            }
        }
        TestGraph graph{count, nodes.data()};
        comparisons += check(&graph);
    }
    std::printf("{\"graphs\":1000,\"predicate_pairs\":%llu,\"mismatches\":0}\n",
                static_cast<unsigned long long>(comparisons));
}
'''.replace("BEFORE", before).replace("AFTER", after)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cpp = args.work_dir / "check_dependencies.cpp"
    cpp.write_text(program)
    executable = args.work_dir / "check_dependencies"
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra",
                    "-I" + str(args.after / "include"), str(cpp),
                    "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
