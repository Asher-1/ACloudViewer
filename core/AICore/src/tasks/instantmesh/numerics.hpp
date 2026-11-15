// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Task-private numerics policy for the InstantMesh ggml graphs.
//
// GGML_HINT_STRICT_F32 and ggml_mul_mat_set_hint are shared repository patch
// extensions, not official APIs of the pinned ggml. InstantMesh reuses the
// single llama patch chain and requests the hint for exact F32 GEMM nodes
// to avoid TF32 truncation. This helper does not prove complete graph coverage,
// quantized-chain performance or backend accuracy; validate CPU/CUDA/Vulkan/Metal
// using the actual manifest chains and operator/model gates.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

namespace instantmesh {

// InstantMesh GEMM uses official F32 accumulation; F32 staging also needs the shared hint.
// This node policy leaves shared devices and other tasks' default arithmetic unchanged;
// component, full-chain and performance gates still apply.
inline ggml_tensor * mul_mat_exact(ggml_context * ctx, ggml_tensor * a,
                                   ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_hint(y, GGML_HINT_STRICT_F32);
    GGML_ASSERT(ggml_prec_set_acc(y, GGML_PREC_F32));
    return y;
}

// The frozen CPU build disabled LLAMAFILE; DINO/LRM and Synthesizer SDF/deformation/RGB
// graphs retain its vec-dot dispatch. The single CPU patch recognizes this name;
// GPU graphs and other tasks' default arithmetic/device state remain unchanged.
inline void preserve_cpu_matmul_dispatch(ggml_cgraph * graph, ggml_backend_t backend) {
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) return;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT) ggml_set_name(node, "instantmesh_cpu_matmul_");
    }
}

} // namespace instantmesh
