// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Owner-scoped llama LLM RMS-norm parity tagging.
//
// Tasks whose upstream reference pins the OLD scalar ggml RMS-norm kernel
// (wevisone, vlx_seek) install this eval callback on llama_context_params.cb_eval
// so the CUDA backend (patch 0033) keeps the pinned scalar sum-of-squares
// accumulation order for their LLM norm nodes. The llm task references the
// pristine NEW vectorized kernel and must NOT install it, otherwise near-tied
// greedy argmax tokens flip and its greedy-text parity gate fails.
//
// Only llama LLM RMS-norm nodes carry the build_norm "norm"/"norm-<il>" naming;
// clip/mtmd ViT norms are unnamed (their trailing MUL/ADD are "norm_w"/"norm_b"),
// so this never touches the vision encoder.
#pragma once

#include "ggml.h"

#include <cstring>

namespace ggml_common {

// ggml_backend_sched_eval_callback. Returns false ("no post-compute inspection
// needed") so the scheduler keeps whole-split compute: the hint is written
// during the ask=true pass that runs BEFORE graph_compute_async, so there is no
// race with the kernel reading op_params and no per-node synchronize overhead.
inline bool llama_norm_parity_eval_cb(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask && t->op == GGML_OP_RMS_NORM) {
        const char * n = t->name;
        if (n != nullptr && std::strncmp(n, "norm", 4) == 0 &&
            (n[4] == '\0' || (n[4] == '-' && n[5] >= '0' && n[5] <= '9'))) {
            ggml_rms_norm_set_hint(t, GGML_HINT_STRICT_F32);
        }
    }
    return false;
}

}  // namespace ggml_common