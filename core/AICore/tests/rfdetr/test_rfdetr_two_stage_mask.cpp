// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// gen_encoder_output_proposals validity mask white-box tests (no GGUF, no
// PyTorch baseline required; the rule is pure arithmetic).
//
// Upstream (rfdetr 1.9.0, rfdetr/models/transformer.py:127-143) zeroes BOTH
// output_proposals and output_memory for every token whose proposal falls
// outside the open interval (0.01, 0.99). Since the centre is
// cx = (w + 0.5) / S for a grid of side S, the condition can only fire when
// S >= 50, where it masks exactly the one-cell border ring (4*S - 4 tokens).
//
// This test has two halves:
//
//  A. `compute_proposal_grid` arithmetic — pure host-side, no model needed.
//     Checks the fired set is exactly the border ring at S = 50/52/64, and
//     that S = 40/49 (every published detection variant and every seg variant
//     below xlarge) is a strict no-op: proposals stay bit-identical to the
//     analytic formula.
//
//  B. `two_stage_forward` graph behaviour — a synthetic 4-channel two-stage
//     head over a 52x52 and a 40x40 grid. enc_output is the identity with a
//     zero bias and enc_output_norm has unit weight plus a distinctive bias
//     `b_ln`, so a token whose MEMORY was zeroed lands on exactly `b_ln`
//     (LayerNorm of the zero vector) and nothing else does. That pins the
//     mask to the memory path specifically, not just to the proposals.
//
// Ported from the upstream rf-detr.cpp test suite (commit d256d3e), adapted
// to AICore's init_backend_ctx / free_backend_ctx lifecycle.

#include <ggml-backend.h>
#include <ggml.h>
#include <gtest/gtest.h>

#include <vector>

#include "tasks/rfdetr/backend.hpp"
#include "tasks/rfdetr/model_loader.hpp"
#include "tasks/rfdetr/trace.hpp"
#include "tasks/rfdetr/two_stage.hpp"

namespace {

bool is_border(int w, int h, int S) {
    return w == 0 || h == 0 || w == S - 1 || h == S - 1;
}

/* ---- Part A: compute_proposal_grid ------------------------------------- */

class RfDetrProposalGrid : public ::testing::TestWithParam<int> {};

TEST_P(RfDetrProposalGrid, MasksOnlyBorderAndPreservesAnalyticInterior) {
    const int S = GetParam();
    const bool expect_masked = S >= 50;
    const float wh = 0.05f;
    std::vector<float> grid((size_t)S * S * 4, -1.0f);
    rfdetr::compute_proposal_grid(S, S, wh, grid.data());

    const float fs = (float)S;
    int n_zeroed = 0;

    for (int h = 0; h < S; ++h) {
        for (int w = 0; w < S; ++w) {
            const float* p = grid.data() + ((size_t)h * S + w) * 4;
            const bool zeroed = (p[0] == 0.0f && p[1] == 0.0f && p[2] == 0.0f &&
                                 p[3] == 0.0f);
            SCOPED_TRACE(::testing::Message() << "cell " << w << "," << h);
            EXPECT_EQ(zeroed, expect_masked && is_border(w, h, S));
            if (zeroed) {
                ++n_zeroed;
            } else {
                // Untouched rows must be exactly equal to the analytic formula.
                EXPECT_EQ(p[0], (static_cast<float>(w) + 0.5f) / fs);
                EXPECT_EQ(p[1], (static_cast<float>(h) + 0.5f) / fs);
                EXPECT_EQ(p[2], wh);
                EXPECT_EQ(p[3], wh);
            }
        }
    }

    const int want_zeroed = expect_masked ? (4 * S - 4) : 0;
    EXPECT_EQ(n_zeroed, want_zeroed);
}

/* ---- Part B: two_stage_forward ----------------------------------------- */

const int kD = 4;                                    /* synthetic model_dim */
const int kNC = 2;                                   /* synthetic num_classes */
const float kBLn[kD] = {0.25f, -1.5f, 3.0f, 0.125f}; /* LayerNorm bias */

/* Deterministic, always-nonzero, never channel-constant input. */
float input_at(int w, int h, int c) {
    return 0.5f + 0.01f * (float)((w * 7 + h * 13 + c * 3) % 17);
}

ggml_tensor* make(ggml_context* ctx,
                  rfdetr::Model& m,
                  const char* name,
                  int64_t ne0,
                  int64_t ne1) {
    ggml_tensor* t = ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
                             : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    if (!t) return nullptr;
    ggml_set_name(t, name);
    m.tensors[name] = t;
    return t;
}

class RfDetrTwoStageMask : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        rfdetr_status status = RFDETR_OK;
        bctx = rfdetr::init_backend_ctx(2, "cpu", &status);
        ASSERT_EQ(status, RFDETR_OK);
        ASSERT_NE(bctx.cpu, nullptr);
    }

    void TearDown() override {
        if (gbuf) ggml_backend_buffer_free(gbuf);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (gctx) ggml_free(gctx);
        if (wctx) ggml_free(wctx);
        rfdetr::free_backend_ctx(bctx);
    }

    rfdetr::BackendCtx bctx;
    ggml_context* wctx = nullptr;
    ggml_context* gctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_buffer_t gbuf = nullptr;
};

TEST_P(RfDetrTwoStageMask, MasksMemoryAndBoxesOnExactlyTheProposalBorder) {
    const int S = GetParam();
    const bool expect_masked = S >= 50;
    rfdetr::Model m;
    m.config.decoder.model_dim = kD;
    m.config.num_classes = kNC;
    ggml_backend_t backend = bctx.cpu;

    const int64_t N = (int64_t)S * S;

    /* --- synthetic weights + the proposal grid, in their own context --- */
    ggml_init_params wp{};
    wp.mem_size = ggml_tensor_overhead() * 64;
    wp.mem_buffer = nullptr;
    wp.no_alloc = true;
    wctx = ggml_init(wp);
    ASSERT_NE(wctx, nullptr);

    ggml_tensor* W_eo = make(wctx, m, "two_stage.enc_output.0.weight", kD, kD);
    ggml_tensor* b_eo = make(wctx, m, "two_stage.enc_output.0.bias", kD, 0);
    ggml_tensor* w_ln =
            make(wctx, m, "two_stage.enc_output_norm.0.weight", kD, 0);
    ggml_tensor* b_ln =
            make(wctx, m, "two_stage.enc_output_norm.0.bias", kD, 0);
    ggml_tensor* W_cls =
            make(wctx, m, "two_stage.enc_out_class_embed.0.weight", kD, kNC);
    ggml_tensor* b_cls =
            make(wctx, m, "two_stage.enc_out_class_embed.0.bias", kNC, 0);
    ggml_tensor* W_b0 = make(
            wctx, m, "two_stage.enc_out_bbox_embed.0.layers.0.weight", kD, kD);
    ggml_tensor* b_b0 = make(
            wctx, m, "two_stage.enc_out_bbox_embed.0.layers.0.bias", kD, 0);
    ggml_tensor* W_b1 = make(
            wctx, m, "two_stage.enc_out_bbox_embed.0.layers.1.weight", kD, kD);
    ggml_tensor* b_b1 = make(
            wctx, m, "two_stage.enc_out_bbox_embed.0.layers.1.bias", kD, 0);
    ggml_tensor* W_b2 = make(
            wctx, m, "two_stage.enc_out_bbox_embed.0.layers.2.weight", kD, kD);
    ggml_tensor* b_b2 = make(
            wctx, m, "two_stage.enc_out_bbox_embed.0.layers.2.bias", kD, 0);

    for (const ggml_tensor* tensor : {W_eo, b_eo, w_ln, b_ln, W_cls, b_cls,
                                      W_b0, b_b0, W_b1, b_b1, W_b2, b_b2}) {
        ASSERT_NE(tensor, nullptr);
    }

    ggml_tensor* prop = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 4, N);
    ASSERT_NE(prop, nullptr);
    ggml_set_name(prop, "two_stage.proposals.grid");
    m.proposals_grid = prop;

    wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ASSERT_NE(wbuf, nullptr);

    auto fill = [](ggml_tensor* t, const std::vector<float>& v) {
        ASSERT_EQ(static_cast<int64_t>(v.size()), ggml_nelements(t));
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    };

    /* enc_output = identity, zero bias, so a zeroed memory row stays zero. */
    std::vector<float> ident((size_t)kD * kD, 0.0f);
    for (int i = 0; i < kD; ++i) ident[(size_t)i * kD + i] = 1.0f;
    ASSERT_NO_FATAL_FAILURE(fill(W_eo, ident));
    ASSERT_NO_FATAL_FAILURE(fill(b_eo, std::vector<float>(kD, 0.0f)));
    ASSERT_NO_FATAL_FAILURE(fill(w_ln, std::vector<float>(kD, 1.0f)));
    ASSERT_NO_FATAL_FAILURE(fill(b_ln, std::vector<float>(kBLn, kBLn + kD)));

    /* Class + bbox heads: arbitrary but nonzero and non-degenerate. */
    std::vector<float> wc((size_t)kD * kNC);
    for (size_t i = 0; i < wc.size(); ++i) wc[i] = 0.1f + 0.05f * (float)i;
    ASSERT_NO_FATAL_FAILURE(fill(W_cls, wc));
    ASSERT_NO_FATAL_FAILURE(fill(b_cls, std::vector<float>(kNC, 0.3f)));

    std::vector<float> wb((size_t)kD * kD);
    for (size_t i = 0; i < wb.size(); ++i) wb[i] = 0.07f + 0.011f * (float)i;
    ASSERT_NO_FATAL_FAILURE(fill(W_b0, wb));
    ASSERT_NO_FATAL_FAILURE(fill(W_b1, wb));
    ASSERT_NO_FATAL_FAILURE(fill(W_b2, wb));
    ASSERT_NO_FATAL_FAILURE(fill(b_b0, std::vector<float>(kD, 0.2f)));
    ASSERT_NO_FATAL_FAILURE(fill(b_b1, std::vector<float>(kD, 0.2f)));
    ASSERT_NO_FATAL_FAILURE(fill(b_b2, std::vector<float>(kD, 0.2f)));

    std::vector<float> grid((size_t)N * 4);
    rfdetr::compute_proposal_grid(S, S, /*wh_value*/ 0.05f, grid.data());
    ASSERT_NO_FATAL_FAILURE(fill(prop, grid));

    /* --- graph --- */
    ggml_init_params gp{};
    gp.mem_size = 64 * 1024 * 1024;
    gp.mem_buffer = nullptr;
    gp.no_alloc = true;
    gctx = ggml_init(gp);
    ASSERT_NE(gctx, nullptr);

    ggml_tensor* proj_in = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, S, S, kD, 1);
    ASSERT_NE(proj_in, nullptr);
    ggml_set_name(proj_in, "projector.output.in");

    rfdetr::TwoStageOutput ts = rfdetr::two_stage_forward(gctx, m, proj_in);
    ASSERT_NE(ts.enc_output_norm_out, nullptr);
    ASSERT_NE(ts.bbox_all, nullptr);

    ggml_cgraph* graph =
            ggml_new_graph_custom(gctx, /*size*/ 8192, /*grads*/ false);
    ASSERT_NE(graph, nullptr);
    ggml_build_forward_expand(graph, ts.enc_output_norm_out);
    ggml_build_forward_expand(graph, ts.bbox_all);

    gbuf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    ASSERT_NE(gbuf, nullptr);

    std::vector<float> in((size_t)S * S * kD);
    for (int c = 0; c < kD; ++c)
        for (int h = 0; h < S; ++h)
            for (int w = 0; w < S; ++w)
                in[((size_t)c * S + h) * S + w] = input_at(w, h, c);
    ggml_backend_tensor_set(proj_in, in.data(), 0, in.size() * sizeof(float));

    ASSERT_EQ(ggml_backend_graph_compute(backend, graph), GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> mem = rfdetr::copy_tensor_to_f32(ts.enc_output_norm_out);
    std::vector<float> bbox = rfdetr::copy_tensor_to_f32(ts.bbox_all);
    ASSERT_EQ(static_cast<int64_t>(mem.size()), N * kD);
    ASSERT_EQ(static_cast<int64_t>(bbox.size()), N * 4);

    int n_mem_masked = 0;
    int n_bbox_zero = 0;

    for (int h = 0; h < S; ++h) {
        for (int w = 0; w < S; ++w) {
            const size_t t = (size_t)h * S + w;
            const bool border = is_border(w, h, S);

            /* LayerNorm(0) == b_ln exactly: only reachable if the MEMORY row
             * was zeroed before enc_output. */
            bool is_bln = true;
            for (int c = 0; c < kD; ++c)
                if (mem[t * kD + c] != kBLn[c]) {
                    is_bln = false;
                    break;
                }
            SCOPED_TRACE(::testing::Message() << "cell " << w << "," << h);
            EXPECT_EQ(is_bln, expect_masked && border) << "masked memory";
            if (is_bln) ++n_mem_masked;

            bool bz = true;
            for (int c = 0; c < 4; ++c)
                if (bbox[t * 4 + c] != 0.0f) {
                    bz = false;
                    break;
                }
            EXPECT_EQ(bz, expect_masked && border) << "zeroed boxes";
            if (bz) ++n_bbox_zero;
        }
    }

    const int want = expect_masked ? (4 * S - 4) : 0;
    EXPECT_EQ(n_mem_masked, want);
    EXPECT_EQ(n_bbox_zero, want);
}

// 40 and 49 are below the masking threshold; 50/52/64 mask one border ring.
INSTANTIATE_TEST_SUITE_P(GridSizes,
                         RfDetrProposalGrid,
                         ::testing::Values(40, 49, 50, 52, 64),
                         ::testing::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(GridSizes,
                         RfDetrTwoStageMask,
                         ::testing::Values(40, 52),
                         ::testing::PrintToStringParamName());

}  // namespace
