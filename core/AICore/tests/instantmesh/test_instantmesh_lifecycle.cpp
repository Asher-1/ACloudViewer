// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "aicore/instantmesh_capi.h"
#include "aicore/runtime_capi.h"
#include "aicore/runtime_raii.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

struct CancellationRequest {
    aicore_cancel_token* token;
    int stage;
    int chunk_mode = 0;  // 0=entry, 1=first completed chunk, 2=final completed
                         // chunk, 3=nonfinal chunk
    bool triggered = false;
    int calls_after_cancel = 0;
};

static void cancel_at_stage(void* user, int stage, int step, int total) {
    auto* request = static_cast<CancellationRequest*>(user);
    if (stage != request->stage) return;
    if (request->triggered) {
        ++request->calls_after_cancel;
        return;
    }
    if (request->chunk_mode && (step <= 0 || total <= 0)) return;
    if (request->chunk_mode == 2 && step != total) return;
    if (request->chunk_mode == 3 && step >= total) return;
    std::printf("cancel_boundary stage=%d done=%d total=%d mode=%d\n", stage,
                step, total, request->chunk_mode);
    request->triggered = true;
    aicore_cancel_token_request(request->token);
}

int main(int argc, char** argv) {
    const char* assets =
            argc > 1 ? argv[1] : std::getenv("CLOUDVIEWER_AICORE_TEST_ASSETS");
    const char* device = argc > 2 ? argv[2] : "cpu";
    const bool full = argc > 3 && std::strcmp(argv[3], "full") == 0;
    const bool rmbg = argc > 3 && std::strcmp(argv[3], "rmbg") == 0;
    if (!assets) {
        std::fprintf(stderr,
                     "Missing real-model directory "
                     "CLOUDVIEWER_AICORE_TEST_ASSETS\n");
        return 77;
    }
    const auto root = std::filesystem::path(assets) / "instantmesh_models";
    const std::string dino = (root / "dino_f16.gguf").string();
    const std::string transformer =
            (root / "lrm_transformer_f16.gguf").string();
    const std::string synthesizer = (root / "synthesizer_f16.gguf").string();
    const std::string cond = (root / "zero123pp_cond_f32.gguf").string();
    const std::string unet = (root / "zero123pp_unet_f16.gguf").string();
    const std::string vae = (root / "zero123pp_vae_f16.gguf").string();
    if (full)
        for (const auto* file : {&cond, &unet, &vae})
            if (!std::filesystem::is_regular_file(*file)) return 77;
    for (const auto* file : {&dino, &transformer, &synthesizer}) {
        if (!std::filesystem::is_regular_file(*file)) {
            std::fprintf(stderr, "Missing model: %s\n", file->c_str());
            return 77;
        }
    }
    auto opts = std::unique_ptr<aicore_instantmesh_options,
                                decltype(&aicore_instantmesh_options_free)>(
            aicore_instantmesh_options_new(), aicore_instantmesh_options_free);
    AICORE_CHECK(opts != nullptr);
    aicore_instantmesh_options_set_device(opts.get(), device);
    aicore_instantmesh_options_set_threads(opts.get(), 4);
    aicore_instantmesh_options_set_steps(opts.get(), 2);
    const std::string rmbgPath =
            (std::filesystem::path(assets) / "rmbg_models" /
             (argc > 4 && rmbg ? argv[4] : "rmbg_f16.gguf"))
                    .string();
    if (rmbg) {
        if (!std::filesystem::is_regular_file(rmbgPath)) return 77;
        aicore_instantmesh_options_set_rmbg_gguf(opts.get(), rmbgPath.c_str());
    }
    if (argc > 4 && std::strcmp(argv[4], "guidance1") == 0)
        aicore_instantmesh_options_set_guidance(opts.get(), 1.0f);
    aicore_instantmesh_model_paths paths = AICORE_INSTANTMESH_MODEL_PATHS_INIT;
    paths.dino_gguf = dino.c_str();
    paths.transformer_gguf = transformer.c_str();
    paths.synthesizer_gguf = synthesizer.c_str();
    if (full) {
        paths.cond_gguf = cond.c_str();
        paths.unet_gguf = unet.c_str();
        paths.vae_gguf = vae.c_str();
    }
    char err[512] = {};
    auto ctx = std::unique_ptr<aicore_instantmesh_ctx,
                               decltype(&aicore_instantmesh_free)>(
            aicore_instantmesh_load_opts_ex(&paths, opts.get(), err,
                                            sizeof(err)),
            aicore_instantmesh_free);
    if (!ctx) {
        std::fprintf(stderr, "Real model load failed: %s\n", err);
        return 1;
    }
    std::printf("actual_backend=%s\n", aicore_instantmesh_backend(ctx.get()));
    auto token = std::unique_ptr<aicore_cancel_token,
                                 decltype(&aicore_cancel_token_free)>(
            aicore_cancel_token_new(), aicore_cancel_token_free);
    AICORE_CHECK(token != nullptr);
    std::vector<uint8_t> pixels(32 * 32 * 4, 255);
    const aicore_image_view image{pixels.data(), 32, 32, 32 * 4,
                                  AICORE_IMAGE_RGBA8};
    aicore_instantmesh_generate_params params =
            AICORE_INSTANTMESH_GENERATE_PARAMS_INIT;
    params.grid_res = 8;
    params.texture_res = 32;
    params.bake_texture = 0;
    if (rmbg) {
        const auto path = std::filesystem::path(assets) /
                          "image_to_mesh_data/examples_images/T.png";
        std::ifstream file(path, std::ios::binary);
        if (!file) return 77;
        const std::vector<uint8_t> encoded(
                (std::istreambuf_iterator<char>(file)), {});
        CancellationRequest request{token.get(), AICORE_INSTANTMESH_STAGE_RMBG};
        {
            aicore::runtime::CancelScope scope(token.get());
            auto* mesh = aicore_instantmesh_generate(
                    ctx.get(), encoded.data(), encoded.size(), &params,
                    cancel_at_stage, &request, err, sizeof(err));
            AICORE_CHECK(request.triggered && request.calls_after_cancel == 0);
            AICORE_CHECK(mesh == nullptr &&
                         std::strstr(err, "cancel") != nullptr);
            aicore_instantmesh_mesh_free(mesh);
        }
        aicore_cancel_token_reset(token.get());
        auto* mesh = aicore_instantmesh_generate(
                ctx.get(), encoded.data(), encoded.size(), &params, nullptr,
                nullptr, err, sizeof(err));
        if (!mesh) {
            std::fprintf(stderr,
                         "Shared RMBG retry after cancellation failed: %s\n",
                         err);
            return 1;
        }
        AICORE_CHECK(aicore_instantmesh_mesh_n_verts(mesh) > 0);
        AICORE_CHECK(aicore_instantmesh_mesh_n_tris(mesh) > 0);
        aicore_pipeline_timings timings{};
        AICORE_CHECK(aicore_instantmesh_last_pipeline_timings(ctx.get(),
                                                              &timings) == 0);
        AICORE_CHECK(timings.preprocess_ms > 0 && timings.inference_ms > 0);
        std::printf("rmbg_model=%s inference_ms=%.3f e2e_ms=%.3f\n",
                    rmbgPath.c_str(), timings.inference_ms, timings.e2e_ms);
        aicore_instantmesh_mesh_free(mesh);
        return failures ? 1 : 0;
    }
    if (argc > 3 && std::strcmp(argv[3], "timing") == 0) {
        // Maximum valid borrowed padded BGRA input makes real wrapper packing
        // cost observable.
        const size_t stride = 4096 * 4 + 16;
        std::vector<uint8_t> largePixels(stride * 4096, 255);
        const aicore_image_view largeImage{largePixels.data(), 4096, 4096,
                                           stride, AICORE_IMAGE_BGRA8};
        const auto start = std::chrono::steady_clock::now();
        auto* mesh = aicore_instantmesh_generate_image_view(
                ctx.get(), &largeImage, &params, nullptr, nullptr, err,
                sizeof(err));
        const double wallMs = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
        AICORE_CHECK(mesh != nullptr);
        aicore_pipeline_timings timings{};
        AICORE_CHECK(aicore_instantmesh_last_pipeline_timings(ctx.get(),
                                                              &timings) == 0);
        // Allow only small call/return gaps, not omission of input
        // copying/format conversion.
        AICORE_CHECK(timings.e2e_ms <= wallMs && wallMs - timings.e2e_ms < 5.0);
        AICORE_CHECK(timings.valid_fields & AICORE_TIMING_PREPROCESS);
        std::printf("api_wall_ms=%.3f e2e_ms=%.3f unaccounted_ms=%.3f\n",
                    wallMs, timings.e2e_ms, wallMs - timings.e2e_ms);
        aicore_instantmesh_mesh_free(mesh);
        return failures ? 1 : 0;
    }
    const auto canceled = [&](int stage, int chunk_mode = 0) {
        const bool before_start = stage < 0;
        CancellationRequest request{token.get(), stage, chunk_mode};
        aicore_cancel_token_reset(token.get());
        if (before_start) aicore_cancel_token_request(token.get());
        aicore::runtime::CancelScope scope(token.get());
        const auto start = std::chrono::steady_clock::now();
        auto* mesh = aicore_instantmesh_generate_image_view(
                ctx.get(), &image, &params,
                before_start ? nullptr : cancel_at_stage, &request, err,
                sizeof(err));
        AICORE_CHECK(mesh == nullptr);
        if (!before_start) AICORE_CHECK(request.triggered);
        AICORE_CHECK(request.calls_after_cancel == 0);
        aicore_instantmesh_mesh_free(mesh);
        AICORE_CHECK(std::strstr(err, "cancel") != nullptr);
        if (before_start || stage == AICORE_INSTANTMESH_STAGE_ENCODE)
            AICORE_CHECK(std::chrono::steady_clock::now() - start <
                         std::chrono::seconds(2));
        aicore_pipeline_timings timings{};
        AICORE_CHECK(aicore_instantmesh_last_pipeline_timings(ctx.get(),
                                                              &timings) == -1);
    };
    canceled(-1);
    if (failures) return 1;
    if (!full) canceled(AICORE_INSTANTMESH_STAGE_ENCODE);
    if (full) {
        canceled(AICORE_INSTANTMESH_STAGE_MULTIVIEW);
        if (failures) return 1;
        canceled(AICORE_INSTANTMESH_STAGE_VAE_DECODE);
    }
    if (failures) return 1;
    if (!full)
        for (int stage :
             {AICORE_INSTANTMESH_STAGE_TRIPLANE, AICORE_INSTANTMESH_STAGE_SYNTH,
              AICORE_INSTANTMESH_STAGE_MESH,
              AICORE_INSTANTMESH_STAGE_TEXTURE}) {
            params.bake_texture = stage == AICORE_INSTANTMESH_STAGE_TEXTURE;
            canceled(stage);
            if (failures) return 1;
        }
    if (!full) {
        params.bake_texture = 0;
        params.grid_res =
                65;  // GPU weight queries cross the 262144-cube chunk boundary.
        canceled(AICORE_INSTANTMESH_STAGE_SYNTH, 3);
        if (failures) return 1;
        params.grid_res = 8;
        params.bake_texture = 1;
        params.texture_res = 256;
        canceled(AICORE_INSTANTMESH_STAGE_TEXTURE, 3);
        if (failures) return 1;
        canceled(AICORE_INSTANTMESH_STAGE_TEXTURE, 2);
        if (failures) return 1;
    }
    params.bake_texture = 0;
    aicore_cancel_token_reset(token.get());
    std::vector<float> first;
    int first_tris = 0;
    for (int iteration = 0; iteration < (full ? 3 : 2); ++iteration) {
        params.seed = full && iteration == 1 ? 42 : 0;
        auto* mesh = aicore_instantmesh_generate_image_view(
                ctx.get(), &image, &params, nullptr, nullptr, err, sizeof(err));
        if (!mesh) {
            std::fprintf(stderr, "Warm-context forward %d failed: %s\n",
                         iteration, err);
            return 1;
        }
        const int nv = aicore_instantmesh_mesh_n_verts(mesh);
        const int nt = aicore_instantmesh_mesh_n_tris(mesh);
        AICORE_CHECK(nv > 0 && nt > 0);
        const auto* verts = aicore_instantmesh_mesh_verts(mesh);
        for (int i = 0; i < nv * 3; ++i) AICORE_CHECK(std::isfinite(verts[i]));
        if (iteration == 0) {
            first.assign(verts, verts + nv * 3);
            first_tris = nt;
        } else if (full && iteration == 1) {
            bool differs = first.size() != static_cast<size_t>(nv) * 3 ||
                           first_tris != nt;
            if (!differs)
                for (size_t i = 0; i < first.size(); ++i)
                    differs = differs || std::abs(first[i] - verts[i]) > 1e-5f;
            AICORE_CHECK(differs);  // Explicit seed=0 must not be replaced by
                                    // context default seed=42.
        } else {
            AICORE_CHECK(first.size() == static_cast<size_t>(nv) * 3);
            AICORE_CHECK(first_tris == nt);
            if (first.size() == static_cast<size_t>(nv) * 3)
                for (size_t i = 0; i < first.size(); ++i)
                    AICORE_CHECK(std::abs(first[i] - verts[i]) < 1e-5f);
        }
        aicore_pipeline_timings timings{};
        AICORE_CHECK(aicore_instantmesh_last_pipeline_timings(ctx.get(),
                                                              &timings) == 0);
        AICORE_CHECK(timings.e2e_ms > 0 && timings.inference_ms > 0);
        std::printf(
                "forward=%d verts=%d tris=%d inference_ms=%.3f e2e_ms=%.3f\n",
                iteration, nv, nt, timings.inference_ms, timings.e2e_ms);
        aicore_instantmesh_mesh_free(mesh);
        auto bad_image = image;
        bad_image.data = nullptr;
        AICORE_CHECK(aicore_instantmesh_generate_image_view(
                             ctx.get(), &bad_image, &params, nullptr, nullptr,
                             err, sizeof(err)) == nullptr);
        AICORE_CHECK(std::strstr(aicore_instantmesh_last_error(ctx.get()),
                                 "image") != nullptr);
        AICORE_CHECK(aicore_instantmesh_last_pipeline_timings(ctx.get(),
                                                              &timings) == -1);
        auto bad_params = params;
        bad_params.struct_size = 0;
        AICORE_CHECK(aicore_instantmesh_generate_image_view(
                             ctx.get(), &image, &bad_params, nullptr, nullptr,
                             nullptr, 0) == nullptr);
        AICORE_CHECK(std::strstr(aicore_instantmesh_last_error(ctx.get()),
                                 "struct_size") != nullptr);
        AICORE_CHECK(aicore_instantmesh_last_pipeline_timings(ctx.get(),
                                                              &timings) == -1);
    }
    aicore_instantmesh_shutdown();
    aicore_instantmesh_shutdown();
    AICORE_CHECK(aicore_instantmesh_is_ready(ctx.get()) == 1);
    canceled(-1);
    ctx.reset();
    ctx.reset(aicore_instantmesh_load_opts_ex(&paths, opts.get(), err,
                                              sizeof(err)));
    AICORE_CHECK(ctx != nullptr);
    return failures ? 1 : 0;
}
