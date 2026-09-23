// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SAM 3D stage-bisect probe (white-box): runs the native e2e pipeline from a
// FIXED condition directory (e.g. one dumped by the upstream sam3d-cli
// --conditions-out), optionally with a FIXED noise directory. With both
// pinned, any output divergence versus the upstream reference is attributable
// to kernel execution rather than to conditioning or sampling noise.
//
// Links AICore_test (static white-box link) because E2eOptions/cmd_e2e are
// internal pipeline surfaces that must never leak through libAICore exports.
//
// usage:
//   sam3d_stage_bisect <models_dir> <cond_dir> [--noise-dir <dir>] [--seed N]
//       [--steps N] [--dtype f16|q8_0|q4_k] [--backend auto|cpu|cuda|vulkan]
//       [--out-ply <path>] [--mesh-vertices <path>] [--mesh-faces <path>]
//       [--dbg-dir <dir>] [--dump-slat-steps] [--ss-flow-only]
//       [--slat-flow-only]
//       [--debug-stage <name>] [--dino-dbg <out.bin>]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "e2e_options.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <models_dir> <cond_dir> [--noise-dir <dir>] "
                     "[--seed N] [--steps N] [--dtype d] [--backend b] "
                     "[--out-ply p] [--mesh-vertices p] [--mesh-faces p] "
                     "[--dbg-dir d] [--ss-flow-only] [--slat-flow-only]\n",
                     argv[0]);
        return 1;
    }

    sam3d::E2eOptions e2e;
    e2e.models_dir = argv[1];
    e2e.cond_dir = argv[2];
    e2e.backend = "cuda";
    e2e.dtype = "q4_k";
    e2e.seed = 42;
    e2e.threads = 8;
    e2e.ss_steps = 25;
    e2e.slat_steps = 25;
    // The image-to-3D production path always sets this (session.cpp).
    e2e.cond_manual_attention = true;
    // Throughput default, same as the C API options contract.
    e2e.ss_strict_attention = false;

    std::string out_ply = "sam3d_stage_bisect.ply";
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return argv[++i]; };
        if (arg == "--noise-dir")
            e2e.noise_dir = next();
        else if (arg == "--stage")
            e2e.stage = next();
        else if (arg == "--debug-stage")
            e2e.debug_stage = next();
        else if (arg == "--dino-dbg") {
            e2e.dino_dbg = true;
            e2e.dino_dbg_out = next();
        } else if (arg == "--seed")
            e2e.seed = static_cast<unsigned>(std::atoi(next()));
        else if (arg == "--steps")
            e2e.ss_steps = e2e.slat_steps = std::atoi(next());
        else if (arg == "--dtype")
            e2e.dtype = next();
        else if (arg == "--backend")
            e2e.backend = next();
        else if (arg == "--out-ply")
            out_ply = next();
        else if (arg == "--mesh-vertices")
            e2e.out_mesh_vertices = next();
        else if (arg == "--mesh-faces")
            e2e.out_mesh_faces = next();
        else if (arg == "--dbg-dir")
            e2e.dbg_dir = next();
        else if (arg == "--dump-slat-steps")
            e2e.dump_slat_steps = true;
        else if (arg == "--ss-flow-only")
            e2e.ss_flow_only = true;
        else if (arg == "--slat-flow-only")
            e2e.slat_flow_only = true;
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    e2e.out_ply = out_ply;

    const int rc = sam3d::cmd_e2e(e2e);
    std::printf(
            "{\"probe\":\"sam3d_stage_bisect\",\"cond_dir\":\"%s\","
            "\"noise_dir\":\"%s\",\"seed\":%u,\"rc\":%d}\n",
            e2e.cond_dir.c_str(), e2e.noise_dir.c_str(), e2e.seed, rc);
    return rc == 0 ? 0 : 1;
}
