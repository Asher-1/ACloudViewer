// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// free_splatter-cli: load a model, run inference over N views, and write the
// gaussians as raw floats (--out) and/or an antimatter15 .splat (--splat) for
// the web viewer. Inputs are either ONE raw .f32 file (view-major NCHW, [0,1])
// or several image files (jpg/png/...), each decoded, center-cropped to a
// square, and resized to the model resolution.
//
//   free_splatter-cli [--device DEV] [--splat OUT.splat] [--out OUT.f32]
//                     [--opacity-threshold T] [--max-splats N] [--dump-taps
//                     DIR] MODEL.gguf  (IMAGES... | INPUT.f32)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "free_splatter.h"

static bool ends_with(const std::string &s, const char *suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// Decode an image and preprocess like FreeSplatter's scene path: center-crop to
// a square, resize to size x size, scale to [0,1], lay out CHW. Appends to
// `out`.
static bool load_image_chw(const char *path,
                           int size,
                           std::vector<float> &out) {
    const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::fprintf(stderr, "decode failed: %s\n", path);
        return false;
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    const int w = rgb.cols, h = rgb.rows;
    const int s = std::min(w, h);
    const int left = (w - s) / 2, top = (h - s) / 2;
    const cv::Mat sq = rgb(cv::Rect(left, top, s, s));

    cv::Mat rz;
    cv::resize(sq, rz, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);

    const size_t base = out.size();
    out.resize(base + (size_t)3 * size * size);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
                out[base + (size_t)c * size * size + (size_t)y * size + x] =
                        rz.at<cv::Vec3b>(y, x)[c] / 255.0f;
    return true;
}

// Convert the engine's activated gaussians [n*gc] to an antimatter15 .splat:
// 32 bytes/splat = pos(3 f32), scale(3 f32), rgba(4 u8), rot(4 u8: w,x,y,z).
// Prunes opacity <= threshold, sorts by importance (opacity*volume), caps to
// max_splats (0 = all).
static bool write_splat(const float *g,
                        size_t n,
                        int gc,
                        float opac_thr,
                        size_t max_splats,
                        const char *path) {
    if (free_splatter_export_splat(g, n, gc, opac_thr, max_splats, path) != 0) {
        std::fprintf(stderr, "cannot write %s\n", path);
        return false;
    }
    std::printf("wrote %s\n", path);
    return true;
}

// Write an accumulated gaussian cloud as an antimatter15 .splat, emitting each
// point's true anisotropic scale + rotation (carried through the Sim(3)) AND
// its activated opacity as the splat alpha — so it renders exactly like the
// single-run write_splat (proper alpha blending), not a fully-opaque, swirling,
// blurry soup. scale_mult scales the radii (1.0 = as-predicted). When capping,
// keep the most important splats (opacity * volume), matching write_splat (not
// a uniform stride).
static bool write_cloud_splat(const free_splatter_point *pts,
                              size_t n,
                              size_t max_splats,
                              float scale_mult,
                              const char *path) {
    if (free_splatter_export_cloud_splat(pts, n, max_splats, scale_mult,
                                         path) != 0) {
        std::fprintf(stderr, "cannot write %s\n", path);
        return false;
    }
    std::printf("wrote %s: up to %zu splats (of %zu cloud points)\n", path,
                max_splats > 0 ? std::min(max_splats, n) : n, n);
    return true;
}

static int usage(const char *a0) {
    std::fprintf(
            stderr,
            "usage: %s [--device DEV] [--splat OUT.splat] [--out OUT.f32]\n"
            "          [--opacity-threshold T] [--max-splats N] [--dump-taps "
            "DIR]\n"
            "          MODEL.gguf (IMAGES... | INPUT.f32)\n"
            "  --device defaults to auto (Vulkan then CPU; Metal then CPU on "
            "macOS).\n"
            "  Use --device cpu to force the portable pure CPU backend.\n"
            "\n"
            "accumulate mode (>=2 images -> one world from the photo stream):\n"
            "  %s --accumulate [--splat-prefix P] [--fuse] [--voxel V] "
            "[--fuse-k K]\n"
            "     [--min-parallax DEG] [--splat-scale S] [--max-splats N] "
            "MODEL.gguf IMG0 IMG1 ...\n"
            "  runs the engine on each consecutive pair, chains the runs "
            "(Sim(3)) into\n"
            "  one accumulating cloud; writes P_<nframes>.splat after each "
            "pair (the\n"
            "  evolving reconstruction) and, with --fuse, a consensus-fused "
            "P_fused.splat.\n"
            "  --min-parallax gates by keyframe: a candidate frame is folded "
            "in only if its\n"
            "  triangulation angle vs the last kept frame is >= DEG (else its "
            "depth is\n"
            "  ill-conditioned); image mode only.\n"
            "\n"
            "parallax (depth conditioning of one pair, after-inference):\n"
            "  %s --parallax MODEL.gguf (IMG0 IMG1 | PAIR.f32)\n",
            a0, a0, a0);
    return 2;
}

int main(int argc, char **argv) {
    const char *device = nullptr, *taps_dir = nullptr, *out_path = nullptr,
               *splat_path = nullptr;
    const char *model = nullptr, *splat_prefix = nullptr;
    float opac_thr = 5e-3f;
    long max_splats = 0;
    bool accumulate = false, fuse = false, parallax = false, tree = false,
         tree_stages = false;
    int tov_block = 4,
        tov_overlap =
                2;  // tree: submap width / shared frames (2/1 = overlap-by-one)
    int fuse_mode = 1;  // 0 averaged, 1 kept, 2 best-frame
    float voxel = 0.02f,
          splat_scale = 1.0f;  // multiplier on the predicted gaussian radii
    int fuse_k = 2;
    bool refine = false;
    int refine_iters = 8;
    float refine_voxel = 0.03f, refine_alpha = 0.5f;  // geometric de-ghost
    float min_parallax = 0.0f;  // deg; >0 gates accumulate by keyframe parallax
    std::vector<std::string> inputs;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--device" && i + 1 < argc)
            device = argv[++i];
        else if (a == "--dump-taps" && i + 1 < argc)
            taps_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc)
            out_path = argv[++i];
        else if (a == "--splat" && i + 1 < argc)
            splat_path = argv[++i];
        else if (a == "--opacity-threshold" && i + 1 < argc)
            opac_thr = (float)atof(argv[++i]);
        else if (a == "--max-splats" && i + 1 < argc)
            max_splats = atol(argv[++i]);
        else if (a == "--accumulate")
            accumulate = true;
        else if (a == "--accumulate-tree") {
            accumulate = true;
            tree = true;
        } else if (a == "--tree-stages") {
            accumulate = true;
            tree_stages = true;
        } else if (a == "--tree-block" && i + 1 < argc)
            tov_block = atoi(argv[++i]);
        else if (a == "--tree-overlap" && i + 1 < argc)
            tov_overlap = atoi(argv[++i]);
        else if (a == "--parallax")
            parallax = true;
        else if (a == "--splat-prefix" && i + 1 < argc)
            splat_prefix = argv[++i];
        else if (a == "--fuse")
            fuse = true;
        else if (a == "--voxel" && i + 1 < argc)
            voxel = (float)atof(argv[++i]);
        else if (a == "--fuse-k" && i + 1 < argc)
            fuse_k = atoi(argv[++i]);
        else if (a == "--fuse-mode" && i + 1 < argc) {
            std::string m = argv[++i];
            fuse_mode = (m == "averaged") ? 0 : (m == "best") ? 2 : 1;
        } else if (a == "--refine")
            refine = true;
        else if (a == "--refine-iters" && i + 1 < argc)
            refine_iters = atoi(argv[++i]);
        else if (a == "--refine-voxel" && i + 1 < argc)
            refine_voxel = (float)atof(argv[++i]);
        else if (a == "--refine-alpha" && i + 1 < argc)
            refine_alpha = (float)atof(argv[++i]);
        else if (a == "--min-parallax" && i + 1 < argc)
            min_parallax = (float)atof(argv[++i]);
        else if (a == "--splat-scale" && i + 1 < argc)
            splat_scale = (float)atof(argv[++i]);
        else if (a == "-h" || a == "--help")
            return usage(argv[0]);
        else if (!model)
            model = argv[i];
        else
            inputs.push_back(argv[i]);
    }
    if (!model) return usage(argv[0]);

    free_splatter_options *opts = free_splatter_options_new();
    if (device) free_splatter_options_set_device(opts, device);
    if (taps_dir) free_splatter_options_set_dump_taps_dir(opts, taps_dir);
    free_splatter_ctx *ctx = free_splatter_load(model, opts);
    free_splatter_options_free(opts);
    if (!ctx) {
        std::fprintf(stderr, "load: out of memory\n");
        return 1;
    }
    if (const char *err = free_splatter_last_error(ctx)) {
        std::fprintf(stderr, "load failed: %s\n", err);
        // Default device is GPU/Vulkan (fail-closed): if none was requested
        // explicitly, point the user at the CPU opt-in instead of guessing.
        if (!device)
            std::fprintf(stderr,
                         "  (no GPU device — pass --device cpu to run on CPU, "
                         "or build with FREE_SPLATTER_VULKAN)\n");
        free_splatter_free(ctx);
        return 1;
    }

    free_splatter_geometry geo;
    free_splatter_geometry_of(ctx, &geo);
    std::printf("model: %dx%d, in=%d, gaussian_channels=%d\n", geo.image_width,
                geo.image_height, geo.in_channels, geo.gaussian_channels);
    if (inputs.empty()) {
        free_splatter_free(ctx);
        return 0;
    }

    // ---- parallax mode: depth-conditioning of a pair (after-inference)
    // -------- Input: exactly two images (engine runs the pair) or one
    // [2,H,W,gc] .f32 gaussian-output dump. Prints the recovered-geometry
    // parallax stats.
    if (parallax) {
        const int gc = geo.gaussian_channels;
        const size_t pair_floats =
                (size_t)2 * geo.image_height * geo.image_width * gc;
        std::vector<float> gbuf;
        if (inputs.size() == 1 && ends_with(inputs[0], ".f32")) {
            std::ifstream f(inputs[0], std::ios::binary | std::ios::ate);
            if (!f) {
                std::fprintf(stderr, "cannot open %s\n", inputs[0].c_str());
                free_splatter_free(ctx);
                return 1;
            }
            const std::streamsize bytes = f.tellg();
            f.seekg(0);
            if ((size_t)(bytes / sizeof(float)) != pair_floats) {
                std::fprintf(stderr,
                             "dump %s: %zu floats, expected 2*%d*%d*%d\n",
                             inputs[0].c_str(), (size_t)(bytes / sizeof(float)),
                             geo.image_height, geo.image_width, gc);
                free_splatter_free(ctx);
                return 1;
            }
            gbuf.resize(pair_floats);
            f.read((char *)gbuf.data(), bytes);
        } else if (inputs.size() == 2) {
            std::vector<float> img;
            for (const std::string &p : inputs)
                if (!load_image_chw(p.c_str(), geo.image_width, img)) {
                    free_splatter_free(ctx);
                    return 1;
                }
            float *out = nullptr;
            size_t n_out = 0;
            if (free_splatter_run(ctx, img.data(), 2, geo.image_height,
                                  geo.image_width, &out, &n_out) != 0) {
                std::fprintf(stderr, "run failed: %s\n",
                             free_splatter_last_error(ctx));
                free_splatter_free(ctx);
                return 1;
            }
            gbuf.assign(out, out + n_out);
            free_splatter_buf_free(out);
        } else {
            std::fprintf(stderr,
                         "--parallax needs exactly 2 images or one .f32 pair "
                         "dump\n");
            free_splatter_free(ctx);
            return 2;
        }
        free_splatter_parallax px;
        if (free_splatter_pair_parallax(gbuf.data(), 2, geo.image_height,
                                        geo.image_width, gc, opac_thr,
                                        &px) != 0) {
            std::fprintf(stderr, "parallax failed\n");
            free_splatter_free(ctx);
            return 1;
        }
        std::printf(
                "parallax: tri_angle=%.3f deg  lateral_angle=%.2f deg  "
                "B/Z=%.4f  baseline=%.4f  median_depth=%.4f  focal=%.1f  "
                "npts=%d\n",
                px.tri_angle_deg, px.lateral_angle_deg, px.baseline_over_depth,
                px.baseline, px.median_depth, px.focal, px.n_points);
        free_splatter_free(ctx);
        return 0;
    }

    // ---- accumulate mode: chain a photo stream into one world
    // ----------------- Two input forms: a stream of IMAGES (the engine runs on
    // each consecutive pair), or pre-computed `.f32` pair dumps (each a
    // [2,H,W,gc] engine output — the engine is skipped, for fast re-bakes /
    // fusion sweeps off cached runs).
    if (accumulate) {
        const int sz = geo.image_width, gc = geo.gaussian_channels;
        const bool dump_mode = std::all_of(
                inputs.begin(), inputs.end(),
                [](const std::string &s) { return ends_with(s, ".f32"); });
        if (!dump_mode && inputs.size() < 2) {
            std::fprintf(
                    stderr,
                    "--accumulate needs >=2 images (or N .f32 pair dumps)\n");
            free_splatter_free(ctx);
            return 2;
        }
        const size_t npairs = dump_mode ? inputs.size() : inputs.size() - 1;
        const int64_t per_view = (int64_t)geo.in_channels * sz * sz;
        const size_t pair_floats = (size_t)2 * sz * sz * gc;

        // image mode: decode every frame once (CHW per view)
        std::vector<std::vector<float>> frames;
        if (!dump_mode)
            for (const std::string &p : inputs) {
                std::vector<float> img;
                if (!load_image_chw(p.c_str(), sz, img)) {
                    free_splatter_free(ctx);
                    return 1;
                }
                frames.push_back(std::move(img));
            }

        free_splatter_accumulator *acc =
                free_splatter_accumulator_new(sz, sz, opac_thr);
        if (!acc) {
            std::fprintf(stderr, "accumulator alloc failed\n");
            free_splatter_free(ctx);
            return 1;
        }

        // --accumulate-tree: collect the (gated) pair engine outputs and merge
        // them up a balanced binary tree instead of chaining linearly (less
        // compounded drift).
        std::vector<std::vector<float>> tree_pairs;

        // emit the accumulated cloud after a pair is folded in (optional
        // de-ghost)
        auto emit_step = [&](int nframes) {
            if (!splat_prefix) return;
            free_splatter_point *cloud = nullptr;
            size_t nc = 0;
            free_splatter_accumulator_cloud(acc, &cloud, &nc);
            if (refine &&
                nframes >= 2)  // geometric de-ghost a copy (internal untouched)
                free_splatter_refine_cloud(cloud, nc, refine_voxel,
                                           refine_iters, refine_alpha);
            char path[1024];
            std::snprintf(path, sizeof path, "%s_%d.splat", splat_prefix,
                          nframes);
            write_cloud_splat(cloud, nc, (size_t)max_splats, splat_scale, path);
            free_splatter_buf_free(cloud);
        };

        if (dump_mode) {
            // fixed pre-computed pairs: cannot re-anchor, so the parallax gate
            // (which works by re-pairing keyframes) does not apply here.
            if (min_parallax > 0.0f)
                std::fprintf(
                        stderr,
                        "note: --min-parallax re-pairs keyframes; ignored for "
                        ".f32 dumps (fixed pairs) -- accumulating all.\n");
            for (size_t k = 0; k < npairs; k++) {
                std::ifstream f(inputs[k], std::ios::binary | std::ios::ate);
                if (!f) {
                    std::fprintf(stderr, "cannot open %s\n", inputs[k].c_str());
                    free_splatter_accumulator_free(acc);
                    free_splatter_free(ctx);
                    return 1;
                }
                const std::streamsize bytes = f.tellg();
                f.seekg(0);
                if ((size_t)(bytes / sizeof(float)) != pair_floats) {
                    std::fprintf(stderr,
                                 "dump %s: %zu floats, expected 2*%d*%d*%d\n",
                                 inputs[k].c_str(),
                                 (size_t)(bytes / sizeof(float)), sz, sz, gc);
                    free_splatter_accumulator_free(acc);
                    free_splatter_free(ctx);
                    return 1;
                }
                std::vector<float> dumpbuf(pair_floats);
                f.read((char *)dumpbuf.data(), bytes);
                if (tree || tree_stages) {
                    tree_pairs.push_back(std::move(dumpbuf));
                    std::printf("pair %zu collected\n", k);
                    continue;
                }
                free_splatter_accumulator_add_pair(acc, dumpbuf.data(), gc);
                const int nframes = free_splatter_accumulator_frame_count(acc);
                std::printf("pair %zu -> %d frames\n", k, nframes);
                emit_step(nframes);
            }
        } else {
            // image mode: keyframe gating. Pair the next candidate against the
            // last KEPT frame; fold it in only if its parallax clears
            // --min-parallax (else its depth is ill-conditioned). Threshold 0
            // -> every consecutive pair, identical to the ungated path.
            size_t last_kept = 0;
            int skipped = 0;
            for (size_t j = 1; j < frames.size(); j++) {
                std::vector<float> pair(2 * per_view);
                std::memcpy(&pair[0], frames[last_kept].data(),
                            per_view * sizeof(float));
                std::memcpy(&pair[per_view], frames[j].data(),
                            per_view * sizeof(float));
                float *runout = nullptr;
                size_t ng = 0;
                if (free_splatter_run(ctx, pair.data(), 2, sz, sz, &runout,
                                      &ng) != 0) {
                    std::fprintf(stderr, "run pair (%zu,%zu) failed: %s\n",
                                 last_kept, j, free_splatter_last_error(ctx));
                    free_splatter_accumulator_free(acc);
                    free_splatter_free(ctx);
                    return 1;
                }
                if (min_parallax > 0.0f) {
                    free_splatter_parallax px;
                    free_splatter_pair_parallax(runout, 2, sz, sz, gc, opac_thr,
                                                &px);
                    if (px.tri_angle_deg < min_parallax) {
                        std::printf(
                                "skip frame %zu: parallax %.1f deg < %.1f (vs "
                                "kept frame %zu)\n",
                                j, px.tri_angle_deg, min_parallax, last_kept);
                        free_splatter_buf_free(runout);
                        skipped++;
                        continue;
                    }
                    std::printf(
                            "keep frame %zu: parallax %.1f deg (vs kept frame "
                            "%zu)\n",
                            j, px.tri_angle_deg, last_kept);
                }
                if (tree || tree_stages) {
                    tree_pairs.emplace_back(runout, runout + ng);
                    free_splatter_buf_free(runout);
                    std::printf("pair (%zu,%zu) collected\n", last_kept, j);
                    last_kept = j;
                    continue;
                }
                free_splatter_accumulator_add_pair(acc, runout, gc);
                free_splatter_buf_free(runout);
                const int nframes = free_splatter_accumulator_frame_count(acc);
                std::printf("pair (%zu,%zu) -> %d frames\n", last_kept, j,
                            nframes);
                emit_step(nframes);
                last_kept = j;
            }
            if (skipped)
                std::printf("gated: skipped %d low-parallax frame(s)\n",
                            skipped);
        }
        if (tree || tree_stages) {
            std::vector<const float *> ps;
            ps.reserve(tree_pairs.size());
            for (auto &v : tree_pairs) ps.push_back(v.data());
            if (ps.empty()) {
                std::fprintf(stderr, "tree: need >=1 pair\n");
                free_splatter_accumulator_free(acc);
                free_splatter_free(ctx);
                return 1;
            }
            const std::string pfx = splat_prefix ? splat_prefix : "stage";

            if (tree_stages) {
                // write one laid-out .splat per merge level (stage 0 = the
                // independent leaf scenes side by side, ... then the single
                // merged world, then a consensus-fused clean scene) + a
                // manifest the viewer steps through.
                std::string dir = ".", base = pfx;
                const size_t slash = pfx.find_last_of('/');
                if (slash != std::string::npos) {
                    dir = pfx.substr(0, slash);
                    base = pfx.substr(slash + 1);
                }
                // per-scene budget: each laid-out scene keeps its own detail (a
                // single shared cap would starve every scene to ~1/N and fill
                // it with floaters).
                const int per_node = (max_splats > 0 && max_splats < 200000)
                                             ? (int)max_splats
                                             : 150000;
                std::vector<std::string> labels;
                auto stage_path = [&](char *p, size_t cap) {
                    std::snprintf(p, cap, "%s_%zu.splat", pfx.c_str(),
                                  labels.size());
                };
                // multi-scene stages of the multi-frame-overlap tree: each
                // laid-out scene gets its own per_node budget.
                for (int L = 0; L <= 64; L++) {
                    free_splatter_point *sc = nullptr;
                    size_t nsc = 0;
                    int nnodes = 0;
                    if (free_splatter_tree_overlap(
                                ps.data(), (int)ps.size(), gc, sz, sz, opac_thr,
                                tov_block, tov_overlap, L,
                                -1.0f /*auto layout*/, per_node, &sc, &nsc,
                                &nnodes) != 0) {
                        std::fprintf(stderr, "tree-stages failed at level %d\n",
                                     L);
                        free_splatter_accumulator_free(acc);
                        free_splatter_free(ctx);
                        return 1;
                    }
                    if (nnodes <= 1) {
                        free_splatter_buf_free(sc);
                        break;
                    }  // single scene: full budget below
                    char path[1024];
                    stage_path(path, sizeof path);
                    write_cloud_splat(sc, nsc, 0 /*already per-node capped*/,
                                      splat_scale, path);
                    free_splatter_buf_free(sc);
                    std::printf("stage %zu: %d scenes\n", labels.size(),
                                nnodes);
                    labels.push_back(L == 0 ? (std::to_string(nnodes) +
                                               " independent scenes")
                                            : ("merge " + std::to_string(L) +
                                               " — " + std::to_string(nnodes) +
                                               " scenes"));
                }
                // the single merged world, at full budget: first the raw union
                // (the overlapping per-frame copies, ghosted), then the
                // consensus-fused (best-frame) clean scene — same contrast as
                // the linear demo's acc/fused.
                free_splatter_point *root = nullptr;
                size_t nroot = 0;
                if (free_splatter_tree_overlap(ps.data(), (int)ps.size(), gc,
                                               sz, sz, opac_thr, tov_block,
                                               tov_overlap, -1, 0.0f, 0, &root,
                                               &nroot, nullptr) == 0) {
                    char path[1024];
                    stage_path(path, sizeof path);
                    write_cloud_splat(root, nroot, (size_t)max_splats,
                                      splat_scale, path);
                    std::printf("stage %zu: merged raw union (%zu pts)\n",
                                labels.size(), nroot);
                    labels.push_back("merged — one scene (raw union)");
                    free_splatter_point *fz = nullptr;
                    size_t nfz = 0;
                    if (free_splatter_fuse_cloud(root, nroot, voxel, fuse_k,
                                                 2 /*best-frame*/, &fz,
                                                 &nfz) == 0 &&
                        nfz > 0) {
                        char path2[1024];
                        stage_path(path2, sizeof path2);
                        write_cloud_splat(fz, nfz, (size_t)max_splats,
                                          splat_scale, path2);
                        std::printf(
                                "stage %zu: consensus-fused (%zu -> %zu pts)\n",
                                labels.size(), nroot, nfz);
                        labels.push_back("consensus-fused — one clean scene");
                        free_splatter_buf_free(fz);
                    }
                    free_splatter_buf_free(root);
                }
                std::ofstream mf(dir + "/manifest.json");
                mf << "{ \"reframe\": true, \"steps\": [\n";
                for (size_t L = 0; L < labels.size(); L++)
                    mf << "    {\"splat\":\"" << base << "_" << L
                       << ".splat\",\"images\":[],\"n\":" << (L + 1)
                       << ",\"label\":\"" << labels[L] << "\"}"
                       << (L + 1 < labels.size() ? "," : "") << "\n";
                mf << "  ] }\n";
                std::printf("tree-stages: %zu stages -> %s/manifest.json\n",
                            labels.size(), dir.c_str());
            } else {
                // one merged world from the overlap tree (block=tov_block,
                // overlap=tov_overlap; 2/1 = the plain overlap-by-one tree).
                // With --fuse, also the de-ghosted scene.
                std::printf("tree-accumulate: %zu pairs, block=%d overlap=%d\n",
                            ps.size(), tov_block, tov_overlap);
                free_splatter_point *tc = nullptr;
                size_t ntc = 0;
                if (free_splatter_tree_overlap(ps.data(), (int)ps.size(), gc,
                                               sz, sz, opac_thr, tov_block,
                                               tov_overlap, -1, 0.0f, 0, &tc,
                                               &ntc, nullptr) != 0) {
                    std::fprintf(stderr, "tree accumulate failed\n");
                    free_splatter_accumulator_free(acc);
                    free_splatter_free(ctx);
                    return 1;
                }
                char path[1024];
                std::snprintf(path, sizeof path, "%s_tree.splat", pfx.c_str());
                write_cloud_splat(tc, ntc, (size_t)max_splats, splat_scale,
                                  path);
                if (fuse) {  // best-frame consensus, de-ghosted
                    free_splatter_point *fz = nullptr;
                    size_t nfz = 0;
                    if (free_splatter_fuse_cloud(tc, ntc, voxel, fuse_k, 2, &fz,
                                                 &nfz) == 0 &&
                        nfz > 0) {
                        std::snprintf(path, sizeof path, "%s_tree_fused.splat",
                                      pfx.c_str());
                        write_cloud_splat(fz, nfz, (size_t)max_splats,
                                          splat_scale, path);
                        free_splatter_buf_free(fz);
                    }
                }
                free_splatter_buf_free(tc);
            }
            free_splatter_accumulator_free(acc);
            free_splatter_free(ctx);
            return 0;
        }
        if (fuse && splat_prefix) {
            if (refine)
                free_splatter_accumulator_refine(acc, refine_voxel,
                                                 refine_iters, refine_alpha);
            free_splatter_point *fc = nullptr;
            size_t nf = 0;
            free_splatter_accumulator_fuse(acc, voxel, fuse_k, fuse_mode, &fc,
                                           &nf);
            char path[1024];
            std::snprintf(path, sizeof path, "%s_fused.splat", splat_prefix);
            write_cloud_splat(fc, nf, (size_t)max_splats, splat_scale, path);
            free_splatter_buf_free(fc);
        }
        free_splatter_accumulator_free(acc);
        free_splatter_free(ctx);
        return 0;
    }

    // assemble input: one raw .f32, or several decoded images
    std::vector<float> buf;
    int32_t n_views = 0;
    const int64_t per_view =
            (int64_t)geo.in_channels * geo.image_height * geo.image_width;
    if (inputs.size() == 1 && ends_with(inputs[0], ".f32")) {
        std::ifstream f(inputs[0], std::ios::binary | std::ios::ate);
        if (!f) {
            std::fprintf(stderr, "cannot open %s\n", inputs[0].c_str());
            free_splatter_free(ctx);
            return 1;
        }
        const std::streamsize bytes = f.tellg();
        f.seekg(0);
        buf.resize(bytes / sizeof(float));
        f.read((char *)buf.data(), bytes);
        if (per_view == 0 || buf.size() % per_view != 0) {
            std::fprintf(stderr, "input size is not a whole number of views\n");
            free_splatter_free(ctx);
            return 1;
        }
        n_views = (int32_t)(buf.size() / per_view);
    } else {
        for (const std::string &p : inputs)
            if (!load_image_chw(p.c_str(), geo.image_width, buf)) {
                free_splatter_free(ctx);
                return 1;
            }
        n_views = (int32_t)inputs.size();
    }

    float *out = nullptr;
    size_t n_out = 0;
    if (free_splatter_run(ctx, buf.data(), n_views, geo.image_height,
                          geo.image_width, &out, &n_out) != 0) {
        std::fprintf(stderr, "run failed: %s\n", free_splatter_last_error(ctx));
        free_splatter_free(ctx);
        return 1;
    }
    std::printf("ran %d views -> %zu gaussian floats\n", n_views, n_out);

    if (out_path) {
        std::ofstream o(out_path, std::ios::binary);
        o.write((const char *)out, (std::streamsize)n_out * sizeof(float));
    }
    if (splat_path)
        write_splat(out, n_out / geo.gaussian_channels, geo.gaussian_channels,
                    opac_thr, (size_t)max_splats, splat_path);

    free_splatter_buf_free(out);
    free_splatter_free(ctx);
    return 0;
}
