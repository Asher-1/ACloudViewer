// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AnyCalib parity probe: run the ggml graph through the C API on the fixed
// reference input and compare against the ONNX Runtime golden outputs
// (anycalib_gen_v1.0.0_322x322.ref.npz, zip STORED entries — the release
// asset is repacked uncompressed so the probe needs no inflate dependency).
// Emits a JSON summary line and exits 0/1 on the requested numeric gate.
//
// Usage: anycalib_parity_probe <gguf_path> <ref.npz> [--max-abs F]
//        [--rel-rms F] [--repeat N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/anycalib_capi.h"

namespace {

uint32_t ReadU32(const std::vector<uint8_t>& b, size_t o) {
    return (uint32_t)b[o] | ((uint32_t)b[o + 1] << 8) |
           ((uint32_t)b[o + 2] << 16) | ((uint32_t)b[o + 3] << 24);
}

uint16_t ReadU16(const std::vector<uint8_t>& b, size_t o) {
    return (uint16_t)(b[o] | (b[o + 1] << 8));
}

// Minimal npz (zip) reader: walks local file headers, inflates or copies
// member data, parses the npy header, returns float32 rows.
bool LoadNpyFromNpz(const std::vector<uint8_t>& zip,
                    const std::string& name,
                    std::vector<float>* out,
                    std::vector<int64_t>* shape) {
    size_t pos = 0;
    while (pos + 30 <= zip.size()) {
        const uint32_t sig = ReadU32(zip, pos);
        if (sig != 0x04034b50) break;  // end of central directory reached
        const uint16_t method = ReadU16(zip, pos + 8);
        const uint32_t csize = ReadU32(zip, pos + 18);
        const uint16_t name_len = ReadU16(zip, pos + 26);
        const uint16_t extra_len = ReadU16(zip, pos + 28);
        const std::string member((const char*)&zip[pos + 30], name_len);
        const size_t data_pos = pos + 30 + name_len + extra_len;
        pos = data_pos + csize;
        // np.savez prefixes names with nothing; np.savez_compressed members
        // are "<name>.npy".
        if (member != name + ".npy") continue;
        std::vector<uint8_t> raw;
        if (method == 0) {
            raw.assign(zip.begin() + data_pos, zip.begin() + data_pos + csize);
        } else {
            return false;  // release assets are ZIP_STORED by convention
        }
        // npy header: \x93NUMPY, major, minor, header_len, dict with
        // 'descr': '<f4', 'shape': (..., ...).
        size_t hpos = 8;
        const uint16_t header_len = ReadU16(raw, 8);
        hpos = 10;
        std::string header((const char*)&raw[hpos], header_len);
        const size_t shape_pos = header.find("'shape'");
        if (shape_pos == std::string::npos ||
            header.find("'<f4'") == std::string::npos) {
            return false;
        }
        const size_t open = header.find('(', shape_pos);
        const size_t close = header.find(')', open);
        std::string dims = header.substr(open + 1, close - open - 1);
        shape->clear();
        size_t start = 0;
        while (start <= dims.size()) {
            size_t comma = dims.find(',', start);
            if (comma == std::string::npos) comma = dims.size();
            const std::string tok = dims.substr(start, comma - start);
            if (!tok.empty()) shape->push_back(atoll(tok.c_str()));
            if (comma == dims.size()) break;
            start = comma + 1;
        }
        int64_t total = 1;
        for (int64_t d : *shape) total *= d;
        out->resize((size_t)total);
        const size_t data_start = hpos + header_len;
        std::memcpy(out->data(), &raw[data_start], total * sizeof(float));
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf;
    std::string ref;
    float max_abs_allowed = 0.05f;
    float rel_rms_allowed = 0.02f;
    int repeat = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--max-abs" && i + 1 < argc)
            max_abs_allowed = atof(argv[++i]);
        else if (a == "--rel-rms" && i + 1 < argc)
            rel_rms_allowed = atof(argv[++i]);
        else if (a == "--repeat" && i + 1 < argc)
            repeat = atoi(argv[++i]);
        else if (gguf.empty())
            gguf = a;
        else if (ref.empty())
            ref = a;
    }
    if (gguf.empty() || ref.empty()) {
        fprintf(stderr,
                "usage: %s <gguf> <ref.npz> [--max-abs F] [--rel-rms F] "
                "[--repeat N]\n",
                argv[0]);
        return 2;
    }

    FILE* fp = fopen(ref.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", ref.c_str());
        return 2;
    }
    std::vector<uint8_t> zip;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0)
        zip.insert(zip.end(), buf, buf + n);
    fclose(fp);

    std::vector<float> image, rays_ref, tangent_ref;
    std::vector<int64_t> shape;
    if (!LoadNpyFromNpz(zip, "image", &image, &shape) ||
        !LoadNpyFromNpz(zip, "rays", &rays_ref, &shape) ||
        !LoadNpyFromNpz(zip, "tangent_coords", &tangent_ref, &shape)) {
        fprintf(stderr, "ref.npz missing members\n");
        return 2;
    }

    aicore_anycalib_options* opts = aicore_anycalib_options_new();
    aicore_anycalib_options_set_device(opts, "cpu");
    aicore_anycalib_ctx* ctx = aicore_anycalib_load_opts(gguf.c_str(), opts);
    aicore_anycalib_options_free(opts);
    if (!aicore_anycalib_is_ready(ctx)) {
        fprintf(stderr, "load failed: %s\n",
                aicore_anycalib_last_error(ctx)
                        ? aicore_anycalib_last_error(ctx)
                        : "unknown");
        return 1;
    }

    double best_rays_rel_rms = 1e30, best_tan_rel_rms = 1e30;
    double best_rays_max_abs = 1e30, best_tan_max_abs = 1e30;
    double p50 = 0, p95 = 0;
    std::vector<double> walls;
    for (int r = 0; r < repeat; ++r) {
        float* rays = nullptr;
        float* tangent = nullptr;
        int32_t count = 0;
        const auto t0 = std::chrono::steady_clock::now();
        const int rc =
                aicore_anycalib_run(ctx, image.data(), &rays, &tangent, &count);
        const auto t1 = std::chrono::steady_clock::now();
        if (rc != 0) {
            fprintf(stderr, "run failed: %s\n",
                    aicore_anycalib_last_error(ctx)
                            ? aicore_anycalib_last_error(ctx)
                            : "unknown");
            return 1;
        }
        walls.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        double se_r = 0, se_t = 0, max_r = 0, max_t = 0, norm2_r = 0,
               norm2_t = 0;
        for (size_t i = 0; i < (size_t)count * 3; ++i) {
            const double d = rays[i] - rays_ref[i];
            se_r += d * d;
            norm2_r += double(rays_ref[i]) * rays_ref[i];
            max_r = std::max(max_r, std::abs(d));
        }
        for (size_t i = 0; i < (size_t)count * 2; ++i) {
            const double d = tangent[i] - tangent_ref[i];
            se_t += d * d;
            norm2_t += double(tangent_ref[i]) * tangent_ref[i];
            max_t = std::max(max_t, std::abs(d));
        }
        best_rays_rel_rms =
                std::min(best_rays_rel_rms, std::sqrt(se_r / norm2_r));
        best_tan_rel_rms =
                std::min(best_tan_rel_rms, std::sqrt(se_t / norm2_t));
        best_rays_max_abs = std::min(best_rays_max_abs, max_r);
        best_tan_max_abs = std::min(best_tan_max_abs, max_t);
        aicore_anycalib_free_buffer(rays);
        aicore_anycalib_free_buffer(tangent);
    }
    std::sort(walls.begin(), walls.end());
    p50 = walls[walls.size() / 2];
    p95 = walls[(size_t)(0.95 * (walls.size() - 1))];
    aicore_pipeline_timings timings{};
    aicore_anycalib_last_pipeline_timings(ctx, &timings);

    printf("{\"gate\":\"anycalib_parity\",\"rays_rel_rms\":%.6g,"
           "\"tangent_rel_rms\":%.6g,\"rays_max_abs\":%.6g,"
           "\"tangent_max_abs\":%.6g,\"wall_ms_p50\":%.3f,"
           "\"wall_ms_p95\":%.3f,\"inference_ms\":%.3f,\"verdict\":\"%s\"}\n",
           best_rays_rel_rms, best_tan_rel_rms, best_rays_max_abs,
           best_tan_max_abs, p50, p95, timings.inference_ms,
           (best_rays_rel_rms <= rel_rms_allowed &&
            best_tan_rel_rms <= rel_rms_allowed &&
            best_rays_max_abs <= max_abs_allowed &&
            best_tan_max_abs <= max_abs_allowed)
                   ? "PASS"
                   : "FAIL");
    aicore_anycalib_free(ctx);
    aicore_anycalib_shutdown();
    return 0;
}
