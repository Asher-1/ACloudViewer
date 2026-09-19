// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ReID diagnostic: feed exact CHW float input (from PyTorch) to the GGML
// graph and output both logits and embedding for comparison.
// ----------------------------------------------------------------------------

#include <QImage>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "tasks/yolo/yolo_graph.hpp"

int main(int argc, char** argv) {
    const char* gguf = nullptr;
    const char* chw_bin = nullptr;
    const char* device = "cuda";
    const char* dump_prefix = nullptr;
    const char* image_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc)
            gguf = argv[++i];
        else if (a == "--chw" && i + 1 < argc)
            chw_bin = argv[++i];
        else if (a == "--device" && i + 1 < argc)
            device = argv[++i];
        else if (a == "--dump" && i + 1 < argc)
            dump_prefix = argv[++i];
        else if (a == "--image" && i + 1 < argc)
            // Preprocess-only mode: run the reid capi's crop→stretch→CHW
            // pipeline on this image + the fixed first box, dump the CHW
            // tensor, and exit (no graph run).
            image_path = argv[++i];
    }

    // ---- preprocess-only mode: dump the CHW the C++ pipeline would feed
    // (mirrors tasks/reid/capi.cpp crop_box_save_one_box + crop_to_chw).
    if (image_path != nullptr) {
        constexpr float kGain = 1.02f;
        constexpr int kPad = 10;
        QImage src(QString::fromUtf8(image_path));
        if (src.isNull()) {
            fprintf(stderr, "cannot open image %s\n", image_path);
            return 3;
        }
        // Fixed first box of the probe geometry.
        const float bx[4] = {40, 60, 140, 220};
        const float w = bx[2] - bx[0], h = bx[3] - bx[1];
        const float cx = (bx[0] + bx[2]) / 2.0f, cy = (bx[1] + bx[3]) / 2.0f;
        const float nw = w * kGain + 2 * kPad, nh = h * kGain + 2 * kPad;
        int x1 = (int)(cx - nw / 2), y1 = (int)(cy - nh / 2);
        int x2 = (int)(cx + nw / 2), y2 = (int)(cy + nh / 2);
        x1 = std::clamp(x1, 0, src.width());
        y1 = std::clamp(y1, 0, src.height());
        x2 = std::clamp(x2, 0, src.width());
        y2 = std::clamp(y2, 0, src.height());
        QImage crop = src.copy(x1, y1, x2 - x1, y2 - y1);
        printf("src format=%d crop format=%d bytesPerLine=%d\n",
               (int)src.format(), (int)crop.format(), (int)crop.bytesPerLine());
        QImage rgb = crop.convertToFormat(QImage::Format_RGB888);
        printf("after convert format=%d bpl=%d\n", (int)rgb.format(),
               (int)rgb.bytesPerLine());
        rgb = rgb.scaled(224, 224, Qt::IgnoreAspectRatio,
                         Qt::SmoothTransformation);
        printf("after scaled format=%d bpl=%d size=%dx%d\n", (int)rgb.format(),
               (int)rgb.bytesPerLine(), rgb.width(), rgb.height());
        const size_t n = (size_t)3 * 224 * 224;
        std::vector<float> chw(n, 0.0f);
        const int plane = 224 * 224;
        for (int y = 0; y < 224; ++y) {
            const uchar* row = rgb.constScanLine(y);
            for (int x = 0; x < 224; ++x) {
                const size_t p = (size_t)y * 224 + x;
                chw[p] = row[x * 3 + 0] / 255.0f;
                chw[plane + p] = row[x * 3 + 1] / 255.0f;
                chw[2 * (size_t)plane + p] = row[x * 3 + 2] / 255.0f;
            }
        }
        const std::string prefix = dump_prefix != nullptr
                                           ? std::string(dump_prefix)
                                           : std::string("/tmp/reid/cpp");
        const std::string chw_path = prefix + ".chw.bin";
        std::ofstream fc(chw_path, std::ios::binary);
        fc.write(reinterpret_cast<const char*>(chw.data()),
                 (std::streamsize)(chw.size() * sizeof(float)));
        printf("preprocess-only: wrote %s (%zu floats, crop %dx%d -> "
               "224x224)\n",
               chw_path.c_str(), chw.size(), crop.width(), crop.height());
        return 0;
    }
    if (!gguf || !chw_bin) {
        fprintf(stderr,
                "usage: reid_diag --gguf <file> --chw <binary> "
                "[--device cuda]\n");
        return 2;
    }

    // Read CHW binary (3 * 224 * 224 floats).
    std::ifstream fin(chw_bin, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "cannot open %s\n", chw_bin);
        return 3;
    }
    const size_t imgsz = 224;
    const size_t n_elems = 3 * imgsz * imgsz;
    std::vector<float> chw(n_elems);
    fin.read(reinterpret_cast<char*>(chw.data()), n_elems * sizeof(float));
    fin.close();

    yolo::SessionOptions sopts;
    sopts.export_embed = true;
    std::string err;
    auto* s = yolo::create_session(gguf, device, sopts);
    if (!s) {
        fprintf(stderr, "session create failed\n");
        return 4;
    }
    if (!yolo::session_ensure_canvas(s, imgsz, imgsz)) {
        fprintf(stderr, "canvas failed\n");
        return 5;
    }
    if (!yolo::session_run(s, chw.data())) {
        fprintf(stderr, "run failed\n");
        return 6;
    }

    // Native reid graphs (converted from the official yolo26*-reid.onnx
    // assets) have no logits: the graph output IS the embedding.
    const bool native_reid = s->model.meta.task == "reid";
    std::vector<float> logits;
    if (!native_reid) {
        if (!yolo::session_read_logits(s, logits)) {
            fprintf(stderr, "logits read failed\n");
            return 7;
        }
    }
    // Read embedding.
    std::vector<float> embed;
    int dim = 0;
    if (!yolo::session_read_embed(s, embed, dim)) {
        fprintf(stderr, "embed read failed\n");
        return 8;
    }

    // Softmax of logits -> top-5 (classify graphs only; native reid graphs
    // carry no classification head).
    std::vector<std::pair<float, int>> probs(logits.size());
    if (logits.empty()) {
        printf("native reid graph: no logits (graph output is the "
               "embedding)\n");
    }
    float max_l = logits.empty()
                          ? 0.0f
                          : *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = {std::exp(logits[i] - max_l), (int)i};
        sum_exp += probs[i].first;
    }
    for (auto& p : probs) p.first /= sum_exp;
    std::sort(probs.begin(), probs.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    printf("=== GGML graph output (exact same CHW input as PyTorch) ===\n");
    printf("logits dim: %zu\n", logits.size());
    printf("embed dim:  %d\n", dim);
    if (!logits.empty()) {
        printf("Top-5 classes:\n");
        for (int i = 0; i < 5; ++i) {
            printf("  rank %d: class=%d prob=%.6f logit=%.4f\n", i,
                   probs[i].second, probs[i].first, logits[probs[i].second]);
        }
    }
    printf("Embed[:10]:");
    for (int i = 0; i < 10 && i < dim; ++i) printf(" %.6f", embed[i]);
    printf("\n");
    double norm = 0;
    for (auto v : embed) norm += v * v;
    printf("Embed norm: %.4f\n", std::sqrt(norm));

    // Optional full-vector dump for the numpy-side algebraic check
    // (logits == W @ embed + b against the same-name torch tensors).
    if (dump_prefix != nullptr) {
        const std::string prefix(dump_prefix);
        std::ofstream fl(prefix + ".logits.bin", std::ios::binary);
        fl.write(reinterpret_cast<const char*>(logits.data()),
                 (std::streamsize)(logits.size() * sizeof(float)));
        std::ofstream fe(prefix + ".embed.bin", std::ios::binary);
        fe.write(reinterpret_cast<const char*>(embed.data()),
                 (std::streamsize)(embed.size() * sizeof(float)));
        printf("dumped: %s.logits.bin (%zu) %s.embed.bin (%d)\n", dump_prefix,
               logits.size(), dump_prefix, dim);
    }

    yolo::free_session(s);
    return 0;
}
