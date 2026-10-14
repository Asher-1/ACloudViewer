// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Rasterization parity of the DB box score against upstream (no model, no
// backend). Every golden row was produced by PaddleX
// DBPostProcess.box_score_fast + cv2.fillPoly (scripts/
// gen_db_box_score_golden.py); the mask must match bit-for-bit and the score
// to double rounding of the summation order.

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ocr_box_score.hpp"
#include "tests/common/test_macros.hpp"

static int failures = 0;

namespace {

std::vector<float> probMap(int w, int h) {
    std::vector<float> map(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint64_t v =
                    ((uint64_t)x * 73856093u) ^ ((uint64_t)y * 19349663u);
            map[static_cast<size_t>(y) * w + x] =
                    static_cast<float>(v % 1000) / 1000.0f;
        }
    return map;
}

uint64_t fnv1a64(const std::vector<uint8_t>& data) {
    uint64_t value = 0xcbf29ce484222325ull;
    for (const uint8_t byte : data) value = (value ^ byte) * 0x100000001b3ull;
    return value;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef AICORE_OCR_BOX_SCORE_GOLDEN
    const char* golden = argc > 1 ? argv[1] : AICORE_OCR_BOX_SCORE_GOLDEN;
#else
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s db_box_score_golden.txt\n", argv[0]);
        return 2;
    }
    const char* golden = argv[1];
#endif
    std::ifstream in(golden);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", golden);
        return 2;
    }
    std::string line;
    int cases = 0, maskMismatch = 0;
    double worstScore = 0.0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        int w = 0, h = 0, count = 0;
        float quad[4][2];
        std::string hashHex;
        double score = 0.0;
        row >> w >> h;
        for (auto& corner : quad) row >> corner[0] >> corner[1];
        row >> count >> hashHex >> score;
        AICORE_CHECK(static_cast<bool>(row));
        if (!row) break;
        ++cases;

        // Mask exactly as dbBoxScoreFast builds it.
        float minX = quad[0][0], maxX = quad[0][0];
        float minY = quad[0][1], maxY = quad[0][1];
        for (const auto& corner : quad) {
            minX = std::fmin(minX, corner[0]);
            maxX = std::fmax(maxX, corner[0]);
            minY = std::fmin(minY, corner[1]);
            maxY = std::fmax(maxY, corner[1]);
        }
        auto clampi = [](int v, int hi) {
            return v < 0 ? 0 : (v > hi ? hi : v);
        };
        const int xmin = clampi((int)std::floor(minX), w - 1);
        const int xmax = clampi((int)std::ceil(maxX), w - 1);
        const int ymin = clampi((int)std::floor(minY), h - 1);
        const int ymax = clampi((int)std::ceil(maxY), h - 1);
        const int mw = xmax - xmin + 1, mh = ymax - ymin + 1;
        int pts[4][2];
        for (int i = 0; i < 4; ++i) {
            pts[i][0] = (int)(quad[i][0] - (float)xmin);
            pts[i][1] = (int)(quad[i][1] - (float)ymin);
        }
        std::vector<uint8_t> mask(static_cast<size_t>(mw) * mh, 0);
        aicore::ocr::fillPolyLine8(pts, 4, mask.data(), mw, mh);
        int got = 0;
        for (const uint8_t m : mask) got += m;
        char hex[17];
        std::snprintf(hex, sizeof(hex), "%016" PRIx64, fnv1a64(mask));
        if (got != count || hashHex != hex) {
            if (maskMismatch++ < 5)
                std::fprintf(stderr,
                             "mask mismatch (row %d): count %d vs %d, hash %s "
                             "vs %s\n",
                             cases, got, count, hex, hashHex.c_str());
        }

        const auto map = probMap(w, h);
        const double s = aicore::ocr::dbBoxScoreFast(map.data(), w, h, quad);
        worstScore = std::fmax(worstScore, std::fabs(s - score));
    }
    AICORE_CHECK(cases >= 600);
    AICORE_CHECK(maskMismatch == 0);
    AICORE_CHECK(worstScore <= 1e-12);
    std::printf(
            "db box score parity: %d cases, %d mask mismatches, worst "
            "|score diff| %.3g\n",
            cases, maskMismatch, worstScore);
    if (failures) {
        std::fprintf(stderr, "test_ocr_box_score: %d failure(s)\n", failures);
        return 1;
    }
    return 0;
}
