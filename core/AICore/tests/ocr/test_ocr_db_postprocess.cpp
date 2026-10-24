// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// DB post-process parity against upstream (no model, no backend). Every
// golden row was produced by PaddleX DBPostProcess.boxes_from_bitmap +
// SortQuadBoxes on a deterministic integer map (scripts/
// gen_db_postprocess_golden.py): raw order, int16 corners, scores and the
// sorted order must match exactly.
//
// `test_ocr_db_postprocess --map <file.f32> <w> <h> <destW> <destH>` prints
// the sorted boxes of a dumped float32 map (one "x0 y0 ... y3 score" row per
// box) for A/B against upstream on real detector output.

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ocr_db_postprocess.hpp"
#include "tests/common/test_macros.hpp"

static int failures = 0;

namespace {

using aicore::ocr::DbBox;

struct Shape {
    long long outer[4][2];
    bool ring;
    long long inner[4][2];
    int value, jitter;
};

bool inside(const long long q[4][2], long long x, long long y) {
    bool pos = true, neg = true;
    for (int i = 0; i < 4; ++i) {
        const long long ax = q[i][0], ay = q[i][1];
        const long long bx = q[(i + 1) % 4][0], by = q[(i + 1) % 4][1];
        const long long c = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
        pos = pos && c >= 0;
        neg = neg && c <= 0;
    }
    return pos || neg;
}

// gen_db_postprocess_golden.py render().
std::vector<float> render(int w,
                          int h,
                          const std::vector<Shape>& shapes,
                          int bgMod,
                          uint64_t salt) {
    std::vector<float> map(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint64_t v = ((uint64_t)x * 73856093u) ^ ((uint64_t)y * 19349663u) ^
                         salt;
            const uint64_t noise = ((v * 2654435761u) & 0xFFFFFFFFu) >> 7;
            long long level = static_cast<long long>(noise % bgMod);
            for (const Shape& s : shapes) {
                bool m = inside(s.outer, x, y);
                if (s.ring) m = m && !inside(s.inner, x, y);
                const long long sv =
                        s.value - static_cast<long long>(noise % s.jitter);
                if (m) level = std::max(level, sv);
            }
            map[static_cast<size_t>(y) * w + x] =
                    static_cast<float>(level) / 255.0f;
        }
    return map;
}

int dumpMap(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s --map file.f32 w h destW destH\n",
                     argv[0]);
        return 2;
    }
    const int w = std::atoi(argv[3]), h = std::atoi(argv[4]);
    const int dw = std::atoi(argv[5]), dh = std::atoi(argv[6]);
    std::vector<float> map(static_cast<size_t>(w) * h);
    FILE* f = std::fopen(argv[2], "rb");
    if (!f ||
        std::fread(map.data(), sizeof(float), map.size(), f) != map.size()) {
        if (f) std::fclose(f);
        std::fprintf(stderr, "cannot read %s\n", argv[2]);
        return 2;
    }
    std::fclose(f);
    std::vector<int8_t> scratch;
    auto boxes = aicore::ocr::dbBoxesFromBitmap(map.data(), w, h, dw, dh, {},
                                                scratch);
    aicore::ocr::sortQuadBoxes(boxes);
    for (const DbBox& b : boxes) {
        for (const auto& c : b.quad) std::printf("%d %d ", c[0], c[1]);
        std::printf("%.17g\n", b.score);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--map") == 0)
        return dumpMap(argc, argv);
#ifdef AICORE_OCR_DB_POSTPROCESS_GOLDEN
    const char* golden = argc > 1 ? argv[1] : AICORE_OCR_DB_POSTPROCESS_GOLDEN;
#else
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s db_postprocess_golden.txt\n", argv[0]);
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
    int cases = 0, boxes = 0, rawMismatch = 0, sortMismatch = 0;
    double worstScore = 0.0;
    std::vector<int8_t> scratch;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        int w = 0, h = 0, dw = 0, dh = 0, bg = 0, k = 0;
        uint64_t salt = 0;
        row >> w >> h >> dw >> dh >> bg >> salt >> k;
        std::vector<Shape> shapes(static_cast<size_t>(k));
        for (Shape& s : shapes) {
            int ring = 0;
            for (auto& c : s.outer) row >> c[0] >> c[1];
            row >> ring;
            s.ring = ring != 0;
            for (auto& c : s.inner) row >> c[0] >> c[1];
            row >> s.value >> s.jitter;
        }
        std::string bar;
        int n = 0;
        row >> bar >> n;
        std::vector<DbBox> want(static_cast<size_t>(n));
        for (DbBox& b : want) {
            for (auto& c : b.quad) row >> c[0] >> c[1];
            row >> b.score;
        }
        int ns = 0;
        row >> bar >> ns;
        std::vector<DbBox> wantSorted(static_cast<size_t>(ns));
        for (DbBox& b : wantSorted)
            for (auto& c : b.quad) row >> c[0] >> c[1];
        AICORE_CHECK(static_cast<bool>(row) || (n == 0 && ns == 0));
        ++cases;
        boxes += n;

        const auto map = render(w, h, shapes, bg, salt);
        auto got = aicore::ocr::dbBoxesFromBitmap(map.data(), w, h, dw, dh, {},
                                                  scratch);
        bool rawOk = got.size() == want.size();
        for (size_t i = 0; rawOk && i < got.size(); ++i) {
            rawOk = std::memcmp(got[i].quad, want[i].quad,
                                sizeof(got[i].quad)) == 0;
            worstScore = std::max(worstScore,
                                  std::abs(got[i].score - want[i].score));
        }
        if (!rawOk && rawMismatch++ < 5) {
            std::fprintf(stderr, "raw mismatch (row %d): %zu vs %zu boxes\n",
                         cases, got.size(), want.size());
            for (const DbBox& b : got)
                std::fprintf(stderr, "  got  %d %d %d %d %d %d %d %d\n",
                             b.quad[0][0], b.quad[0][1], b.quad[1][0],
                             b.quad[1][1], b.quad[2][0], b.quad[2][1],
                             b.quad[3][0], b.quad[3][1]);
            for (const DbBox& b : want)
                std::fprintf(stderr, "  want %d %d %d %d %d %d %d %d\n",
                             b.quad[0][0], b.quad[0][1], b.quad[1][0],
                             b.quad[1][1], b.quad[2][0], b.quad[2][1],
                             b.quad[3][0], b.quad[3][1]);
        }
        // Sort the upstream raw list so the sort is checked on its own.
        aicore::ocr::sortQuadBoxes(want);
        bool sortOk = want.size() == wantSorted.size();
        for (size_t i = 0; sortOk && i < want.size(); ++i)
            sortOk = std::memcmp(want[i].quad, wantSorted[i].quad,
                                 sizeof(want[i].quad)) == 0;
        if (!sortOk && sortMismatch++ < 5)
            std::fprintf(stderr, "sort mismatch (row %d)\n", cases);
    }
    AICORE_CHECK(cases >= 400);
    AICORE_CHECK(rawMismatch == 0);
    AICORE_CHECK(sortMismatch == 0);
    AICORE_CHECK(worstScore <= 1e-12);
    std::printf(
            "db postprocess parity: %d cases, %d boxes, %d raw / %d sort "
            "mismatches, worst |score diff| %.3g\n",
            cases, boxes, rawMismatch, sortMismatch, worstScore);
    if (failures) {
        std::fprintf(stderr, "test_ocr_db_postprocess: %d failure(s)\n",
                     failures);
        return 1;
    }
    return 0;
}
