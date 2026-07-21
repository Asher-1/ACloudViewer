// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Standalone harness for scripts/parity_glb.py: reads a synthetic input blob
// and writes a .glb via aicore::depth::write_glb. Not a ctest (takes CLI args).
//
// Input blob layout (little-endian):
//   int32 N, H, W
//   float32 depth[N*H*W]
//   float32 conf [N*H*W]
//   float32 K    [N*9]
//   float32 ext  [N*16]
//   uint8   img  [N*H*W*3]
//
// Usage: glb_parity_dump <in.bin> <out.glb> [show_cameras=1]
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "glb_export.hpp"

template <class T>
static bool read_n(std::ifstream& f, T* dst, size_t n) {
    f.read(reinterpret_cast<char*>(dst), (std::streamsize)(n * sizeof(T)));
    return (bool)f;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <in.bin> <out.glb> [show_cameras]\n",
                     argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    int32_t N = 0, H = 0, W = 0;
    if (!read_n(f, &N, 1) || !read_n(f, &H, 1) || !read_n(f, &W, 1)) return 2;
    const size_t plane = (size_t)H * W;
    const size_t tot = (size_t)N * plane;

    std::vector<float> depth(tot), conf(tot);
    std::vector<std::array<float, 9>> K(N);
    std::vector<std::array<float, 16>> ext(N);
    std::vector<std::vector<uint8_t>> imgs(N, std::vector<uint8_t>(plane * 3));

    if (!read_n(f, depth.data(), tot)) return 2;
    if (!read_n(f, conf.data(), tot)) return 2;
    for (int i = 0; i < N; ++i)
        if (!read_n(f, K[i].data(), 9)) return 2;
    for (int i = 0; i < N; ++i)
        if (!read_n(f, ext[i].data(), 16)) return 2;
    for (int i = 0; i < N; ++i)
        if (!read_n(f, imgs[i].data(), plane * 3)) return 2;

    std::vector<const uint8_t*> img_ptrs(N);
    for (int i = 0; i < N; ++i) img_ptrs[i] = imgs[i].data();

    aicore::depth::GlbOptions opt;
    if (argc >= 4) opt.show_cameras = (std::atoi(argv[3]) != 0);

    if (!aicore::depth::write_glb(argv[2], depth, conf, K, ext, img_ptrs, H, W,
                                  N, opt)) {
        std::fprintf(stderr, "write_glb failed\n");
        return 1;
    }
    return 0;
}
