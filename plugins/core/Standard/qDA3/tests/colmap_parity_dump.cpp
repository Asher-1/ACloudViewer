// Standalone harness for scripts/parity_colmap.py: reads a synthetic input
// blob and writes a COLMAP model via da::write_colmap. Not a ctest (takes CLI
// args).
//
// Input blob layout (little-endian):
//   int32 N, H, W
//   float32 depth[N*H*W]
//   float32 conf [N*H*W]
//   float32 K    [N*9]
//   float32 ext  [N*16]
//   uint8   img  [N*H*W*3]
//   per frame: int32 orig_w, int32 orig_h
//   per frame: int32 name_len, char name[name_len]
//
// Usage: colmap_parity_dump <in.bin> <out_dir> [binary=1]
#include "colmap_export.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

template <class T>
static bool read_n(std::ifstream& f, T* dst, size_t n) {
    f.read(reinterpret_cast<char*>(dst), (std::streamsize)(n * sizeof(T)));
    return (bool)f;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <in.bin> <out_dir> [binary]\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

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
    for (int i = 0; i < N; ++i) if (!read_n(f, K[i].data(), 9)) return 2;
    for (int i = 0; i < N; ++i) if (!read_n(f, ext[i].data(), 16)) return 2;
    for (int i = 0; i < N; ++i) if (!read_n(f, imgs[i].data(), plane * 3)) return 2;

    std::vector<std::pair<int, int>> orig_wh(N);
    for (int i = 0; i < N; ++i) {
        int32_t ow = 0, oh = 0;
        if (!read_n(f, &ow, 1) || !read_n(f, &oh, 1)) return 2;
        orig_wh[i] = {ow, oh};
    }
    std::vector<std::string> names(N);
    for (int i = 0; i < N; ++i) {
        int32_t len = 0;
        if (!read_n(f, &len, 1)) return 2;
        std::string s((size_t)len, '\0');
        if (len > 0 && !read_n(f, &s[0], (size_t)len)) return 2;
        names[i] = s;
    }

    std::vector<const uint8_t*> img_ptrs(N);
    for (int i = 0; i < N; ++i) img_ptrs[i] = imgs[i].data();

    bool binary = true;
    if (argc >= 4) binary = (std::atoi(argv[3]) != 0);

    if (!da::write_colmap(argv[2], depth, conf, K, ext, img_ptrs, names,
                          orig_wh, H, W, N, binary)) {
        std::fprintf(stderr, "write_colmap failed\n");
        return 1;
    }
    return 0;
}
