#include "depth_export.hpp"
#include <QImage>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace da {

bool write_pfm(const std::string& path, const std::vector<float>& depth, int H, int W) {
    if (H <= 0 || W <= 0 || depth.size() != (size_t)H * (size_t)W) return false;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "Pf\n%d %d\n-1.0\n", W, H);
    bool ok = true;
    for (int y = H - 1; y >= 0 && ok; --y) {
        size_t n = std::fwrite(depth.data() + (size_t)y * W, sizeof(float), (size_t)W, f);
        ok = (n == (size_t)W);
    }
    std::fclose(f);
    return ok;
}

bool write_depth_png(const std::string& path, const std::vector<float>& depth, int H, int W, bool invert) {
    if (H <= 0 || W <= 0 || depth.size() != (size_t)H * (size_t)W) return false;
    float dmin = depth[0], dmax = depth[0];
    for (float v : depth) { dmin = std::min(dmin, v); dmax = std::max(dmax, v); }
    const float range = dmax - dmin;

    QImage img(W, H, QImage::Format_Grayscale8);
    for (int y = 0; y < H; ++y) {
        uchar* line = img.scanLine(y);
        for (int x = 0; x < W; ++x) {
            float t = (range > 0.0f) ? (depth[y * W + x] - dmin) / range : 0.0f;
            if (invert) t = 1.0f - t;
            int v = static_cast<int>(t * 255.0f + 0.5f);
            line[x] = static_cast<uchar>(std::min(255, std::max(0, v)));
        }
    }
    return img.save(QString::fromStdString(path), "PNG");
}

}  // namespace da
