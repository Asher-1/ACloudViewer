#include "landmark_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "align.hpp"
#include "antispoof_graph.hpp"   // run_onnx_graph (shared interpreter)
#include "detect.hpp"
#include "model_loader.hpp"
#include "preprocess.hpp"

namespace fd {

namespace {

const FaceConfig::LandmarkHead& pick(const ModelLoader& ml, bool three_d) {
    const FaceConfig& cfg = ml.config();
    return three_d ? cfg.landmark_3d : cfg.landmark_2d;
}

} // namespace

bool landmark_crop(const Image& img, const Detection& d, int size,
                   std::array<float, 6>& M, Image& out) {
    // insightface Landmark.get: _scale = size / (max(w,h) * 1.5), centered on the
    // detection box. face_align.transform with rotation 0 reduces to a pure
    // scale-about-center similarity (the same geometry as the genderage crop).
    const float w = d.x2 - d.x1, h = d.y2 - d.y1;
    const float cx = (d.x1 + d.x2) * 0.5f, cy = (d.y1 + d.y2) * 0.5f;
    const float s = (float)size / (std::max(w, h) * 1.5f);
    M = { s, 0.0f, -s * cx + size * 0.5f,
          0.0f, s, -s * cy + size * 0.5f };
    return warp_affine(img, M, out, size, size);
}

std::vector<float> landmark_forward(const ModelLoader& ml, bool three_d,
                                    const Image& crop, Backend& be) {
    const FaceConfig::LandmarkHead& h = pick(ml, three_d);
    if (!h.present || h.graph.empty())
        throw std::runtime_error("landmark: head not present in pack");
    const int sz = (int)h.input_size;
    // Both heads feed raw [0,255] R,G,B (input_mean 0 / input_std 1): 2d106det
    // carries (x-127.5)/128 in-graph (Sub/Mul leaves), 1k3d68 absorbs the input
    // scale into its leading bn_data BatchNormalization. swap_rb=false on the RGB
    // crop lands on the R,G,B planes the reference's swapRB=True produces.
    std::vector<float> blob = to_blob(crop, sz, h.input_mean, h.input_std,
                                      /*swap_rb=*/false);
    const std::string prefix = three_d ? "l3d." : "l2d.";
    // bn_eps fallback is unused: every BatchNormalization node carries its own `e=`
    // (2d106det 1e-3, 1k3d68 2e-5) embedded by the converter.
    return run_onnx_graph(ml, prefix, h.graph, h.graph_output, h.graph_input,
                          blob, sz, be, /*bn_eps=*/1e-5f);
}

std::vector<LandmarkPoint> landmark_decode_crop(const ModelLoader& ml, bool three_d,
                                                const std::vector<float>& raw) {
    const FaceConfig::LandmarkHead& h = pick(ml, three_d);
    const int dim = (int)h.dim;
    const int npts = (int)h.num_points;
    if (dim < 2 || npts <= 0)
        throw std::runtime_error("landmark: bad head geometry");
    if ((int)(raw.size() % dim) != 0)
        throw std::runtime_error("landmark: output not divisible by dim");
    const int rows = (int)raw.size() / dim;
    if (rows < npts)
        throw std::runtime_error("landmark: fewer rows than points");
    // insightface: reshape (-1, dim) then take the LAST num_points rows (1k3d68
    // emits 1103 rows, of which 68 are the landmarks). The half-scale is
    // input_size // 2 (96 for the 192 input), applied after the +1 shift.
    const int base = rows - npts;
    const float half = (float)(h.input_size / 2);  // integer division, like insightface
    std::vector<LandmarkPoint> out(npts);
    for (int i = 0; i < npts; ++i) {
        const float* p = &raw[(size_t)(base + i) * dim];
        out[i].x = (p[0] + 1.0f) * half;
        out[i].y = (p[1] + 1.0f) * half;
        out[i].z = (dim == 3) ? (p[2] * half) : 0.0f;
    }
    return out;
}

std::vector<LandmarkPoint> landmark_to_image(const std::vector<LandmarkPoint>& pts,
                                             const std::array<float, 6>& M) {
    // Invert the 2x3 forward affine (cv2.invertAffineTransform), then apply it as
    // insightface trans_points: x,y by the full inverse, z scaled by the inverse's
    // uniform scale. Mirrors the inversion in fd::warp_affine.
    const double det = (double)M[0] * M[4] - (double)M[1] * M[3];
    if (std::fabs(det) < 1e-12)
        throw std::runtime_error("landmark: singular crop transform");
    const double i00 = M[4] / det, i01 = -M[1] / det;
    const double i10 = -M[3] / det, i11 = M[0] / det;
    const double i02 = -(i00 * M[2] + i01 * M[5]);
    const double i12 = -(i10 * M[2] + i11 * M[5]);
    const double scale = std::sqrt(i00 * i00 + i01 * i01);
    std::vector<LandmarkPoint> out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        out[i].x = (float)(i00 * pts[i].x + i01 * pts[i].y + i02);
        out[i].y = (float)(i10 * pts[i].x + i11 * pts[i].y + i12);
        out[i].z = (float)(pts[i].z * scale);
    }
    return out;
}

std::vector<LandmarkPoint> landmarks(const ModelLoader& ml, bool three_d,
                                     const Image& img, const Detection& d, Backend& be) {
    const FaceConfig::LandmarkHead& h = pick(ml, three_d);
    std::array<float, 6> M{};
    Image crop;
    if (!landmark_crop(img, d, (int)h.input_size, M, crop))
        throw std::runtime_error("landmark: crop failed");
    std::vector<float> raw = landmark_forward(ml, three_d, crop, be);
    std::vector<LandmarkPoint> cp = landmark_decode_crop(ml, three_d, raw);
    return landmark_to_image(cp, M);
}

} // namespace fd
