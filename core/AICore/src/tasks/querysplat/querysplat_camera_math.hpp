// Host-side camera math for the QuerySplat chain: VGGT-Omega pose encoding ->
// cameras -> Plucker rays. Ports scripts/utils/data.py (get_rays) and the
// vendored vggt_omega.utils.pose_enc.encoding_to_camera + QuerySplat camera
// plumbing exactly, in f32, op-for-op.
//
// Layout contract:
//   pose_enc: torch (S, 9) C-order = [absT(3) | quaR(4, XYZW scalar-last) |
//             fov_h | fov_w] per view.
//   w2c/c2w:  per view 16 floats, row-major 4x4 (w2c = [[R,T],[0,1]], the
//             Python `w2c` tensor; cam_view == w2c.transpose(-2,-1)).
//   intrinsics: per view [fx, fy, cx, cy] (QuerySplat stacks K[0,0], K[1,1],
//             K[0,2], K[1,2] in that order).
//   plucker:  torch (S, 6, H, W) C-order == ggml ne {W, H, 6, S} — channel
//             major per view, spatial row-major; channels [cross(o,d) | d].
#pragma once

#include <cstddef>
#include <vector>

namespace qs {

struct CameraData {
    int S = 0;
    std::vector<float> w2c;        // S * 16, row-major
    std::vector<float> c2w;        // S * 16, row-major (== inv(w2c))
    std::vector<float> intrinsics; // S * 4: fx, fy, cx, cy
};

// Mirrors vggt_omega.utils.pose_enc.encoding_to_camera + QuerySplat's
// _decode_cameras: extrinsic = [R|T] (camera-from-world, OpenCV), w2c built
// from it, cam_view = w2c^T, c2w = inv(cam_view^T) == inv(w2c) (general
// inverse, as torch.linalg.inv); fy = (H/2)/tan(fov_h/2), fx = (W/2)/tan(fov_w/2).
CameraData decode_cameras(const float* pose_enc, int S, int H, int W);

// General 4x4 inverse (row-major): f64 Gauss-Jordan, one f32 rounding, so
// within 0.5 ulp of the exact inverse of the f32 input.
void invert4(const float* m, float* out);

// evaluations/cameras.py align_cameras_to_reference: the similarity transform
// (orientation average via SVD with det correction, center-spread scale)
// that maps the first n_ref of n_views independently predicted cameras onto
// the n_ref reference (input-reconstruction) cameras, applied to all n_views.
// w2c / ref_w2c / result: per view 16 floats row-major (cam_view^T). f64
// internally, inverses via invert4.
std::vector<float> align_cameras_to_reference(const float* w2c, int n_views,
                                              const float* ref_w2c, int n_ref);
// The rotation part on its own: R = U diag(1,1,sign det(U V^T)) V^T of
// C = U S V^T (row-major double 3x3).
void procrustes_rotation(const double C[3][3], double R[3][3]);

// Mirrors scripts/utils/data.py get_rays + QuerySplat._plucker_from_vggt_cameras:
// pixel centers (+0.5), directions ((x-cx)/fx, (y-cy)/fy, 1) normalized,
// d_world = R_c2w * d (== directions @ c2w.R^T), o_world = c2w.t,
// plucker = [cross(o, d_world), d_world].
std::vector<float> plucker_from_cameras(const CameraData& cam, int H, int W);

// ImageNet normalization over an (S,3,H,W) [0,1] buffer (torch
// transforms.Normalize mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]).
// This is what feeds patch_embed (NOT the VGGT encoder, which normalizes
// in-graph with its own ResNet mean/std). The channel planes are PER-FRAME
// segments ((s,c) blocks of H*W), not 3 contiguous planes across S.
std::vector<float> imagenet_normalize(const float* rgb, int S, size_t n_elements);

}  // namespace qs
