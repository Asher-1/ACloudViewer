#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace aicore {
namespace depth {

// Write a COLMAP sparse model (cameras / images / points3D) to directory `dir`,
// mirroring the reference exporter `utils/export/colmap.py` (which drives
// pycolmap's `reconstruction.write`). Inputs mirror back_project():
//   depth, conf : N*H*W row-major (frame, row, col).
//   K           : per-frame 3x3 intrinsics, row-major (at processed size HxW).
//   ext         : per-frame 4x4 world-to-camera extrinsics, row-major.
//   images_u8   : per-frame pointer to H*W*3 RGB uint8 image (processed size).
//   image_names : per-frame output image name (basename).
//   orig_wh     : per-frame original (width, height) for intrinsic rescale.
// Behaviour (faithful to colmap.py):
//   - conf_thr = percentile_linear(conf over all frames, 40).
//   - back_project -> world points + colors; point3D ids 1..num_points in
//     back-projection order.
//   - Per frame (camera_id = image_id = frame+1): PINHOLE model. Intrinsics
//     rescaled to original size: fx,cx *= orig_w/W; fy,cy *= orig_h/H.
//     params = [fx,fy,cx,cy]; width = orig_w; height = orig_h.
//   - Extrinsic -> qvec = rotmat2qvec(R = ext[:3,:3]) (COLMAP order qw,qx,qy,qz),
//     tvec = ext[:3,3].
//   - point2D x,y = int-truncated (u*orig_w/W, v*orig_h/H), matching the
//     reference's int32 in-place scaling; linked to the point3D id; the
//     point3D track records (image_id, point2D_idx).
// When `binary` is true (default) emits cameras.bin/images.bin/points3D.bin in
// the little-endian COLMAP layout; otherwise emits the .txt variants. Returns
// false on I/O error.
bool write_colmap(const std::string& dir,
                  const std::vector<float>& depth,
                  const std::vector<float>& conf,
                  const std::vector<std::array<float, 9>>& K,
                  const std::vector<std::array<float, 16>>& ext,
                  const std::vector<const uint8_t*>& images_u8,
                  const std::vector<std::string>& image_names,
                  const std::vector<std::pair<int, int>>& orig_wh,
                  int H, int W, int N, bool binary = true);

} // namespace depth
} // namespace aicore
