#include "cam_pose.hpp"
#include "ggml_extend.hpp"
#include <cmath>
#include <algorithm>

namespace da {

bool CamPose::pose(const std::vector<float>& cam_token, int H, int W,
                   std::array<float,9>& pose_enc,
                   std::array<float,12>& extrinsics,
                   std::array<float,9>& intrinsics) {
    auto t = [&](const std::string& n) { return ml_.tensor(n); };
    // cam token width = 2*embed (base 1536, giant 3072); validate against bb0 input dim.
    ggml_tensor* bb0 = t("cam.bb0.weight");
    if (cam_token.empty() || !bb0 || (int64_t)cam_token.size() != bb0->ne[0]) return false;
    const int64_t D = (int64_t)cam_token.size();

    // ---- MLP / heads via single ggml graph -> 9-vector pose_enc ----
    GraphInputPool pool;
    std::vector<float> pe;  // [t(3), q(4), fov(2)]
    bool ok = be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ne[1] = { D };
        ggml_tensor* feat = be_.add_graph_input_nd(ctx, pool, cam_token.data(), ne, 1);
        // backbone: relu(linear(bb0)); relu(linear(bb2))
        feat = ggml_relu(ctx, linear(ctx, t("cam.bb0.weight"), feat, t("cam.bb0.bias")));
        feat = ggml_relu(ctx, linear(ctx, t("cam.bb2.weight"), feat, t("cam.bb2.bias")));
        // heads: t (no relu), q (no relu), fov (relu AFTER fc_fov)
        ggml_tensor* th  = linear(ctx, t("cam.fc_t.weight"),   feat, t("cam.fc_t.bias"));   // [3]
        ggml_tensor* qh  = linear(ctx, t("cam.fc_q.weight"),   feat, t("cam.fc_q.bias"));   // [4]
        ggml_tensor* fvh = ggml_relu(ctx,
                            linear(ctx, t("cam.fc_fov.weight"), feat, t("cam.fc_fov.bias")));// [2]
        ggml_tensor* tq  = ggml_concat(ctx, th, qh, 0);   // [7]
        ggml_tensor* out = ggml_concat(ctx, tq, fvh, 0);  // [9]
        return out;
    }, pe);
    if (!ok || pe.size() != 9) return false;

    for (int i = 0; i < 9; ++i) pose_enc[i] = pe[i];

    // ---- pose_encoding_to_extri_intri ----
    const float Tx = pe[0], Ty = pe[1], Tz = pe[2];
    // quaternion XYZW, scalar-LAST
    const float qi = pe[3], qj = pe[4], qk = pe[5], qr = pe[6];
    const float fov_h = pe[7], fov_w = pe[8];

    // quat_to_mat -> R (3x3 row-major)
    const float s = 2.0f / (qi*qi + qj*qj + qk*qk + qr*qr);
    float R[3][3];
    R[0][0] = 1.0f - s*(qj*qj + qk*qk); R[0][1] = s*(qi*qj - qk*qr);        R[0][2] = s*(qi*qk + qj*qr);
    R[1][0] = s*(qi*qj + qk*qr);        R[1][1] = 1.0f - s*(qi*qi + qk*qk); R[1][2] = s*(qj*qk - qi*qr);
    R[2][0] = s*(qi*qk - qj*qr);        R[2][1] = s*(qj*qk + qi*qr);        R[2][2] = 1.0f - s*(qi*qi + qj*qj);

    // c2w = [R | T]
    const float Tc[3] = { Tx, Ty, Tz };

    // intrinsics K
    const float fy = (H / 2.0f) / std::max(std::tan(fov_h / 2.0f), 1e-6f);
    const float fx = (W / 2.0f) / std::max(std::tan(fov_w / 2.0f), 1e-6f);
    float K[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    K[0][0] = fx; K[1][1] = fy; K[0][2] = W / 2.0f; K[1][2] = H / 2.0f; K[2][2] = 1.0f;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            intrinsics[r*3 + c] = K[r][c];

    // affine_inverse(c2w) -> ext: [R^T | -R^T * Tc]
    float RT[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            RT[r][c] = R[c][r];
    float negRTt[3];
    for (int r = 0; r < 3; ++r) {
        float acc = 0.0f;
        for (int c = 0; c < 3; ++c) acc += RT[r][c] * Tc[c];
        negRTt[r] = -acc;
    }
    for (int r = 0; r < 3; ++r) {
        extrinsics[r*4 + 0] = RT[r][0];
        extrinsics[r*4 + 1] = RT[r][1];
        extrinsics[r*4 + 2] = RT[r][2];
        extrinsics[r*4 + 3] = negRTt[r];
    }
    return true;
}

} // namespace da
