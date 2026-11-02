#include "querysplat_camera_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace qs {

namespace {

// vggt_omega.utils.rotation.quat_to_mat — XYZW scalar-last (i, j, k, r).
void quat_to_mat(const float* q, float* R) {
    const float i = q[0], j = q[1], k = q[2], r = q[3];
    const float two_s = 2.0f / (i * i + j * j + k * k + r * r);
    R[0] = 1 - two_s * (j * j + k * k);
    R[1] = two_s * (i * j - k * r);
    R[2] = two_s * (i * k + j * r);
    R[3] = two_s * (i * j + k * r);
    R[4] = 1 - two_s * (i * i + k * k);
    R[5] = two_s * (j * k - i * r);
    R[6] = two_s * (i * k - j * r);
    R[7] = two_s * (j * k + i * r);
    R[8] = 1 - two_s * (i * i + j * j);
}

}  // namespace

// Gauss-Jordan with partial pivoting in f64. O(4^3) per camera.
void invert4(const float* m, float* out) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            a[r][c] = c < 4 ? (double)m[r * 4 + c] : (c - 4 == r ? 1.0 : 0.0);
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
        if (piv != col)
            for (int c = 0; c < 8; c++) std::swap(a[col][c], a[piv][c]);
        const double inv_p = 1.0 / a[col][col];
        for (int c = 0; c < 8; c++) a[col][c] *= inv_p;
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            const double f = a[r][col];
            for (int c = 0; c < 8; c++) a[r][c] -= f * a[col][c];
        }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) out[r * 4 + c] = (float)a[r][c + 4];
}

namespace {

// Symmetric 3x3 eigen-decomposition by cyclic Jacobi rotations (f64):
// A = V diag(w) V^T, eigenvalues sorted descending, V's columns the
// eigenvectors. Converges quadratically; 3x3 needs < 10 sweeps.
void jacobi_eigen3(double A[3][3], double w[3], double V[3][3]) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) V[r][c] = r == c ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 64; sweep++) {
        const double off = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
        if (off < 1e-300) break;
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                if (A[p][q] == 0.0) continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; k++) {  // A <- A J
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - s * akq;
                    A[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++) {  // A <- J^T A
                    const double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - s * aqk;
                    A[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++) {  // V <- V J
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    int order[3] = {0, 1, 2};
    for (int a = 0; a < 3; a++)
        for (int b = a + 1; b < 3; b++)
            if (A[order[b]][order[b]] > A[order[a]][order[a]]) std::swap(order[a], order[b]);
    double Vs[3][3];
    for (int k = 0; k < 3; k++) {
        w[k] = A[order[k]][order[k]];
        for (int r = 0; r < 3; r++) Vs[r][k] = V[r][order[k]];
    }
    std::memcpy(V, Vs, sizeof(Vs));
}

}  // namespace

// R = U diag(1, 1, d) V^T from the SVD of C, d = sign(det(U V^T)). With
// C^T C = V diag(s^2) V^T: u_k = C v_k / s_k for the two leading singular
// vectors, and the (corrected) third column is det(V) * (u0 x u1) in both
// branches — the identity that keeps R in SO(3) without touching s_2, so a
// near-rank-2 covariance cannot destabilize it.
void procrustes_rotation(const double C[3][3], double R[3][3]) {
    double A[3][3], w[3], V[3][3];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            A[r][c] = 0;
            for (int k = 0; k < 3; k++) A[r][c] += C[k][r] * C[k][c];
        }
    jacobi_eigen3(A, w, V);
    double U[3][3];
    for (int k = 0; k < 2; k++) {
        double u[3];
        for (int r = 0; r < 3; r++)
            u[r] = C[r][0] * V[0][k] + C[r][1] * V[1][k] + C[r][2] * V[2][k];
        if (k == 1) {  // Gram-Schmidt against u0 (exact in theory, cheap insurance)
            const double d = u[0] * U[0][0] + u[1] * U[1][0] + u[2] * U[2][0];
            for (int r = 0; r < 3; r++) u[r] -= d * U[r][0];
        }
        const double n = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        for (int r = 0; r < 3; r++) U[r][k] = u[r] / n;
    }
    const double detV = V[0][0] * (V[1][1] * V[2][2] - V[1][2] * V[2][1]) -
                        V[0][1] * (V[1][0] * V[2][2] - V[1][2] * V[2][0]) +
                        V[0][2] * (V[1][0] * V[2][1] - V[1][1] * V[2][0]);
    const double sgn = detV < 0 ? -1.0 : 1.0;
    U[0][2] = sgn * (U[1][0] * U[2][1] - U[2][0] * U[1][1]);
    U[1][2] = sgn * (U[2][0] * U[0][1] - U[0][0] * U[2][1]);
    U[2][2] = sgn * (U[0][0] * U[1][1] - U[1][0] * U[0][1]);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R[r][c] = U[r][0] * V[c][0] + U[r][1] * V[c][1] + U[r][2] * V[c][2];
}

std::vector<float> align_cameras_to_reference(const float* w2c, int n_views,
                                              const float* ref_w2c, int n_ref) {
    std::vector<float> c2w((size_t)n_views * 16), ref((size_t)n_ref * 16);
    for (int v = 0; v < n_views; v++) invert4(w2c + (size_t)v * 16, &c2w[(size_t)v * 16]);
    for (int v = 0; v < n_ref; v++) invert4(ref_w2c + (size_t)v * 16, &ref[(size_t)v * 16]);
    // covariance = sum_v R_target,v R_source,v^T over the overlapping inputs
    double C[3][3] = {};
    double smean[3] = {}, tmean[3] = {};
    for (int v = 0; v < n_ref; v++) {
        const float* s = &c2w[(size_t)v * 16];
        const float* t = &ref[(size_t)v * 16];
        for (int i = 0; i < 3; i++)
            for (int k = 0; k < 3; k++)
                for (int j = 0; j < 3; j++) C[i][k] += (double)t[i * 4 + j] * s[k * 4 + j];
        for (int i = 0; i < 3; i++) {
            smean[i] += s[i * 4 + 3] / (double)n_ref;
            tmean[i] += t[i * 4 + 3] / (double)n_ref;
        }
    }
    double R[3][3];
    procrustes_rotation(C, R);
    double svar = 0, tvar = 0;
    for (int v = 0; v < n_ref; v++)
        for (int i = 0; i < 3; i++) {
            const double ds = c2w[(size_t)v * 16 + i * 4 + 3] - smean[i];
            const double dt = ref[(size_t)v * 16 + i * 4 + 3] - tmean[i];
            svar += ds * ds;
            tvar += dt * dt;
        }
    double scale = std::sqrt(tvar / std::max(svar, 1e-8));
    if (!std::isfinite(scale) || !(tvar > 1e-8)) scale = 1.0;
    double trans[3];
    for (int i = 0; i < 3; i++)
        trans[i] = tmean[i] - scale * (R[i][0] * smean[0] + R[i][1] * smean[1] + R[i][2] * smean[2]);
    std::vector<float> out((size_t)n_views * 16);
    for (int v = 0; v < n_views; v++) {
        const float* m = &c2w[(size_t)v * 16];
        float a[16];
        std::memcpy(a, m, sizeof(a));  // bottom row kept (aligned = c2w.clone())
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++)
                a[i * 4 + k] = (float)(R[i][0] * m[0 * 4 + k] + R[i][1] * m[1 * 4 + k] +
                                       R[i][2] * m[2 * 4 + k]);
            a[i * 4 + 3] = (float)(scale * (R[i][0] * m[3] + R[i][1] * m[7] + R[i][2] * m[11]) +
                                   trans[i]);
        }
        invert4(a, &out[(size_t)v * 16]);
    }
    return out;
}

CameraData decode_cameras(const float* pose_enc, int S, int H, int W) {
    CameraData out;
    out.S = S;
    out.w2c.assign((size_t)S * 16, 0.0f);
    out.c2w.assign((size_t)S * 16, 0.0f);
    out.intrinsics.resize((size_t)S * 4);
    for (int s = 0; s < S; s++) {
        const float* p = pose_enc + (size_t)s * 9;
        float R[9];
        quat_to_mat(p + 3, R);  // quat at [3:7]
        // w2c = [[R, T], [0, 1]] — extrinsic is camera-from-world
        float* w2c = &out.w2c[(size_t)s * 16];
        float* c2w = &out.c2w[(size_t)s * 16];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                w2c[r * 4 + c] = R[r * 3 + c];
            }
            w2c[r * 4 + 3] = p[r];       // translation
            w2c[12 + r] = 0.0f;
        }
        w2c[15] = 1.0f;
        // c2w = inv(w2c). The official chain uses torch.linalg.inv (general
        // LU), not the rigid R^T form: the f32 quaternion R is orthonormal
        // only to ~1e-7, so R^T deviates from the true inverse by up to
        // ~12 ulp. Invert in f64 and round once -> within 0.5 ulp of the
        // exact inverse, i.e. as close to torch's f32 LU as f32 allows.
        invert4(w2c, c2w);
        // intrinsics from FoV (fov_h uses H, fov_w uses W)
        const float fov_h = p[7], fov_w = p[8];
        out.intrinsics[(size_t)s * 4 + 0] = (W / 2.0f) / std::tan(fov_w / 2.0f);  // fx
        out.intrinsics[(size_t)s * 4 + 1] = (H / 2.0f) / std::tan(fov_h / 2.0f);  // fy
        out.intrinsics[(size_t)s * 4 + 2] = W / 2.0f;                             // cx
        out.intrinsics[(size_t)s * 4 + 3] = H / 2.0f;                             // cy
    }
    return out;
}

std::vector<float> plucker_from_cameras(const CameraData& cam, int H, int W) {
    const int S = cam.S;
    std::vector<float> out((size_t)S * 6 * H * W);
    for (int s = 0; s < S; s++) {
        const float fx = cam.intrinsics[(size_t)s * 4 + 0];
        const float fy = cam.intrinsics[(size_t)s * 4 + 1];
        const float cx = cam.intrinsics[(size_t)s * 4 + 2];
        const float cy = cam.intrinsics[(size_t)s * 4 + 3];
        const float* c2w = &cam.c2w[(size_t)s * 16];
        const float ox = c2w[3], oy = c2w[7], oz = c2w[11];  // c2w.t
        float* dst = &out[(size_t)s * 6 * H * W];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                // directions = ((x - cx) / fx, (y - cy) / fy, 1), normalized
                const float px = (float)x + 0.5f;
                const float py = (float)y + 0.5f;
                float dx = (px - cx) / fx;
                float dy = (py - cy) / fy;
                float dz = 1.0f;
                const float inv = 1.0f / std::sqrt(dx * dx + dy * dy + dz * dz);
                dx *= inv;
                dy *= inv;
                dz *= inv;
                // d_world = R_c2w * d (c2w rotation is at [:3, :3])
                const float wx = c2w[0] * dx + c2w[1] * dy + c2w[2] * dz;
                const float wy = c2w[4] * dx + c2w[5] * dy + c2w[6] * dz;
                const float wz = c2w[8] * dx + c2w[9] * dy + c2w[10] * dz;
                // plucker = [cross(rays_o, rays_d), rays_d]
                const float cxp = oy * wz - oz * wy;
                const float cyp = oz * wx - ox * wz;
                const float czp = ox * wy - oy * wx;
                // channel-major layout: out[(c * H + y) * W + x]
                float* col = dst + (size_t)y * W + x;
                col[(size_t)0 * H * W] = cxp;
                col[(size_t)1 * H * W] = cyp;
                col[(size_t)2 * H * W] = czp;
                col[(size_t)3 * H * W] = wx;
                col[(size_t)4 * H * W] = wy;
                col[(size_t)5 * H * W] = wz;
            }
        }
    }
    return out;
}

std::vector<float> imagenet_normalize(const float* rgb, int S, size_t n_elements) {
    // (S,3,H,W) layout: per-frame channel planes of H*W; the per-channel
    // segment offset is s*(3*H*W) + c*(H*W) — NOT c*S*H*W (an earlier
    // 3-contiguous-planes read silently mis-normalized every frame past s=0).
    static const float kMean[3] = {0.485f, 0.456f, 0.406f};
    static const float kStd[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> out(rgb, rgb + n_elements);
    const size_t plane = n_elements / (size_t)(S * 3);
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < 3; c++) {
            float* p = out.data() + ((size_t)s * 3 + c) * plane;
            for (size_t i = 0; i < plane; i++) p[i] = (p[i] - kMean[c]) / kStd[c];
        }
    }
    return out;
}

}  // namespace qs
