// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gs_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

// GaussianAdapter — host port of depth_anything_3.model.gs_adapter.forward and
// its geometry helpers (utils/geometry.py, model/utils/transform.py,
// utils/sh_helpers.py).  Everything is f64 internally for stability, emitted
// f32.
//
// Reference channel layout of raw_gs (37, channels-last), per forward():
//   ch 0,1     : offset_xy        (stripped from the FRONT, second)
//   ch 2,3,4   : scales (logits)
//   ch 5..8    : quaternion xyzw
//   ch 9..35   : sh (3 colors x 9 coeff; color-major: ch 9 + color*9 + coeff)
//   ch 36      : offset_depth     (stripped from the END, first)

namespace aicore {
namespace depth {
namespace {

// ---- small dense linear algebra (row-major) --------------------------------
static void mat3_inverse(const double m[9], double inv[9]) {
    double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6],
           h = m[7], i = m[8];
    double A = (e * i - f * h), B = -(d * i - f * g), C = (d * h - e * g);
    double det = a * A + b * B + c * C;
    double id = (det != 0.0) ? 1.0 / det : 0.0;
    inv[0] = A * id;
    inv[1] = (c * h - b * i) * id;
    inv[2] = (b * f - c * e) * id;
    inv[3] = B * id;
    inv[4] = (a * i - c * g) * id;
    inv[5] = (c * d - a * f) * id;
    inv[6] = C * id;
    inv[7] = (b * g - a * h) * id;
    inv[8] = (a * e - b * d) * id;
}
// out[n*n] = A[n*n] @ B[n*n]
static void matmul(const double* A, const double* B, double* out, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
            out[i * n + j] = s;
        }
}

// Generic matrix exponential (scaling-and-squaring + Taylor), n<=5.
static void mat_exp(const double* A, int n, double* out) {
    double nrm = 0;
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += std::fabs(A[i * n + j]);
        nrm = std::max(nrm, s);
    }
    int k = 0;
    double scale = 1.0;
    while (nrm * scale > 0.25) {
        scale *= 0.5;
        ++k;
    }
    double B[25];
    for (int i = 0; i < n * n; i++) B[i] = A[i] * scale;
    double term[25] = {0}, acc[25] = {0}, tmp[25];
    for (int i = 0; i < n; i++) {
        term[i * n + i] = 1.0;
        acc[i * n + i] = 1.0;
    }
    for (int m = 1; m <= 18; m++) {
        matmul(term, B, tmp, n);
        for (int i = 0; i < n * n; i++) term[i] = tmp[i] / (double)m;
        for (int i = 0; i < n * n; i++) acc[i] += term[i];
    }
    for (int s = 0; s < k; s++) {
        matmul(acc, acc, tmp, n);
        std::memcpy(acc, tmp, sizeof(double) * n * n);
    }
    std::memcpy(out, acc, sizeof(double) * n * n);
}

// ---- quaternion <-> matrix (pytorch3d convention, scalar-LAST xyzw) ---------
// quat_to_mat: input wxyz here is handled by callers; this expects xyzw
// (i,j,k,r).
static void quat_xyzw_to_mat(const double q[4], double R[9]) {
    double i = q[0], j = q[1], k = q[2], r = q[3];
    double two_s = 2.0 / (i * i + j * j + k * k + r * r);
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

// mat_to_quat: returns xyzw (ijkr), standardized (real part >= 0).
static void mat_to_quat_xyzw(const double m[9], double out_xyzw[4]) {
    double m00 = m[0], m01 = m[1], m02 = m[2], m10 = m[3], m11 = m[4],
           m12 = m[5], m20 = m[6], m21 = m[7], m22 = m[8];
    double qa[4];
    qa[0] = 1.0 + m00 + m11 + m22;
    qa[1] = 1.0 + m00 - m11 - m22;
    qa[2] = 1.0 - m00 + m11 - m22;
    qa[3] = 1.0 - m00 - m11 + m22;
    for (int t = 0; t < 4; t++) qa[t] = qa[t] > 0 ? std::sqrt(qa[t]) : 0.0;
    // candidate rows (rijk order), pick the best-conditioned (largest qa)
    double cand[4][4] = {
            {qa[0] * qa[0], m21 - m12, m02 - m20, m10 - m01},
            {m21 - m12, qa[1] * qa[1], m10 + m01, m02 + m20},
            {m02 - m20, m10 + m01, qa[2] * qa[2], m12 + m21},
            {m10 - m01, m20 + m02, m21 + m12, qa[3] * qa[3]},
    };
    int best = 0;
    for (int t = 1; t < 4; t++)
        if (qa[t] > qa[best]) best = t;
    double flr = std::max(qa[best], 0.1);
    double denom = 2.0 * flr;
    double rijk[4];
    for (int t = 0; t < 4; t++) rijk[t] = cand[best][t] / denom;
    // rijk -> ijkr  (out indices [1,2,3,0])
    double x = rijk[1], y = rijk[2], z = rijk[3], w = rijk[0];
    if (w < 0) {
        x = -x;
        y = -y;
        z = -z;
        w = -w;
    }  // standardize
    out_xyzw[0] = x;
    out_xyzw[1] = y;
    out_xyzw[2] = z;
    out_xyzw[3] = w;
}

// cam_quat_xyzw_to_world_quat_wxyz: rotate the camera-space quaternion into
// world space using the cam2world rotation, returning wxyz.
static void cam_quat_to_world_wxyz(const double cam_xyzw[4],
                                   const double Rc2w[9],
                                   double out_wxyz[4]) {
    // xyzw -> wxyz, then build rotation matrix via quat_to_mat (which itself
    // takes xyzw). The python re-labels (w,x,y,z) as the (i,j,k,r) input to
    // quat_to_mat, i.e. it feeds [w,x,y,z] positionally as xyzw.
    double q_relabel[4] = {cam_xyzw[3], cam_xyzw[0], cam_xyzw[1],
                           cam_xyzw[2]};  // (w,x,y,z)
    double Rcam[9];
    quat_xyzw_to_mat(q_relabel, Rcam);
    double Rworld[9];
    matmul(Rc2w, Rcam, Rworld, 3);
    // NOTE (historical quirk in cam_quat_xyzw_to_world_quat_wxyz): the python
    // returns mat_to_quat's raw output (xyzw / ijkr order) but *labels* it
    // wxyz; there is NO reordering. So we emit mat_to_quat's xyzw result
    // verbatim into the stored "rotations" buffer to match the reference
    // exactly.
    mat_to_quat_xyzw(Rworld, out_wxyz);
}

// ---- e3nn angle / wigner machinery (rotate_sh) -----------------------------
// matrix_y / matrix_x rotations.
static void matrix_y(double a, double R[9]) {
    double c = std::cos(a), s = std::sin(a);
    R[0] = c;
    R[1] = 0;
    R[2] = s;
    R[3] = 0;
    R[4] = 1;
    R[5] = 0;
    R[6] = -s;
    R[7] = 0;
    R[8] = c;
}
static void matrix_x(double a, double R[9]) {
    double c = std::cos(a), s = std::sin(a);
    R[0] = 1;
    R[1] = 0;
    R[2] = 0;
    R[3] = 0;
    R[4] = c;
    R[5] = -s;
    R[6] = 0;
    R[7] = s;
    R[8] = c;
}
// angles_to_matrix = Ry(a) @ Rx(b) @ Ry(g)
static void angles_to_matrix(double a, double b, double g, double R[9]) {
    double Ya[9], Xb[9], Yg[9], t[9];
    matrix_y(a, Ya);
    matrix_x(b, Xb);
    matrix_y(g, Yg);
    matmul(Ya, Xb, t, 3);
    matmul(t, Yg, R, 3);
}
// matrix_to_angles (e3nn convention): returns alpha,beta,gamma.
static void matrix_to_angles(const double R[9],
                             double& alpha,
                             double& beta,
                             double& gamma) {
    // x = R @ [0,1,0]  -> second column of R
    double x0 = R[1], x1 = R[4], x2 = R[7];
    double nrm = std::sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    if (nrm > 0) {
        x0 /= nrm;
        x1 /= nrm;
        x2 /= nrm;
    }
    x1 = std::max(-1.0, std::min(1.0, x1));
    beta = std::acos(x1);
    alpha = std::atan2(x0, x2);
    // R2 = angles_to_matrix(a,b,0)^T @ R ; gamma = atan2(R2[0,2], R2[0,0])
    double A[9];
    angles_to_matrix(alpha, beta, 0.0, A);
    double AT[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) AT[i * 3 + j] = A[j * 3 + i];
    double R2[9];
    matmul(AT, R, R2, 3);
    gamma = std::atan2(R2[2], R2[0]);
}
// _z_rot_mat(angle,l): (2l+1)x(2l+1).
static void z_rot_mat(double angle, int l, double* M) {
    int d = 2 * l + 1;
    for (int i = 0; i < d * d; i++) M[i] = 0;
    for (int idx = 0; idx < d; idx++) {
        int rev = d - 1 - idx;
        double freq = (double)(l - idx);
        M[idx * d + rev] = std::sin(freq * angle);
        M[idx * d + idx] = std::cos(freq * angle);
    }
}
// so3 x-generator X0 for l (exact entries) -> dense (2l+1)^2.
static void x_gen(int l, double* X0) {
    int d = 2 * l + 1;
    for (int i = 0; i < d * d; i++) X0[i] = 0;
    if (l == 1) {
        // [[0,0,0],[0,0,-1],[0,1,0]]
        X0[1 * 3 + 2] = -1.0;
        X0[2 * 3 + 1] = 1.0;
    } else {  // l==2
        double s3 = std::sqrt(3.0);
        // [[0,1,0,0,0],[-1,0,0,0,0],[0,0,0,-s3,0],[0,0,s3,0,-1],[0,0,0,1,0]]
        X0[0 * 5 + 1] = 1.0;
        X0[1 * 5 + 0] = -1.0;
        X0[2 * 5 + 3] = -s3;
        X0[3 * 5 + 2] = s3;
        X0[3 * 5 + 4] = -1.0;
        X0[4 * 5 + 3] = 1.0;
    }
}
// wigner_D(l,a,b,g) = z_rot(a) @ expm(b*X0) @ z_rot(g)   (matches e3nn wigner_D
// via so3 generators: matrix_exp(a*Xz)@matrix_exp(b*Xx)@matrix_exp(g*Xz), with
// Xz closed form = z_rot_mat).
static void wigner_D(int l, double a, double b, double g, double* D) {
    int d = 2 * l + 1;
    double Za[25], Zg[25], X0[25], Eb[25], t[25];
    z_rot_mat(a, l, Za);
    z_rot_mat(g, l, Zg);
    x_gen(l, X0);
    double bX0[25];
    for (int i = 0; i < d * d; i++) bX0[i] = b * X0[i];
    mat_exp(bX0, d, Eb);
    matmul(Za, Eb, t, d);
    matmul(t, Zg, D, d);
}

}  // namespace

bool GsAdapter::build(const std::vector<float>& raw_gs,
                      const std::vector<float>& depth,
                      const std::vector<float>& gs_conf,
                      const std::array<float, 12>& ext,
                      const std::array<float, 9>& intr,
                      int H,
                      int W,
                      Gaussians& out) const {
    const size_t HW = (size_t)H * (size_t)W;
    if (raw_gs.size() != HW * 37 || depth.size() != HW || gs_conf.size() != HW)
        return false;
    const int dsh = d_sh;  // 9
    const int N = (int)HW;

    // --- extrinsics (w2c, 3x4) -> homogeneous 4x4 -> cam2world =
    // affine_inverse. c2w rotation Rc2w = R^T, translation Tc2w = -R^T t  (R,t
    // are the w2c block).
    double R[9], tt[3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) R[i * 3 + j] = ext[i * 4 + j];
        tt[i] = ext[i * 4 + 3];
    }
    double Rc2w[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) Rc2w[i * 3 + j] = R[j * 3 + i];  // R^T
    double Tc2w[3];
    for (int i = 0; i < 3; i++) {
        double s = 0;
        for (int j = 0; j < 3; j++) s += Rc2w[i * 3 + j] * tt[j];
        Tc2w[i] = -s;
    }

    // --- intr_normed = K with row0/=W, row1/=H ; inv used for unproject +
    // scale mult.
    double Kn[9];
    for (int j = 0; j < 3; j++) {
        Kn[0 * 3 + j] = (double)intr[0 * 3 + j] / W;
        Kn[1 * 3 + j] = (double)intr[1 * 3 + j] / H;
        Kn[2 * 3 + j] = (double)intr[2 * 3 + j];
    }
    double Kninv[9];
    mat3_inverse(Kn, Kninv);
    // get_scale_multiplier: 0.1 * sum( inv(Kn[:2,:2]) @ (1/W, 1/H) )
    double K2[4] = {Kn[0], Kn[1], Kn[3], Kn[4]};
    double det2 = K2[0] * K2[3] - K2[1] * K2[2];
    double id2 = (det2 != 0.0) ? 1.0 / det2 : 0.0;
    double K2inv[4] = {K2[3] * id2, -K2[1] * id2, -K2[2] * id2, K2[0] * id2};
    double ps[2] = {1.0 / W, 1.0 / H};
    double mx0 = K2inv[0] * ps[0] + K2inv[1] * ps[1];
    double mx1 = K2inv[2] * ps[0] + K2inv[3] * ps[1];
    double multiplier = 0.1 * (mx0 + mx1);

    // --- SH rotation matrices (same cam2world rotation for all pixels)
    // -------- rotate_sh: permute axes yzx->xyz (P^-1 R P), -> e3nn angles ->
    // wigner. P = [[0,0,1],[1,0,0],[0,1,0]]; P^-1 = P^T.
    double D1[9], D2[25];
    {
        double P[9] = {0, 0, 1, 1, 0, 0, 0, 1, 0};
        double Pinv[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) Pinv[i * 3 + j] = P[j * 3 + i];
        double t1[9], permR[9];
        matmul(Pinv, Rc2w, t1, 3);
        matmul(t1, P, permR, 3);
        // project_to_so3_strict is ~identity for a proper rotation; skip.
        double a, b, g;
        matrix_to_angles(permR, a, b, g);
        wigner_D(1, a, -b, g, D1);
        wigner_D(2, a, -b, g, D2);
    }

    // sh_mask[deg^2:(deg+1)^2] = 0.1*0.25^deg ; index 0 -> 1.
    double sh_mask[9];
    sh_mask[0] = 1.0;
    for (int deg = 1; deg <= sh_degree; ++deg) {
        double v = 0.1 * std::pow(0.25, deg);
        for (int k = deg * deg; k < (deg + 1) * (deg + 1); ++k) sh_mask[k] = v;
    }

    out.N = N;
    out.means.resize((size_t)N * 3);
    out.scales.resize((size_t)N * 3);
    out.rotations.resize((size_t)N * 4);
    out.harmonics.resize((size_t)N * 3 * dsh);
    out.opacities.resize((size_t)N);

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const size_t pix = (size_t)h * W + w;
            const float* rg = &raw_gs[pix * 37];

            // depth offset (ch 36) and depth.
            double gs_depth = (double)depth[pix] + (double)rg[36];

            // xy offset (ch 0,1). sample_image_grid: x=(w+0.5)/W, y=(h+0.5)/H.
            double xr = (w + 0.5) / W + (double)rg[0] * (1.0 / W);
            double yr = (h + 0.5) / H + (double)rg[1] * (1.0 / H);

            // unproject -> camera dir = normalize(Kn^-1 @ (xr,yr,1)).
            double dx = Kninv[0] * xr + Kninv[1] * yr + Kninv[2];
            double dy = Kninv[3] * xr + Kninv[4] * yr + Kninv[5];
            double dz = Kninv[6] * xr + Kninv[7] * yr + Kninv[8];
            double dn = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (dn > 0) {
                dx /= dn;
                dy /= dn;
                dz /= dn;
            }
            // world dir = Rc2w @ dir (homogeneous vector, w=0).
            double wdx = Rc2w[0] * dx + Rc2w[1] * dy + Rc2w[2] * dz;
            double wdy = Rc2w[3] * dx + Rc2w[4] * dy + Rc2w[5] * dz;
            double wdz = Rc2w[6] * dx + Rc2w[7] * dy + Rc2w[8] * dz;
            out.means[pix * 3 + 0] = (float)(Tc2w[0] + wdx * gs_depth);
            out.means[pix * 3 + 1] = (float)(Tc2w[1] + wdy * gs_depth);
            out.means[pix * 3 + 2] = (float)(Tc2w[2] + wdz * gs_depth);

            // scales (ch 2,3,4) sigmoid -> [min,max] * depth * multiplier.
            for (int d = 0; d < 3; d++) {
                double sg = 1.0 / (1.0 + std::exp(-(double)rg[2 + d]));
                double sc = scale_min + (scale_max - scale_min) * sg;
                out.scales[pix * 3 + d] = (float)(sc * gs_depth * multiplier);
            }

            // quaternion (ch 5,6,7,8) xyzw normalized -> world wxyz.
            double q[4] = {rg[5], rg[6], rg[7], rg[8]};
            double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] +
                                  q[3] * q[3]) +
                        1e-8;
            for (int d = 0; d < 4; d++) q[d] /= qn;
            double world_wxyz[4];
            cam_quat_to_world_wxyz(q, Rc2w, world_wxyz);
            for (int d = 0; d < 4; d++)
                out.rotations[pix * 4 + d] = (float)world_wxyz[d];

            // SH (ch 9.. ) : color-major (color*9 + coeff). mask then rotate.
            for (int color = 0; color < 3; ++color) {
                double sh[9];
                for (int k = 0; k < dsh; ++k)
                    sh[k] = (double)rg[9 + color * dsh + k] * sh_mask[k];
                double rot[9];
                rot[0] = sh[0];  // degree 0 unchanged
                // degree 1: D1 (3x3) on sh[1..3]
                for (int i = 0; i < 3; i++) {
                    double s = 0;
                    for (int j = 0; j < 3; j++) s += D1[i * 3 + j] * sh[1 + j];
                    rot[1 + i] = s;
                }
                // degree 2: D2 (5x5) on sh[4..8]
                for (int i = 0; i < 5; i++) {
                    double s = 0;
                    for (int j = 0; j < 5; j++) s += D2[i * 5 + j] * sh[4 + j];
                    rot[4 + i] = s;
                }
                float* hp = &out.harmonics[(pix * 3 + color) * dsh];
                for (int k = 0; k < dsh; ++k) hp[k] = (float)rot[k];
            }

            // opacity = map_pdf_to_opacity(conf) with global_step=0 ->
            // identity.
            double pdf = (double)gs_conf[pix];
            out.opacities[pix] =
                    (float)(0.5 * (1.0 - (1.0 - pdf) + pdf));  // == pdf
        }
    }
    return true;
}

}  // namespace depth
}  // namespace aicore
