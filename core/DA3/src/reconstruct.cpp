#include "reconstruct.hpp"

#include <algorithm>
#include <cmath>

namespace da {

// ---------------------------------------------------------------------------
// Linear algebra helpers (double internally for accuracy, return float).
// ---------------------------------------------------------------------------

bool inv3(const std::array<float, 9>& m, std::array<float, 9>& out) {
    double a = m[0], b = m[1], c = m[2];
    double d = m[3], e = m[4], f = m[5];
    double g = m[6], h = m[7], i = m[8];
    double A =  (e * i - f * h);
    double B = -(d * i - f * g);
    double C =  (d * h - e * g);
    double det = a * A + b * B + c * C;
    if (det == 0.0 || !std::isfinite(det)) return false;
    double inv = 1.0 / det;
    // adjugate (transpose of cofactor matrix) * 1/det
    out[0] = (float)(A * inv);
    out[1] = (float)(-(b * i - c * h) * inv);
    out[2] = (float)((b * f - c * e) * inv);
    out[3] = (float)(B * inv);
    out[4] = (float)((a * i - c * g) * inv);
    out[5] = (float)(-(a * f - c * d) * inv);
    out[6] = (float)(C * inv);
    out[7] = (float)(-(a * h - b * g) * inv);
    out[8] = (float)((a * e - b * d) * inv);
    return true;
}

bool inv4(const std::array<float, 16>& m, std::array<float, 16>& out) {
    // Gauss-Jordan elimination on [m | I] with partial pivoting, in double.
    double a[4][8];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) a[r][c] = m[r * 4 + c];
        for (int c = 0; c < 4; ++c) a[r][4 + c] = (r == c) ? 1.0 : 0.0;
    }
    for (int col = 0; col < 4; ++col) {
        // find pivot
        int piv = col;
        double best = std::fabs(a[col][col]);
        for (int r = col + 1; r < 4; ++r) {
            double v = std::fabs(a[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best == 0.0 || !std::isfinite(best)) return false;
        if (piv != col) for (int c = 0; c < 8; ++c) std::swap(a[col][c], a[piv][c]);
        double d = a[col][col];
        for (int c = 0; c < 8; ++c) a[col][c] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            double factor = a[r][col];
            if (factor == 0.0) continue;
            for (int c = 0; c < 8; ++c) a[r][c] -= factor * a[col][c];
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[r * 4 + c] = (float)a[r][4 + c];
    return true;
}

// ---------------------------------------------------------------------------
// percentile_linear: numpy.percentile linear interpolation.
// ---------------------------------------------------------------------------

double percentile_linear(const std::vector<float>& v, double q_percent) {
    const size_t n = v.size();
    if (n == 0) return 0.0;
    if (n == 1) return (double)v[0];
    std::vector<float> s(v);
    std::sort(s.begin(), s.end());
    double idx = (q_percent / 100.0) * (double)(n - 1);
    double lo = std::floor(idx);
    double hi = std::ceil(idx);
    double frac = idx - lo;
    size_t li = (size_t)lo, hii = (size_t)hi;
    if (hii >= n) hii = n - 1;
    return (double)s[li] + frac * ((double)s[hii] - (double)s[li]);
}

// ---------------------------------------------------------------------------
// rotmat2qvec: symmetric-matrix eigenvector method (Jacobi), COLMAP order.
// Ported from read_write_model.py's rotmat2qvec.
// ---------------------------------------------------------------------------

std::array<float, 4> rotmat2qvec(const std::array<float, 9>& R) {
    // R indexed row-major: R[row*3+col]. Reference uses R.flat order
    // Rxx,Ryx,Rzx, Rxy,Ryy,Rzy, Rxz,Ryz,Rzz, i.e. row-major as well.
    double Rxx = R[0], Ryx = R[1], Rzx = R[2];
    double Rxy = R[3], Ryy = R[4], Rzy = R[5];
    double Rxz = R[6], Ryz = R[7], Rzz = R[8];

    // Build the symmetric 4x4 K matrix. The reference fills the lower triangle
    // and np.linalg.eigh (UPLO='L') treats it as symmetric; mirror it here.
    double K[4][4];
    K[0][0] = (Rxx - Ryy - Rzz) / 3.0;
    K[1][1] = (Ryy - Rxx - Rzz) / 3.0;
    K[2][2] = (Rzz - Rxx - Ryy) / 3.0;
    K[3][3] = (Rxx + Ryy + Rzz) / 3.0;
    K[1][0] = K[0][1] = (Ryx + Rxy) / 3.0;
    K[2][0] = K[0][2] = (Rzx + Rxz) / 3.0;
    K[2][1] = K[1][2] = (Rzy + Ryz) / 3.0;
    K[3][0] = K[0][3] = (Ryz - Rzy) / 3.0;
    K[3][1] = K[1][3] = (Rzx - Rxz) / 3.0;
    K[3][2] = K[2][3] = (Rxy - Ryx) / 3.0;

    // Jacobi eigenvalue algorithm for symmetric 4x4.
    double V[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    for (int sweep = 0; sweep < 100; ++sweep) {
        // off-diagonal magnitude
        double off = 0.0;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q) off += K[p][q] * K[p][q];
        if (off < 1e-30) break;
        for (int p = 0; p < 4; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                if (std::fabs(K[p][q]) < 1e-300) continue;
                double theta = (K[q][q] - K[p][p]) / (2.0 * K[p][q]);
                double t = (theta >= 0 ? 1.0 : -1.0) /
                           (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(t * t + 1.0);
                double s = t * c;
                // rotate K
                for (int i = 0; i < 4; ++i) {
                    double kip = K[i][p], kiq = K[i][q];
                    K[i][p] = c * kip - s * kiq;
                    K[i][q] = s * kip + c * kiq;
                }
                for (int i = 0; i < 4; ++i) {
                    double kpi = K[p][i], kqi = K[q][i];
                    K[p][i] = c * kpi - s * kqi;
                    K[q][i] = s * kpi + c * kqi;
                }
                // accumulate eigenvectors
                for (int i = 0; i < 4; ++i) {
                    double vip = V[i][p], viq = V[i][q];
                    V[i][p] = c * vip - s * viq;
                    V[i][q] = s * vip + c * viq;
                }
            }
        }
    }
    // eigenvalues on diagonal; pick column of largest eigenvalue
    int best = 0;
    double bestVal = K[0][0];
    for (int i = 1; i < 4; ++i) {
        if (K[i][i] > bestVal) { bestVal = K[i][i]; best = i; }
    }
    // eigenvector column `best`; reference reorders rows [3,0,1,2] -> (qw,qx,qy,qz)
    double ev[4] = {V[0][best], V[1][best], V[2][best], V[3][best]};
    std::array<float, 4> q;
    q[0] = (float)ev[3]; // qw
    q[1] = (float)ev[0]; // qx
    q[2] = (float)ev[1]; // qy
    q[3] = (float)ev[2]; // qz
    if (q[0] < 0) { q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; }
    return q;
}

// ---------------------------------------------------------------------------
// back_project
// ---------------------------------------------------------------------------

WorldPoints back_project(const std::vector<float>& depth,
                         const std::vector<float>& conf,
                         const std::vector<std::array<float, 9>>& K,
                         const std::vector<std::array<float, 16>>& ext_w2c,
                         const std::vector<const uint8_t*>& images_u8,
                         int H, int W, int N, float conf_thr) {
    WorldPoints wp;
    const size_t plane = (size_t)H * (size_t)W;
    const bool have_conf = !conf.empty();

    for (int i = 0; i < N; ++i) {
        std::array<float, 9> Kinv;
        std::array<float, 16> c2w;
        if (!inv3(K[i], Kinv)) continue;
        if (!inv4(ext_w2c[i], c2w)) continue;

        const float* d_frame = depth.data() + (size_t)i * plane;
        const float* c_frame = have_conf ? conf.data() + (size_t)i * plane : nullptr;
        const uint8_t* img = images_u8[i];

        // double-precision copies for math
        double ki[9];
        for (int k = 0; k < 9; ++k) ki[k] = (double)Kinv[k];
        double cw[16];
        for (int k = 0; k < 16; ++k) cw[k] = (double)c2w[k];

        for (int v = 0; v < H; ++v) {
            for (int u = 0; u < W; ++u) {
                size_t pix = (size_t)v * (size_t)W + (size_t)u;
                float d = d_frame[pix];
                if (!std::isfinite(d) || d <= 0.0f) continue;
                if (have_conf && !(c_frame[pix] >= conf_thr)) continue;

                // ray = Kinv @ [u,v,1]
                double rx = ki[0] * u + ki[1] * v + ki[2];
                double ry = ki[3] * u + ki[4] * v + ki[5];
                double rz = ki[6] * u + ki[7] * v + ki[8];
                // Xc = ray * d
                double xc = rx * d, yc = ry * d, zc = rz * d;
                // Xw = (c2w @ [Xc;1])[:3]
                double xw = cw[0] * xc + cw[1] * yc + cw[2] * zc + cw[3];
                double yw = cw[4] * xc + cw[5] * yc + cw[6] * zc + cw[7];
                double zw = cw[8] * xc + cw[9] * yc + cw[10] * zc + cw[11];

                wp.xyz.push_back((float)xw);
                wp.xyz.push_back((float)yw);
                wp.xyz.push_back((float)zw);
                wp.rgb.push_back(img[3 * pix + 0]);
                wp.rgb.push_back(img[3 * pix + 1]);
                wp.rgb.push_back(img[3 * pix + 2]);
                wp.frame.push_back(i);
                wp.u.push_back(u);
                wp.v.push_back(v);
            }
        }
    }
    return wp;
}

} // namespace da
