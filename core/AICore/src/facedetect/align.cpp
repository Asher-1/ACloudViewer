#include "align.hpp"
#include "common.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace fd {

// insightface arcface_dst (the 5-point template for a 112x112 aligned face).
// Source: insightface/utils/face_align.py `arcface_dst`.
const Landmarks5 kArcFaceRefLandmarks112 = {{
    {{38.2946f, 51.6963f}},
    {{73.5318f, 51.5014f}},
    {{56.0252f, 71.7366f}},
    {{41.5493f, 92.3655f}},
    {{70.7299f, 92.2041f}},
}};

// Robust 2x2 SVD: A = U * diag(s0, s1) * Vt with s0 >= s1 >= 0. V (a rotation)
// diagonalizes A^T A; U follows from A*V/s. This avoids the sign/branch fragility
// of the two-angle closed form (an earlier closed-form attempt drifted from the
// golden crop), and is validated numerically against skimage's _umeyama.
static void svd2x2(double a, double b, double c, double d,
                   double U[2][2], double S[2], double Vt[2][2]) {
    double ata00 = a*a + c*c;
    double ata01 = a*b + c*d;
    double ata11 = b*b + d*d;
    double phi   = 0.5 * std::atan2(2.0*ata01, ata00 - ata11);
    double cphi  = std::cos(phi), sphi = std::sin(phi);
    double t1 = ata00 + ata11;
    double t2 = std::hypot(ata00 - ata11, 2.0*ata01);
    double s0 = std::sqrt(std::max((t1 + t2) * 0.5, 0.0));
    double s1 = std::sqrt(std::max((t1 - t2) * 0.5, 0.0));
    // V columns are eigenvectors of A^T A (v0 along phi).
    double V[2][2] = {{cphi, -sphi}, {sphi, cphi}};
    // AV = A * V (columns), then U[:,j] = AV[:,j] / s_j.
    double av00 = a*V[0][0] + b*V[1][0], av01 = a*V[0][1] + b*V[1][1];
    double av10 = c*V[0][0] + d*V[1][0], av11 = c*V[0][1] + d*V[1][1];
    if (s0 > 1e-12) { U[0][0] = av00/s0; U[1][0] = av10/s0; }
    else            { U[0][0] = av00;    U[1][0] = av10;    }
    if (s1 > 1e-12) { U[0][1] = av01/s1; U[1][1] = av11/s1; }
    else            { U[0][1] = av01;    U[1][1] = av11;    }
    S[0] = s0; S[1] = s1;
    Vt[0][0] = V[0][0]; Vt[0][1] = V[1][0];   // Vt = V^T
    Vt[1][0] = V[0][1]; Vt[1][1] = V[1][1];
}

// Umeyama similarity (rotation+uniform scale+translation, no shear), src->dst,
// matching skimage SimilarityTransform.estimate. Returns the 2x3 affine M.
//
// Reference: Umeyama 1991 ("Least-squares estimation of transformation
// parameters between two point patterns"), as implemented in skimage's _umeyama
// (the path insightface estimate_norm uses). The 2x2 rotation comes from the SVD
// of the cross-covariance H = dst_demean^T * src_demean / N, with a reflection
// correction D = diag(1, sign(det H)). The scale is sum(S * D) / var(src).
static std::array<float,6> estimate_norm(const Landmarks5& lmk) {
    const int N = 5;
    double sx=0,sy=0,dx=0,dy=0;
    for (int i=0;i<N;++i){ sx+=lmk[i][0]; sy+=lmk[i][1];
        dx+=kArcFaceRefLandmarks112[i][0]; dy+=kArcFaceRefLandmarks112[i][1]; }
    sx/=N; sy/=N; dx/=N; dy/=N;
    double var_s=0; double H[2][2]={{0,0},{0,0}};   // cov = dst_demean^T * src_demean / N
    for (int i=0;i<N;++i){
        double sxx=lmk[i][0]-sx, syy=lmk[i][1]-sy;
        double dxx=kArcFaceRefLandmarks112[i][0]-dx, dyy=kArcFaceRefLandmarks112[i][1]-dy;
        var_s += sxx*sxx + syy*syy;
        H[0][0]+=dxx*sxx; H[0][1]+=dxx*syy;
        H[1][0]+=dyy*sxx; H[1][1]+=dyy*syy;
    }
    var_s/=N; H[0][0]/=N; H[0][1]/=N; H[1][0]/=N; H[1][1]/=N;
    double U[2][2], S[2], Vt[2][2];
    svd2x2(H[0][0], H[0][1], H[1][0], H[1][1], U, S, Vt);
    double detH = H[0][0]*H[1][1] - H[0][1]*H[1][0];
    double D0 = 1.0, D1 = (detH < 0 ? -1.0 : 1.0);   // reflection-correcting D
    // R = U * diag(D0, D1) * Vt.
    double Rm[2][2];
    Rm[0][0]=U[0][0]*D0*Vt[0][0]+U[0][1]*D1*Vt[1][0];
    Rm[0][1]=U[0][0]*D0*Vt[0][1]+U[0][1]*D1*Vt[1][1];
    Rm[1][0]=U[1][0]*D0*Vt[0][0]+U[1][1]*D1*Vt[1][0];
    Rm[1][1]=U[1][0]*D0*Vt[0][1]+U[1][1]*D1*Vt[1][1];
    double scale = (var_s>0)? (S[0]*D0 + S[1]*D1)/var_s : 1.0;
    std::array<float,6> M{};
    M[0]=(float)(scale*Rm[0][0]); M[1]=(float)(scale*Rm[0][1]);
    M[3]=(float)(scale*Rm[1][0]); M[4]=(float)(scale*Rm[1][1]);
    M[2]=(float)(dx - (M[0]*sx + M[1]*sy));
    M[5]=(float)(dy - (M[3]*sx + M[4]*sy));
    return M;   // [m00 m01 m02 ; m10 m11 m12]
}

bool warp_affine(const Image& src, const std::array<float,6>& M, Image& out,
                 int out_w, int out_h) {
    if (src.empty() || out_w <= 0 || out_h <= 0) return false;
    // Invert M (2x3) to map dst->src for backward warpAffine sampling.
    double det = (double)M[0]*M[4] - (double)M[1]*M[3];
    if (std::fabs(det) < 1e-12) return false;
    double i00= M[4]/det, i01=-M[1]/det, i10=-M[3]/det, i11=M[0]/det;
    double i02=-(i00*M[2]+i01*M[5]), i12=-(i10*M[2]+i11*M[5]);

    // Reproduce OpenCV's warpAffine INTER_LINEAR sampling so the crop is
    // byte-faithful (<= 1 LSB) to cv2.warpAffine, which is what the insightface
    // golden aligned_crop was generated with. OpenCV quantizes the source
    // coordinate to a 5-bit sub-pixel grid (INTER_BITS) via 10-bit fixed point
    // (AB_BITS); a plain float bilinear drifts by up to 2 LSB and fails the gate.
    // See OpenCV modules/imgproc/src/imgwarp.cpp (warpAffine / remap bilinear).
    const int INTER_BITS = 5, INTER_TAB = 1 << INTER_BITS;      // 32 sub-pixel steps
    const int AB_BITS = 10, AB_SCALE = 1 << AB_BITS;            // coordinate fixed point
    const int ROUND_DELTA = AB_SCALE / INTER_TAB / 2;           // 16
    auto cvRound = [](double v) -> long { return std::lrint(v); }; // round half to even

    out.width=out_w; out.height=out_h;
    out.rgb.assign((size_t)out_w*out_h*3, 0);
    std::vector<long> adelta(out_w), bdelta(out_w);
    for (int dxx=0; dxx<out_w; ++dxx) {
        adelta[dxx] = cvRound(i00 * dxx * AB_SCALE);
        bdelta[dxx] = cvRound(i10 * dxx * AB_SCALE);
    }
    auto S=[&](int xx,int yy,int ch)->double{
        if (xx<0||yy<0||xx>=src.width||yy>=src.height) return 0.0; // BORDER_CONSTANT 0
        return src.rgb[((size_t)yy*src.width+xx)*3+ch]; };
    for (int dyy=0; dyy<out_h; ++dyy) {
      long X0 = cvRound((i01*dyy + i02) * AB_SCALE) + ROUND_DELTA;
      long Y0 = cvRound((i11*dyy + i12) * AB_SCALE) + ROUND_DELTA;
      for (int dxx=0; dxx<out_w; ++dxx) {
        long X = (X0 + adelta[dxx]) >> (AB_BITS - INTER_BITS);
        long Y = (Y0 + bdelta[dxx]) >> (AB_BITS - INTER_BITS);
        int x0 = (int)(X >> INTER_BITS), y0 = (int)(Y >> INTER_BITS);
        double ax = (double)(X & (INTER_TAB-1)) / INTER_TAB;   // quantized sub-pixel
        double ay = (double)(Y & (INTER_TAB-1)) / INTER_TAB;
        for (int ch=0; ch<3; ++ch) {
          double v = S(x0,y0,ch)*(1-ax)*(1-ay) + S(x0+1,y0,ch)*ax*(1-ay)
                   + S(x0,y0+1,ch)*(1-ax)*ay   + S(x0+1,y0+1,ch)*ax*ay;
          out.rgb[((size_t)dyy*out_w+dxx)*3+ch] = (uint8_t)std::lround(v);
        }
      }
    }
    return true;
}

bool norm_crop(const Image& src, const Landmarks5& lmk, Image& out, int size) {
    if (src.empty() || size <= 0) return false;
    return warp_affine(src, estimate_norm(lmk), out, size, size);
}

} // namespace fd
