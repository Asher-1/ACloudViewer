
// ---------------------------------------------------------------------------
// Exact replication of Pillow 12.2.0's 8bpc BICUBIC resample — the official
// load_and_preprocess_images path. Bit-level agreement with PIL matters:
// the plugin's end-to-end output must not differ from the upstream engine
// fed by the official Python preprocessing. Reference:
// Pillow-12.2.0/src/libImaging/Resample.c:
//   - precompute_coeffs: center = (out+0.5)*scale, support = 2*filterscale,
//     window rounded via +0.5, float coeffs normalized by their sum
//   - normalize_coeffs_8bpc: coefficients fixed-point (<< PRECISION_BITS=22)
//   - passes: u8 in -> u8 out with rounding bias (1 << (PRECISION_BITS-1))
//     and a clip8 lookup; identity sizes skip the pass entirely.
// ---------------------------------------------------------------------------
constexpr int kPillowPrecisionBits = 32 - 8 - 2;  // 22

inline double pillow_bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

inline uint8_t pillow_clip8(int in) {
    const int v = in >> kPillowPrecisionBits;  // arithmetic shift
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

struct PillowCoeffs {
    int ksize = 0;
    std::vector<int> bounds;  // [out * 2] = {xmin, xmax(count)}
    std::vector<int> kk;      // fixed-point coefficients
};

static PillowCoeffs pillow_precompute_coeffs(int inSize, int outSize) {
    const double scale = static_cast<double>(inSize) / outSize;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 2.0 * filterscale;  // BICUBIC support = 2.0
    PillowCoeffs c;
    c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    c.bounds.assign(static_cast<size_t>(outSize) * 2, 0);
    c.kk.assign(static_cast<size_t>(outSize) * c.ksize, 0);
    std::vector<double> prekk(static_cast<size_t>(outSize) * c.ksize, 0.0);
    for (int xx = 0; xx < outSize; ++xx) {
        const double center = (xx + 0.5) * scale;
        double ww = 0.0;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        double* k = prekk.data() + static_cast<size_t>(xx) * c.ksize;
        for (int x = 0; x < xmax; ++x) {
            const double w = pillow_bicubic((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; ++x) {
            if (ww != 0.0) k[x] /= ww;
        }
        for (int x = xmax; x < c.ksize; ++x) k[x] = 0.0;
        c.bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
        c.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    // normalize_coeffs_8bpc
    for (size_t i = 0; i < prekk.size(); ++i) {
        c.kk[i] = static_cast<int>(
                (prekk[i] >= 0 ? 0.5 : -0.5) +
                prekk[i] * static_cast<double>(1 << kPillowPrecisionBits));
    }
    return c;
}

// Horizontal pass over a tightly-packed 3-channel u8 image.
static void pillow_resample_horizontal(const uint8_t* in, int inW, int inH,
                                       uint8_t* out, int outW,
                                       const PillowCoeffs& c) {
    const int rounding = 1 << (kPillowPrecisionBits - 1);
    for (int y = 0; y < inH; ++y) {
        const uint8_t* inRow = in + static_cast<size_t>(y) * inW * 3;
        uint8_t* outRow = out + static_cast<size_t>(y) * outW * 3;
        for (int xx = 0; xx < outW; ++xx) {
            const int xmin = c.bounds[static_cast<size_t>(xx) * 2 + 0];
            const int xmax = c.bounds[static_cast<size_t>(xx) * 2 + 1];
            const int* k = c.kk.data() + static_cast<size_t>(xx) * c.ksize;
            int ss0 = rounding, ss1 = rounding, ss2 = rounding;
            for (int x = 0; x < xmax; ++x) {
                const uint8_t* px = inRow + static_cast<size_t>(x + xmin) * 3;
                ss0 += px[0] * k[x];
                ss1 += px[1] * k[x];
                ss2 += px[2] * k[x];
            }
            outRow[static_cast<size_t>(xx) * 3 + 0] = pillow_clip8(ss0);
            outRow[static_cast<size_t>(xx) * 3 + 1] = pillow_clip8(ss1);
            outRow[static_cast<size_t>(xx) * 3 + 2] = pillow_clip8(ss2);
        }
    }
}

// Vertical pass over a tightly-packed 3-channel u8 image.
static void pillow_resample_vertical(const uint8_t* in, int inW, int inH,
                                     uint8_t* out, int outH,
                                     const PillowCoeffs& c) {
    const int rounding = 1 << (kPillowPrecisionBits - 1);
    for (int yy = 0; yy < outH; ++yy) {
        const int ymin = c.bounds[static_cast<size_t>(yy) * 2 + 0];
        const int ymax = c.bounds[static_cast<size_t>(yy) * 2 + 1];
        const int* k = c.kk.data() + static_cast<size_t>(yy) * c.ksize;
        uint8_t* outRow = out + static_cast<size_t>(yy) * inW * 3;
        for (int xx = 0; xx < inW; ++xx) {
            int ss0 = rounding, ss1 = rounding, ss2 = rounding;
            for (int y = 0; y < ymax; ++y) {
                const uint8_t* px =
                        in + static_cast<size_t>(y + ymin) * inW * 3 +
                        static_cast<size_t>(xx) * 3;
                ss0 += px[0] * k[y];
                ss1 += px[1] * k[y];
                ss2 += px[2] * k[y];
            }
            outRow[static_cast<size_t>(xx) * 3 + 0] = pillow_clip8(ss0);
            outRow[static_cast<size_t>(xx) * 3 + 1] = pillow_clip8(ss1);
            outRow[static_cast<size_t>(xx) * 3 + 2] = pillow_clip8(ss2);
        }
    }
}
