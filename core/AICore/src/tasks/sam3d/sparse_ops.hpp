// Host-side index tables for the SLat sparse stages (see sparse_ops.cpp).
// The token set is fixed for the whole sampling loop (coords come from the
// SS stage occupancy), so every gather/scatter table is built once.
#pragma once

#include <cstdint>
#include <vector>

namespace sam3d {

struct SlatTables {
    int64_t nf = 0;   // fine tokens (one per active cell at resolution 64)
    int64_t nc = 0;   // coarse tokens (after the 2x mean-pool downsample)

    // fine token -> owning coarse token row (torch scatter_reduce idx; this
    // is also the SparseUpsample gather table: fine_i = coarse[idx[i]])
    std::vector<int32_t> fine_to_coarse;
    // coarse token -> its children rows in the fine table, (8*nc) with the
    // child index fastest (c*8+j); missing children point at the zero
    // sentinel. Coarse grids are partial (a 2x2x2 block need not be full),
    // so this is NOT a reshape of fine_to_coarse.
    std::vector<int32_t> coarse_children;
    std::vector<float> coarse_count;       // (nc) |children|+1 divisor for the mean
    // SubMConv3d neighbor rows, offset-major flattened (n*27 + o) with
    // o = kd*9 + kh*3 + kw matching the (Cout, 3,3,3, Cin) weight flatten;
    // missing neighbors point at the zero sentinel row (index N in a
    // (N+1)-row feature table).
    std::vector<int32_t> conv_fine;    // (27*nf)
    std::vector<int32_t> conv_coarse;  // (27*nc)
    // spconv-compatible tables. Indices are offset-major (27*n), missing
    // neighbours are -1, and masks are a stable ascending ordering of one
    // valid-neighbour bitset per output token. argsort maps each sorted mask
    // row back to the corresponding token, matching spconv implicit GEMM.
    std::vector<int32_t> conv_fine_spconv;
    std::vector<int32_t> conv_coarse_spconv;
    std::vector<int32_t> conv_fine_mask;
    std::vector<int32_t> conv_coarse_mask;
    std::vector<int32_t> conv_fine_argsort;
    std::vector<int32_t> conv_coarse_argsort;
    // AbsolutePositionEmbedder output for the coarse grid, (1024, nc) with
    // channel-fastest layout (a plain graph input).
    std::vector<float> ape;

    // coords: (nf, 4) [batch, x, y, z]; builds every table. Returns false on
    // out-of-range coordinates.
    bool build(const int32_t* coords, int64_t n_fine, int channels = 1024,
               bool dump_tables = false);
};

// Swin-window attention tables for the GS decoder (slat_decoder_gs). The
// sparse structure is fixed for the whole decode, so the window grouping is
// built once per shift (blocks alternate shift_window 0 / 4).
struct GsWindowTables {
    int64_t nf = 0;
    // bucket-order gather: out[:, i] = x[:, bkt_idx[i]] reorders tokens so
    // that same-length windows sit contiguous (window-major, token within
    // window in original table order - attention is order invariant)
    std::vector<int32_t> bkt_idx;
    std::vector<int32_t> bwd_idx;  // bucket order -> original order
    struct Bucket {
        int32_t len;      // tokens per window in this bucket
        int32_t off;      // first row in bucket-order token space
        int32_t n_win;    // windows in this bucket (batch dim)
    };
    std::vector<Bucket> buckets;
};

struct GsTables {
    GsWindowTables shifts[2];  // [0]: shift (0,0,0), [1]: shift (4,4,4)
    // AbsolutePositionEmbedder output for the fine coords, (768, nf)
    // token-major rows (channel fastest) - a plain graph input.
    std::vector<float> ape;

    // coords: (nf, 4) [batch, x, y, z]; window_size fixed at 8 (matches
    // slat_decoder_gs.yaml).
    bool build(const int32_t* coords, int64_t n_fine);
};

// Sparse coordinate and neighbour tables for SLatMeshDecoder.  Unlike the
// Gaussian decoder, this branch expands every active 64^3 cell twice.  The
// expansion order is exactly torch.nonzero(torch.ones(2, 2, 2)): z varies
// fastest, then y, then x.  This preserves the feature/coordinate ownership
// expected by SparseSubdivide and by the later FlexiCubes extractor.
struct MeshConvLevel {
    int64_t n = 0;
    int resolution = 0;
    // (n, 4), token-major [batch, x, y, z]
    std::vector<int32_t> coords;
    // (n * 27), token-major 3x3x3 SubMConv gather table. Missing neighbours
    // refer to row n, the zero sentinel appended by MeshDecoderGraph.
    std::vector<int32_t> conv;
};

struct MeshTables {
    // The Mesh and Gaussian decoders share the same 64^3 Swin torso.
    GsTables swin;
    // levels[0] is the input sparse support at 64^3; levels[1] and [2] are
    // the 128^3 and 256^3 supports after the two SparseSubdivide calls.
    MeshConvLevel levels[3];

    bool build(const int32_t* coords, int64_t n_fine, int base_resolution = 64);
};

}  // namespace sam3d
