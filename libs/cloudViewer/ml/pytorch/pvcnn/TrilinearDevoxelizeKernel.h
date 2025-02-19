#pragma once

/// This function performs trilinear devoxelization operation.
/// It computes aggregated features from the voxel grid for each
/// point passed in the input.
///
/// \param b    The batch size.
/// \param c    Feature dimension of voxel grid.
/// \param n    Number of points per batch.
/// \param r    Resolution of the grid.
/// \param r2   r squared.
/// \param r3   r cubed.
/// \param is_training  Whether model is in training phase.
/// \param coords   Array with the point positions. The shape is
///        [b, 3, n]
/// \param feat    Array with the voxel grid. The shape is
///        [b, c, r, r, r]
/// \param inds    The voxel coordinates of point cube [b, 8, n]
/// \param wgts    weight for trilinear interpolation [b, 8, n]
/// \param outs    Outputs, FloatTensor[b, c, n]
///
void TrilinearDevoxelize(int b,
                         int c,
                         int n,
                         int r,
                         int r2,
                         int r3,
                         bool is_training,
                         const float *coords,
                         const float *feat,
                         int *inds,
                         float *wgts,
                         float *outs);

/// This function computes gradient for trilinear devoxelization op.
/// It computes gradient for the input voxelgrid.
///
/// \param b    The batch size.
/// \param c    Feature dimension of voxel grid.
/// \param n    Number of points per batch.
/// \param r3   resolution cubed.
/// \param inds    The voxel coordinates of point cube [b, 8, n]
/// \param wgts    weight for trilinear interpolation [b, 8, n]
/// \param grad_y    The gradient passed from top.
/// \param grad_x   The computed gradient for voxelgrid.
///
void TrilinearDevoxelizeGrad(int b,
                             int c,
                             int n,
                             int r3,
                             const int *inds,
                             const float *wgts,
                             const float *grad_y,
                             float *grad_x);
