// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "base/camera.h"
#include "util/alignment.h"
#include "util/bitmap.h"

namespace colmap {

// Warp source image to target image by projecting the pixels of the target
// image up to infinity and projecting it down into the source image
// (i.e. an inverse mapping). The function allocates the target image.
void WarpImageBetweenCameras(const Camera& source_camera,
                             const Camera& target_camera,
                             const Bitmap& source_image,
                             Bitmap* target_image);

// Warp an image with the given homography, where H defines the pixel mapping
// from the target to source image. Note that the pixel centers are assumed to
// have coordinates (0.5, 0.5).
void WarpImageWithHomography(const Eigen::Matrix3d& H,
                             const Bitmap& source_image,
                             Bitmap* target_image);

// First, warp source image to target image by projecting the pixels of the
// target image up to infinity and projecting it down into the source image
// (i.e. an inverse mapping). Second, warp the coordinates from the first
// warping with the given homography. The function allocates the target image.
void WarpImageWithHomographyBetweenCameras(const Eigen::Matrix3d& H,
                                           const Camera& source_camera,
                                           const Camera& target_camera,
                                           const Bitmap& source_image,
                                           Bitmap* target_image);

// Resample row-major image using bilinear interpolation.
void ResampleImageBilinear(const float* data,
                           const int rows,
                           const int cols,
                           const int new_rows,
                           const int new_cols,
                           float* resampled);

// Smooth row-major image using a Gaussian filter kernel.
void SmoothImage(const float* data,
                 const int rows,
                 const int cols,
                 const float sigma_r,
                 const float sigma_c,
                 float* smoothed);

// Downsample row-major image by first smoothing and then resampling.
void DownsampleImage(const float* data,
                     const int rows,
                     const int cols,
                     const int new_rows,
                     const int new_cols,
                     float* downsampled);

}  // namespace colmap
