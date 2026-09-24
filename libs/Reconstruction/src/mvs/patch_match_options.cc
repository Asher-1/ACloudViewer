// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mvs/patch_match_options.h"

#include "util/logging.h"
#include "util/misc.h"

namespace colmap {
namespace mvs {

#define PrintOption(option) \
  std::cout << #option ": " << option << std::endl

void PatchMatchOptions::Print() const {
  PrintHeading2("PatchMatchOptions");
  PrintOption(max_image_size);
  PrintOption(gpu_index);
  PrintOption(depth_min);
  PrintOption(depth_max);
  PrintOption(window_radius);
  PrintOption(window_step);
  PrintOption(sigma_spatial);
  PrintOption(sigma_color);
  PrintOption(num_samples);
  PrintOption(ncc_sigma);
  PrintOption(min_triangulation_angle);
  PrintOption(incident_angle_sigma);
  PrintOption(num_iterations);
  PrintOption(geom_consistency);
  PrintOption(geom_consistency_regularizer);
  PrintOption(geom_consistency_max_cost);
  PrintOption(filter);
  PrintOption(filter_min_ncc);
  PrintOption(filter_min_triangulation_angle);
  PrintOption(filter_min_num_consistent);
  PrintOption(filter_geom_consistency_max_cost);
  PrintOption(write_consistency_graph);
  PrintOption(allow_missing_files);
  PrintOption(photometric_force_recompute);
  PrintOption(photometric_use_existing_as_init);
  PrintOption(skip_photometric_pass);
}

}  // namespace mvs
}  // namespace colmap
