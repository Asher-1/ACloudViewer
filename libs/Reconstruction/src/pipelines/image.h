// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "controllers/incremental_mapper.h"

namespace cloudViewer {

int DeleteImage(const std::string& input_path,
                const std::string& output_path,
                const std::string& image_ids_path = "",
                const std::string& image_names_path = "");

int FilterImage(const std::string& input_path,
                const std::string& output_path,
                double min_focal_length_ratio = 0.1,
                double max_focal_length_ratio = 10.0,
                double max_extra_param = 100.0,
                std::size_t min_num_observations = 10);

int RectifyImage(const std::string& image_path,
                 const std::string& input_path,
                 const std::string& output_path,
                 const std::string& stereo_pairs_list,
                 double blank_pixels = 0.0,
                 double min_scale = 0.2,
                 double max_scale = 2.0,
                 int max_image_size = -1);

int RegisterImage(
        const std::string& database_path,
        const std::string& input_path,
        const std::string& output_path,
        const colmap::IncrementalMapperOptions& incremental_mapper_options =
                colmap::IncrementalMapperOptions());

int UndistortImage(const std::string& image_path,
                   const std::string& input_path,
                   const std::string& output_path,
                   const std::string& image_list_path = "",
                   const std::string& output_type = "COLMAP",
                   const std::string& copy_policy = "copy",
                   int num_patch_match_src_images = 20,
                   double blank_pixels = 0.0,
                   double min_scale = 0.2,
                   double max_scale = 2.0,
                   int max_image_size = -1,
                   double roi_min_x = 0.0,
                   double roi_min_y = 0.0,
                   double roi_max_x = 1.0,
                   double roi_max_y = 1.0);

int UndistortImageStandalone(const std::string& image_path,
                             const std::string& input_file,
                             const std::string& output_path,
                             double blank_pixels = 0.0,
                             double min_scale = 0.2,
                             double max_scale = 2.0,
                             int max_image_size = -1,
                             double roi_min_x = 0.0,
                             double roi_min_y = 0.0,
                             double roi_max_x = 1.0,
                             double roi_max_y = 1.0);

}  // namespace cloudViewer
