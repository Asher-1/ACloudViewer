// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include <string>

namespace cloudViewer {

int AlignModel(const std::string& input_path,
               const std::string& output_path,
               const std::string& database_path = "",
               const std::string& ref_images_path = "",
               const std::string& transform_path = "",
               const std::string& alignment_type = "plane",
               double max_error = 0.0,
               int min_common_images = 3,
               bool robust_alignment = true,
               bool estimate_scale = true);

int AnalyzeModel(const std::string& input_path);

int CompareModel(const std::string& input_path1,
                 const std::string& input_path2,
                 const std::string& output_path = "",
                 double min_inlier_observations = 0.3,
                 double max_reproj_error = 8.0);

int ConvertModel(const std::string& input_path,
                 const std::string& output_path,
                 const std::string& output_type,
                 bool skip_distortion = false);

int CropModel(const std::string& input_path,
              const std::string& output_path,
              const std::string& boundary,
              const std::string& gps_transform_path = "");

int MergeModel(const std::string& input_path1,
               const std::string& input_path2,
               const std::string& output_path,
               double max_reproj_error = 64.0);

int AlignModelOrientation(const std::string& image_path,
                          const std::string& input_path,
                          const std::string& output_path,
                          std::string method = "MANHATTAN-WORLD",
                          int max_image_size = 1024);

int SplitModel(const std::string& input_path,
               const std::string& output_path,
               const std::string& split_type,
               const std::string& split_params,
               const std::string& gps_transform_path = "",
               std::size_t min_reg_images = 10,
               std::size_t min_num_points = 100,
               double overlap_ratio = 0.0,
               double min_area_ratio = 0.0,
               int num_threads = -1);

int TransformModel(const std::string& input_path,
                   const std::string& output_path,
                   const std::string& transform_path,
                   bool is_inverse = false);

}  // namespace cloudViewer
