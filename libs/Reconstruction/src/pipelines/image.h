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

#include <string>

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

int RegisterImage(const std::string& database_path,
                  const std::string& input_path,
                  const std::string& output_path);

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
