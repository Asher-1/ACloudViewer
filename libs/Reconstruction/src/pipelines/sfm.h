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

int AutomaticReconstruct(const std::string& workspace_path,
                         const std::string& image_path,
                         const std::string& mask_path = "",
                         const std::string& vocab_tree_path = "",
                         const std::string& data_type = "individual",
                         const std::string& quality = "high",
                         const std::string& mesher = "poisson",
                         const std::string& camera_model = "SIMPLE_RADIAL",
                         bool single_camera = false,
                         bool sparse = true,
                         bool dense = true,
                         int num_threads = -1,
                         bool use_gpu = true,
                         const std::string& gpu_index = "-1");

int BundleAdjust(const std::string& input_path, const std::string& output_path);

int ExtractColor(const std::string& image_path,
                 const std::string& output_path,
                 const std::string& input_path = "");

int NormalMapper(const std::string& database_path,
                 const std::string& image_path,
                 const std::string& output_path,
                 const std::string& input_path = "",
                 const std::string& image_list_path = "");

int HierarchicalMapper(const std::string& database_path,
                       const std::string& image_path,
                       const std::string& output_path,
                       int num_workers = -1,
                       int image_overlap = 50,
                       int leaf_max_num_images = 500);

int FilterPoints(const std::string& input_path,
                 const std::string& output_path,
                 std::size_t min_track_len = 2,
                 double max_reproj_error = 4.0,
                 double min_tri_angle = 1.5);

int TriangulatePoints(const std::string& database_path,
                      const std::string& image_path,
                      const std::string& input_path,
                      const std::string& output_path,
                      bool clear_points = false);

int RigBundleAdjust(const std::string& input_path,
                    const std::string& output_path,
                    const std::string& rig_config_path,
                    bool estimate_rig_relative_poses = true,
                    bool refine_relative_poses = true);

}  // namespace cloudViewer
