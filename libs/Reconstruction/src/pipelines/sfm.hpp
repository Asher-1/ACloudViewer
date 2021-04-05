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

#include "exe/sfm.h"
#include "option_utils.hpp"

namespace cloudViewer {

int RunAutomaticReconstructor(const std::string& workspace_path,
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
                              const std::string& gpu_index = "-1"
                              ) {
#ifdef CUDA_ENABLED
    dense = true;
#else
    dense = false;
#endif

  OptionsParser parser;
  parser.registerOption("workspace_path", &workspace_path);
  parser.registerOption("image_path", &image_path);
  parser.registerOption("mask_path", &mask_path);
  parser.registerOption("vocab_tree_path", &vocab_tree_path);
  // supported {individual, video, internet}
  parser.registerOption("data_type", &data_type);
  // supported {low, medium, high, extreme}
  parser.registerOption("quality", &quality);
  parser.registerOption("mesher", &mesher);
  parser.registerOption("camera_model", &camera_model);
  parser.registerOption("single_camera", &single_camera);
  parser.registerOption("sparse", &sparse);
  parser.registerOption("dense", &dense);
  parser.registerOption("num_threads", &num_threads);
  parser.registerOption("use_gpu", &use_gpu);
  parser.registerOption("gpu_index", &gpu_index);
  if (!parser.parseOptions())
    return EXIT_FAILURE;

  return colmap::RunAutomaticReconstructor(parser.getArgc(), parser.getArgv());
}

int RunBundleAdjuster(const std::string& input_path,
                      const std::string& output_path) {

//  colmap::OptionManager options;
//  options.AddBundleAdjustmentOptions();

  OptionsParser parser;
  parser.registerOption("input_path", &input_path);
  parser.registerOption("output_path", &output_path);
  if (!parser.parseOptions())
    return EXIT_FAILURE;

  return colmap::RunBundleAdjuster(parser.getArgc(), parser.getArgv());

}

int RunColorExtractor(const std::string& image_path,
                      const std::string& input_path,
                      const std::string& output_path) {

  OptionsParser parser;
  parser.registerOption("image_path", &image_path);
  parser.registerOption("input_path", &input_path);
  parser.registerOption("output_path", &output_path);
  if (!parser.parseOptions())
    return EXIT_FAILURE;

  return colmap::RunColorExtractor(parser.getArgc(), parser.getArgv());
}

int RunMapper(const std::string& database_path,
              const std::string& image_path,
              const std::string& input_path,
              const std::string& output_path,
              const std::string& image_list_path = "") {
//  colmap::OptionManager options;
//  options.AddMapperOptions();
  OptionsParser parser;
  parser.registerOption("database_path", &database_path);
  parser.registerOption("image_path", &image_path);
  parser.registerOption("input_path", &input_path);
  parser.registerOption("output_path", &output_path);
  parser.registerOption("image_list_path", &image_list_path);
  if (!parser.parseOptions())
    return EXIT_FAILURE;

  return colmap::RunMapper(parser.getArgc(), parser.getArgv());
}

int RunHierarchicalMapper(const std::string& database_path,
                          const std::string& image_path,
                          const std::string& output_path,
                          int num_workers = -1,
                          int image_overlap = 50,
                          int leaf_max_num_images = 500) {
  //  colmap::OptionManager options;
  //  options.AddMapperOptions();
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("num_workers", &num_workers);
    parser.registerOption("image_overlap", &image_overlap);
    parser.registerOption("leaf_max_num_images", &leaf_max_num_images);
    if (!parser.parseOptions())
      return EXIT_FAILURE;

    return colmap::RunHierarchicalMapper(parser.getArgc(), parser.getArgv());
}

int RunPointFiltering(const std::string& input_path,
                      const std::string& output_path,
                      std::size_t min_track_len = 2,
                      double max_reproj_error = 4.0,
                      double min_tri_angle = 1.5) {
  OptionsParser parser;
  parser.registerOption("input_path", &input_path);
  parser.registerOption("output_path", &output_path);
  parser.registerOption("min_track_len", &min_track_len);
  parser.registerOption("max_reproj_error", &max_reproj_error);
  parser.registerOption("min_tri_angle", &min_tri_angle);
  if (!parser.parseOptions())
    return EXIT_FAILURE;

  return colmap::RunPointFiltering(parser.getArgc(), parser.getArgv());
}

int RunPointTriangulator(const std::string& database_path,
                         const std::string& image_path,
                         const std::string& input_path,
                         const std::string& output_path,
                         bool clear_points = false) {
  //  colmap::OptionManager options;
  //  options.AddMapperOptions();
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    // Whether to clear all existing points and observations
    parser.registerOption("clear_points", &clear_points);
    if (!parser.parseOptions())
      return EXIT_FAILURE;

    return colmap::RunPointTriangulator(parser.getArgc(), parser.getArgv());

}

int RunRigBundleAdjuster(const std::string& input_path,
                         const std::string& output_path,
                         const std::string& rig_config_path,
                         bool estimate_rig_relative_poses = true,
                         bool refine_relative_poses = true) {
  //  colmap::OptionManager options;
  //  options.AddBundleAdjustmentOptions();
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("rig_config_path", &rig_config_path);
    // Whether to optimize the relative poses of the camera rigs.
    parser.registerOption("estimate_rig_relative_poses", &estimate_rig_relative_poses);
    parser.registerOption("RigBundleAdjustment.refine_relative_poses", &refine_relative_poses);
    if (!parser.parseOptions())
      return EXIT_FAILURE;

    return colmap::RunRigBundleAdjuster(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer
