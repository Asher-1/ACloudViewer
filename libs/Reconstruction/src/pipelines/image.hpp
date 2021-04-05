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

#include "exe/image.h"
#include "option_utils.hpp"

namespace cloudViewer {

int RunImageDeleter(const std::string& input_path,
                    const std::string& output_path,
                    const std::string& image_ids_path,
                    const std::string& image_names_path) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    // Path to text file containing one image_id to delete per line
    parser.registerOption("image_ids_path", &image_ids_path);
    // Path to text file containing one image name to delete per line
    parser.registerOption("image_names_path", &image_names_path);
    if (!parser.parseOptions())
        return EXIT_FAILURE;

    return colmap::RunImageDeleter(parser.getArgc(), parser.getArgv());
}

int RunImageFilterer(const std::string& input_path,
                     const std::string& output_path,
                     double min_focal_length_ratio = 0.1,
                     double max_focal_length_ratio = 10.0,
                     double max_extra_param = 100.0,
                     std::size_t min_num_observations = 10) {

  OptionsParser parser;
  parser.registerOption("input_path", &input_path);
  parser.registerOption("output_path", &output_path);
  parser.registerOption("min_focal_length_ratio", &min_focal_length_ratio);
  parser.registerOption("max_focal_length_ratio", &max_focal_length_ratio);
  parser.registerOption("max_extra_param", &max_extra_param);
  parser.registerOption("min_num_observations", &min_num_observations);
  if (!parser.parseOptions())
      return EXIT_FAILURE;

  return colmap::RunImageFilterer(parser.getArgc(), parser.getArgv());

}

int RunImageRectifier(const std::string& image_path,
                      const std::string& input_path,
                      const std::string& output_path,
                      const std::string& stereo_pairs_list,
                      double blank_pixels = 0.0,
                      double min_scale = 0.2,
                      double max_scale = 2.0,
                      int max_image_size = -1) {

  OptionsParser parser;
  parser.registerOption("image_path", &image_path);
  parser.registerOption("input_path", &input_path);
  parser.registerOption("output_path", &output_path);
  parser.registerOption("stereo_pairs_list", &stereo_pairs_list);
  parser.registerOption("blank_pixels", &blank_pixels);
  parser.registerOption("min_scale", &min_scale);
  parser.registerOption("max_scale", &max_scale);
  parser.registerOption("max_image_size", &max_image_size);
  if (!parser.parseOptions())
      return EXIT_FAILURE;

  return colmap::RunImageRectifier(parser.getArgc(), parser.getArgv());
}

int RunImageRegistrator(const std::string& database_path,
                        const std::string& input_path,
                        const std::string& output_path) {
//  colmap::OptionManager options;
//  options.AddMapperOptions();

  OptionsParser parser;
  parser.registerOption("database_path", &database_path);
  parser.registerOption("input_path", &input_path);
  parser.registerOption("output_path", &output_path);
  if (!parser.parseOptions())
      return EXIT_FAILURE;

  return colmap::RunImageRegistrator(parser.getArgc(), parser.getArgv());

}

int RunImageUndistorter(const std::string& image_path,
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
                        double roi_max_y = 1.0) {
  OptionsParser parser;
  parser.registerOption("image_path", &image_path);
  parser.registerOption("input_path", &input_path);
  parser.registerOption("output_path", &output_path);
  parser.registerOption("image_list_path", &image_list_path);
  // supported {COLMAP, PMVS, CMP-MVS}
  parser.registerOption("output_type", &output_type);
  // supported {copy, soft-link, hard-link}
  parser.registerOption("copy_policy", &copy_policy);
  parser.registerOption("num_patch_match_src_images", &num_patch_match_src_images);
  parser.registerOption("blank_pixels", &blank_pixels);
  parser.registerOption("min_scale", &min_scale);
  parser.registerOption("max_scale", &max_scale);
  parser.registerOption("max_image_size", &max_image_size);
  parser.registerOption("roi_min_x", &roi_min_x);
  parser.registerOption("roi_min_y", &roi_min_y);
  parser.registerOption("roi_max_x", &roi_max_x);
  parser.registerOption("roi_max_y", &roi_max_y);

  if (!parser.parseOptions())
      return EXIT_FAILURE;

  return colmap::RunImageUndistorter(parser.getArgc(), parser.getArgv());

}

int RunImageUndistorterStandalone(const std::string& image_path,
                                  const std::string& input_file,
                                  const std::string& output_path,
                                  double blank_pixels = 0.0,
                                  double min_scale = 0.2,
                                  double max_scale = 2.0,
                                  int max_image_size = -1,
                                  double roi_min_x = 0.0,
                                  double roi_min_y = 0.0,
                                  double roi_max_x = 1.0,
                                  double roi_max_y = 1.0) {

  OptionsParser parser;
  parser.registerOption("image_path", &image_path);
  parser.registerOption("input_file", &input_file);
  parser.registerOption("output_path", &output_path);
  parser.registerOption("blank_pixels", &blank_pixels);
  parser.registerOption("min_scale", &min_scale);
  parser.registerOption("max_scale", &max_scale);
  parser.registerOption("max_image_size", &max_image_size);
  parser.registerOption("roi_min_x", &roi_min_x);
  parser.registerOption("roi_min_y", &roi_min_y);
  parser.registerOption("roi_max_x", &roi_max_x);
  parser.registerOption("roi_max_y", &roi_max_y);

  if (!parser.parseOptions())
      return EXIT_FAILURE;

  return colmap::RunImageUndistorter(parser.getArgc(), parser.getArgv());

}

}  // namespace cloudViewer
