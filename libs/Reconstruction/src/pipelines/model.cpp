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

#include "exe/model.h"

#include "pipelines/model.h"
#include "pipelines/option_utils.h"

namespace cloudViewer {

int AlignModel(const std::string& input_path,
               const std::string& output_path,
               const std::string& database_path /*= ""*/,
               const std::string& ref_images_path /*= ""*/,
               const std::string& transform_path /*= ""*/,
               const std::string& alignment_type /*= "plane"*/,
               double max_error /* = 0.0*/,
               int min_common_images /*= 3*/,
               bool robust_alignment /*= true*/,
               bool estimate_scale /*= true*/) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("ref_images_path", &ref_images_path);
    parser.registerOption("transform_path", &transform_path);
    // supported {plane, ecef, enu, enu-unscaled, custom}
    parser.registerOption("alignment_type", &alignment_type);
    parser.registerOption("robust_alignment_max_error", &max_error);
    parser.registerOption("min_common_images", &min_common_images);
    parser.registerOption("robust_alignment", &robust_alignment);
    parser.registerOption("estimate_scale", &estimate_scale);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunModelAligner(parser.getArgc(), parser.getArgv());
}

int AnalyzeModel(const std::string& input_path) {
    OptionsParser parser;
    parser.registerOption("path", &input_path); // equal to input_path
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunModelAnalyzer(parser.getArgc(), parser.getArgv());
}

int CompareModel(const std::string& input_path1,
                 const std::string& input_path2,
                 const std::string& output_path /*= ""*/,
                 double min_inlier_observations /*= 0.3*/,
                 double max_reproj_error /*= 8.0*/) {
    OptionsParser parser;
    parser.registerOption("input_path1", &input_path1);
    parser.registerOption("input_path2", &input_path2);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("min_inlier_observations", &min_inlier_observations);
    parser.registerOption("max_reproj_error", &max_reproj_error);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunModelComparer(parser.getArgc(), parser.getArgv());
}

int ConvertModel(const std::string& input_path,
                 const std::string& output_path,
                 const std::string& output_type,
                 bool skip_distortion /*= false*/) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    // supported type {BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM}
    parser.registerOption("output_type", &output_type);
    parser.registerOption("skip_distortion", &skip_distortion);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunModelConverter(parser.getArgc(), parser.getArgv());
}

int CropModel(const std::string& input_path,
              const std::string& output_path,
              const std::string& boundary,
              const std::string& gps_transform_path /* = ""*/) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("boundary", &boundary);
    parser.registerOption("gps_transform_path", &gps_transform_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunModelCropper(parser.getArgc(), parser.getArgv());
}

int MergeModel(const std::string& input_path1,
               const std::string& input_path2,
               const std::string& output_path,
               double max_reproj_error /* = 64.0*/) {
    OptionsParser parser;
    parser.registerOption("input_path1", &input_path1);
    parser.registerOption("input_path2", &input_path2);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("max_reproj_error", &max_reproj_error);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunModelMerger(parser.getArgc(), parser.getArgv());
}

int AlignModelOrientation(const std::string& image_path,
                          const std::string& input_path,
                          const std::string& output_path,
                          std::string method /*= "MANHATTAN-WORLD"*/,
                          int max_image_size /*= 1024*/) {
    OptionsParser parser;
    parser.registerOption("image_path", &image_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    // supported {MANHATTAN-WORLD, IMAGE-ORIENTATION}
    parser.registerOption("method", &method);
    parser.registerOption("max_image_size", &max_image_size);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunModelOrientationAligner(parser.getArgc(),
                                              parser.getArgv());
}

int SplitModel(const std::string& input_path,
               const std::string& output_path,
               const std::string& split_type,
               const std::string& split_params,
               const std::string& gps_transform_path /*= ""*/,
               std::size_t min_reg_images /*= 10*/,
               std::size_t min_num_points /*= 100*/,
               double overlap_ratio /*= 0.0*/,
               double min_area_ratio /*= 0.0*/,
               int num_threads /*= -1*/) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    // supported {tiles, extent, parts}
    parser.registerOption("split_type", &split_type);
    parser.registerOption("split_params", &split_params);
    parser.registerOption("gps_transform_path", &gps_transform_path);
    parser.registerOption("min_reg_images", &min_reg_images);
    parser.registerOption("min_num_points", &min_num_points);
    parser.registerOption("overlap_ratio", &overlap_ratio);
    parser.registerOption("min_area_ratio", &min_area_ratio);
    parser.registerOption("num_threads", &num_threads);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunModelSplitter(parser.getArgc(), parser.getArgv());
}

int TransformModel(const std::string& input_path,
                   const std::string& output_path,
                   const std::string& transform_path,
                   bool is_inverse /*= false*/) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("transform_path", &transform_path);
    parser.registerOption("is_inverse", &is_inverse);

    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunModelTransformer(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer
