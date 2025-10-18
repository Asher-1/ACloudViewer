// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "exe/image.h"

#include "pipelines/image.h"
#include "pipelines/option_utils.h"

namespace cloudViewer {

int DeleteImage(const std::string& input_path,
                const std::string& output_path,
                const std::string& image_ids_path /*= ""*/,
                const std::string& image_names_path /*= ""*/) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    // Path to text file containing one image_id to delete per line
    parser.registerOption("image_ids_path", &image_ids_path);
    // Path to text file containing one image name to delete per line
    parser.registerOption("image_names_path", &image_names_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunImageDeleter(parser.getArgc(), parser.getArgv());
}

int FilterImage(const std::string& input_path,
                const std::string& output_path,
                double min_focal_length_ratio /*= 0.1*/,
                double max_focal_length_ratio /*= 10.0*/,
                double max_extra_param /*= 100.0*/,
                std::size_t min_num_observations /*= 10*/) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("min_focal_length_ratio", &min_focal_length_ratio);
    parser.registerOption("max_focal_length_ratio", &max_focal_length_ratio);
    parser.registerOption("max_extra_param", &max_extra_param);
    parser.registerOption("min_num_observations", &min_num_observations);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunImageFilterer(parser.getArgc(), parser.getArgv());
}

int RectifyImage(const std::string& image_path,
                 const std::string& input_path,
                 const std::string& output_path,
                 const std::string& stereo_pairs_list,
                 double blank_pixels /*= 0.0*/,
                 double min_scale /*= 0.2*/,
                 double max_scale /*= 2.0*/,
                 int max_image_size /*= -1*/) {
    OptionsParser parser;
    parser.registerOption("image_path", &image_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("stereo_pairs_list", &stereo_pairs_list);
    parser.registerOption("blank_pixels", &blank_pixels);
    parser.registerOption("min_scale", &min_scale);
    parser.registerOption("max_scale", &max_scale);
    parser.registerOption("max_image_size", &max_image_size);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunImageRectifier(parser.getArgc(), parser.getArgv());
}

int RegisterImage(
        const std::string& database_path,
        const std::string& input_path,
        const std::string& output_path,
        const colmap::IncrementalMapperOptions& incremental_mapper_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.addMapperOptions(incremental_mapper_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunImageRegistrator(parser.getArgc(), parser.getArgv());
}

int UndistortImage(const std::string& image_path,
                   const std::string& input_path,
                   const std::string& output_path,
                   const std::string& image_list_path /*= ""*/,
                   const std::string& output_type /*= "COLMAP"*/,
                   const std::string& copy_policy /*= "copy"*/,
                   int num_patch_match_src_images /* = 20*/,
                   double blank_pixels /*= 0.0*/,
                   double min_scale /*= 0.2*/,
                   double max_scale /*= 2.0*/,
                   int max_image_size /*= -1 */,
                   double roi_min_x /*= 0.0*/,
                   double roi_min_y /*= 0.0*/,
                   double roi_max_x /*= 1.0*/,
                   double roi_max_y /*= 1.0*/) {
    OptionsParser parser;
    parser.registerOption("image_path", &image_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("image_list_path", &image_list_path);
    // supported {COLMAP, PMVS, CMP-MVS}
    parser.registerOption("output_type", &output_type);
    // supported {copy, soft-link, hard-link}
    parser.registerOption("copy_policy", &copy_policy);
    parser.registerOption("num_patch_match_src_images",
                          &num_patch_match_src_images);
    parser.registerOption("blank_pixels", &blank_pixels);
    parser.registerOption("min_scale", &min_scale);
    parser.registerOption("max_scale", &max_scale);
    parser.registerOption("max_image_size", &max_image_size);
    parser.registerOption("roi_min_x", &roi_min_x);
    parser.registerOption("roi_min_y", &roi_min_y);
    parser.registerOption("roi_max_x", &roi_max_x);
    parser.registerOption("roi_max_y", &roi_max_y);

    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunImageUndistorter(parser.getArgc(), parser.getArgv());
}

int UndistortImageStandalone(const std::string& image_path,
                             const std::string& input_file,
                             const std::string& output_path,
                             double blank_pixels /*= 0.0*/,
                             double min_scale /*= 0.2*/,
                             double max_scale /*= 2.0*/,
                             int max_image_size /*= -1 */,
                             double roi_min_x /*= 0.0*/,
                             double roi_min_y /*= 0.0*/,
                             double roi_max_x /*= 1.0*/,
                             double roi_max_y /*= 1.0*/) {
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

    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunImageUndistorterStandalone(parser.getArgc(),
                                                 parser.getArgv());
}

}  // namespace cloudViewer
