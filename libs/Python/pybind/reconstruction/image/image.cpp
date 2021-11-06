// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/image/image.h"

#include "pipelines/image.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/reconstruction_options.h"

namespace cloudViewer {
namespace reconstruction {
namespace image {

// Reconstruction image functions have similar arguments, sharing arg
// docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"input_path",
                 "The input path containing cameras.bin/txt, images.bin/txt "
                 "and points3D.bin/txt."},
                {"output_path",
                 "The output path containing target cameras.bin/txt, "
                 "images.bin/txt and points3D.bin/txt."},
                {"min_focal_length_ratio",
                 "Minimum ratio of focal length over minimum sensor "
                 "dimension."},
                {"max_focal_length_ratio",
                 "Maximum ratio of focal length over maximum sensor "
                 "dimension."},
                {"max_extra_param",
                 "Maximum magnitude of each extra parameter."},
                {"min_num_observations", "The maximum number of observations."},
                {"image_ids_path",
                 "Path to text file containing one image_id to delete per "
                 "line."},
                {"image_names_path",
                 "Path to text file containing one image name to delete per "
                 "line."},
                {"stereo_pairs_list",
                 "A text file path containing "
                 "stereo image pair names from. The text file is expected to "
                 "have one image pair per line, e.g.:\n"
                 "image_name1.jpg image_name2.jpg\n"
                 "image_name3.jpg image_name4.jpg\n"
                 "image_name5.jpg image_name6.jpg"},
                {"blank_pixels",
                 "The amount of blank pixels in the undistorted image in the "
                 "range [0, 1]."},
                {"min_scale",
                 "Minimum scale change of camera used to satisfy the blank "
                 "pixel constraint."},
                {"max_scale",
                 "Maximum scale change of camera used to satisfy the blank "
                 "pixel constraint."},
                {"max_image_size",
                 "Maximum image size in terms of width or height of the "
                 "undistorted camera."},
                {"output_type",
                 "Output file format: supported values are {'COLMAP', 'PMVS', "
                 "'CMP-MVS'}."},
                {"image_list_path",
                 "A text file path containing image file path."},
                {"copy_policy",
                 "Supported copy policy are {copy, soft-link, hard-link}."},
                {"num_patch_match_src_images",
                 "The number of patch match source images."},
                {"roi_min_x",
                 "The value in the range [0, 1] that define the ROI (region "
                 "of interest) minimum x in original image."},
                {"roi_min_y",
                 "The value in the range [0, 1] that define the ROI (region of "
                 "interest) minimum y in original image."},
                {"roi_max_x",
                 "The value in the range [0, 1] that define the ROI (region "
                 "of interest) maximum x in original image."},
                {"roi_max_y",
                 "The value in the range [0, 1] that define the ROI (region of "
                 "interest) maximum y in original image."}};

void pybind_image_methods(py::module &m) {
    m.def("delete_image", &DeleteImage,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the deletion of images", "input_path"_a,
          "output_path"_a, "image_ids_path"_a = "", "image_names_path"_a = "");
    docstring::FunctionDocInject(m, "delete_image",
                                 map_shared_argument_docstrings);

    m.def("filter_image", &FilterImage,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the filtering of images", "input_path"_a,
          "output_path"_a, "min_focal_length_ratio"_a = 0.1,
          "max_focal_length_ratio"_a = 10.0, "max_extra_param"_a = 100.0,
          "min_num_observations"_a = 10);
    docstring::FunctionDocInject(m, "filter_image",
                                 map_shared_argument_docstrings);

    m.def("rectify_image", &RectifyImage,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the rectification of images", "image_path"_a, "input_path"_a,
          "output_path"_a, "stereo_pairs_list"_a, "blank_pixels"_a = 0.0,
          "min_scale"_a = 0.2, "max_scale"_a = 2.0, "max_image_size"_a = -1);
    docstring::FunctionDocInject(m, "rectify_image",
                                 map_shared_argument_docstrings);

    m.def("register_image", &RegisterImage,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the registeration of images", "database_path"_a,
          "input_path"_a, "output_path"_a,
          "incremental_mapper_options"_a = colmap::IncrementalMapperOptions());
    docstring::FunctionDocInject(m, "register_image",
                                 map_shared_argument_docstrings);

    m.def("undistort_image", &UndistortImage,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the undistortion of images", "image_path"_a, "input_path"_a,
          "output_path"_a, "image_list_path"_a = "", "output_type"_a = "COLMAP",
          "copy_policy"_a = "copy", "num_patch_match_src_images"_a = 20,
          "blank_pixels"_a = 0.0, "min_scale"_a = 0.2, "max_scale"_a = 2.0,
          "max_image_size"_a = -1, "roi_min_x"_a = 0.0, "roi_min_y"_a = 0.0,
          "roi_max_x"_a = 1.0, "roi_max_y"_a = 1.0);
    docstring::FunctionDocInject(m, "undistort_image",
                                 map_shared_argument_docstrings);

    m.def("undistort_image_standalone", &UndistortImageStandalone,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the standalone undistortion of images", "image_path"_a, "input_path"_a,
          "output_path"_a, "blank_pixels"_a = 0.0, "min_scale"_a = 0.2,
          "max_scale"_a = 2.0, "max_image_size"_a = -1, "roi_min_x"_a = 0.0,
          "roi_min_y"_a = 0.0, "roi_max_x"_a = 1.0, "roi_max_y"_a = 1.0);
    docstring::FunctionDocInject(m, "undistort_image_standalone",
                                 map_shared_argument_docstrings);
}

void pybind_image(py::module &m) {
    py::module m_submodule = m.def_submodule("image", "Reconstruction Images.");
    pybind_image_methods(m_submodule);
}

}  // namespace image
}  // namespace reconstruction
}  // namespace cloudViewer
