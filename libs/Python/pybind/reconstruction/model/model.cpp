// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/model/model.h"

#include "pipelines/model.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/reconstruction_options.h"

namespace cloudViewer {
namespace reconstruction {
namespace model {

// Reconstruction model functions have similar arguments, sharing arg
// docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"input_path",
                 "The input path containing cameras.bin/txt, images.bin/txt "
                 "and points3D.bin/txt."},
                {"output_path",
                 "The output path containing target cameras.bin/txt, "
                 "images.bin/txt and points3D.bin/txt."},
                {"database_path",
                 "Path to database in which to store the extracted data."},
                {"ref_images_path",
                 "Path to text file containing reference images per line."},
                {"transform_path",
                 "The alignment transformation matrix saving path."},
                {"alignment_type",
                 "Alignment type: supported values are {plane, ecef, enu, "
                 "enu-unscaled, custom}."},
                {"max_error",
                 "Maximum error for a sample to be considered as an inlier. "
                 "Note that the residual of an estimator corresponds to a "
                 "squared error."},
                {"min_common_images", "Minimum common images."},
                {"robust_alignment", "Whether align robustly or not."},
                {"estimate_scale", "Whether estimate scale or not."},
                {"min_inlier_observations",
                 "The threshold determines how many observations in a common "
                 "image must reproject within the given threshold.."},
                {"max_reproj_error", "The Maximum re-projection error."},
                {"output_type",
                 "The supported output type values are {BIN, TXT, NVM, "
                 "Bundler, VRML, PLY, R3D, CAM}."},
                {"skip_distortion",
                 "Whether skip distortion or no. When skip_distortion == true "
                 "it supports all camera models with the caveat that it's "
                 "using the mean focal length which will be inaccurate for "
                 "camera models with two focal lengths and distortion."},
                {"boundary", "The cropping boundary coordinates."},
                {"gps_transform_path",
                 "The gps transformation parameters file path."},
                {"method",
                 "The supported Model Orientation Alignment values are "
                 "{MANHATTAN-WORLD, IMAGE-ORIENTATION}."},
                {"max_image_size",
                 "The maximum image size for line detection."},
                {"split_type",
                 "The supported split type values are {tiles, extent, parts}."},
                {"split_params", "The split parameters file path."},
                {"num_threads", "The number of cpu thread."},
                {"min_reg_images", "The minimum number of reg images."},
                {"min_num_points", "The minimum number of points."},
                {"overlap_ratio", "The overlapped ratio."},
                {"min_area_ratio", "The minimum area ratio."},
                {"is_inverse", "Whether inverse or not."}};

void pybind_model_methods(py::module &m) {
    m.def("align_model", &AlignModel, py::call_guard<py::gil_scoped_release>(),
          "Function for the alignment of model", "input_path"_a,
          "output_path"_a, "database_path"_a = "", "ref_images_path"_a = "",
          "transform_path"_a = "", "alignment_type"_a = "plane",
          "max_error"_a = 0.0, "min_common_images"_a = 3,
          "robust_alignment"_a = true, "estimate_scale"_a = true);
    docstring::FunctionDocInject(m, "align_model",
                                 map_shared_argument_docstrings);

    m.def("analyze_model", &AnalyzeModel,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the analyse of model", "input_path"_a);
    docstring::FunctionDocInject(m, "analyze_model",
                                 map_shared_argument_docstrings);

    m.def("compare_model", &CompareModel,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the comparison of model", "input_path1"_a,
          "input_path2"_a, "output_path"_a = "",
          "min_inlier_observations"_a = 0.3, "max_reproj_error"_a = 8.0);
    docstring::FunctionDocInject(m, "compare_model",
                                 map_shared_argument_docstrings);

    m.def("convert_model", &ConvertModel,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the convertion of model", "input_path"_a,
          "output_path"_a, "output_type"_a, "skip_distortion"_a = false);
    docstring::FunctionDocInject(m, "convert_model",
                                 map_shared_argument_docstrings);

    m.def("crop_model", &CropModel, py::call_guard<py::gil_scoped_release>(),
          "Function for the cropping of model", "input_path"_a, "output_path"_a,
          "boundary"_a, "gps_transform_path"_a = "");
    docstring::FunctionDocInject(m, "crop_model",
                                 map_shared_argument_docstrings);

    m.def("merge_model", &MergeModel, py::call_guard<py::gil_scoped_release>(),
          "Function for the merging of model", "input_path1"_a, "input_path2"_a,
          "output_path"_a, "max_reproj_error"_a = 64.0);
    docstring::FunctionDocInject(m, "merge_model",
                                 map_shared_argument_docstrings);

    m.def("align_model_orientation", &AlignModelOrientation,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the orientation alignment of model", "image_path"_a,
          "input_path"_a, "output_path"_a, "method"_a = "MANHATTAN-WORLD",
          "max_image_size"_a = 1024);
    docstring::FunctionDocInject(m, "align_model_orientation",
                                 map_shared_argument_docstrings);

    m.def("split_model", &SplitModel, py::call_guard<py::gil_scoped_release>(),
          "Function for the splitting of model", "input_path"_a,
          "output_path"_a, "split_type"_a, "split_params"_a,
          "gps_transform_path"_a = "", "min_reg_images"_a = 10,
          "min_num_points"_a = 100, "overlap_ratio"_a = 0.0,
          "min_area_ratio"_a = 0.0, "num_threads"_a = -1);
    docstring::FunctionDocInject(m, "split_model",
                                 map_shared_argument_docstrings);

    m.def("transform_model", &TransformModel, py::call_guard<py::gil_scoped_release>(),
          "Function for the transformation of model", "input_path"_a,
          "output_path"_a, "transform_path"_a, "is_inverse"_a = false);
    docstring::FunctionDocInject(m, "transform_model",
                                 map_shared_argument_docstrings);
}

void pybind_model(py::module &m) {
    py::module m_submodule = m.def_submodule("model", "Reconstruction model.");
    pybind_model_methods(m_submodule);
}

}  // namespace model
}  // namespace reconstruction
}  // namespace cloudViewer
