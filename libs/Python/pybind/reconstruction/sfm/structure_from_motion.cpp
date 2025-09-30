// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/sfm/structure_from_motion.h"

#include "pipelines/sfm.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/reconstruction_options.h"

namespace cloudViewer {
namespace reconstruction {
namespace sfm {

// Reconstruction structure from motion functions have similar arguments,
// sharing arg docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"database_path",
                 "Path to database in which to store the extracted data"},
                {"workspace_path",
                 "The path to the workspace folder in which all results are "
                 "stored."},
                {"image_path",
                 "The path to the image folder which are used as input."},
                {"mask_path",
                 "The path to the mask folder which are used as input."},
                {"vocab_tree_path",
                 "The path to the vocabulary tree for feature matching."},
                {"data_type",
                 "Supported data types are {individual, video, internet}."},
                {"quality",
                 "Supported quality types are {low, medium, high, extreme}."},
                {"camera_model", "Which camera model to use for images."},
                {"single_camera", "Whether to use shared intrinsics or not."},
                {"sparse", "Whether to perform sparse mapping."},
                {"dense", "Whether to perform dense mapping."},
                {"mesher",
                 "Supported meshing algorithm types are {poisson, delaunay}."},
                {"num_threads", "The number of threads to use in all stages."},
                {"use_gpu",
                 "Whether to use the GPU in feature extraction and matching."},
                {"gpu_index",
                 "Index of the GPU used for GPU stages. For multi-GPU "
                 "computation, you should separate multiple GPU indices by "
                 "comma, e.g., ``0,1,2,3``. By default, all GPUs will be used "
                 "in all stages."},
                {"input_path",
                 "The input path containing cameras.bin/txt, images.bin/txt "
                 "and points3D.bin/txt."},
                {"output_path",
                 "The output path containing target cameras.bin/txt, "
                 "images.bin/txt and points3D.bin/txt."},
                {"image_list_path",
                 "A text file path containing image file path."},
                {"num_workers",
                 "The number of workers used to reconstruct clusters in "
                 "parallel."},
                {"image_overlap",
                 "The number of overlapping images between child clusters."},
                {"leaf_max_num_images",
                 "The maximum number of images in a leaf node cluster, "
                 "otherwise the cluster is further partitioned using the given "
                 "branching factor. "
                 "Note that a cluster leaf node will have at most "
                 "`leaf_max_num_images + overlap` images to satisfy the "
                 "overlap constraint."},
                {"min_track_len", "The minimum track length."},
                {"max_reproj_error", "The maximum re-projection error."},
                {"min_tri_angle", "The minimum tri angle."},
                {"clear_points",
                 "Whether to clear all existing points and observations."},
                {"rig_config_path", "The rig config path."},
                {"estimate_rig_relative_poses",
                 "Whether to estimate rig relative poses."},
                {"refine_relative_poses",
                 "Whether to optimize the relative poses of the camera "
                 "rigs."}};

void pybind_sfm_methods(py::module &m) {
    m.def("auto_reconstruction", &AutomaticReconstruct,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the automatic reconstruction", "workspace_path"_a,
          "image_path"_a, "mask_path"_a = "", "vocab_tree_path"_a = "",
          "data_type"_a = "individual", "quality"_a = "high",
          "mesher"_a = "poisson", "camera_model"_a = "SIMPLE_RADIAL",
          "single_camera"_a = false, "sparse"_a = true, "dense"_a = true,
          "num_threads"_a = -1, "use_gpu"_a = true, "gpu_index"_a = "-1");
    docstring::FunctionDocInject(m, "auto_reconstruction",
                                 map_shared_argument_docstrings);

    m.def("bundle_adjustment", &BundleAdjustment,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the bundle adjustment", "input_path"_a, "output_path"_a,
          "bundle_adjustment_options"_a = colmap::BundleAdjustmentOptions());
    docstring::FunctionDocInject(m, "bundle_adjustment",
                                 map_shared_argument_docstrings);

    m.def("extract_color", &ExtractColor,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the extraction of images color", "image_path"_a,
          "input_path"_a, "output_path"_a);
    docstring::FunctionDocInject(m, "extract_color",
                                 map_shared_argument_docstrings);

    m.def("normal_mapper", &NormalMapper,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the normal mapper", "database_path"_a, "image_path"_a,
          "input_path"_a, "output_path"_a, "image_list_path"_a = "",
          "incremental_mapper_options"_a = colmap::IncrementalMapperOptions());
    docstring::FunctionDocInject(m, "normal_mapper",
                                 map_shared_argument_docstrings);

    m.def("hierarchical_mapper", &HierarchicalMapper,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the hierarchical mapper", "database_path"_a,
          "image_path"_a, "output_path"_a, "num_workers"_a = -1,
          "image_overlap"_a = 50, "leaf_max_num_images"_a = 500,
          "incremental_mapper_options"_a = colmap::IncrementalMapperOptions());
    docstring::FunctionDocInject(m, "hierarchical_mapper",
                                 map_shared_argument_docstrings);

    m.def("filter_points", &FilterPoints,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the filtering of points", "input_path"_a,
          "output_path"_a, "min_track_len"_a = 2, "max_reproj_error"_a = 4.0,
          "min_tri_angle"_a = 1.5);
    docstring::FunctionDocInject(m, "filter_points",
                                 map_shared_argument_docstrings);

    m.def("triangulate_points", &TriangulatePoints,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the triangulation of points", "database_path"_a,
          "image_path"_a, "input_path"_a, "output_path"_a,
          "clear_points"_a = false,
          "incremental_mapper_options"_a = colmap::IncrementalMapperOptions());
    docstring::FunctionDocInject(m, "triangulate_points",
                                 map_shared_argument_docstrings);

    m.def("rig_bundle_adjustment", &RigBundleAdjust,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the rig bundle adjustment", "input_path"_a,
          "output_path"_a, "rig_config_path"_a,
          "estimate_rig_relative_poses"_a = true,
          "refine_relative_poses"_a = true,
          "bundle_adjustment_options"_a = colmap::BundleAdjustmentOptions());
    docstring::FunctionDocInject(m, "rig_bundle_adjustment",
                                 map_shared_argument_docstrings);
}

void pybind_structure_from_motion(py::module &m) {
    py::module m_submodule =
            m.def_submodule("sfm", "Reconstruction structure from motion.");
    pybind_sfm_methods(m_submodule);
}

}  // namespace sfm
}  // namespace reconstruction
}  // namespace cloudViewer
