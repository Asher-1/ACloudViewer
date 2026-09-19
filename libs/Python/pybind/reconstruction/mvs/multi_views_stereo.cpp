// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/mvs/multi_views_stereo.h"

#include <cstring>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "mvs/depth_map.h"
#include "mvs/model.h"
#include "mvs/normal_map.h"
#include "pipelines/mvs.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/reconstruction_options.h"

namespace cloudViewer {
namespace reconstruction {
namespace mvs {

// Reconstruction multiple views stereo functions have similar arguments,
// sharing arg docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"input_path",
                 "Path to either the dense workspace folder or the sparse "
                 "reconstruction."},
                {"output_path",
                 "The output path containing target cameras.bin/txt, "
                 "images.bin/txt and points3D.bin/txt."},
                {"input_type",
                 "Supported input type values are {dense, sparse}."},
                {"stereo_input_type",
                 "Supported stereo input type values are {photometric, "
                 "geometric}."},
                {"output_type",
                 "Supported output type values are {BIN, TXT, PLY}."},
                {"workspace_path",
                 "Path to the folder containing the undistorted images."},
                {"workspace_format",
                 "Supported workspace format values are {COLMAP, PMVS}."},
                {"pmvs_option_name", "The pmvs option name."},
                {"config_path", "The config path."},
                {"bbox_path", "The bounds file path."}};

void pybind_multi_views_stereo_methods(py::module& m) {
    m.def("mesh_delaunay", &MeshDelaunay,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the delaunay of mesh", "input_path"_a, "output_path"_a,
          "input_type"_a = "dense",
          "delaunay_meshing_options"_a = colmap::mvs::DelaunayMeshingOptions(),
          "mesh_post_processing_options"_a =
                  colmap::mvs::MeshPostProcessingOptions());
    docstring::FunctionDocInject(m, "mesh_delaunay",
                                 map_shared_argument_docstrings);

    m.def("stereo_patch_match", &StereoPatchMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the stereo path-match of mesh", "workspace_path"_a,
          "config_path"_a = "", "workspace_format"_a = "COLMAP",
          "pmvs_option_name"_a = "option-all",
          "patch_match_options"_a = colmap::mvs::PatchMatchOptions());
    docstring::FunctionDocInject(m, "stereo_patch_match",
                                 map_shared_argument_docstrings);

    m.def("poisson_mesh", &MeshPoisson,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the poisson of mesh", "input_path"_a, "output_path"_a,
          "poisson_meshing_options"_a = colmap::mvs::PoissonMeshingOptions(),
          "mesh_post_processing_options"_a =
                  colmap::mvs::MeshPostProcessingOptions());
    docstring::FunctionDocInject(m, "poisson_mesh",
                                 map_shared_argument_docstrings);

    m.def("stereo_fuse", &StereoFuse, py::call_guard<py::gil_scoped_release>(),
          "Function for the stereo path-match of mesh", "workspace_path"_a,
          "output_path"_a, "bbox_path"_a = "",
          "stereo_input_type"_a = "geometric", "output_type"_a = "PLY",
          "workspace_format"_a = "COLMAP", "pmvs_option_name"_a = "option-all",
          "stereo_fusion_options"_a = colmap::mvs::StereoFusionOptions());
    docstring::FunctionDocInject(m, "stereo_fuse",
                                 map_shared_argument_docstrings);
}

void pybind_multi_views_stereo(py::module& m) {
    py::module m_submodule =
            m.def_submodule("mvs", "Reconstruction multiple views stereo.");
    pybind_multi_views_stereo_methods(m_submodule);

    // Upstream pycolmap parity (src/pycolmap/mvs): the dense-reconstruction
    // value types. Bitmap comes from the sensor module; numpy views are
    // materialized copies (the engine stores row-major packed data).
    using colmap::mvs::DepthMap;
    using colmap::mvs::Mat;
    using colmap::mvs::Model;
    using colmap::mvs::NormalMap;

    py::class_<DepthMap> depth_map(m_submodule, "DepthMap",
                                   "A per-pixel depth map.");
    depth_map.def(py::init<>())
            .def(py::init([](size_t width, size_t height, float depth_min,
                             float depth_max) {
                     return DepthMap(width, height, depth_min, depth_max);
                 }),
                 "width"_a, "height"_a, "depth_min"_a, "depth_max"_a)
            .def_property_readonly("width", &DepthMap::GetWidth)
            .def_property_readonly("height", &DepthMap::GetHeight)
            .def_property_readonly("depth_min", &DepthMap::GetDepthMin)
            .def_property_readonly("depth_max", &DepthMap::GetDepthMax)
            .def(
                    "get",
                    [](const DepthMap& self, size_t row, size_t col) {
                        return self.Get(row, col);
                    },
                    "row"_a, "col"_a, "The depth at the given pixel.")
            .def(
                    "array",
                    [](const DepthMap& self) {
                        py::array_t<float> arr(
                                {static_cast<py::ssize_t>(self.GetHeight()),
                                 static_cast<py::ssize_t>(self.GetWidth())});
                        std::memcpy(arr.mutable_data(), self.GetPtr(),
                                    self.GetNumBytes());
                        return arr;
                    },
                    "A (height, width) float32 copy of the depth values.")
            .def(
                    "set_array",
                    [](DepthMap& self,
                       const py::array_t<float, py::array::c_style |
                                                        py::array::forcecast>&
                               arr) {
                        THROW_CHECK_EQ(arr.ndim(), 2);
                        THROW_CHECK_EQ(static_cast<size_t>(arr.shape(0)),
                                       self.GetHeight());
                        THROW_CHECK_EQ(static_cast<size_t>(arr.shape(1)),
                                       self.GetWidth());
                        std::memcpy(self.GetPtr(), arr.data(),
                                    self.GetNumBytes());
                    },
                    "array"_a,
                    "Copy a (height, width) float32 array into the map.")
            .def(
                    "read",
                    [](DepthMap& self, const std::string& path) {
                        self.Read(path);
                    },
                    "path"_a, "Read the depth map from a file.")
            .def(
                    "write",
                    [](const DepthMap& self, const std::string& path) {
                        self.Write(path);
                    },
                    "path"_a, "Write the depth map to a file.")
            .def("rescale", &DepthMap::Rescale, "factor"_a)
            .def("downsize", &DepthMap::Downsize, "max_width"_a, "max_height"_a)
            .def(
                    "to_bitmap",
                    [](const DepthMap& self, float min_percentile,
                       float max_percentile) {
                        return self.ToBitmap(min_percentile, max_percentile);
                    },
                    "min_percentile"_a = 5, "max_percentile"_a = 95,
                    "Convert the depth map to a grayscale Bitmap.")
            .def("__repr__", [](const DepthMap& self) {
                return "DepthMap(width=" + std::to_string(self.GetWidth()) +
                       ", height=" + std::to_string(self.GetHeight()) + ")";
            });

    py::class_<NormalMap> normal_map(m_submodule, "NormalMap",
                                     "A per-pixel normal map (MxNx3).");
    normal_map.def(py::init<>())
            .def(py::init([](size_t width, size_t height) {
                     return NormalMap(width, height);
                 }),
                 "width"_a, "height"_a)
            .def_property_readonly("width", &NormalMap::GetWidth)
            .def_property_readonly("height", &NormalMap::GetHeight)
            .def(
                    "array",
                    [](const NormalMap& self) {
                        // The engine stores channel planes (depth_ = 3); copy
                        // per pixel into the numpy-idiomatic (H, W, 3) view.
                        py::array_t<float> arr(
                                {static_cast<py::ssize_t>(self.GetHeight()),
                                 static_cast<py::ssize_t>(self.GetWidth()),
                                 py::ssize_t(3)});
                        auto buf = arr.mutable_unchecked<3>();
                        for (py::ssize_t r = 0; r < buf.shape(0); ++r) {
                            for (py::ssize_t c = 0; c < buf.shape(1); ++c) {
                                for (py::ssize_t s = 0; s < 3; ++s) {
                                    buf(r, c, s) =
                                            self.Get(static_cast<size_t>(r),
                                                     static_cast<size_t>(c),
                                                     static_cast<size_t>(s));
                                }
                            }
                        }
                        return arr;
                    },
                    "A (height, width, 3) float32 copy of the normals.")
            .def(
                    "set_array",
                    [](NormalMap& self,
                       const py::array_t<float, py::array::c_style |
                                                        py::array::forcecast>&
                               arr) {
                        THROW_CHECK_EQ(arr.ndim(), 3);
                        THROW_CHECK_EQ(static_cast<size_t>(arr.shape(2)),
                                       size_t(3));
                        THROW_CHECK_EQ(static_cast<size_t>(arr.shape(0)),
                                       self.GetHeight());
                        THROW_CHECK_EQ(static_cast<size_t>(arr.shape(1)),
                                       self.GetWidth());
                        auto buf = arr.unchecked<3>();
                        for (py::ssize_t r = 0; r < buf.shape(0); ++r) {
                            for (py::ssize_t c = 0; c < buf.shape(1); ++c) {
                                for (py::ssize_t s = 0; s < 3; ++s) {
                                    self.Set(static_cast<size_t>(r),
                                             static_cast<size_t>(c),
                                             static_cast<size_t>(s),
                                             buf(r, c, s));
                                }
                            }
                        }
                    },
                    "array"_a,
                    "Copy a (height, width, 3) float32 array into the map.")
            .def(
                    "read",
                    [](NormalMap& self, const std::string& path) {
                        self.Read(path);
                    },
                    "path"_a)
            .def(
                    "write",
                    [](const NormalMap& self, const std::string& path) {
                        self.Write(path);
                    },
                    "path"_a)
            .def("rescale", &NormalMap::Rescale, "factor"_a)
            .def("downsize", &NormalMap::Downsize, "max_width"_a,
                 "max_height"_a)
            .def(
                    "to_bitmap",
                    [](const NormalMap& self) { return self.ToBitmap(); },
                    "Convert the normal map to a color Bitmap.")
            .def("__repr__", [](const NormalMap& self) {
                return "NormalMap(width=" + std::to_string(self.GetWidth()) +
                       ", height=" + std::to_string(self.GetHeight()) + ")";
            });

    py::class_<Model> mvs_model(m_submodule, "MVSModel",
                                "The dense reconstruction workspace model.");
    py::class_<Model::Point> mvs_point(mvs_model, "Point",
                                       "A dense point with its visibility "
                                       "track.");
    mvs_point.def(py::init<>())
            .def_readwrite("x", &Model::Point::x)
            .def_readwrite("y", &Model::Point::y)
            .def_readwrite("z", &Model::Point::z)
            .def_readwrite("track", &Model::Point::track);
    mvs_model.def(py::init<>())
            .def(
                    "read",
                    [](Model& self, const std::string& path,
                       const std::string& format) { self.Read(path, format); },
                    "path"_a, "format"_a)
            .def(
                    "read_from_colmap",
                    [](Model& self, const std::string& path,
                       const std::string& sparse_path,
                       const std::string& images_path) {
                        self.ReadFromCOLMAP(path, sparse_path, images_path);
                    },
                    "path"_a, "sparse_path"_a = "sparse",
                    "images_path"_a = "images",
                    "Read the model from a COLMAP dense workspace.")
            .def(
                    "read_from_pmvs",
                    [](Model& self, const std::string& path) {
                        self.ReadFromPMVS(path);
                    },
                    "path"_a)
            .def("get_image_idx", &Model::GetImageIdx, "name"_a,
                 "Get the image index for the given image name.")
            .def("get_image_name", &Model::GetImageName, "image_idx"_a,
                 "Get the image name for the given image index.")
            .def("get_max_overlapping_images", &Model::GetMaxOverlappingImages,
                 "num_images"_a, "min_triangulation_angle"_a,
                 "Maximally overlapping images per image, sorted by shared "
                 "points subject to a minimum triangulation angle.")
            .def("compute_depth_ranges", &Model::ComputeDepthRanges,
                 "Robust per-image (min, max) depth ranges.")
            .def("compute_shared_points", &Model::ComputeSharedPoints,
                 "The number of shared points between all overlapping "
                 "images.")
            .def("compute_triangulation_angles",
                 &Model::ComputeTriangulationAngles, "percentile"_a = 50,
                 "Median triangulation angles between all overlapping "
                 "images.")
            .def_property_readonly(
                    "images", [](Model& self) { return &self.images; },
                    py::return_value_policy::reference_internal)
            .def_property_readonly(
                    "points", [](Model& self) { return &self.points; },
                    py::return_value_policy::reference_internal)
            .def_property_readonly(
                    "image_names",
                    [](const Model& self) {
                        std::vector<std::string> names;
                        names.reserve(self.images.size());
                        for (size_t i = 0; i < self.images.size(); ++i) {
                            names.push_back(
                                    self.GetImageName(static_cast<int>(i)));
                        }
                        return names;
                    })
            .def("__repr__", [](const Model& self) {
                return "MVSModel(num_images=" +
                       std::to_string(self.images.size()) +
                       ", num_points=" + std::to_string(self.points.size()) +
                       ")";
            });
}

}  // namespace mvs
}  // namespace reconstruction
}  // namespace cloudViewer
