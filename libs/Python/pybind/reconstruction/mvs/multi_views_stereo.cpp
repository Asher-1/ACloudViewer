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

#include "pybind/reconstruction/mvs/multi_views_stereo.h"

#include "pipelines/mvs.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/reconstruction_options.h"

namespace cloudViewer {
namespace reconstruction {
namespace mvs {

// Reconstruction multiple views stereo functions have similar arguments, sharing arg
// docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"input_path",
                 "Path to either the dense workspace folder or the sparse reconstruction."},
                {"output_path",
                 "The output path containing target cameras.bin/txt, "
                 "images.bin/txt and points3D.bin/txt."},
                {"input_type",
                 "Supported input type values are {dense, sparse}."},
                {"stereo_input_type",
                 "Supported stereo input type values are {photometric, geometric}."},
                {"output_type",
                 "Supported output type values are {BIN, TXT, PLY}."},
                {"workspace_path",
                 "Path to the folder containing the undistorted images."},
                {"workspace_format",
                 "Supported workspace format values are {COLMAP, PMVS}."},
                {"pmvs_option_name","The pmvs option name."},
                {"config_path", "The config path."},
                {"bbox_path", "The bounds file path."}};

void pybind_multi_views_stereo_methods(py::module &m) {
    m.def("mesh_delaunay", &MeshDelaunay,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the delaunay of mesh", "input_path"_a,
          "output_path"_a, "input_type"_a = "dense",
          "delaunay_meshing_options"_a = colmap::mvs::DelaunayMeshingOptions());
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
          "Function for the poisson of mesh", "input_path"_a,
          "output_path"_a, "poisson_meshing_options"_a = colmap::mvs::PoissonMeshingOptions());
    docstring::FunctionDocInject(m, "poisson_mesh",
                                 map_shared_argument_docstrings);

    m.def("stereo_fuse", &StereoFuse,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the stereo path-match of mesh", "workspace_path"_a,
          "output_path"_a, "bbox_path"_a = "", "stereo_input_type"_a = "geometric",
          "output_type"_a = "PLY", "workspace_format"_a = "COLMAP",
          "pmvs_option_name"_a = "option-all",
          "stereo_fusion_options"_a = colmap::mvs::StereoFusionOptions());
    docstring::FunctionDocInject(m, "stereo_fuse",
                                 map_shared_argument_docstrings);
}

void pybind_multi_views_stereo(py::module &m) {
    py::module m_submodule = m.def_submodule("mvs", "Reconstruction multiple views stereo.");
    pybind_multi_views_stereo_methods(m_submodule);
}

}  // namespace mvs
}  // namespace reconstruction
}  // namespace cloudViewer
