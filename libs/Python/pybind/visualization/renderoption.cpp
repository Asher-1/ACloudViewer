// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// clang-format off
#include "pybind/visualization/visualization_trampoline.h"  // must include first
// clang-format on

#include "visualization/visualizer/RenderOption.h"

#include <IJsonConvertibleIO.h>

#include "pybind/docstring.h"
#include "pybind/visualization/visualization.h"

namespace cloudViewer {
namespace visualization {

void pybind_renderoption(py::module &m) {
    // cloudViewer.visualization.RenderOption
    py::class_<visualization::RenderOption,
               std::shared_ptr<visualization::RenderOption>>
            renderoption(m, "RenderOption",
                         "Defines rendering options for visualizer.");
    py::detail::bind_default_constructor<visualization::RenderOption>(
            renderoption);
    renderoption
            .def("__repr__",
                 [](const visualization::RenderOption &vc) {
                     return std::string("RenderOption");
                 })
            .def(
                    "load_from_json",
                    [](visualization::RenderOption &ro,
                       const std::string &filename) {
                        io::ReadIJsonConvertible(filename, ro);
                    },
                    "Function to load visualization::RenderOption from a JSON "
                    "file.",
                    "filename"_a)
            .def(
                    "save_to_json",
                    [](visualization::RenderOption &ro,
                       const std::string &filename) {
                        io::WriteIJsonConvertible(filename, ro);
                    },
                    "Function to save visualization::RenderOption to a JSON "
                    "file.",
                    "filename"_a)
            .def_readwrite(
                    "background_color",
                    &visualization::RenderOption::background_color_,
                    "float numpy array of size ``(3,)``: Background RGB color.")
            .def_readwrite("light_on", &visualization::RenderOption::light_on_,
                           "bool: Whether to turn on Phong lighting.")
            .def_readwrite("point_size",
                           &visualization::RenderOption::point_size_,
                           "float: Point size for ``PointCloud``.")
            .def_readwrite("line_width",
                           &visualization::RenderOption::line_width_,
                           "float: Line width for ``LineSet``.")
            .def_readwrite("point_show_normal",
                           &visualization::RenderOption::point_show_normal_,
                           "bool: Whether to show normal for ``PointCloud``.")
            .def_readwrite("show_coordinate_frame",
                           &visualization::RenderOption::show_coordinate_frame_,
                           "bool: Whether to show coordinate frame.")
            .def_readwrite(
                    "mesh_show_back_face",
                    &visualization::RenderOption::mesh_show_back_face_,
                    "bool: Whether to show back faces for ``TriangleMesh``.")
            .def_readwrite(
                    "mesh_show_wireframe",
                    &visualization::RenderOption::mesh_show_wireframe_,
                    "bool: Whether to show wireframe for ``TriangleMesh``.")
            .def_readwrite("point_color_option",
                           &visualization::RenderOption::point_color_option_,
                           "``PointColorOption``: Point color option for "
                           "``PointCloud``.")
            .def_readwrite("mesh_shade_option",
                           &visualization::RenderOption::mesh_shade_option_,
                           "``MeshShadeOption``: Mesh shading option for "
                           "``TriangleMesh``.")
            .def_readwrite(
                    "mesh_color_option",
                    &visualization::RenderOption::mesh_color_option_,
                    "``MeshColorOption``: Color option for ``TriangleMesh``.");
    docstring::ClassMethodDocInject(m, "RenderOption", "load_from_json",
                                    {{"filename", "Path to file."}});
    docstring::ClassMethodDocInject(m, "RenderOption", "save_to_json",
                                    {{"filename", "Path to file."}});

    // This is a nested class, but now it's bind to the module
    // cv3d.visualization.PointColorOption
    py::native_enum<visualization::RenderOption::PointColorOption>
            enum_point_color_option(
                    m, "PointColorOption", "enum.Enum",
                    "Enum class for point color for ``PointCloud``.");
    enum_point_color_option
            .value("Default",
                   visualization::RenderOption::PointColorOption::Default)
            .value("Color",
                   visualization::RenderOption::PointColorOption::Color)
            .value("XCoordinate",
                   visualization::RenderOption::PointColorOption::XCoordinate)
            .value("YCoordinate",
                   visualization::RenderOption::PointColorOption::YCoordinate)
            .value("ZCoordinate",
                   visualization::RenderOption::PointColorOption::ZCoordinate)
            .value("Normal",
                   visualization::RenderOption::PointColorOption::Normal)
            .export_values()
            .finalize();

    // This is a nested class, but now it's bind to the module
    // cv3d.visualization.MeshShadeOption
    py::native_enum<visualization::RenderOption::MeshShadeOption>
            enum_mesh_shade_option(
                    m, "MeshShadeOption", "enum.Enum",
                    "Enum class for mesh shading for ``TriangleMesh``.");
    enum_mesh_shade_option
            .value("Default",
                   visualization::RenderOption::MeshShadeOption::FlatShade)
            .value("Color",
                   visualization::RenderOption::MeshShadeOption::SmoothShade)
            .export_values()
            .finalize();

    // This is a nested class, but now it's bind to the module
    // cv3d.visualization.MeshColorOption
    py::native_enum<visualization::RenderOption::MeshColorOption>
            enum_mesh_clor_option(m, "MeshColorOption", "enum.Enum",
                                  "Enum class for color for ``TriangleMesh``.");
    enum_mesh_clor_option
            .value("Default",
                   visualization::RenderOption::MeshColorOption::Default)
            .value("Color", visualization::RenderOption::MeshColorOption::Color)
            .value("XCoordinate",
                   visualization::RenderOption::MeshColorOption::XCoordinate)
            .value("YCoordinate",
                   visualization::RenderOption::MeshColorOption::YCoordinate)
            .value("ZCoordinate",
                   visualization::RenderOption::MeshColorOption::ZCoordinate)
            .value("Normal",
                   visualization::RenderOption::MeshColorOption::Normal)
            .export_values()
            .finalize();
}

void pybind_renderoption_method(py::module &m) {}

}  // namespace visualization
}  // namespace cloudViewer