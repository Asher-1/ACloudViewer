// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io -
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

#include <string>
#include <unordered_map>

#include "pybind/docstring.h"
#include "pybind/t/io/io.h"
#include "t/geometry/PointCloud.h"
#include "t/io/ImageIO.h"
#include "t/io/PointCloudIO.h"
#include "t/io/TriangleMeshIO.h"

namespace cloudViewer {
namespace t {
namespace io {

// IO functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"filename", "Path to file."},
                // Write options
                {"compressed",
                 "Set to ``True`` to write in compressed format."},
                {"format",
                 "The format of the input file. When not specified or set as "
                 "``auto``, the format is inferred from file extension name."},
                {"remove_nan_points",
                 "If true, all points that include a NaN are removed from "
                 "the PointCloud."},
                {"remove_infinite_points",
                 "If true, all points that include an infinite value are "
                 "removed from the PointCloud."},
                {"quality", "Quality of the output file."},
                {"write_ascii",
                 "Set to ``True`` to output in ascii format, otherwise binary "
                 "format will be used."},
                {"write_vertex_normals",
                 "Set to ``False`` to not write any vertex normals, even if "
                 "present on the mesh."},
                {"write_vertex_colors",
                 "Set to ``False`` to not write any vertex colors, even if "
                 "present on the mesh."},
                {"write_triangle_uvs",
                 "Set to ``False`` to not write any triangle uvs, even if "
                 "present on the mesh. For ``obj`` format, mtl file is saved "
                 "only when ``True`` is set."},
                // Entities
                {"config", "AzureKinectSensor's config file."},
                {"pointcloud", "The ``PointCloud`` object for I/O."},
                {"mesh", "The ``TriangleMesh`` object for I/O."},
                {"line_set", "The ``LineSet`` object for I/O."},
                {"image", "The ``Image`` object for I/O."},
                {"voxel_grid", "The ``VoxelGrid`` object for I/O."},
                {"trajectory",
                 "The ``PinholeCameraTrajectory`` object for I/O."},
                {"intrinsic", "The ``PinholeCameraIntrinsic`` object for I/O."},
                {"parameters",
                 "The ``PinholeCameraParameters`` object for I/O."},
                {"pose_graph", "The ``PoseGraph`` object for I/O."},
                {"feature", "The ``Feature`` object for I/O."},
                {"print_progress",
                 "If set to true a progress bar is visualized in the console."},
};

void pybind_class_io(py::module &m_io) {
    m_io.def(
            "read_point_cloud",
            [](const std::string &filename, const std::string &format,
               bool remove_nan_points, bool remove_infinite_points,
               bool print_progress) {
                py::gil_scoped_release release;
                t::geometry::PointCloud pcd;
                ReadPointCloud(filename, pcd,
                               {format, remove_nan_points,
                                remove_infinite_points, print_progress});
                return pcd;
            },
            "Function to read PointCloud with tensor attributes from file.",
            "filename"_a, "format"_a = "auto", "remove_nan_points"_a = false,
            "remove_infinite_points"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_point_cloud",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_point_cloud",
            [](const std::string &filename,
               const t::geometry::PointCloud &pointcloud, bool write_ascii,
               bool compressed, bool print_progress) {
                py::gil_scoped_release release;
                return WritePointCloud(
                        filename, pointcloud,
                        {write_ascii, compressed, print_progress});
            },
            "Function to write PointCloud with tensor attributes to file.",
            "filename"_a, "pointcloud"_a, "write_ascii"_a = false,
            "compressed"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_point_cloud",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_image",
            [](const std::string &filename) {
                py::gil_scoped_release release;
                geometry::Image image;
                ReadImage(filename, image);
                return image;
            },
            "Function to read image from file.", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_image",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_image",
            [](const std::string &filename, const geometry::Image &image,
               int quality) {
                py::gil_scoped_release release;
                return WriteImage(filename, image, quality);
            },
            "Function to write Image to file.", "filename"_a, "image"_a,
            "quality"_a = kCloudViewerImageIODefaultQuality);
    docstring::FunctionDocInject(m_io, "write_image",
                                 map_shared_argument_docstrings);

    // cloudViewer::geometry::ccMesh
    m_io.def(
            "read_triangle_mesh",
            [](const std::string &filename, bool enable_post_processing,
               bool print_progress) {
                py::gil_scoped_release release;
                t::geometry::TriangleMesh mesh;
                cloudViewer::io::ReadTriangleMeshOptions opt;
                opt.enable_post_processing = enable_post_processing;
                opt.print_progress = print_progress;
                ReadTriangleMesh(filename, mesh, opt);
                return mesh;
            },
            "Function to read TriangleMesh from file", "filename"_a,
            "enable_post_processing"_a = false, "print_progress"_a = false,
            R"doc(The general entrance for reading a TriangleMesh from a file.
        The function calls read functions based on the extension name of filename.
        Supported formats are `obj, ply, stl, off, gltf, glb, fbx`.

        The following example reads a triangle mesh with the .ply extension::
        import cloudViewer as cv3d
        mesh = cv3d.t.io.read_triangle_mesh('mesh.ply')

        Args:
        filename (str): Path to the mesh file.
        enable_post_processing (bool): If True enables post-processing.
                Post-processing will
                - triangulate meshes with polygonal faces
                - remove redundant materials
                - pretransform vertices
                - generate face normals if needed

                For more information see ASSIMPs documentation on the flags
                `aiProcessPreset_TargetRealtime_Fast, aiProcess_RemoveRedundantMaterials,
                aiProcess_OptimizeMeshes, aiProcess_PreTransformVertices`.

                Note that identical vertices will always be joined regardless of whether
                post-processing is enabled or not, which changes the number of vertices
                in the mesh.

                The `ply`-format is not affected by the post-processing.

        print_progress (bool): If True print the reading progress to the terminal.

        Returns:
        Returns the mesh object. On failure an empty mesh is returned.
        )doc");

    m_io.def(
            "write_triangle_mesh",
            [](const std::string &filename, const t::geometry::TriangleMesh &mesh,
               bool write_ascii, bool compressed, bool write_vertex_normals,
               bool write_vertex_colors, bool write_triangle_uvs,
               bool print_progress) {
                py::gil_scoped_release release;
                return WriteTriangleMesh(filename, mesh, write_ascii,
                                         compressed, write_vertex_normals,
                                         write_vertex_colors,
                                         write_triangle_uvs, print_progress);
            },
            "Function to write TriangleMesh to file", "filename"_a, "mesh"_a,
            "write_ascii"_a = false, "compressed"_a = false,
            "write_vertex_normals"_a = true, "write_vertex_colors"_a = true,
            "write_triangle_uvs"_a = true, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_triangle_mesh",
                                 map_shared_argument_docstrings);
}

}  // namespace io
}  // namespace t
}  // namespace cloudViewer
