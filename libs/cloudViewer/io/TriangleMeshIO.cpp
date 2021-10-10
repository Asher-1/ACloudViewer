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

#include "io/TriangleMeshIO.h"

#include <Eigen/Geometry>
#include <fstream>
#include <numeric>
#include <unordered_map>

// CV_CORE_LIB
#include <FileSystem.h>
#include <Helper.h>
#include <Logging.h>
#include <ProgressReporters.h>

// ECV_DB_LIB
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// ECV_IO_LIB
#include <ImageIO.h>
#include <rply.h>

#include "io/FileFormatIO.h"
#include <tiny_gltf.h>
#include <tiny_obj_loader.h>

namespace cloudViewer {

namespace {
using namespace io;

namespace ply_trianglemesh_reader {

struct PLYReaderState {
    utility::CountingProgressReporter* progress_bar;
    ccMesh* mesh_ptr;
    long vertex_index;
    long vertex_num;
    long normal_index;
    long normal_num;
    long color_index;
    long color_num;
    std::vector<unsigned int> face;
    long face_index;
    long face_num;
    double colors[3];
    double normals[3];
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState* state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void**>(&state_ptr),
                               &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    CCVector3* P_ptr = const_cast<CCVector3*>(
            state_ptr->mesh_ptr->getAssociatedCloud()->getPoint(
                    static_cast<unsigned int>(state_ptr->vertex_index)));
    P_ptr->u[index] = value;
    if (index == 2) {  // reading 'z'
        state_ptr->vertex_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

int ReadNormalCallback(p_ply_argument argument) {
    PLYReaderState* state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void**>(&state_ptr),
                               &index);
    if (state_ptr->normal_index >= state_ptr->normal_num) {
        return 0;
    }

    state_ptr->normals[index] = ply_get_argument_value(argument);

    if (index == 2) {  // reading 'nz'
        state_ptr->mesh_ptr->setVertexNormal(
                state_ptr->normal_index,
                Eigen::Vector3d(state_ptr->normals[0], state_ptr->normals[1],
                                state_ptr->normals[2]));
        state_ptr->normal_index++;
    }
    return 1;
}

int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState* state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void**>(&state_ptr),
                               &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->colors[index] = value / 255.0;

    if (index == 2) {  // reading 'blue'
        state_ptr->mesh_ptr->setVertexColor(
                state_ptr->color_index,
                Eigen::Vector3d(state_ptr->colors[0], state_ptr->colors[1],
                                state_ptr->colors[2]));
        state_ptr->color_index++;
    }
    return 1;
}

int ReadFaceCallBack(p_ply_argument argument) {
    PLYReaderState* state_ptr;
    long dummy, length, index;
    ply_get_argument_user_data(argument, reinterpret_cast<void**>(&state_ptr),
                               &dummy);
    double value = ply_get_argument_value(argument);
    if (state_ptr->face_index >= state_ptr->face_num) {
        return 0;
    }

    ply_get_argument_property(argument, nullptr, &length, &index);
    if (index == -1) {
        state_ptr->face.clear();
    } else {
        state_ptr->face.push_back(int(value));
    }
    if (long(state_ptr->face.size()) == length) {
        if (!AddTrianglesByEarClipping(*state_ptr->mesh_ptr, state_ptr->face)) {
            utility::LogWarning(
                    "Read PLY failed: A polygon in the mesh could not be "
                    "decomposed into triangles.");
            return 0;
        }
        state_ptr->face_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

}  // namespace ply_trianglemesh_reader

static const std::unordered_map<
        std::string,
        std::function<bool(
                const std::string&, ccMesh&, const ReadTriangleMeshOptions&)>>
        file_extension_to_trianglemesh_read_function{
                {"ply", ReadTriangleMeshFromPLY},
                {"stl", ReadTriangleMeshUsingASSIMP},
                {"obj", ReadTriangleMeshUsingASSIMP},
                {"off", ReadTriangleMeshFromOFF},
                {"gltf", ReadTriangleMeshUsingASSIMP},
                {"glb", ReadTriangleMeshUsingASSIMP},
                {"fbx", ReadTriangleMeshUsingASSIMP},
                {"vtk", AutoReadMesh},
                {"bin", AutoReadMesh},
        };

static const std::unordered_map<std::string,
                                std::function<bool(const std::string&,
                                                   const ccMesh&,
                                                   const bool,
                                                   const bool,
                                                   const bool,
                                                   const bool,
                                                   const bool,
                                                   const bool)>>
        file_extension_to_trianglemesh_write_function{
                {"ply", WriteTriangleMeshToPLY},
                {"stl", WriteTriangleMeshToSTL},
                {"obj", WriteTriangleMeshToOBJ},
                {"off", WriteTriangleMeshToOFF},
                {"gltf", WriteTriangleMeshToGLTF},
                {"glb", WriteTriangleMeshToGLTF},
                {"vtk", AutoWriteMesh},
                {"bin", AutoWriteMesh},
        };

}  // unnamed namespace

namespace io {
using namespace cloudViewer;
std::shared_ptr<ccMesh> CreateMeshFromFile(const std::string& filename,
                                           bool print_progress) {
    auto mesh = cloudViewer::make_shared<ccMesh>();
    mesh->createInternalCloud();

    ReadTriangleMeshOptions opt;
    opt.print_progress = print_progress;
    ReadTriangleMesh(filename, *mesh, opt);
    return mesh;
}

bool ReadTriangleMesh(const std::string& filename,
                      ccMesh& mesh,
                      ReadTriangleMeshOptions params /*={}*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(
                    filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read ccMesh failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trianglemesh_read_function.find(filename_ext);
    if (map_itr == file_extension_to_trianglemesh_read_function.end()) {
        utility::LogWarning(
                "Read ccMesh failed: unknown file "
                "extension.");
        return false;
    }

    if (params.print_progress) {
        auto progress_text = std::string("Reading ") +
                             utility::ToUpper(filename_ext) +
                             " file: " + filename;
        auto pbar = utility::ConsoleProgressBar(100, progress_text, true);
        params.update_progress = [pbar](double percent) mutable -> bool {
            pbar.setCurrentCount(size_t(percent));
            return true;
        };
    }

    bool success = map_itr->second(filename, mesh, params);
    utility::LogDebug(
            "Read ccMesh: {:d} triangles and {:d} vertices.", mesh.size(),
            mesh.getVerticeSize());
    if (mesh.hasVertices() && !mesh.hasTriangles()) {
        utility::LogWarning(
                "ccMesh appears to be a geometry::PointCloud "
                "(only contains vertices, but no triangles).");
    }
    return success;
}

bool WriteTriangleMesh(const std::string& filename,
                       const ccMesh& mesh,
                       bool write_ascii /* = false*/,
                       bool compressed /* = false*/,
                       bool write_vertex_normals /* = true*/,
                       bool write_vertex_colors /* = true*/,
                       bool write_triangle_uvs /* = true*/,
                       bool print_progress /* = false*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(
                    filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write ccMesh failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trianglemesh_write_function.find(filename_ext);
    if (map_itr == file_extension_to_trianglemesh_write_function.end()) {
        utility::LogWarning(
                "Write ccMesh failed: unknown file "
                "extension.");
        return false;
    }
    bool success = map_itr->second(filename, mesh, write_ascii, compressed,
                                   write_vertex_normals, write_vertex_colors,
                                   write_triangle_uvs, print_progress);
    utility::LogDebug(
            "Write ccMesh: {:d} triangles and {:d} vertices.", mesh.size(),
            mesh.getVerticeSize());
    return success;
}

// Reference: https://stackoverflow.com/a/43896965
bool IsPointInsidePolygon(const Eigen::MatrixX2d& polygon, double x, double y) {
    bool inside = false;
    for (int i = 0; i < polygon.rows(); ++i) {
        // i and j are the indices of the first and second vertices.
        int j = (i + 1) % polygon.rows();

        // The vertices of the edge that will be checked.
        double vx0 = polygon(i, 0);
        double vy0 = polygon(i, 1);
        double vx1 = polygon(j, 0);
        double vy1 = polygon(j, 1);

        // Check whether the edge intersects a line from (-inf,y) to (x,y).
        // First, check if the line crosses the horizontal line at y in either
        // direction.
        if (((vy0 <= y) && (vy1 > y)) || ((vy1 <= y) && (vy0 > y))) {
            // If so, get the point where it crosses that line.
            double cross = (vx1 - vx0) * (y - vy0) / (vy1 - vy0) + vx0;

            // Finally, check if it crosses to the left of the test point.
            if (cross < x) inside = !inside;
        }
    }
    return inside;
}

bool AddTrianglesByEarClipping(ccMesh& mesh,
                               std::vector<unsigned int>& indices) {
    int n = int(indices.size());
    Eigen::Vector3d face_normal = Eigen::Vector3d::Zero();
    if (n > 3) {
        for (int i = 0; i < n; i++) {
            const Eigen::Vector3d& v1 = mesh.getVertice(indices[(i + 1) % n]) -
                                        mesh.getVertice(indices[i % n]);
            const Eigen::Vector3d& v2 = mesh.getVertice(indices[(i + 2) % n]) -
                                        mesh.getVertice(indices[(i + 1) % n]);
            face_normal += v1.cross(v2);
        }
        double l = std::sqrt(face_normal.dot(face_normal));
        face_normal *= (1.0 / l);
    }

    bool found_ear = true;
    while (n > 3) {
        if (!found_ear) {
            // If no ear is found after all indices are looped through, the
            // polygon is not triangulable.
            return false;
        }

        found_ear = false;
        for (int i = 1; i < n - 2; i++) {
            const Eigen::Vector3d& v1 = mesh.getVertice(indices[i]) -
                                        mesh.getVertice(indices[i - 1]);
            const Eigen::Vector3d& v2 = mesh.getVertice(indices[i + 1]) -
                                        mesh.getVertice(indices[i]);
            bool is_convex = (face_normal.dot(v1.cross(v2)) > 0.0);
            bool is_ear = true;
            if (is_convex) {
                // If convex, check if vertex is an ear
                // (no vertices within triangle v[i-1], v[i], v[i+1])
                Eigen::MatrixX2d polygon(3, 2);
                for (int j = 0; j < 3; j++) {
                    polygon(j, 0) = mesh.getVertice(indices[i + j - 1])(0);
                    polygon(j, 1) = mesh.getVertice(indices[i + j - 1])(1);
                }

                for (int j = 0; j < n; j++) {
                    if (j == i - 1 || j == i || j == i + 1) {
                        continue;
                    }
                    const Eigen::Vector3d& v = mesh.getVertice(indices[j]);
                    if (IsPointInsidePolygon(polygon, v(0), v(1))) {
                        is_ear = false;
                        break;
                    }
                }

                if (is_ear) {
                    found_ear = true;
                    mesh.addTriangle(Eigen::Vector3i(indices[i - 1], indices[i],
                                                     indices[i + 1]));
                    indices.erase(indices.begin() + i);
                    n = int(indices.size());
                    break;
                }
            }
        }
    }
    mesh.addTriangle(Eigen::Vector3i(indices[0], indices[1], indices[2]));

    return true;
}

bool ReadTriangleMeshFromPLY(const std::string& filename,
                             ccMesh& mesh,
                             const ReadTriangleMeshOptions& params /*={}*/) {
    using namespace ply_trianglemesh_reader;

    p_ply ply_file = ply_open(filename.c_str(), nullptr, 0, nullptr);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: {}",
                            filename);
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    state.mesh_ptr = &mesh;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x",
                                       ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx",
                                       ReadNormalCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "ny", ReadNormalCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "nz", ReadNormalCallback, &state, 2);

    state.color_num = ply_set_read_cb(ply_file, "vertex", "red",
                                      ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

    if (state.vertex_num <= 0) {
        utility::LogWarning("Read PLY failed: number of vertex <= 0.");
        ply_close(ply_file);
        return false;
    }

    state.face_num = ply_set_read_cb(ply_file, "face", "vertex_indices",
                                     ReadFaceCallBack, &state, 0);
    if (state.face_num == 0) {
        state.face_num = ply_set_read_cb(ply_file, "face", "vertex_index",
                                         ReadFaceCallBack, &state, 0);
    }

    state.vertex_index = 0;
    state.normal_index = 0;
    state.color_index = 0;
    state.face_index = 0;

    mesh.clear();
    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    assert(cloud);
    cloud->resize(state.vertex_num);
    if (state.normal_num > 0) {
        cloud->resizeTheNormsTable();
    }
    if (state.color_num > 0) {
        cloud->resizeTheRGBTable();
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(state.vertex_num + state.face_num);
    state.progress_bar = &reporter;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}",
                            filename);
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    return true;
}

bool WriteTriangleMeshToPLY(const std::string& filename,
                            const ccMesh& mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/,
                            bool write_vertex_normals /* = true*/,
                            bool write_vertex_colors /* = true*/,
                            bool write_triangle_uvs /* = true*/,
                            bool print_progress) {
    if (write_triangle_uvs && mesh.hasTriangleUvs()) {
        utility::LogWarning(
                "This file format currently does not support writing textures "
                "and uv coordinates. Consider using .obj");
    }

    if (mesh.size() == 0) {
        utility::LogWarning("Write PLY failed: mesh has 0 vertices.");
        return false;
    }

    p_ply ply_file = ply_create(filename.c_str(),
                                write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                                nullptr, 0, nullptr);
    if (!ply_file) {
        utility::LogWarning("Write PLY failed: unable to open file: {}",
                            filename);
        return false;
    }

    write_vertex_normals = write_vertex_normals && mesh.hasNormals();
    write_vertex_colors = write_vertex_colors && mesh.hasColors();

    ply_add_comment(ply_file, "Created by cloudViewer");
    ply_add_element(ply_file, "vertex",
                    static_cast<long>(mesh.getVerticeSize()));
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    if (write_vertex_normals) {
        ply_add_property(ply_file, "nx", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "ny", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "nz", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    }
    if (write_vertex_colors) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }
    ply_add_element(ply_file, "face", static_cast<long>(mesh.size()));
    ply_add_property(ply_file, "vertex_indices", PLY_LIST, PLY_UCHAR, PLY_UINT);
    if (!ply_write_header(ply_file)) {
        utility::LogWarning("Write PLY failed: unable to write header.");
        ply_close(ply_file);
        return false;
    }

    utility::ConsoleProgressBar progress_bar(
            static_cast<size_t>(mesh.getVerticeSize() + mesh.size()),
            "Writing PLY: ", print_progress);
    bool printed_color_warning = false;
    for (size_t i = 0; i < mesh.getVerticeSize(); i++) {
        const auto& vertex = mesh.getVertice(i);
        ply_write(ply_file, vertex(0));
        ply_write(ply_file, vertex(1));
        ply_write(ply_file, vertex(2));
        if (write_vertex_normals) {
            const auto& normal = mesh.getVertexNormal(i);
            ply_write(ply_file, normal(0));
            ply_write(ply_file, normal(1));
            ply_write(ply_file, normal(2));
        }
        if (write_vertex_colors) {
            const auto& color = mesh.getVertexColor(i);
            if (!printed_color_warning &&
                (color(0) < 0 || color(0) > 1 || color(1) < 0 || color(1) > 1 ||
                 color(2) < 0 || color(2) > 1)) {
                utility::LogWarning(
                        "Write Ply clamped color value to valid range");
                printed_color_warning = true;
            }
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(0) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(1) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(2) * 255.0)));
        }
        ++progress_bar;
    }
    for (unsigned int i = 0; i < mesh.size(); i++) {
        Eigen::Vector3i triangle;
        mesh.getTriangleVertIndexes(i, triangle);
        ply_write(ply_file, 3);
        ply_write(ply_file, triangle(0));
        ply_write(ply_file, triangle(1));
        ply_write(ply_file, triangle(2));
        ++progress_bar;
    }

    ply_close(ply_file);
    return true;
}

FileGeometry ReadFileGeometryTypeFBX(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

FileGeometry ReadFileGeometryTypeSTL(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

bool ReadTriangleMeshFromSTL(const std::string& filename,
                             ccMesh& mesh,
                             bool print_progress) {
    FILE* myFile =
            utility::filesystem::FOpen(filename.c_str(), "rb");

    if (!myFile) {
        utility::LogWarning(
                "Read STL failed: unable to open file.");
        fclose(myFile);
        return false;
    }

    int num_of_triangles;
    if (myFile) {
        char header[80] = "";
        if (fread(header, sizeof(char), 80, myFile) != 80) {
            utility::LogWarning(
                    "[TriangleMeshIO::ReadTriangleMeshFromSTL] header IO "
                    "error!");
        }
        if (fread(&num_of_triangles, sizeof(unsigned int), 1, myFile) != 1) {
            utility::LogWarning(
                    "[TriangleMeshIO::ReadTriangleMeshFromSTL] triangles IO "
                    "error!");
        }
    } else {
        utility::LogWarning(
                "Read STL failed: unable to read header.");
        fclose(myFile);
        return false;
    }

    if (num_of_triangles == 0) {
        utility::LogWarning("Read STL failed: empty file.");
        fclose(myFile);
        return false;
    }

    mesh.clear();
    mesh.reservePerTriangleNormalIndexes();
    mesh.reserve(num_of_triangles);
    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    assert(cloud && cloud->reserveThePointsTable(num_of_triangles * 3));

    utility::ConsoleProgressBar progress_bar(
            num_of_triangles, "Reading STL: ", print_progress);
    for (int i = 0; i < num_of_triangles; i++) {
        char buffer[50];
        float* float_buffer;
        if (myFile) {
            if (fread(buffer, sizeof(char), 50, myFile) != 50) {
                utility::LogWarning(
                        "[TriangleMeshIO::ReadTriangleMeshFromSTL] buffer IO "
                        "error!");
            }
            float_buffer = reinterpret_cast<float*>(buffer);
            mesh.addTriangleNorm(
                    Eigen::Map<Eigen::Vector3f>(float_buffer).cast<double>());
            for (int j = 0; j < 3; j++) {
                float_buffer = reinterpret_cast<float*>(buffer + 12 * (j + 1));
                Eigen::Vector3d temp = Eigen::Map<Eigen::Vector3f>(float_buffer)
                                               .cast<double>();
                *cloud->getPointPtr(static_cast<unsigned int>(i * 3 + j)) =
                        temp;
            }
            mesh.addTriangle(Eigen::Vector3i(i * 3 + 0, i * 3 + 1, i * 3 + 2));
            // ignore buffer[48] and buffer [49] because it is rarely used.

        } else {
            utility::LogWarning(
                    "Read STL failed: not enough triangles.");
            fclose(myFile);
            return false;
        }
        ++progress_bar;
    }

    // do some cleaning
    {
        cloud->shrinkToFit();
        mesh.shrinkToFit();
        NormsIndexesTableType* normals = mesh.getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    cloud->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    cloud->setLocked(false);

    fclose(myFile);
    return true;
}

bool WriteTriangleMeshToSTL(const std::string& filename,
                            const ccMesh& mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/,
                            bool write_vertex_normals /* = true*/,
                            bool write_vertex_colors /* = true*/,
                            bool write_triangle_uvs /* = true*/,
                            bool print_progress) {
    if (write_triangle_uvs && mesh.hasTriangleUvs()) {
        utility::LogWarning(
                "This file format does not support writing textures and uv "
                "coordinates. Consider using .obj");
    }

    if (write_ascii) {
        utility::LogError("Writing ascii STL file is not supported yet.");
    }

    std::ofstream myFile(filename.c_str(), std::ios::out | std::ios::binary);

    if (!myFile) {
        utility::LogWarning(
                "Write STL failed: unable to open file.");
        return false;
    }

    if (!mesh.hasTriNormals()) {
        utility::LogWarning(
                "Write STL failed: compute normals first.");
        return false;
    }

    size_t num_of_triangles = mesh.size();
    if (num_of_triangles == 0) {
        utility::LogWarning("Write STL failed: empty file.");
        return false;
    }
    char header[80] = "Created by cloudViewer";
    myFile.write(header, 80);
    myFile.write((char*)(&num_of_triangles), 4);

    utility::ConsoleProgressBar progress_bar(
            num_of_triangles, "Writing STL: ", print_progress);
    for (size_t i = 0; i < num_of_triangles; i++) {
        Eigen::Vector3f float_vector3f =
                mesh.getTriangleNorm(static_cast<unsigned int>(i))
                        .cast<float>();
        myFile.write(reinterpret_cast<const char*>(float_vector3f.data()), 12);

        std::vector<Eigen::Vector3d> vN3;
        mesh.getTriangleVertices(static_cast<unsigned int>(i), vN3[0].data(),
                                 vN3[1].data(), vN3[2].data());
        for (int j = 0; j < 3; j++) {
            Eigen::Vector3f float_vector3f = vN3[j].cast<float>();
            myFile.write(reinterpret_cast<const char*>(float_vector3f.data()),
                         12);
        }
        char blank[2] = {0, 0};
        myFile.write(blank, 2);
        ++progress_bar;
    }
    return true;
}

FileGeometry ReadFileGeometryTypeOBJ(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

bool ReadTriangleMeshFromOBJ(const std::string& filename,
                             ccMesh& mesh,
                             const ReadTriangleMeshOptions& /*={}*/) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    std::string mtl_base_path =
            utility::filesystem::GetFileParentDirectory(filename);
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                filename.c_str(), mtl_base_path.c_str());
    if (!warn.empty()) {
        utility::LogWarning("Read OBJ failed: {}", warn);
    }
    if (!err.empty()) {
        utility::LogWarning("Read OBJ failed: {}", err);
    }

    if (!ret) {
        return false;
    }

    mesh.clear();

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    assert(cloud);

    cloud->reserveThePointsTable(
            static_cast<unsigned int>(attrib.vertices.size()));

    if (attrib.colors.size() > 0) {
        cloud->reserveTheRGBTable();
        cloud->showColors(true);
    }

    // copy vertex and data
    for (size_t vidx = 0; vidx < attrib.vertices.size(); vidx += 3) {
        tinyobj::real_t vx = attrib.vertices[vidx + 0];
        tinyobj::real_t vy = attrib.vertices[vidx + 1];
        tinyobj::real_t vz = attrib.vertices[vidx + 2];
        cloud->addEigenPoint(Eigen::Vector3d(vx, vy, vz));
    }

    for (size_t vidx = 0; vidx < attrib.colors.size(); vidx += 3) {
        tinyobj::real_t r = attrib.colors[vidx + 0];
        tinyobj::real_t g = attrib.colors[vidx + 1];
        tinyobj::real_t b = attrib.colors[vidx + 2];
        cloud->addEigenColor(Eigen::Vector3d(r, g, b));
    }

    // resize normal data and create bool indicator vector
    if (!attrib.normals.empty()) {
        cloud->resizeTheNormsTable();
    }
    std::vector<bool> normals_indicator(cloud->size(), false);

    // copy face data and copy normals data
    // append face-wise uv data
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            if (fv != 3) {
                utility::LogWarning(
                        "Read OBJ failed: facet with number of vertices not "
                        "equal to 3");
                return false;
            }

            Eigen::Vector3i facet;
            for (int v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                unsigned int vidx = idx.vertex_index;
                facet(v) = vidx;

                if (!attrib.normals.empty() && !normals_indicator[vidx] &&
                    (3 * idx.normal_index + 2) < int(attrib.normals.size())) {
                    tinyobj::real_t nx =
                            attrib.normals[3 * idx.normal_index + 0];
                    tinyobj::real_t ny =
                            attrib.normals[3 * idx.normal_index + 1];
                    tinyobj::real_t nz =
                            attrib.normals[3 * idx.normal_index + 2];

                    cloud->setPointNormal(vidx, CCVector3(nx, ny, nz));
                    normals_indicator[vidx] = true;
                }

                if (!attrib.texcoords.empty() &&
                    2 * idx.texcoord_index + 1 < int(attrib.texcoords.size())) {
                    tinyobj::real_t tx =
                            attrib.texcoords[2 * idx.texcoord_index + 0];
                    tinyobj::real_t ty =
                            attrib.texcoords[2 * idx.texcoord_index + 1];
                    mesh.triangle_uvs_.push_back(Eigen::Vector2d(tx, ty));
                }
            }

            mesh.addTriangle(facet);
            mesh.triangle_material_ids_.push_back(
                    shapes[s].mesh.material_ids[f]);
            index_offset += fv;
        }
    }

    // if not all normals have been set, then remove the vertex normals
    bool all_normals_set =
            std::accumulate(normals_indicator.begin(), normals_indicator.end(),
                            true, [](bool a, bool b) { return a && b; });
    if (!all_normals_set) {
        cloud->unallocateNorms();
    }

    // if not all triangles have corresponding uvs, then remove uvs
    if (3 * mesh.size() != mesh.triangle_uvs_.size()) {
        mesh.triangle_uvs_.clear();
    }

    auto textureLoader = [&mtl_base_path](std::string& relativePath) {
        auto image = io::CreateImageFromFile(mtl_base_path + relativePath);
        return image->HasData() ? image : std::shared_ptr<geometry::Image>();
    };

    using MaterialParameter = ccMesh::Material::MaterialParameter;

    for (auto& material : materials) {
        auto& meshMaterial = mesh.materials_[material.name];

        meshMaterial.baseColor = MaterialParameter::CreateRGB(
                material.diffuse[0], material.diffuse[1], material.diffuse[2]);

        if (!material.normal_texname.empty()) {
            meshMaterial.normalMap = textureLoader(material.normal_texname);
        } else if (!material.bump_texname.empty()) {
            // try bump, because there is often a misunderstanding of
            // what bump map or normal map is
            meshMaterial.normalMap = textureLoader(material.bump_texname);
        }

        if (!material.ambient_texname.empty()) {
            meshMaterial.ambientOcclusion =
                    textureLoader(material.ambient_texname);
        }

        if (!material.diffuse_texname.empty()) {
            meshMaterial.albedo = textureLoader(material.diffuse_texname);

            // Legacy texture map support
            if (meshMaterial.albedo) {
                mesh.textures_.push_back(*meshMaterial.albedo->FlipVertical());
            }
        }

        if (!material.metallic_texname.empty()) {
            meshMaterial.metallic = textureLoader(material.metallic_texname);
        }

        if (!material.roughness_texname.empty()) {
            meshMaterial.roughness = textureLoader(material.roughness_texname);
        }

        if (!material.sheen_texname.empty()) {
            meshMaterial.reflectance = textureLoader(material.sheen_texname);
        }

        // NOTE: We want defaults of 0.0 and 1.0 for baseMetallic and
        // baseRoughness respectively but 0.0 is a valid value for both and
        // tiny_obj_loader defaults to 0.0 for both. So, we will assume that
        // only if one of the values is greater than 0.0 that there are
        // non-default values set in the .mtl file
        if (material.roughness > 0.f || material.metallic > 0.f) {
            meshMaterial.baseMetallic = material.metallic;
            meshMaterial.baseRoughness = material.roughness;
        }

        if (material.sheen > 0.f) {
            meshMaterial.baseReflectance = material.sheen;
        }

        // NOTE: We will unconditionally copy the following parameters because
        // the TinyObj defaults match CloudViewer's internal defaults
        meshMaterial.baseClearCoat = material.clearcoat_thickness;
        meshMaterial.baseClearCoatRoughness = material.clearcoat_roughness;
        meshMaterial.baseAnisotropy = material.anisotropy;
    }

    return true;
}

bool WriteTriangleMeshToOBJ(const std::string& filename,
                            const ccMesh& mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/,
                            bool write_vertex_normals /* = true*/,
                            bool write_vertex_colors /* = true*/,
                            bool write_triangle_uvs /* = true*/,
                            bool print_progress) {
    std::string object_name = utility::filesystem::GetFileNameWithoutExtension(
            utility::filesystem::GetFileNameWithoutDirectory(filename));

    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
    if (!file) {
        utility::LogWarning("Write OBJ failed: unable to open file.");
        return false;
    }

    if (mesh.hasTriNormals()) {
        utility::LogWarning("Write OBJ can not include triangle normals.");
    }

    file << "# Created by cloudViewer " << std::endl;
    file << "# object name: " << object_name << std::endl;
    file << "# number of vertices: " << mesh.getVerticeSize() << std::endl;
    file << "# number of triangles: " << mesh.size() << std::endl;

    // always write material filename in obj file, regardless of uvs or textures
    file << "mtllib " << object_name << ".mtl" << std::endl;

    utility::ConsoleProgressBar progress_bar(
            mesh.getVerticeSize() + mesh.size(),
            "Writing OBJ: ", print_progress);

    write_vertex_normals = write_vertex_normals && mesh.hasNormals();
    write_vertex_colors = write_vertex_colors && mesh.hasColors();
    for (size_t vidx = 0; vidx < mesh.getVerticeSize(); ++vidx) {
        const Eigen::Vector3d& vertex = mesh.getVertice(vidx);
        file << "v " << vertex(0) << " " << vertex(1) << " " << vertex(2);
        if (write_vertex_colors) {
            const Eigen::Vector3d& color = mesh.getVertexColor(vidx);
            file << " " << color(0) << " " << color(1) << " " << color(2);
        }
        file << std::endl;

        if (write_vertex_normals) {
            const Eigen::Vector3d& normal = mesh.getVertexNormal(vidx);
            file << "vn " << normal(0) << " " << normal(1) << " " << normal(2)
                 << std::endl;
        }

        ++progress_bar;
    }

    // we are less strict and allows writing to uvs without known material
    // potentially this will be useful for exporting conformal map generation
    write_triangle_uvs = write_triangle_uvs && mesh.hasTriangleUvs();

    // we don't compress uvs into vertex-wise representation.
    // loose triangle-wise representation is provided
    if (write_triangle_uvs) {
        for (auto& uv : mesh.triangle_uvs_) {
            file << "vt " << uv(0) << " " << uv(1) << std::endl;
        }
    }

    // write faces with (possibly multiple) material ids
    // map faces with material ids
    std::map<int, std::vector<size_t>> material_id_faces_map;
    if (mesh.hasTriangleMaterialIds()) {
        for (size_t i = 0; i < mesh.triangle_material_ids_.size(); ++i) {
            int mi = mesh.triangle_material_ids_[i];
            auto it = material_id_faces_map.find(mi);
            if (it == material_id_faces_map.end()) {
                material_id_faces_map[mi] = {i};
            } else {
                it->second.push_back(i);
            }
        }
    } else {  // every face falls to the default material
        material_id_faces_map[0].resize(mesh.size());
        std::iota(material_id_faces_map[0].begin(),
                  material_id_faces_map[0].end(), 0);
    }

    // enumerate ids and their corresponding faces
    for (auto it = material_id_faces_map.begin();
         it != material_id_faces_map.end(); ++it) {
        // write the mtl name
        if (write_triangle_uvs) {
            std::string mtl_name =
                    object_name + "_" + std::to_string(it->first);
            file << "usemtl " << mtl_name << std::endl;
        }

        // write the corresponding faces
        for (auto tidx : it->second) {
            Eigen::Vector3i triangle;
            mesh.getTriangleVertIndexes(tidx, triangle);
            if (write_vertex_normals && write_triangle_uvs) {
                file << "f ";
                file << triangle(0) + 1 << "/" << 3 * tidx + 1 << "/"
                     << triangle(0) + 1 << " ";
                file << triangle(1) + 1 << "/" << 3 * tidx + 2 << "/"
                     << triangle(1) + 1 << " ";
                file << triangle(2) + 1 << "/" << 3 * tidx + 3 << "/"
                     << triangle(2) + 1 << std::endl;
            } else if (!write_vertex_normals && write_triangle_uvs) {
                file << "f ";
                file << triangle(0) + 1 << "/" << 3 * tidx + 1 << " ";
                file << triangle(1) + 1 << "/" << 3 * tidx + 2 << " ";
                file << triangle(2) + 1 << "/" << 3 * tidx + 3 << std::endl;
            } else if (write_vertex_normals && !write_triangle_uvs) {
                file << "f " << triangle(0) + 1 << "//" << triangle(0) + 1
                     << " " << triangle(1) + 1 << "//" << triangle(1) + 1 << " "
                     << triangle(2) + 1 << "//" << triangle(2) + 1 << std::endl;
            } else {
                file << "f " << triangle(0) + 1 << " " << triangle(1) + 1 << " "
                     << triangle(2) + 1 << std::endl;
            }
            ++progress_bar;
        }
    }
    // end of writing obj.
    //////

    //////
    // start to write to mtl and texture
    if (write_triangle_uvs) {
        std::string parent_dir =
                utility::filesystem::GetFileParentDirectory(filename);
        std::string mtl_filename = parent_dir + object_name + ".mtl";

        // write headers
        std::ofstream mtl_file(mtl_filename.c_str(), std::ios::out);
        if (!mtl_file) {
            utility::LogWarning(
                    "Write OBJ successful, but failed to write material file.");
            return true;
        }
        mtl_file << "# Created by cloudViewer " << std::endl;
        mtl_file << "# object name: " << object_name << std::endl;

        // write textures (if existing)
        for (size_t i = 0; i < mesh.textures_.size(); ++i) {
            std::string mtl_name = object_name + "_" + std::to_string(i);
            mtl_file << "newmtl " << mtl_name << std::endl;
            mtl_file << "Ka 1.000 1.000 1.000" << std::endl;
            mtl_file << "Kd 1.000 1.000 1.000" << std::endl;
            mtl_file << "Ks 0.000 0.000 0.000" << std::endl;

            std::string tex_filename = parent_dir + mtl_name + ".png";
            if (!io::WriteImage(tex_filename,
                                *mesh.textures_[i].FlipVertical())) {
                utility::LogWarning(
                        "Write OBJ successful, but failed to write texture "
                        "file.");
                return true;
            }
            mtl_file << "map_Kd " << mtl_name << ".png\n";
        }

        // write the default material
        if (!mesh.hasEigenTextures()) {
            std::string mtl_name = object_name + "_0";
            mtl_file << "newmtl " << mtl_name << std::endl;
            mtl_file << "Ka 1.000 1.000 1.000" << std::endl;
            mtl_file << "Kd 1.000 1.000 1.000" << std::endl;
            mtl_file << "Ks 0.000 0.000 0.000" << std::endl;
        }
    }
    return true;
}

FileGeometry ReadFileGeometryTypeOFF(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

bool ReadTriangleMeshFromOFF(const std::string& filename,
                             ccMesh& mesh,
                             const ReadTriangleMeshOptions &params) {
    std::ifstream file(filename.c_str(), std::ios::in);
    if (!file) {
        utility::LogWarning("Read OFF failed: unable to open file: {}",
                            filename);
        return false;
    }

    auto GetNextLine = [](std::ifstream& file) -> std::string {
        for (std::string line; std::getline(file, line);) {
            line = utility::StripString(line);
            if (!line.empty() && line[0] != '#') {
                return line;
            }
        }
        return "";
    };

    std::string header = GetNextLine(file);
    if (header != "OFF" && header != "COFF" && header != "NOFF" &&
        header != "CNOFF") {
        utility::LogWarning(
                "Read OFF failed: header keyword '{}' not supported.", header);
        return false;
    }

    std::string info = GetNextLine(file);
    unsigned int num_of_vertices, num_of_faces, num_of_edges;
    std::istringstream iss(info);
    if (!(iss >> num_of_vertices >> num_of_faces >> num_of_edges)) {
        utility::LogWarning("Read OFF failed: could not read file info.");
        return false;
    }

    if (num_of_vertices == 0 || num_of_faces == 0) {
        utility::LogWarning("Read OFF failed: mesh has no vertices or faces.");
        return false;
    }

    mesh.clear();
    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    assert(cloud);

    cloud->resize(num_of_vertices);
    bool parse_vertex_normals = false;
    bool parse_vertex_colors = false;
    if (header == "NOFF" || header == "CNOFF") {
        parse_vertex_normals = true;
        cloud->resizeTheNormsTable();
    }
    if (header == "COFF" || header == "CNOFF") {
        parse_vertex_colors = true;
        cloud->resizeTheRGBTable();
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(num_of_vertices + num_of_faces);

    float vx, vy, vz;
    float nx, ny, nz;
    float r, g, b, alpha;
    for (unsigned int vidx = 0; vidx < num_of_vertices; vidx++) {
        std::string line = GetNextLine(file);
        std::istringstream iss(line);
        if (!(iss >> vx >> vy >> vz)) {
            utility::LogWarning(
                    "Read OFF failed: could not read all vertex values.");
            return false;
        }
        *cloud->getPointPtr(vidx) = Eigen::Vector3d(vx, vy, vz);

        if (parse_vertex_normals) {
            if (!(iss >> nx >> ny >> nz)) {
                utility::LogWarning(
                        "Read OFF failed: could not read all vertex normal "
                        "values.");
                return false;
            }
            cloud->setPointNormal(vidx, CCVector3(nx, ny, nz));
        }
        if (parse_vertex_colors) {
            if (!(iss >> r >> g >> b >> alpha)) {
                utility::LogWarning(
                        "Read OFF failed: could not read all vertex color "
                        "values.");
                return false;
            }
            cloud->setPointColor(vidx,
                                 ecvColor::Rgb(static_cast<ColorCompType>(r),
                                               static_cast<ColorCompType>(g),
                                               static_cast<ColorCompType>(b)));
        }

        ++reporter;
    }

    unsigned int n, vertex_index;
    std::vector<unsigned int> indices;
    for (size_t tidx = 0; tidx < num_of_faces; tidx++) {
        std::string line = GetNextLine(file);
        std::istringstream iss(line);
        iss >> n;
        indices.clear();
        for (size_t vidx = 0; vidx < n; vidx++) {
            if (!(iss >> vertex_index)) {
                utility::LogWarning(
                        "Read OFF failed: could not read all vertex "
                        "indices.");
                return false;
            }
            indices.push_back(vertex_index);
        }
        if (!AddTrianglesByEarClipping(mesh, indices)) {
            utility::LogWarning(
                    "Read OFF failed: A polygon in the mesh could not be "
                    "decomposed into triangles. Vertex indices: {}",
                    indices);
            return false;
        }
        ++reporter;
    }

    file.close();
    return true;
}

bool WriteTriangleMeshToOFF(const std::string& filename,
                            const ccMesh& mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/,
                            bool write_vertex_normals /* = true*/,
                            bool write_vertex_colors /* = true*/,
                            bool write_triangle_uvs /* = true*/,
                            bool print_progress) {
    if (write_triangle_uvs && mesh.hasTriangleUvs()) {
        utility::LogWarning(
                "This file format does not support writing textures and uv "
                "coordinates. Consider using .obj");
    }

    std::ofstream file(filename.c_str(), std::ios::out);
    if (!file) {
        utility::LogWarning("Write OFF failed: unable to open file.");
        return false;
    }

    if (mesh.hasTriNormals()) {
        utility::LogWarning("Write OFF cannot include triangle normals.");
    }

    size_t num_of_vertices = mesh.getVerticeSize();
    size_t num_of_triangles = mesh.size();
    if (num_of_vertices == 0 || num_of_triangles == 0) {
        utility::LogWarning("Write OFF failed: empty file.");
        return false;
    }

    write_vertex_normals = write_vertex_normals && mesh.hasNormals();
    write_vertex_colors = write_vertex_colors && mesh.hasColors();
    if (write_vertex_colors) {
        file << "C";
    }
    if (write_vertex_normals) {
        file << "N";
    }
    file << "OFF" << std::endl;
    file << num_of_vertices << " " << num_of_triangles << " 0" << std::endl;

    utility::ConsoleProgressBar progress_bar(num_of_vertices + num_of_triangles,
                                             "Writing OFF: ", print_progress);
    for (size_t vidx = 0; vidx < num_of_vertices; ++vidx) {
        const Eigen::Vector3d& vertex = mesh.getVertice(vidx);
        file << vertex(0) << " " << vertex(1) << " " << vertex(2);
        if (write_vertex_normals) {
            const Eigen::Vector3d& normal = mesh.getVertexNormal(vidx);
            file << " " << normal(0) << " " << normal(1) << " " << normal(2);
        }
        if (write_vertex_colors) {
            const Eigen::Vector3d& color = mesh.getVertexColor(vidx);
            file << " " << std::round(color(0) * 255.0) << " "
                 << std::round(color(1) * 255.0) << " "
                 << std::round(color(2) * 255.0) << " 255";
        }
        file << std::endl;
        ++progress_bar;
    }

    for (size_t tidx = 0; tidx < num_of_triangles; ++tidx) {
        Eigen::Vector3i triangle;
        mesh.getTriangleVertIndexes(tidx, triangle);
        file << "3 " << triangle(0) << " " << triangle(1) << " " << triangle(2)
             << std::endl;
        ++progress_bar;
    }

    file.close();
    return true;
}

// Adapts an array of bytes to an array of T. Will advance of byte_stride each
// elements.
template <typename T>
struct ArrayAdapter {
    // Pointer to the bytes
    const unsigned char* data_ptr;
    // Number of elements in the array
    const size_t elem_count;
    // Stride in bytes between two elements
    const size_t stride;

    // Construct an array adapter.
    // \param ptr Pointer to the start of the data, with offset applied
    // \param count Number of elements in the array
    // \param byte_stride Stride betweens elements in the array
    ArrayAdapter(const unsigned char* ptr, size_t count, size_t byte_stride)
        : data_ptr(ptr), elem_count(count), stride(byte_stride) {}

    // Returns a *copy* of a single element. Can't be used to modify it.
    T operator[](size_t pos) const {
        if (pos >= elem_count)
            throw std::out_of_range(
                    "Tried to access beyond the last element of an array "
                    "adapter with count " +
                    std::to_string(elem_count) +
                    " while getting element number " + std::to_string(pos));
        return *(reinterpret_cast<const T*>(data_ptr + pos * stride));
    }
};

// Interface of any adapted array that returns integer data
struct IntArrayBase {
    virtual ~IntArrayBase() = default;
    virtual unsigned int operator[](size_t) const = 0;
    virtual size_t size() const = 0;
};

// An array that loads integer types, and returns them as int
template <class T>
struct IntArray : public IntArrayBase {
    ArrayAdapter<T> adapter;

    IntArray(const ArrayAdapter<T>& a) : adapter(a) {}
    unsigned int operator[](size_t position) const override {
        return static_cast<unsigned int>(adapter[position]);
    }

    size_t size() const override { return adapter.elem_count; }
};

FileGeometry ReadFileGeometryTypeGLTF(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

bool ReadTriangleMeshFromGLTF(const std::string& filename,
                              ccMesh& mesh,
                              const ReadTriangleMeshOptions& params /*={}*/) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn;
    std::string err;

    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    bool ret;
    if (filename_ext == "glb") {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
    } else {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());
    }

    if (!warn.empty() || !err.empty()) {
        utility::LogWarning("Read GLTF failed: unable to open file {}",
                            filename);
    }
    if (!ret) {
        return false;
    }

    if (model.meshes.size() > 1) {
        utility::LogInfo(
                "The file contains more than one mesh. All meshes will be "
                "loaded as a single mesh.");
    }

    mesh.clear();
    ccPointCloud* baseVertice = new ccPointCloud("vertices");
    baseVertice->setEnabled(false);

    ccMesh mesh_temp(baseVertice);
    mesh_temp.addChild(baseVertice);

    for (const tinygltf::Node& gltf_node : model.nodes) {
        if (gltf_node.mesh != -1) {
            mesh_temp.clear();
            const tinygltf::Mesh& gltf_mesh = model.meshes[gltf_node.mesh];

            for (const tinygltf::Primitive& primitive : gltf_mesh.primitives) {
                for (const auto& attribute : primitive.attributes) {
                    if (attribute.first == "POSITION") {
                        tinygltf::Accessor& positions_accessor =
                                model.accessors[attribute.second];
                        tinygltf::BufferView& positions_view =
                                model.bufferViews[positions_accessor
                                                          .bufferView];
                        const tinygltf::Buffer& positions_buffer =
                                model.buffers[positions_view.buffer];
                        const float* positions = reinterpret_cast<const float*>(
                                &positions_buffer
                                         .data[positions_view.byteOffset +
                                               positions_accessor.byteOffset]);

                        for (size_t i = 0; i < positions_accessor.count; ++i) {
                            baseVertice->addEigenPoint(Eigen::Vector3d(
                                    positions[i * 3 + 0], positions[i * 3 + 1],
                                    positions[i * 3 + 2]));
                        }
                    }

                    if (attribute.first == "NORMAL") {
                        tinygltf::Accessor& normals_accessor =
                                model.accessors[attribute.second];
                        tinygltf::BufferView& normals_view =
                                model.bufferViews[normals_accessor.bufferView];
                        const tinygltf::Buffer& normals_buffer =
                                model.buffers[normals_view.buffer];
                        const float* normals = reinterpret_cast<const float*>(
                                &normals_buffer
                                         .data[normals_view.byteOffset +
                                               normals_accessor.byteOffset]);

                        for (size_t i = 0; i < normals_accessor.count; ++i) {
                            baseVertice->addEigenNorm(Eigen::Vector3d(
                                    normals[i * 3 + 0], normals[i * 3 + 1],
                                    normals[i * 3 + 2]));
                        }
                    }

                    if (attribute.first == "COLOR_0") {
                        tinygltf::Accessor& colors_accessor =
                                model.accessors[attribute.second];
                        tinygltf::BufferView& colors_view =
                                model.bufferViews[colors_accessor.bufferView];
                        const tinygltf::Buffer& colors_buffer =
                                model.buffers[colors_view.buffer];

                        size_t byte_stride = colors_view.byteStride;
                        if (byte_stride == 0) {
                            // According to glTF 2.0 specs:
                            // When byteStride==0, it means that accessor
                            // elements are tightly packed.
                            byte_stride =
                                    colors_accessor.type *
                                    tinygltf::GetComponentSizeInBytes(
                                            colors_accessor.componentType);
                        }
                        switch (colors_accessor.componentType) {
                            case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                                for (size_t i = 0; i < colors_accessor.count;
                                     ++i) {
                                    const float* colors =
                                            reinterpret_cast<const float*>(
                                                    colors_buffer.data.data() +
                                                    colors_view.byteOffset +
                                                    i * byte_stride);

                                    baseVertice->addRGBColor(
                                            ecvColor::Rgb::FromEigen(
                                                    Eigen::Vector3d(
                                                            colors[0],
                                                            colors[1],
                                                            colors[2])));
                                }
                                break;
                            }
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                                double max_val = (double)
                                        std::numeric_limits<uint8_t>::max();
                                for (size_t i = 0; i < colors_accessor.count;
                                     ++i) {
                                    const uint8_t* colors =
                                            reinterpret_cast<const uint8_t*>(
                                                    colors_buffer.data.data() +
                                                    colors_view.byteOffset +
                                                    i * byte_stride);

                                    baseVertice->addRGBColor(
                                            ecvColor::Rgb::FromEigen(
                                                    Eigen::Vector3d(
                                                            colors[0] / max_val,
                                                            colors[1] / max_val,
                                                            colors[2] /
                                                                    max_val)));
                                }
                                break;
                            }
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                                double max_val = (double)
                                        std::numeric_limits<uint16_t>::max();
                                for (size_t i = 0; i < colors_accessor.count;
                                     ++i) {
                                    const uint16_t* colors =
                                            reinterpret_cast<const uint16_t*>(
                                                    colors_buffer.data.data() +
                                                    colors_view.byteOffset +
                                                    i * byte_stride);
                                    baseVertice->addRGBColor(
                                            ecvColor::Rgb::FromEigen(
                                                    Eigen::Vector3d(
                                                            colors[0] / max_val,
                                                            colors[1] / max_val,
                                                            colors[2] /
                                                                    max_val)));
                                }
                                break;
                            }
                            default: {
                                utility::LogWarning(
                                        "Unrecognized component type for "
                                        "vertex colors");
                                break;
                            }
                        }
                    }
                }

                // Load triangles
                std::unique_ptr<IntArrayBase> indices_array_pointer = nullptr;
                {
                    const tinygltf::Accessor& indices_accessor =
                            model.accessors[primitive.indices];
                    const tinygltf::BufferView& indices_view =
                            model.bufferViews[indices_accessor.bufferView];
                    const tinygltf::Buffer& indices_buffer =
                            model.buffers[indices_view.buffer];
                    const auto data_address = indices_buffer.data.data() +
                                              indices_view.byteOffset +
                                              indices_accessor.byteOffset;
                    const auto byte_stride =
                            indices_accessor.ByteStride(indices_view);
                    const auto count = indices_accessor.count;

                    // Allocate the index array in the pointer-to-base
                    // declared in the parent scope
                    switch (indices_accessor.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_BYTE:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<char>>(
                                            new IntArray<char>(
                                                    ArrayAdapter<char>(
                                                            data_address, count,
                                                            byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<unsigned char>>(
                                            new IntArray<unsigned char>(
                                                    ArrayAdapter<unsigned char>(
                                                            data_address, count,
                                                            byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_SHORT:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<short>>(
                                            new IntArray<short>(
                                                    ArrayAdapter<short>(
                                                            data_address, count,
                                                            byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            indices_array_pointer = std::unique_ptr<
                                    IntArray<unsigned short>>(
                                    new IntArray<unsigned short>(
                                            ArrayAdapter<unsigned short>(
                                                    data_address, count,
                                                    byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_INT:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<int>>(
                                            new IntArray<int>(ArrayAdapter<int>(
                                                    data_address, count,
                                                    byte_stride)));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            indices_array_pointer =
                                    std::unique_ptr<IntArray<unsigned int>>(
                                            new IntArray<unsigned int>(
                                                    ArrayAdapter<unsigned int>(
                                                            data_address, count,
                                                            byte_stride)));
                            break;
                        default:
                            break;
                    }
                    const auto& indices = *indices_array_pointer;

                    switch (primitive.mode) {
                        case TINYGLTF_MODE_TRIANGLES:
                            for (size_t i = 0; i < indices_accessor.count;
                                 i += 3) {
                                mesh_temp.addTriangle(Eigen::Vector3i(
                                        indices[i], indices[i + 1],
                                        indices[i + 2]));
                            }
                            break;
                        case TINYGLTF_MODE_TRIANGLE_STRIP:
                            for (size_t i = 2; i < indices_accessor.count;
                                 ++i) {
                                mesh_temp.addTriangle(Eigen::Vector3i(
                                        indices[i - 2], indices[i - 1],
                                        indices[i]));
                            }
                            break;
                        case TINYGLTF_MODE_TRIANGLE_FAN:
                            for (size_t i = 2; i < indices_accessor.count;
                                 ++i) {
                                mesh_temp.addTriangle(Eigen::Vector3i(
                                        indices[0], indices[i - 1],
                                        indices[i]));
                            }
                            break;
                    }
                }
            }

            if (gltf_node.matrix.size() > 0) {
                std::vector<double> matrix = gltf_node.matrix;
                Eigen::Matrix4d transform =
                        Eigen::Map<Eigen::Matrix4d>(&matrix[0], 4, 4);
                mesh_temp.transform(transform);
            } else {
                // The specification states that first the scale is
                // applied to the vertices, then the rotation, and then the
                // translation.
                if (gltf_node.scale.size() > 0) {
                    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                    transform(0, 0) = gltf_node.scale[0];
                    transform(1, 1) = gltf_node.scale[1];
                    transform(2, 2) = gltf_node.scale[2];
                    mesh_temp.transform(transform);
                }
                if (gltf_node.rotation.size() > 0) {
                    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                    // glTF represents a quaternion as qx, qy, qz, qw, while
                    // Eigen::Quaterniond orders the parameters as qw, qx,
                    // qy, qz.
                    transform.topLeftCorner<3, 3>() =
                            Eigen::Quaterniond(gltf_node.rotation[3],
                                               gltf_node.rotation[0],
                                               gltf_node.rotation[1],
                                               gltf_node.rotation[2])
                                    .toRotationMatrix();
                    mesh_temp.transform(transform);
                }
                if (gltf_node.translation.size() > 0) {
                    mesh_temp.translate(Eigen::Vector3d(
                            gltf_node.translation[0], gltf_node.translation[1],
                            gltf_node.translation[2]));
                }
            }
            mesh += mesh_temp;
        }
    }

    return true;
}

bool WriteTriangleMeshToGLTF(const std::string& filename,
                             const ccMesh& mesh,
                             bool write_ascii /* = false*/,
                             bool compressed /* = false*/,
                             bool write_vertex_normals /* = true*/,
                             bool write_vertex_colors /* = true*/,
                             bool write_triangle_uvs /* = true*/,
                             bool print_progress) {
    if (write_triangle_uvs && mesh.hasTriangleUvs()) {
        utility::LogWarning(
                "This file format does not support writing textures and uv "
                "coordinates. Consider using .obj");
    }
    tinygltf::Model model;
    model.asset.generator = "cloudViewer";
    model.asset.version = "2.0";
    model.defaultScene = 0;

    size_t byte_length;
    size_t num_of_vertices = mesh.getVerticeSize();
    size_t num_of_triangles = mesh.size();

    float float_temp;
    unsigned char* temp = nullptr;

    tinygltf::BufferView indices_buffer_view_array;
    bool save_indices_as_uint32 = num_of_vertices > 65536;
    indices_buffer_view_array.name = save_indices_as_uint32
                                             ? "buffer-0-bufferview-uint"
                                             : "buffer-0-bufferview-ushort";
    indices_buffer_view_array.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
    indices_buffer_view_array.buffer = 0;
    indices_buffer_view_array.byteLength = 0;
    model.bufferViews.push_back(indices_buffer_view_array);
    size_t indices_buffer_view_index = model.bufferViews.size() - 1;

    tinygltf::BufferView buffer_view_array;
    buffer_view_array.name = "buffer-0-bufferview-vec3",
    buffer_view_array.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    buffer_view_array.buffer = 0;
    buffer_view_array.byteLength = 0;
    buffer_view_array.byteOffset = 0;
    buffer_view_array.byteStride = 12;
    model.bufferViews.push_back(buffer_view_array);
    size_t mesh_attributes_buffer_view_index = model.bufferViews.size() - 1;

    tinygltf::Scene gltf_scene;
    gltf_scene.nodes.push_back(0);
    model.scenes.push_back(gltf_scene);

    tinygltf::Node gltf_node;
    gltf_node.mesh = 0;
    model.nodes.push_back(gltf_node);

    tinygltf::Mesh gltf_mesh;
    tinygltf::Primitive gltf_primitive;

    tinygltf::Accessor indices_accessor;
    indices_accessor.name = "buffer-0-accessor-indices-buffer-0-mesh-0";
    indices_accessor.type = TINYGLTF_TYPE_SCALAR;
    indices_accessor.componentType =
            save_indices_as_uint32 ? TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT
                                   : TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
    indices_accessor.count = 3 * num_of_triangles;
    byte_length =
            3 * num_of_triangles *
            (save_indices_as_uint32 ? sizeof(uint32_t) : sizeof(uint16_t));

    indices_accessor.bufferView = int(indices_buffer_view_index);
    indices_accessor.byteOffset =
            model.bufferViews[indices_buffer_view_index].byteLength;
    model.bufferViews[indices_buffer_view_index].byteLength += byte_length;

    std::vector<unsigned char> index_buffer;
    for (size_t tidx = 0; tidx < num_of_triangles; ++tidx) {
        Eigen::Vector3i triangle;
        mesh.getTriangleVertIndexes(tidx, triangle);
        size_t uint_size =
                save_indices_as_uint32 ? sizeof(uint32_t) : sizeof(uint16_t);
        for (size_t i = 0; i < 3; ++i) {
            temp = (unsigned char*)&(triangle(i));
            for (size_t j = 0; j < uint_size; ++j) {
                index_buffer.push_back(temp[j]);
            }
        }
    }

    indices_accessor.minValues.push_back(0);
    indices_accessor.maxValues.push_back(3 * int(num_of_triangles) - 1);
    model.accessors.push_back(indices_accessor);
    gltf_primitive.indices = int(model.accessors.size()) - 1;

    tinygltf::Accessor positions_accessor;
    positions_accessor.name = "buffer-0-accessor-position-buffer-0-mesh-0";
    positions_accessor.type = TINYGLTF_TYPE_VEC3;
    positions_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    positions_accessor.count = num_of_vertices;
    byte_length = 3 * num_of_vertices * sizeof(float);
    positions_accessor.bufferView = int(mesh_attributes_buffer_view_index);
    positions_accessor.byteOffset =
            model.bufferViews[mesh_attributes_buffer_view_index].byteLength;
    model.bufferViews[mesh_attributes_buffer_view_index].byteLength +=
            byte_length;

    std::vector<unsigned char> mesh_attribute_buffer;
    for (size_t vidx = 0; vidx < num_of_vertices; ++vidx) {
        Eigen::Vector3d vertex = mesh.getVertice(vidx);
        for (size_t i = 0; i < 3; ++i) {
            float_temp = (float)vertex(i);
            temp = (unsigned char*)&(float_temp);
            for (size_t j = 0; j < sizeof(float); ++j) {
                mesh_attribute_buffer.push_back(temp[j]);
            }
        }
    }

    Eigen::Vector3d min_bound = mesh.getMinBound();
    positions_accessor.minValues.push_back(min_bound[0]);
    positions_accessor.minValues.push_back(min_bound[1]);
    positions_accessor.minValues.push_back(min_bound[2]);
    Eigen::Vector3d max_bound = mesh.getMaxBound();
    positions_accessor.maxValues.push_back(max_bound[0]);
    positions_accessor.maxValues.push_back(max_bound[1]);
    positions_accessor.maxValues.push_back(max_bound[2]);
    model.accessors.push_back(positions_accessor);
    gltf_primitive.attributes.insert(std::make_pair(
            "POSITION", static_cast<int>(model.accessors.size()) - 1));

    write_vertex_normals = write_vertex_normals && mesh.hasNormals();
    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    if (write_vertex_normals) {
        tinygltf::Accessor normals_accessor;
        normals_accessor.name = "buffer-0-accessor-normal-buffer-0-mesh-0";
        normals_accessor.type = TINYGLTF_TYPE_VEC3;
        normals_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        normals_accessor.count = mesh.getVerticeSize();
        size_t byte_length = 3 * mesh.getVerticeSize() * sizeof(float);
        normals_accessor.bufferView = int(mesh_attributes_buffer_view_index);
        normals_accessor.byteOffset =
                model.bufferViews[mesh_attributes_buffer_view_index].byteLength;
        model.bufferViews[mesh_attributes_buffer_view_index].byteLength +=
                byte_length;

        if (cloud) {
            for (size_t vidx = 0; vidx < num_of_vertices; ++vidx) {
                const Eigen::Vector3d& normal = cloud->getEigenNormal(vidx);
                for (size_t i = 0; i < 3; ++i) {
                    float_temp = (float)normal(i);
                    temp = (unsigned char*)&(float_temp);
                    for (size_t j = 0; j < sizeof(float); ++j) {
                        mesh_attribute_buffer.push_back(temp[j]);
                    }
                }
            }
        }

        model.accessors.push_back(normals_accessor);
        gltf_primitive.attributes.insert(std::make_pair(
                "NORMAL", static_cast<int>(model.accessors.size()) - 1));
    }

    write_vertex_colors = write_vertex_colors && mesh.hasColors();
    if (write_vertex_colors) {
        tinygltf::Accessor colors_accessor;
        colors_accessor.name = "buffer-0-accessor-color-buffer-0-mesh-0";
        colors_accessor.type = TINYGLTF_TYPE_VEC3;
        colors_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        colors_accessor.count = mesh.getVerticeSize();
        size_t byte_length = 3 * mesh.getVerticeSize() * sizeof(float);
        colors_accessor.bufferView = int(mesh_attributes_buffer_view_index);
        colors_accessor.byteOffset =
                model.bufferViews[mesh_attributes_buffer_view_index].byteLength;
        model.bufferViews[mesh_attributes_buffer_view_index].byteLength +=
                byte_length;

        if (cloud) {
            for (size_t vidx = 0; vidx < num_of_vertices; ++vidx) {
                const ecvColor::Rgb& col =
                        cloud->getPointColor(static_cast<unsigned int>(vidx));
                const Eigen::Vector3d& color = ecvColor::Rgb::ToEigen(col);
                for (size_t i = 0; i < 3; ++i) {
                    float_temp = (float)color(i);
                    temp = (unsigned char*)&(float_temp);
                    for (size_t j = 0; j < sizeof(float); ++j) {
                        mesh_attribute_buffer.push_back(temp[j]);
                    }
                }
            }
        }

        model.accessors.push_back(colors_accessor);
        gltf_primitive.attributes.insert(std::make_pair(
                "COLOR_0", static_cast<int>(model.accessors.size()) - 1));
    }

    gltf_primitive.mode = TINYGLTF_MODE_TRIANGLES;
    gltf_mesh.primitives.push_back(gltf_primitive);
    model.meshes.push_back(gltf_mesh);

    model.bufferViews[0].byteOffset = 0;
    model.bufferViews[1].byteOffset = index_buffer.size();

    tinygltf::Buffer buffer;
    buffer.uri = filename.substr(0, filename.find_last_of(".")) + ".bin";
    buffer.data.resize(index_buffer.size() + mesh_attribute_buffer.size());
    memcpy(buffer.data.data(), index_buffer.data(), index_buffer.size());
    memcpy(buffer.data.data() + index_buffer.size(),
           mesh_attribute_buffer.data(), mesh_attribute_buffer.size());
    model.buffers.push_back(buffer);

    tinygltf::TinyGLTF loader;
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext == "glb") {
        if (!loader.WriteGltfSceneToFile(&model, filename, false, true, true,
                                         true)) {
            utility::LogWarning("Write GLTF failed.");
            return false;
        }
    } else {
        if (!loader.WriteGltfSceneToFile(&model, filename, false, true, true,
                                         false)) {
            utility::LogWarning("Write GLTF failed.");
            return false;
        }
    }

    return true;
}

}  // namespace io
}  // namespace cloudViewer
