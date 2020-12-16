// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#pragma once

#include "eCV_io.h"
#include <memory>
#include <string>
#include <vector>

class ccMesh;
namespace cloudViewer {
namespace io {

/// Factory function to create a mesh from a file (TriangleMeshFactory.cpp)
/// Return an empty mesh if fail to read the file.
std::shared_ptr<ccMesh> ECV_IO_LIB_API CreateMeshFromFile(const std::string &filename,
                                                          bool print_progress = false);

/// The general entrance for reading a TriangleMesh from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ECV_IO_LIB_API ReadTriangleMesh(const std::string &filename,
                                     ccMesh &mesh,
                                     bool print_progress = false);

/// The general entrance for writing a TriangleMesh to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports binary encoding and compression, the later
/// two parameters will be used. Otherwise they will be ignored.
/// At current only .obj format supports uv coordinates (triangle_uvs) and
/// textures.
/// \return return true if the write function is successful, false otherwise.
bool ECV_IO_LIB_API WriteTriangleMesh(const std::string &filename,
                                      const ccMesh &mesh,
                                      bool write_ascii = false,
                                      bool compressed = false,
                                      bool write_vertex_normals = true,
                                      bool write_vertex_colors = true,
                                      bool write_triangle_uvs = true,
                                      bool print_progress = false);

bool ECV_IO_LIB_API ReadTriangleMeshFromPLY(const std::string &filename,
                                            ccMesh &mesh,
                                            bool print_progress);

bool ECV_IO_LIB_API WriteTriangleMeshToPLY(const std::string &filename,
                                           const ccMesh &mesh,
                                           bool write_ascii,
                                           bool compressed,
                                           bool write_vertex_normals,
                                           bool write_vertex_colors,
                                           bool write_triangle_uvs,
                                           bool print_progress);

bool ECV_IO_LIB_API ReadTriangleMeshFromSTL(const std::string &filename,
                                            ccMesh &mesh,
                                            bool print_progress);

bool ECV_IO_LIB_API WriteTriangleMeshToSTL(const std::string &filename,
                                           const ccMesh &mesh,
                                           bool write_ascii,
                                           bool compressed,
                                           bool write_vertex_normals,
                                           bool write_vertex_colors,
                                           bool write_triangle_uvs,
                                           bool print_progress);

bool ECV_IO_LIB_API ReadTriangleMeshFromOBJ(const std::string &filename,
                                            ccMesh &mesh,
                                            bool print_progress);

bool ECV_IO_LIB_API WriteTriangleMeshToOBJ(const std::string &filename,
                                           const ccMesh &mesh,
                                           bool write_ascii,
                                           bool compressed,
                                           bool write_vertex_normals,
                                           bool write_vertex_colors,
                                           bool write_triangle_uvs,
                                           bool print_progress);

bool ECV_IO_LIB_API ReadTriangleMeshFromOFF(const std::string &filename,
                                            ccMesh &mesh,
                                            bool print_progress);

bool ECV_IO_LIB_API WriteTriangleMeshToOFF(const std::string &filename,
                                           const ccMesh &mesh,
                                           bool write_ascii,
                                           bool compressed,
                                           bool write_vertex_normals,
                                           bool write_vertex_colors,
                                           bool write_triangle_uvs,
                                           bool print_progress);

bool ECV_IO_LIB_API ReadTriangleMeshFromGLTF(const std::string &filename,
                                             ccMesh &mesh,
                                             bool print_progress);

bool ECV_IO_LIB_API WriteTriangleMeshToGLTF(const std::string &filename,
                                            const ccMesh &mesh,
                                            bool write_ascii,
                                            bool compressed,
                                            bool write_vertex_normals,
                                            bool write_vertex_colors,
                                            bool write_triangle_uvs,
                                            bool print_progress);

/// Function to convert a polygon into a collection of
/// triangles whose vertices are only those of the polygon.
/// Assume that the vertices are connected by edges based on their order, and
/// the final vertex connected to the first.
/// The triangles are added to the mesh that is passed as reference. The mesh
/// should contain all vertices prior to calling this function.
/// \return return true if triangulation is successful, false otherwise.
bool ECV_IO_LIB_API AddTrianglesByEarClipping(ccMesh &mesh,
                                              std::vector<unsigned int> &indices);

}  // namespace io
}  // namespace cloudViewer
