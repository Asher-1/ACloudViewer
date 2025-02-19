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

#pragma once

#include <memory>
#include <string>

#include "FileIO.h"

class ccMesh;
class ccHObject;

//! Generic file read and write utility for python interface
/** Gives static access to file loader and writer
 **/
namespace cloudViewer {
namespace io {

struct ReadTriangleMeshOptions {
    /// Enables post-processing on the mesh
    bool enable_post_processing = false;
    /// Print progress to stdout about loading progress.
    /// Also see \p update_progress if you want to have your own progress
    /// indicators or to be able to cancel loading.
    bool print_progress = false;
    /// Callback to invoke as reading is progressing, parameter is percentage
    /// completion (0.-100.) return true indicates to continue loading, false
    /// means to try to stop loading and cleanup
    std::function<bool(double)> update_progress;
};

/// Factory function to create a ccHObject from a file.
/// \return return an empty ccHObject if fail to read the file.
std::shared_ptr<ccHObject> ECV_IO_LIB_API
CreateEntityFromFile(const std::string& filename,
                     const std::string& format = "auto",
                     bool print_progress = false);

/// The general entrance for reading a ccHObject from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ECV_IO_LIB_API ReadEntity(const std::string& filename,
                               ccHObject& obj,
                               const std::string& format = "auto",
                               bool print_progress = false);

/// The general entrance for writing a ccHObject to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports binary encoding and compression, the later
/// two parameters will be used. Otherwise they will be ignored.
/// \return return true if the write function is successful, false otherwise.
bool ECV_IO_LIB_API WriteEntity(const std::string& filename,
                                const ccHObject& obj,
                                bool write_ascii = false,
                                bool compressed = false,
                                bool print_progress = false);

bool ECV_IO_LIB_API AutoReadEntity(const std::string& filename,
                                   ccHObject& entity,
                                   const ReadPointCloudOption& params);

bool ECV_IO_LIB_API AutoWriteEntity(const std::string& filename,
                                    const ccHObject& entity,
                                    const WritePointCloudOption& params);

bool ECV_IO_LIB_API AutoReadMesh(const std::string& filename,
                                 ccMesh& mesh,
                                 const ReadTriangleMeshOptions& params = {});

bool ECV_IO_LIB_API AutoWriteMesh(const std::string& filename,
                                  const ccMesh& mesh,
                                  bool write_ascii = false,
                                  bool compressed = false,
                                  bool write_vertex_normals = true,
                                  bool write_vertex_colors = true,
                                  bool write_triangle_uvs = true,
                                  bool print_progress = false);

}  // namespace io
}  // namespace cloudViewer
