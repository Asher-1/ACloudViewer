// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
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

#include "OctreeIO.h"
#include "IJsonConvertibleIO.h"

#include <Logging.h>
#include <FileSystem.h>

#include <unordered_map>

namespace cloudViewer {
namespace io {
static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::Octree &)>>
        file_extension_to_octree_read_function{
                {"json", ReadOctreeFromJson},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const geometry::Octree &)>>
        file_extension_to_octree_write_function{
                {"json", WriteOctreeToJson},
        };

std::shared_ptr<geometry::Octree> CreateOctreeFromFile(
        const std::string &filename, const std::string &format) {
    auto octree = cloudViewer::make_shared<geometry::Octree>();
    WriteOctree(filename, *octree);
    return octree;
}

bool ReadOctree(const std::string &filename,
                geometry::Octree &octree,
                const std::string &format) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::Octree failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_octree_read_function.find(filename_ext);
    if (map_itr == file_extension_to_octree_read_function.end()) {
        utility::LogWarning(
                "Read geometry::Octree failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, octree);
    utility::LogDebug("Read geometry::Octree.");
    return success;
}

bool WriteOctree(const std::string &filename, const geometry::Octree &octree) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::Octree failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_octree_write_function.find(filename_ext);
    if (map_itr == file_extension_to_octree_write_function.end()) {
        utility::LogWarning(
                "Write geometry::Octree failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, octree);
    utility::LogDebug("Write geometry::Octree.");
    return success;
}

bool ReadOctreeFromJson(const std::string &filename, geometry::Octree &octree) {
    return ReadIJsonConvertible(filename, octree);
}

bool WriteOctreeToJson(const std::string &filename,
                       const geometry::Octree &octree) {
    return WriteIJsonConvertibleToJSON(filename, octree);
}
}  // namespace io
}  // namespace cloudViewer
