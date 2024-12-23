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

#include "io/PoseGraphIO.h"

#include <unordered_map>

#include <Logging.h>
#include <FileSystem.h>
#include <IJsonConvertibleIO.h>

namespace cloudViewer {

namespace {
using namespace io;

bool ReadPoseGraphFromJSON(const std::string &filename,
                           pipelines::registration::PoseGraph &pose_graph) {
    return ReadIJsonConvertible(filename, pose_graph);
}

bool WritePoseGraphToJSON(const std::string &filename,
                          const pipelines::registration::PoseGraph &pose_graph) {
    return WriteIJsonConvertibleToJSON(filename, pose_graph);
}

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, pipelines::registration::PoseGraph &)>>
        file_extension_to_pose_graph_read_function{
                {"json", ReadPoseGraphFromJSON},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const pipelines::registration::PoseGraph &)>>
        file_extension_to_pose_graph_write_function{
                {"json", WritePoseGraphToJSON},
        };

}  // unnamed namespace

namespace io {
using namespace cloudViewer;

std::shared_ptr<pipelines::registration::PoseGraph> CreatePoseGraphFromFile(
        const std::string &filename) {
    auto pose_graph = cloudViewer::make_shared<pipelines::registration::PoseGraph>();
    ReadPoseGraph(filename, *pose_graph);
    return pose_graph;
}

bool ReadPoseGraph(const std::string &filename,
                   pipelines::registration::PoseGraph &pose_graph) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read pipelines::registration::PoseGraph failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_pose_graph_read_function.find(filename_ext);
    if (map_itr == file_extension_to_pose_graph_read_function.end()) {
        utility::LogWarning(
                "Read pipelines::registration::PoseGraph failed: unknown file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, pose_graph);
}

bool WritePoseGraph(const std::string &filename,
                    const pipelines::registration::PoseGraph &pose_graph) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write pipelines::registration::PoseGraph failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_pose_graph_write_function.find(filename_ext);
    if (map_itr == file_extension_to_pose_graph_write_function.end()) {
        utility::LogWarning(
                "Write pipelines::registration::PoseGraph failed: unknown file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, pose_graph);
}

}  // namespace io
}  // namespace cloudViewer
