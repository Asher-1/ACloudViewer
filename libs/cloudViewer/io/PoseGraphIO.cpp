// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
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
