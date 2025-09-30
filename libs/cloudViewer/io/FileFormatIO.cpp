// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "io/FileFormatIO.h"

#include <map>

#include <FileSystem.h>

namespace cloudViewer {
namespace io {
static std::map<std::string, FileGeometry (*)(const std::string&)> gExt2Func = {
        {"glb", ReadFileGeometryTypeGLTF},
        {"gltf", ReadFileGeometryTypeGLTF},
        {"obj", ReadFileGeometryTypeOBJ},
        {"fbx", ReadFileGeometryTypeFBX},
        {"off", ReadFileGeometryTypeOFF},
        {"pcd", ReadFileGeometryTypePCD},
        {"ply", ReadFileGeometryTypePLY},
        {"pts", ReadFileGeometryTypePTS},
        {"stl", ReadFileGeometryTypeSTL},
        {"xyz", ReadFileGeometryTypeXYZ},
        {"xyzn", ReadFileGeometryTypeXYZN},
        {"xyzrgb", ReadFileGeometryTypeXYZRGB},
};

FileGeometry ReadFileGeometryType(const std::string& path) {
    auto ext = cloudViewer::utility::filesystem::GetFileExtensionInLowerCase(path);
    auto it = gExt2Func.find(ext);
    if (it != gExt2Func.end()) {
        return it->second(path);
    } else {
        return CONTENTS_UNKNOWN;
    }
}

}  // namespace io
}  // namespace cloudViewer
