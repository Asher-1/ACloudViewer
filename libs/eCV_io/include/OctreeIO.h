// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "eCV_io.h"

#include <string>

#include <Octree.h>

namespace cloudViewer {
    namespace geometry {

    }
namespace io {
/// Factory function to create a octree from a file.
/// \return return an empty octree if fail to read the file.
std::shared_ptr<geometry::Octree> ECV_IO_LIB_API CreateOctreeFromFile(
        const std::string &filename, const std::string &format = "auto");

/// The general entrance for reading a Octree from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ECV_IO_LIB_API ReadOctree(const std::string &filename,
                               geometry::Octree &octree,
                               const std::string &format = "auto");

/// The general entrance for writing a Octree to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports binary encoding and compression, the later
/// two parameters will be used. Otherwise they will be ignored.
/// \return return true if the write function is successful, false otherwise.
bool ECV_IO_LIB_API WriteOctree(const std::string &filename, const geometry::Octree &octree);

bool ECV_IO_LIB_API ReadOctreeFromJson(const std::string &filename, geometry::Octree &octree);

bool ECV_IO_LIB_API WriteOctreeToJson(const std::string &filename,
                                      const geometry::Octree &octree);

}  // namespace io
}  // namespace cloudViewer
