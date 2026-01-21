// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <VoxelGrid.h>

#include <string>

#include "CV_io.h"

namespace cloudViewer {
namespace io {

/// Factory function to create a voxelgrid from a file.
/// \return return an empty voxelgrid if fail to read the file.
std::shared_ptr<geometry::VoxelGrid> CV_IO_LIB_API
CreateVoxelGridFromFile(const std::string &filename,
                        const std::string &format = "auto",
                        bool print_progress = false);

/// The general entrance for reading a VoxelGrid from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool CV_IO_LIB_API ReadVoxelGrid(const std::string &filename,
                                 geometry::VoxelGrid &voxelgrid,
                                 const std::string &format = "auto",
                                 bool print_progress = false);

/// The general entrance for writing a VoxelGrid to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports binary encoding and compression, the later
/// two parameters will be used. Otherwise they will be ignored.
/// \return return true if the write function is successful, false otherwise.
bool CV_IO_LIB_API WriteVoxelGrid(const std::string &filename,
                                  const geometry::VoxelGrid &voxelgrid,
                                  bool write_ascii = false,
                                  bool compressed = false,
                                  bool print_progress = false);

bool CV_IO_LIB_API ReadVoxelGridFromPLY(const std::string &filename,
                                        geometry::VoxelGrid &voxelgrid,
                                        bool print_progress = false);

bool CV_IO_LIB_API WriteVoxelGridToPLY(const std::string &filename,
                                       const geometry::VoxelGrid &voxelgrid,
                                       bool write_ascii = false,
                                       bool compressed = false,
                                       bool print_progress = false);

}  // namespace io
}  // namespace cloudViewer
