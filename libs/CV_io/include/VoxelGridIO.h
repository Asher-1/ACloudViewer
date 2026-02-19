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

/**
 * @file VoxelGridIO.h
 * @brief Voxel grid file I/O utilities
 * 
 * Provides functions for reading and writing voxel grids, which are
 * 3D regular grids where each cell (voxel) can store occupancy or other data.
 * Voxel grids are commonly used for volumetric representations and collision detection.
 */

namespace cloudViewer {
namespace io {

/**
 * @brief Factory function to create voxel grid from file
 * 
 * Automatically detects file format from extension.
 * @param filename Input file path
 * @param format Format hint ("auto" for auto-detection)
 * @param print_progress Print loading progress
 * @return Shared pointer to loaded voxel grid (empty if loading fails)
 */
std::shared_ptr<geometry::VoxelGrid> CV_IO_LIB_API
CreateVoxelGridFromFile(const std::string &filename,
                        const std::string &format = "auto",
                        bool print_progress = false);

/**
 * @brief Read voxel grid from file (general entrance)
 * 
 * Automatically selects appropriate reader based on file extension.
 * @param filename Input file path
 * @param voxelgrid Output voxel grid object
 * @param format Format hint ("auto" for auto-detection)
 * @param print_progress Print loading progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadVoxelGrid(const std::string &filename,
                                 geometry::VoxelGrid &voxelgrid,
                                 const std::string &format = "auto",
                                 bool print_progress = false);

/**
 * @brief Write voxel grid to file (general entrance)
 * 
 * Automatically selects appropriate writer based on file extension.
 * Binary/compression parameters are used if supported by the format.
 * @param filename Output file path
 * @param voxelgrid Voxel grid to write
 * @param write_ascii Write in ASCII format (if supported)
 * @param compressed Use compression (if supported)
 * @param print_progress Print saving progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteVoxelGrid(const std::string &filename,
                                  const geometry::VoxelGrid &voxelgrid,
                                  bool write_ascii = false,
                                  bool compressed = false,
                                  bool print_progress = false);

/**
 * @brief Read voxel grid from PLY file
 * @param filename Input PLY file path
 * @param voxelgrid Output voxel grid object
 * @param print_progress Print loading progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadVoxelGridFromPLY(const std::string &filename,
                                        geometry::VoxelGrid &voxelgrid,
                                        bool print_progress = false);

/**
 * @brief Write voxel grid to PLY file
 * @param filename Output PLY file path
 * @param voxelgrid Voxel grid to write
 * @param write_ascii Write in ASCII format
 * @param compressed Use compression
 * @param print_progress Print saving progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteVoxelGridToPLY(const std::string &filename,
                                       const geometry::VoxelGrid &voxelgrid,
                                       bool write_ascii = false,
                                       bool compressed = false,
                                       bool print_progress = false);

}  // namespace io
}  // namespace cloudViewer
