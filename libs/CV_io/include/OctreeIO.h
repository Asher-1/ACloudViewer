// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Octree.h>

#include <string>

#include "CV_io.h"

/**
 * @file OctreeIO.h
 * @brief Octree file I/O utilities
 * 
 * Provides functions for reading and writing octree spatial data structures,
 * which are used for efficient spatial partitioning and nearest neighbor searches.
 */

namespace cloudViewer {
namespace geometry {}
namespace io {

/**
 * @brief Factory function to create octree from file
 * 
 * Automatically detects file format from extension.
 * @param filename Input file path
 * @param format Format hint ("auto" for auto-detection)
 * @return Shared pointer to loaded octree (empty if loading fails)
 */
std::shared_ptr<geometry::Octree> CV_IO_LIB_API CreateOctreeFromFile(
        const std::string &filename, const std::string &format = "auto");

/**
 * @brief Read octree from file (general entrance)
 * 
 * Automatically selects appropriate reader based on file extension.
 * @param filename Input file path
 * @param octree Output octree object
 * @param format Format hint ("auto" for auto-detection)
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadOctree(const std::string &filename,
                              geometry::Octree &octree,
                              const std::string &format = "auto");

/**
 * @brief Write octree to file (general entrance)
 * 
 * Automatically selects appropriate writer based on file extension.
 * @param filename Output file path
 * @param octree Octree to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteOctree(const std::string &filename,
                               const geometry::Octree &octree);

/**
 * @brief Read octree from JSON file
 * @param filename Input JSON file path
 * @param octree Output octree object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadOctreeFromJson(const std::string &filename,
                                      geometry::Octree &octree);

/**
 * @brief Write octree to JSON file
 * @param filename Output JSON file path
 * @param octree Octree to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteOctreeToJson(const std::string &filename,
                                     const geometry::Octree &octree);

}  // namespace io
}  // namespace cloudViewer
