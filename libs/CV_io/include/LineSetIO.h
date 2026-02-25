// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <LineSet.h>

#include <string>

#include "CV_io.h"

/**
 * @file LineSetIO.h
 * @brief LineSet file I/O utilities
 *
 * Provides functions for reading and writing line set geometries,
 * which represent collections of connected line segments.
 */

namespace cloudViewer {
namespace io {

/**
 * @brief Factory function to create line set from file
 *
 * Automatically detects file format from extension.
 * @param filename Input file path
 * @param format Format hint ("auto" for auto-detection)
 * @param print_progress Print loading progress
 * @return Shared pointer to loaded line set (empty if loading fails)
 */
std::shared_ptr<geometry::LineSet> CV_IO_LIB_API
CreateLineSetFromFile(const std::string &filename,
                      const std::string &format = "auto",
                      bool print_progress = false);

/**
 * @brief Read line set from file (general entrance)
 *
 * Automatically selects appropriate reader based on file extension.
 * @param filename Input file path
 * @param lineset Output line set object
 * @param format Format hint ("auto" for auto-detection)
 * @param print_progress Print loading progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadLineSet(const std::string &filename,
                               geometry::LineSet &lineset,
                               const std::string &format = "auto",
                               bool print_progress = false);

/**
 * @brief Write line set to file (general entrance)
 *
 * Automatically selects appropriate writer based on file extension.
 * Binary/compression parameters are used if supported by the format.
 * @param filename Output file path
 * @param lineset Line set to write
 * @param write_ascii Write in ASCII format (if supported)
 * @param compressed Use compression (if supported)
 * @param print_progress Print saving progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteLineSet(const std::string &filename,
                                const geometry::LineSet &lineset,
                                bool write_ascii = false,
                                bool compressed = false,
                                bool print_progress = false);

/**
 * @brief Read line set from PLY file
 * @param filename Input PLY file path
 * @param lineset Output line set object
 * @param print_progress Print loading progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadLineSetFromPLY(const std::string &filename,
                                      geometry::LineSet &lineset,
                                      bool print_progress = false);

/**
 * @brief Write line set to PLY file
 * @param filename Output PLY file path
 * @param lineset Line set to write
 * @param write_ascii Write in ASCII format
 * @param compressed Use compression
 * @param print_progress Print saving progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteLineSetToPLY(const std::string &filename,
                                     const geometry::LineSet &lineset,
                                     bool write_ascii = false,
                                     bool compressed = false,
                                     bool print_progress = false);

}  // namespace io
}  // namespace cloudViewer
