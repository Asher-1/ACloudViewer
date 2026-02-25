// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>

#include "FileIO.h"

class ccMesh;
class ccHObject;

/**
 * @file AutoIO.h
 * @brief Automatic file I/O utilities for Python interface
 *
 * Provides high-level, format-agnostic file I/O functions that automatically
 * detect file formats and dispatch to appropriate readers/writers. Designed
 * primarily for Python bindings but also useful for C++ applications.
 */

namespace cloudViewer {
namespace io {

/**
 * @struct ReadTriangleMeshOptions
 * @brief Options for reading triangle meshes
 */
struct ReadTriangleMeshOptions {
    /// Enables post-processing on the mesh (smoothing, validation, etc.)
    bool enable_post_processing = false;

    /// Print progress to stdout about loading progress
    /// Also see \p update_progress if you want custom progress indicators
    /// or the ability to cancel loading
    bool print_progress = false;

    /// Callback invoked as reading progresses
    /// Parameter is percentage completion (0.0-100.0)
    /// Return true to continue loading, false to stop and cleanup
    std::function<bool(double)> update_progress;
};

/**
 * @brief Factory function to create entity from file
 *
 * Automatically detects file format and creates appropriate entity.
 * @param filename Input file path
 * @param format Format hint ("auto" for auto-detection, or specific format)
 * @param print_progress Print loading progress to stdout
 * @return Shared pointer to loaded entity (empty if loading fails)
 */
std::shared_ptr<ccHObject> CV_IO_LIB_API
CreateEntityFromFile(const std::string& filename,
                     const std::string& format = "auto",
                     bool print_progress = false);

/**
 * @brief Read entity from file (general entrance)
 *
 * Automatically selects appropriate reader based on file extension.
 * @param filename Input file path
 * @param obj Output entity object
 * @param format Format hint ("auto" for auto-detection)
 * @param print_progress Print loading progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadEntity(const std::string& filename,
                              ccHObject& obj,
                              const std::string& format = "auto",
                              bool print_progress = false);

/**
 * @brief Write entity to file (general entrance)
 *
 * Automatically selects appropriate writer based on file extension.
 * Binary/compression parameters are used if supported by the format.
 * @param filename Output file path
 * @param obj Entity to write
 * @param write_ascii Write in ASCII format (if supported)
 * @param compressed Use compression (if supported)
 * @param print_progress Print saving progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteEntity(const std::string& filename,
                               const ccHObject& obj,
                               bool write_ascii = false,
                               bool compressed = false,
                               bool print_progress = false);

/**
 * @brief Auto-read entity with custom parameters
 * @param filename Input file path
 * @param entity Output entity
 * @param params Point cloud reading parameters
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API AutoReadEntity(const std::string& filename,
                                  ccHObject& entity,
                                  const ReadPointCloudOption& params);

/**
 * @brief Auto-write entity with custom parameters
 * @param filename Output file path
 * @param entity Entity to write
 * @param params Point cloud writing parameters
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API AutoWriteEntity(const std::string& filename,
                                   const ccHObject& entity,
                                   const WritePointCloudOption& params);

/**
 * @brief Auto-read mesh with options
 * @param filename Input mesh file path
 * @param mesh Output mesh
 * @param params Reading options
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API AutoReadMesh(const std::string& filename,
                                ccMesh& mesh,
                                const ReadTriangleMeshOptions& params = {});

/**
 * @brief Auto-write mesh with detailed options
 * @param filename Output mesh file path
 * @param mesh Mesh to write
 * @param write_ascii Write in ASCII format (if supported)
 * @param compressed Use compression (if supported)
 * @param write_vertex_normals Include vertex normals
 * @param write_vertex_colors Include vertex colors
 * @param write_triangle_uvs Include texture coordinates
 * @param print_progress Print saving progress
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API AutoWriteMesh(const std::string& filename,
                                 const ccMesh& mesh,
                                 bool write_ascii = false,
                                 bool compressed = false,
                                 bool write_vertex_normals = true,
                                 bool write_vertex_colors = true,
                                 bool write_triangle_uvs = true,
                                 bool print_progress = false);

}  // namespace io
}  // namespace cloudViewer
