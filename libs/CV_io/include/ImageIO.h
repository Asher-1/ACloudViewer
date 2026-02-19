// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Image.h>

#include <string>

#include "CV_io.h"

/**
 * @file ImageIO.h
 * @brief Image file I/O utilities
 * 
 * Provides functions for reading and writing images in various formats
 * (PNG, JPEG, etc.), including support for in-memory image data.
 */

namespace cloudViewer {
namespace io {

/**
 * @brief Factory function to create image from file
 * 
 * Automatically detects image format from file extension.
 * @param filename Input image file path
 * @return Shared pointer to loaded image (empty if loading fails)
 */
std::shared_ptr<geometry::Image> CV_IO_LIB_API
CreateImageFromFile(const std::string &filename);

/**
 * @brief Factory function to create image from memory
 * 
 * Decodes image data from memory buffer.
 * @param image_format Image format ("png", "jpg", etc.)
 * @param image_data_ptr Pointer to image data in memory
 * @param image_data_size Size of image data in bytes
 * @return Shared pointer to decoded image (empty if decoding fails)
 */
std::shared_ptr<geometry::Image> CV_IO_LIB_API
CreateImageFromMemory(const std::string &image_format,
                      const unsigned char *image_data_ptr,
                      size_t image_data_size);

/**
 * @brief Read image from file (general entrance)
 * 
 * Automatically selects appropriate reader based on file extension.
 * @param filename Input image file path
 * @param image Output image object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadImage(const std::string &filename,
                             geometry::Image &image);

/**
 * @brief Read image from memory buffer
 * 
 * Decodes image data from memory based on specified format.
 * @param image_format Image format ("png", "jpg", etc.)
 * @param image_data_ptr Pointer to image data in memory
 * @param image_data_size Size of image data in bytes
 * @param image Output image object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadImageFromMemory(const std::string &image_format,
                                       const unsigned char *image_data_ptr,
                                       size_t image_data_size,
                                       geometry::Image &image);

/// Default quality value (-1 = use format-specific defaults)
constexpr int kCloudViewerImageIODefaultQuality = -1;

/**
 * @brief Write image to file (general entrance)
 * 
 * Automatically selects appropriate writer based on file extension.
 * Quality parameter is format-specific:
 * - PNG: [0-9] where <=2 is fast write, >=3 is balanced (default)
 * - JPEG: [0-100] where 70-95 is typical, 90 is default
 * 
 * @param filename Output image file path
 * @param image Image to write
 * @param quality Quality parameter (format-specific, -1 for default)
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteImage(const std::string &filename,
                              const geometry::Image &image,
                              int quality = kCloudViewerImageIODefaultQuality);

/**
 * @brief Read PNG image from file
 * @param filename Input PNG file path
 * @param image Output image object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadImageFromPNG(const std::string &filename,
                                    geometry::Image &image);

/**
 * @brief Read PNG image from memory
 * @param image_data_ptr Pointer to PNG data in memory
 * @param image_data_size Size of PNG data in bytes
 * @param image Output image object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadPNGFromMemory(const unsigned char *image_data_ptr,
                                     size_t image_data_size,
                                     geometry::Image &image);

/**
 * @brief Write image to PNG file
 * @param filename Output PNG file path
 * @param image Image to write
 * @param quality Compression level [0-9] (default: -1)
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API
WriteImageToPNG(const std::string &filename,
                const geometry::Image &image,
                int quality = kCloudViewerImageIODefaultQuality);

/**
 * @brief Read JPEG image from file
 * @param filename Input JPEG file path
 * @param image Output image object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadImageFromJPG(const std::string &filename,
                                    geometry::Image &image);

/**
 * @brief Read JPEG image from memory
 * @param image_data_ptr Pointer to JPEG data in memory
 * @param image_data_size Size of JPEG data in bytes
 * @param image Output image object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadJPGFromMemory(const unsigned char *image_data_ptr,
                                     size_t image_data_size,
                                     geometry::Image &image);

/**
 * @brief Write image to JPEG file
 * @param filename Output JPEG file path
 * @param image Image to write
 * @param quality JPEG quality [0-100] (default: 90)
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API
WriteImageToJPG(const std::string &filename,
                const geometry::Image &image,
                int quality = kCloudViewerImageIODefaultQuality);

}  // namespace io
}  // namespace cloudViewer
