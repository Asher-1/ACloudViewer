// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>
#include <functional>

#include "CV_io.h"

/**
 * @brief File I/O utility class for managing application metadata
 * 
 * Provides static methods to set and retrieve information about the
 * application that writes/reads files, including version and timestamp.
 */
class FileIO {
public:
    /**
     * @brief Set writer information
     * @param applicationName Name of the application
     * @param version Version string of the application
     */
    CV_IO_LIB_API static void setWriterInfo(const QString &applicationName,
                                            const QString &version);
    
    /**
     * @brief Get complete writer information string
     * @return Writer information including application name and version
     */
    CV_IO_LIB_API static QString writerInfo();

    /**
     * @brief Get application name
     * @return Application name
     */
    CV_IO_LIB_API static QString applicationName();
    
    /**
     * @brief Get application version
     * @return Version string
     */
    CV_IO_LIB_API static QString version();

    /**
     * @brief Get "created by" information
     * @return Creator information string
     */
    CV_IO_LIB_API static QString createdBy();
    
    /**
     * @brief Get creation date and time
     * @return Creation timestamp as string
     */
    CV_IO_LIB_API static QString createdDateTime();

private:
    FileIO() = delete;  ///< Deleted constructor (static-only class)

    static QString s_applicationName;  ///< Application name
    static QString s_version;          ///< Version string
    static QString s_writerInfo;       ///< Writer information
};

namespace cloudViewer {
namespace io {
/**
 * @struct ReadPointCloudOption
 * @brief Optional parameters for reading point clouds
 * 
 * Provides configuration options for reading point cloud data from files,
 * including format specification, NaN/inf filtering, and progress tracking.
 */
struct CV_IO_LIB_API ReadPointCloudOption {
    /**
     * @brief Constructor with full options
     * @param format File format ("auto" for auto-detection based on extension)
     * @param remove_nan_points Remove NaN points (default: false)
     * @param remove_infinite_points Remove infinite points (default: false)
     * @param print_progress Print progress to stdout (default: false)
     * @param update_progress Progress callback function
     * @note When updating defaults, also update docstrings in pybind/io/class_io.cpp
     */
    ReadPointCloudOption(
            std::string format = "auto",
            bool remove_nan_points = false,
            bool remove_infinite_points = false,
            bool print_progress = false,
            std::function<bool(double)> update_progress = {})
        : format(format),
          remove_nan_points(remove_nan_points),
          remove_infinite_points(remove_infinite_points),
          print_progress(print_progress),
          update_progress(update_progress) {};
    
    /**
     * @brief Constructor with progress callback only
     * @param up Progress update callback
     */
    ReadPointCloudOption(std::function<bool(double)> up)
        : ReadPointCloudOption() {
        update_progress = up;
    };
    
    std::string format;                             ///< File format ("auto" for auto-detection)
    bool remove_nan_points;                         ///< Remove NaN points
    bool remove_infinite_points;                    ///< Remove infinite points
    bool print_progress;                            ///< Print progress to stdout
    std::function<bool(double)> update_progress;    ///< Progress callback (0-100%, return false to cancel)
};

/**
 * @struct WritePointCloudOption
 * @brief Optional parameters for writing point clouds
 * 
 * Provides configuration options for writing point cloud data to files,
 * including format, ASCII/binary mode, compression, and progress tracking.
 */
struct CV_IO_LIB_API WritePointCloudOption {
    /**
     * @brief ASCII or binary output mode
     */
    enum class IsAscii : bool { Binary = false, Ascii = true };
    
    /**
     * @brief Compression mode
     */
    enum class Compressed : bool { Uncompressed = false, Compressed = true };
    
    /**
     * @brief Constructor with full options
     * @param format File format ("auto" for auto-detection based on extension)
     * @param write_ascii ASCII or binary mode (default: Binary)
     * @param compressed Compression mode (default: Uncompressed)
     * @param print_progress Print progress to stdout (default: false)
     * @param update_progress Progress callback function
     * @note When updating defaults, also update docstrings in pybind/io/class_io.cpp
     */
    WritePointCloudOption(
            std::string format = "auto",
            IsAscii write_ascii = IsAscii::Binary,
            Compressed compressed = Compressed::Uncompressed,
            bool print_progress = false,
            std::function<bool(double)> update_progress = {})
        : format(format),
          write_ascii(write_ascii),
          compressed(compressed),
          print_progress(print_progress),
          update_progress(update_progress) {};
    
    /**
     * @brief Constructor for compatibility (bool parameters)
     * @param write_ascii ASCII mode flag
     * @param compressed Compression flag (default: false)
     * @param print_progress Print progress (default: false)
     * @param update_progress Progress callback
     */
    WritePointCloudOption(bool write_ascii,
                          bool compressed = false,
                          bool print_progress = false,
                          std::function<bool(double)> update_progress = {})
        : write_ascii(IsAscii(write_ascii)),
          compressed(Compressed(compressed)),
          print_progress(print_progress),
          update_progress(update_progress) {};
    
    /**
     * @brief Constructor for compatibility (format + bool parameters)
     * @param format File format
     * @param write_ascii ASCII mode flag
     * @param compressed Compression flag (default: false)
     * @param print_progress Print progress (default: false)
     * @param update_progress Progress callback
     */
    WritePointCloudOption(std::string format,
                          bool write_ascii,
                          bool compressed = false,
                          bool print_progress = false,
                          std::function<bool(double)> update_progress = {})
        : format(format),
          write_ascii(IsAscii(write_ascii)),
          compressed(Compressed(compressed)),
          print_progress(print_progress),
          update_progress(update_progress) {};
    
    /**
     * @brief Constructor with progress callback only
     * @param up Progress update callback
     */
    WritePointCloudOption(std::function<bool(double)> up)
        : WritePointCloudOption() {
        update_progress = up;
    };
    
    std::string format;                             ///< File format ("auto" for auto-detection)
    IsAscii write_ascii;                            ///< ASCII or binary mode
    Compressed compressed;                          ///< Compression mode (PCD only)
    bool print_progress;                            ///< Print progress to stdout
    std::function<bool(double)> update_progress;    ///< Progress callback (0-100%, return false to cancel)
};
}  // namespace io
}  // namespace cloudViewer
