// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef FILEIO_H
#define FILEIO_H

// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                               #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: CLOUDVIEWER  project                               #
// #                                                                        #
// ##########################################################################

#include <QString>
#include <functional>

#include "eCV_io.h"

class FileIO {
public:
    ECV_IO_LIB_API static void setWriterInfo(const QString &applicationName,
                                             const QString &version);
    ECV_IO_LIB_API static QString writerInfo();

    ECV_IO_LIB_API static QString applicationName();
    ECV_IO_LIB_API static QString version();

    ECV_IO_LIB_API static QString createdBy();
    ECV_IO_LIB_API static QString createdDateTime();

private:
    FileIO() = delete;

    static QString s_applicationName;
    static QString s_version;
    static QString s_writerInfo;
};

namespace cloudViewer {
namespace io {
/// \struct ReadPointCloudOption
/// \brief Optional parameters to ReadPointCloud
struct ECV_IO_LIB_API ReadPointCloudOption {
    ReadPointCloudOption(
            // Attention: when you update the defaults, update the docstrings in
            // pybind/io/class_io.cpp
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
    ReadPointCloudOption(std::function<bool(double)> up)
        : ReadPointCloudOption() {
        update_progress = up;
    };
    /// Specifies what format the contents of the file are (and what loader to
    /// use), default "auto" means to go off of file extension.
    /// Note: "auto" is incompatible when reading directly from memory.
    std::string format;
    /// Whether to remove all points that have nan
    bool remove_nan_points;
    /// Whether to remove all points that have +-inf
    bool remove_infinite_points;
    /// Print progress to stdout about loading progress.
    /// Also see \p update_progress if you want to have your own progress
    /// indicators or to be able to cancel loading.
    bool print_progress;
    /// Callback to invoke as reading is progressing, parameter is percentage
    /// completion (0.-100.) return true indicates to continue loading, false
    /// means to try to stop loading and cleanup
    std::function<bool(double)> update_progress;
};

/// \struct WritePointCloudOption
/// \brief Optional parameters to WritePointCloud
struct ECV_IO_LIB_API WritePointCloudOption {
    enum class IsAscii : bool { Binary = false, Ascii = true };
    enum class Compressed : bool { Uncompressed = false, Compressed = true };
    WritePointCloudOption(
            // Attention: when you update the defaults, update the docstrings in
            // pybind/io/class_io.cpp
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
    // for compatibility
    WritePointCloudOption(bool write_ascii,
                          bool compressed = false,
                          bool print_progress = false,
                          std::function<bool(double)> update_progress = {})
        : write_ascii(IsAscii(write_ascii)),
          compressed(Compressed(compressed)),
          print_progress(print_progress),
          update_progress(update_progress) {};
    // for compatibility
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
    WritePointCloudOption(std::function<bool(double)> up)
        : WritePointCloudOption() {
        update_progress = up;
    };
    /// Specifies what format the contents of the file are (and what writer to
    /// use), default "auto" means to go off of file extension.
    /// Note: "auto" is incompatible when reading directly from memory.
    std::string format;
    /// Whether to save in Ascii or Binary.  Some savers are capable of doing
    /// either, other ignore this.
    IsAscii write_ascii;
    /// Whether to save Compressed or Uncompressed.  Currently, only PCD is
    /// capable of compressing, and only if using IsAscii::Binary, all other
    /// formats ignore this.
    Compressed compressed;
    /// Print progress to stdout about loading progress.  Also see
    /// \p update_progress if you want to have your own progress indicators or
    /// to be able to cancel loading.
    bool print_progress;
    /// Callback to invoke as reading is progressing, parameter is percentage
    /// completion (0.-100.) return true indicates to continue loading, false
    /// means to try to stop loading and cleanup
    std::function<bool(double)> update_progress;
};
}  // namespace io
}  // namespace cloudViewer

#endif
