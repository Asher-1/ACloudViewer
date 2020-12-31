// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "t/io/PointCloudIO.h"
#include "io/PointCloudIO.h"

#include <iostream>
#include <unordered_map>

#include <Helper.h>
#include <Console.h>
#include <FileSystem.h>
#include <ProgressReporters.h>

namespace cloudViewer {
namespace t {
namespace io {

    using namespace CVLib;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           geometry::PointCloud&,
                           const cloudViewer::io::ReadPointCloudOption &)>>
        file_extension_to_pointcloud_read_function{
                {"xyzi", ReadPointCloudFromXYZI},
                {"ply", ReadPointCloudFromPLY},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const geometry::PointCloud &,
                           const cloudViewer::io::WritePointCloudOption &)>>
        file_extension_to_pointcloud_write_function{
                {"xyzi", WritePointCloudToXYZI},
                {"ply", WritePointCloudToPLY},
        };

std::shared_ptr<geometry::PointCloud> CreatetPointCloudFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    ReadPointCloud(filename, *pointcloud, {format, true, true, print_progress});
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    geometry::PointCloud &pointcloud,
                    const cloudViewer::io::ReadPointCloudOption &params) {
    std::string format = params.format;
    if (format == "auto") {
        format = utility::filesystem::GetFileExtensionInLowerCase(filename);
    }

    utility::LogDebug("Format {} File {}", params.format, filename);

    bool success = false;
    auto map_itr = file_extension_to_pointcloud_read_function.find(format);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        ccPointCloud legacy_pointcloud;
        success =
                cloudViewer::io::ReadPointCloud(filename, legacy_pointcloud, params);
        if (!success) return false;
        pointcloud = geometry::PointCloud::FromLegacyPointCloud(
                legacy_pointcloud, core::Dtype::Float64);
    } else {
        success = map_itr->second(filename, pointcloud, params);
        utility::LogDebug("Read geometry::PointCloud: {:d} vertices.",
                          (int)pointcloud.GetPoints().GetLength());
        if (params.remove_nan_points || params.remove_infinite_points) {
            utility::LogError(
                    "remove_nan_points and remove_infinite_points options are "
                    "unimplemented.");
            return false;
        }
    }
    return success;
}

bool ReadPointCloud(const std::string &filename,
                    geometry::PointCloud &pointcloud,
                    const std::string &file_format,
                    bool remove_nan_points,
                    bool remove_infinite_points,
                    bool print_progress) {
    std::string format = file_format;
    if (format == "auto") {
        format = utility::filesystem::GetFileExtensionInLowerCase(filename);
    }

    cloudViewer::io::ReadPointCloudOption p;
    p.format = format;
    p.remove_nan_points = remove_nan_points;
    p.remove_infinite_points = remove_infinite_points;
    utility::ConsoleProgressUpdater progress_updater(
            std::string("Reading ") + utility::ToUpper(format) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;
    return ReadPointCloud(filename, pointcloud, p);
}

bool WritePointCloud(const std::string &filename,
                     const geometry::PointCloud &pointcloud,
                     const cloudViewer::io::WritePointCloudOption &params) {
    std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    auto map_itr = file_extension_to_pointcloud_write_function.find(format);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        return cloudViewer::io::WritePointCloud(
                filename, pointcloud.ToLegacyPointCloud(), params);
    }

    bool success = map_itr->second(filename, pointcloud, params);
    utility::LogDebug("Write geometry::PointCloud: {:d} vertices.",
                      (int)pointcloud.GetPoints().GetLength());
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const geometry::PointCloud &pointcloud,
                     bool write_ascii /* = false*/,
                     bool compressed /* = false*/,
                     bool print_progress) {
    cloudViewer::io::WritePointCloudOption p;
    p.write_ascii = cloudViewer::io::WritePointCloudOption::IsAscii(write_ascii);
    p.compressed = cloudViewer::io::WritePointCloudOption::Compressed(compressed);
    std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    utility::ConsoleProgressUpdater progress_updater(
            std::string("Writing ") + utility::ToUpper(format) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;
    return WritePointCloud(filename, pointcloud, p);
}

}  // namespace io
}  // namespace t
}  // namespace cloudViewer
