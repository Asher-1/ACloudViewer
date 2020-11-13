// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
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

#include "PointCloudIO.h"

// CV_CORE_LIB
#include <Helper.h>
#include <Console.h>
#include <FileSystem.h>
#include <ProgressReporters.h>

// ECV_DB_LIB
#include <ecvPointCloud.h>

// ECV_IO_LIB
#include <AutoIO.h>

// SYSTEM
#include <iostream>
#include <unordered_map>

namespace cloudViewer {
namespace io {
    using namespace CVLib;
static const std::unordered_map<std::string,
        std::function<bool(const std::string &, 
                           ccPointCloud &, 
                           const ReadPointCloudOption&)>>
        file_extension_to_pointcloud_read_function{
				{"xyz", ReadPointCloudFromXYZ},
				{"txt", ReadPointCloudFromXYZ},
				{"xyzn", ReadPointCloudFromXYZN},
				{"xyzrgb", ReadPointCloudFromXYZRGB},
				{"ply", ReadPointCloudFromPLY},
				{"pcd", ReadPointCloudFromPCD},
				{"pts", ReadPointCloudFromPTS},
                {"ptx", AutoReadEntity},
				{"vtk", AutoReadEntity},
                {"bin", AutoReadEntity},
        };

static const std::unordered_map<std::string,
        std::function<bool(const std::string &,
                           const ccPointCloud &,
                           const WritePointCloudOption&)>>
        file_extension_to_pointcloud_write_function{
				{"xyz", WritePointCloudToXYZ},
				{"txt", WritePointCloudToXYZ},
				{"xyzn", WritePointCloudToXYZN},
				{"xyzrgb", WritePointCloudToXYZRGB},
				{"ply", WritePointCloudToPLY},
				{"pcd", WritePointCloudToPCD},
				{"pts", WritePointCloudToPTS},
                {"ptx", AutoWriteEntity},
                {"vtk", AutoWriteEntity},
                {"bin", AutoWriteEntity},
        };
std::shared_ptr<ccPointCloud> CreatePointCloudFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<ccPointCloud>("pointCloud");
    ReadPointCloud(filename, *pointcloud, { format, true, true, print_progress });
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    ccPointCloud &pointcloud,
                    const ReadPointCloudOption& params) {
    std::string format = params.format;
    if (format == "auto") {
        format = CVLib::utility::filesystem::GetFileExtensionInLowerCase(filename);
    }

    utility::LogDebug("Format {} File {}", params.format, filename);

    auto map_itr = file_extension_to_pointcloud_read_function.find(format);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        utility::LogWarning(
            "Read ccPointCloud failed: unknown file extension for "
            "{} (format: {}).",
            filename, params.format);
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, params);
    CVLib::utility::LogDebug("Read ccPointCloud: {:d} vertices.",
                             pointcloud.size());
    if (params.remove_nan_points || params.remove_infinite_points) {
        pointcloud.removeNonFinitePoints(params.remove_nan_points,
            params.remove_infinite_points);
    }
    return success;
}

bool ReadPointCloud(const std::string& filename,
                    ccPointCloud& pointcloud,
                    const std::string& file_format,
                    bool remove_nan_points,
                    bool remove_infinite_points,
                    bool print_progress) {
    std::string format = file_format;
    if (format == "auto") {
        format = utility::filesystem::GetFileExtensionInLowerCase(filename);
    }

    ReadPointCloudOption p;
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

bool WritePointCloud(const std::string& filename,
    const ccPointCloud& pointcloud,
    const WritePointCloudOption& params) {
    std::string format =
        utility::filesystem::GetFileExtensionInLowerCase(filename);
    auto map_itr = file_extension_to_pointcloud_write_function.find(format);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        utility::LogWarning(
            "Write ccPointCloud failed: unknown file extension {} "
            "for file {}.",
            format, filename);
        return false;
    }

    bool success = map_itr->second(filename, pointcloud, params);
    CVLib::utility::LogDebug("Write ccPointCloud: {:d} vertices.",
                             pointcloud.size());
    return success;
}

bool WritePointCloud(const std::string& filename,
    const ccPointCloud& pointcloud,
    bool write_ascii /* = false*/,
    bool compressed /* = false*/,
    bool print_progress) {
    WritePointCloudOption p;
    p.write_ascii = WritePointCloudOption::IsAscii(write_ascii);
    p.compressed = WritePointCloudOption::Compressed(compressed);
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
}  // namespace cloudViewer
