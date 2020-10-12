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
#include <Console.h>
#include <FileSystem.h>

// ECV_DB_LIB
#include <ecvPointCloud.h>

// ECV_IO_LIB
#include <AutoIO.h>

// SYSTEM
#include <iostream>
#include <unordered_map>

namespace cloudViewer {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, ccPointCloud &, bool)>>
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
                                                   const bool,
                                                   const bool,
                                                   const bool)>>
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

}  // unnamed namespace

namespace io {

std::shared_ptr<ccPointCloud> CreatePointCloudFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<ccPointCloud>("pointCloud");
    ReadPointCloud(filename, *pointcloud, format, print_progress);
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    ccPointCloud &pointcloud,
                    const std::string &format,
                    bool remove_nan_points,
                    bool remove_infinite_points,
                    bool print_progress) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                CVLib::utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }

    std::cout << "Format = " << format << std::endl;
    std::cout << "Extension = " << filename_ext << std::endl;

    if (filename_ext.empty()) {
        CVLib::utility::LogWarning(
                "Read ccPointCloud failed: unknown file extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_pointcloud_read_function.find(filename_ext);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        CVLib::utility::LogWarning(
                "Read ccPointCloud failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, print_progress);
    CVLib::utility::LogDebug("Read ccPointCloud: {:d} vertices.",
                      (int)pointcloud.size());
    if (remove_nan_points || remove_infinite_points) {
        pointcloud.removeNonFinitePoints(remove_nan_points,
                                         remove_infinite_points);
    }
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const ccPointCloud &pointcloud,
                     bool write_ascii /* = false*/,
                     bool compressed /* = false*/,
                     bool print_progress) {
    std::string filename_ext =
            CVLib::utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        CVLib::utility::LogWarning(
                "Write ccPointCloud failed: unknown file extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_pointcloud_write_function.find(filename_ext);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        CVLib::utility::LogWarning(
                "Write ccPointCloud failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, write_ascii,
                                   compressed, print_progress);
    CVLib::utility::LogDebug("Write ccPointCloud: {:d} vertices.",
                      (int)pointcloud.size());
    return success;
}

}  // namespace io
}  // namespace cloudViewer
