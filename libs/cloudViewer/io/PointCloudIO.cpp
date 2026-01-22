// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "io/PointCloudIO.h"

// CV_CORE_LIB
#include <FileSystem.h>
#include <Helper.h>
#include <Logging.h>
#include <ProgressReporters.h>

// CV_DB_LIB
#include <ecvPointCloud.h>

// CV_IO_LIB
#include <AutoIO.h>

// SYSTEM
#include <iostream>
#include <unordered_map>

namespace cloudViewer {
namespace io {
static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           ccPointCloud &,
                           const ReadPointCloudOption &)>>
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

static const std::unordered_map<
        std::string,
        std::function<bool(const unsigned char *,
                           const size_t,
                           ccPointCloud &,
                           const ReadPointCloudOption &)>>
        in_memory_to_pointcloud_read_function{
                {"mem::xyz", ReadPointCloudInMemoryFromXYZ},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const ccPointCloud &,
                           const WritePointCloudOption &)>>
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

static const std::unordered_map<
        std::string,
        std::function<bool(unsigned char *&,
                           size_t &,
                           const ccPointCloud &,
                           const WritePointCloudOption &)>>
        in_memory_to_pointcloud_write_function{
                {"mem::xyz", WritePointCloudInMemoryToXYZ},
        };

std::shared_ptr<ccPointCloud> CreatePointCloudFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<ccPointCloud>("pointCloud");
    ReadPointCloud(filename, *pointcloud, {format, true, true, print_progress});
    return pointcloud;
}

std::shared_ptr<ccPointCloud> CreatePointCloudFromMemory(
        const unsigned char *buffer,
        const size_t length,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<ccPointCloud>();
    ReadPointCloud(buffer, length, *pointcloud,
                   {format, true, true, print_progress});
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    ccPointCloud &pointcloud,
                    const ReadPointCloudOption &params) {
    std::string format = params.format;
    if (format == "auto") {
        format = cloudViewer::utility::filesystem::GetFileExtensionInLowerCase(
                filename);
    }

    cloudViewer::utility::LogDebug("Format {} File {}", params.format,
                                   filename);

    auto map_itr = file_extension_to_pointcloud_read_function.find(format);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        cloudViewer::utility::LogWarning(
                "Read ccPointCloud failed: unknown file extension for "
                "{} (format: {}).",
                filename, params.format);
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, params);
    cloudViewer::utility::LogDebug("Read ccPointCloud: {:d} vertices.",
                                   pointcloud.size());
    if (params.remove_nan_points || params.remove_infinite_points) {
        pointcloud.RemoveNonFinitePoints(params.remove_nan_points,
                                         params.remove_infinite_points);
    }
    return success;
}

bool ReadPointCloud(const std::string &filename,
                    ccPointCloud &pointcloud,
                    const std::string &file_format,
                    bool remove_nan_points,
                    bool remove_infinite_points,
                    bool print_progress) {
    std::string format = file_format;
    if (format == "auto") {
        format = cloudViewer::utility::filesystem::GetFileExtensionInLowerCase(
                filename);
    }

    ReadPointCloudOption p;
    p.format = format;
    p.remove_nan_points = remove_nan_points;
    p.remove_infinite_points = remove_infinite_points;
    cloudViewer::utility::ConsoleProgressUpdater progress_updater(
            std::string("Reading ") + cloudViewer::utility::ToUpper(format) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;
    return ReadPointCloud(filename, pointcloud, p);
}

bool ReadPointCloud(const unsigned char *buffer,
                    const size_t length,
                    ccPointCloud &pointcloud,
                    const ReadPointCloudOption &params) {
    std::string format = params.format;
    if (format == "auto") {
        cloudViewer::utility::LogWarning(
                "Read ccPointCloud failed: unknown format for "
                "(format: {}).",
                params.format);
        return false;
    }

    cloudViewer::utility::LogDebug("Format {}", params.format);

    auto map_itr = in_memory_to_pointcloud_read_function.find(format);
    if (map_itr == in_memory_to_pointcloud_read_function.end()) {
        cloudViewer::utility::LogWarning(
                "Read ccPointCloud failed: unknown format for "
                "(format: {}).",
                params.format);
        return false;
    }
    bool success = map_itr->second(buffer, length, pointcloud, params);
    cloudViewer::utility::LogDebug("Read ccPointCloud: {} vertices.",
                                   pointcloud.size());
    if (params.remove_nan_points || params.remove_infinite_points) {
        pointcloud.RemoveNonFinitePoints(params.remove_nan_points,
                                         params.remove_infinite_points);
    }
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const ccPointCloud &pointcloud,
                     const WritePointCloudOption &params) {
    std::string format =
            cloudViewer::utility::filesystem::GetFileExtensionInLowerCase(
                    filename);
    auto map_itr = file_extension_to_pointcloud_write_function.find(format);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        cloudViewer::utility::LogWarning(
                "Write ccPointCloud failed: unknown file extension {} "
                "for file {}.",
                format, filename);
        return false;
    }

    bool success = map_itr->second(filename, pointcloud, params);
    cloudViewer::utility::LogDebug("Write ccPointCloud: {:d} vertices.",
                                   pointcloud.size());
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const ccPointCloud &pointcloud,
                     const std::string &file_format /* = "auto"*/,
                     bool write_ascii /* = false*/,
                     bool compressed /* = false*/,
                     bool print_progress) {
    WritePointCloudOption p;
    p.write_ascii = WritePointCloudOption::IsAscii(write_ascii);
    p.compressed = WritePointCloudOption::Compressed(compressed);
    std::string format = file_format;
    if (format == "auto") {
        format = cloudViewer::utility::filesystem::GetFileExtensionInLowerCase(
                filename);
    }
    cloudViewer::utility::ConsoleProgressUpdater progress_updater(
            std::string("Writing ") + cloudViewer::utility::ToUpper(format) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;
    return WritePointCloud(filename, pointcloud, p);
}

bool WritePointCloud(unsigned char *&buffer,
                     size_t &length,
                     const ccPointCloud &pointcloud,
                     const WritePointCloudOption &params) {
    std::string format = params.format;
    if (format == "auto") {
        cloudViewer::utility::LogWarning(
                "Write ccPointCloud failed: unknown format for "
                "(format: {}).",
                params.format);
        return false;
    }
    auto map_itr = in_memory_to_pointcloud_write_function.find(format);
    if (map_itr == in_memory_to_pointcloud_write_function.end()) {
        cloudViewer::utility::LogWarning(
                "Write ccPointCloud failed: unknown format {}.", format);
        return false;
    }

    bool success = map_itr->second(buffer, length, pointcloud, params);
    cloudViewer::utility::LogDebug("Write ccPointCloud: {} vertices.",
                                   pointcloud.size());
    return success;
}

}  // namespace io
}  // namespace cloudViewer
