// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <FileIO.h>

#include <memory>
#include <string>

class ccPointCloud;
namespace cloudViewer {
namespace io {

/// Factory function to create a pointcloud from a file (PointCloudFactory.cpp)
/// Return an empty pointcloud if fail to read the file.
std::shared_ptr<ccPointCloud> CreatePointCloudFromFile(
        const std::string &filename,
        const std::string &format = "auto",
        bool print_progress = false);

/// Factory function to create a pointcloud from memory
/// Return an empty pointcloud if fail to read from buffer.
std::shared_ptr<ccPointCloud> CreatePointCloudFromMemory(
        const unsigned char *buffer,
        const size_t length,
        const std::string &format,
        bool print_progress = false);

/// The general entrance for reading a PointCloud from a file
/// The function calls read functions based on the extension name of filename.
/// See \p ReadPointCloudOption for additional options you can pass.
/// \return return true if the read function is successful, false otherwise.
bool ReadPointCloud(const std::string &filename,
                    ccPointCloud &pointcloud,
                    const ReadPointCloudOption &params = {});

/// The general entrance for reading a PointCloud from memory
/// The function calls read functions based on the format.
/// See \p ReadPointCloudOption for additional options you can pass.
/// \return return true if the read function is successful, false otherwise.
bool ReadPointCloud(const unsigned char *buffer,
                    const size_t length,
                    ccPointCloud &pointcloud,
                    const ReadPointCloudOption &params = {});

/// The general entrance for writing a PointCloud to a file
/// The function calls write functions based on the extension name of filename.
/// See \p WritePointCloudOption for additional options you can pass.
/// \return return true if the write function is successful, false otherwise.
bool WritePointCloud(const std::string &filename,
                     const ccPointCloud &pointcloud,
                     const WritePointCloudOption &params = {});

/// The general entrance for writing a PointCloud to memory
/// The function calls write functions based on the format.
/// WARNING: buffer gets initialized by WritePointCloud, you need to
/// delete it when finished when ret is true
/// See \p WritePointCloudOption for additional options you can pass.
/// \return return true if the write function is
/// successful, false otherwise.
bool WritePointCloud(unsigned char *&buffer,
                     size_t &length,
                     const ccPointCloud &pointcloud,
                     const WritePointCloudOption &params = {});

bool ReadPointCloudFromXYZ(const std::string &filename,
                           ccPointCloud &pointcloud,
                           const ReadPointCloudOption &params);

bool ReadPointCloudInMemoryFromXYZ(const unsigned char *buffer,
                                   const size_t length,
                                   ccPointCloud &pointcloud,
                                   const ReadPointCloudOption &params);

bool WritePointCloudToXYZ(const std::string &filename,
                          const ccPointCloud &pointcloud,
                          const WritePointCloudOption &params);

bool WritePointCloudInMemoryToXYZ(unsigned char *&buffer,
                                  size_t &length,
                                  const ccPointCloud &pointcloud,
                                  const WritePointCloudOption &params);

bool ReadPointCloudFromXYZN(const std::string &filename,
                            ccPointCloud &pointcloud,
                            const ReadPointCloudOption &params);

bool WritePointCloudToXYZN(const std::string &filename,
                           const ccPointCloud &pointcloud,
                           const WritePointCloudOption &params);

bool ReadPointCloudFromXYZRGB(const std::string &filename,
                              ccPointCloud &pointcloud,
                              const ReadPointCloudOption &params);

bool WritePointCloudToXYZRGB(const std::string &filename,
                             const ccPointCloud &pointcloud,
                             const WritePointCloudOption &params);

bool ReadPointCloudFromPLY(const std::string &filename,
                           ccPointCloud &pointcloud,
                           const ReadPointCloudOption &params);

bool WritePointCloudToPLY(const std::string &filename,
                          const ccPointCloud &pointcloud,
                          const WritePointCloudOption &params);

bool ReadPointCloudFromPCD(const std::string &filename,
                           ccPointCloud &pointcloud,
                           const ReadPointCloudOption &params);

bool WritePointCloudToPCD(const std::string &filename,
                          const ccPointCloud &pointcloud,
                          const WritePointCloudOption &params);

bool ReadPointCloudFromPTS(const std::string &filename,
                           ccPointCloud &pointcloud,
                           const ReadPointCloudOption &params);

bool WritePointCloudToPTS(const std::string &filename,
                          const ccPointCloud &pointcloud,
                          const WritePointCloudOption &params);

}  // namespace io
}  // namespace cloudViewer
