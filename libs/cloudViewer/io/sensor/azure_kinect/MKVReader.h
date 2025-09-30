// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <RGBDImage.h>
#include "io/sensor/azure_kinect/MKVMetadata.h"
#include <IJsonConvertible.h>

struct _k4a_playback_t;        // typedef _k4a_playback_t* k4a_playback_t;
struct _k4a_capture_t;         // typedef _k4a_capture_t* k4a_capture_t;
struct _k4a_transformation_t;  // typedef _k4a_transformation_t*
                               // k4a_transformation_t;

namespace cloudViewer {
namespace io {

/// \class MKVReader
///
/// AzureKinect mkv file reader.
class MKVReader {
public:
    /// \brief Default Constructor.
    MKVReader();
    virtual ~MKVReader() {}

    /// Check If the mkv file is opened.
    bool IsOpened();
    /// Check if the mkv file is all read.
    bool IsEOF() { return is_eof_; }

    /// Open an mkv playback.
    ///
    /// \param filename Path to the mkv file.
    bool Open(const std::string &filename);
    /// Close the opened mkv playback.
    void Close();

    /// Get metadata of the mkv playback.
    MKVMetadata &GetMetadata() { return metadata_; }
    /// Seek to the timestamp (in us).
    bool SeekTimestamp(size_t timestamp);
    /// Get next frame from the mkv playback and returns the RGBD object.
    std::shared_ptr<geometry::RGBDImage> NextFrame();

private:
    _k4a_playback_t *handle_;
    _k4a_transformation_t *transformation_;
    MKVMetadata metadata_;
    bool is_eof_ = false;

    Json::Value GetMetadataJson();
    std::string GetTagInMetadata(const std::string &tag_name);
};
}  // namespace io
}  // namespace cloudViewer
