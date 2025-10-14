// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_PHOTOSCAN_FILTER_HEADER
#define CC_PHOTOSCAN_FILTER_HEADER

// qCC_io
#include <FileIOFilter.h>

//! Photoscan (PSZ) file I/O filter
class PhotoScanFilter : public FileIOFilter {
public:
    PhotoScanFilter();

    // inherited from FileIOFilter
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters);
};

#endif  // CC_PHOTOSCAN_FILTER_HEADER
