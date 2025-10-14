// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_DEPTH_MAP_FILE_FILTER_HEADER
#define ECV_DEPTH_MAP_FILE_FILTER_HEADER

#include "FileIOFilter.h"

class ccGBLSensor;

//! Depth map I/O filter
class ECV_IO_LIB_API DepthMapFileFilter : public FileIOFilter {
public:
    DepthMapFileFilter();

    // static accessors
    static inline QString GetFileFilter() {
        return "Depth Map [ascii] (*.txt *.asc)";
    }

    // inherited from FileIOFilter
    virtual bool canSave(CV_CLASS_ENUM type,
                         bool& multiple,
                         bool& exclusive) const override;
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;

    // direct method to save a sensor (depth map)
    CC_FILE_ERROR saveToFile(const QString& filename, ccGBLSensor* sensor);
};

#endif  // ECV_DEPTH_MAP_FILE_FILTER_HEADER
