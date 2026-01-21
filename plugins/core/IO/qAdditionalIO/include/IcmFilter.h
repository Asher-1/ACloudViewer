// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

//! Calibrated images and cloud meta-file I/O filter
class /*CV_IO_LIB_API*/ IcmFilter : public FileIOFilter {
public:
    IcmFilter();

    // inherited from FileIOFilter
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) override;

protected:
    static int LoadCalibratedImages(ccHObject* entities,
                                    const QString& path,
                                    const QString& imageDescFilename,
                                    const ccBBox& globalBBox);
};
