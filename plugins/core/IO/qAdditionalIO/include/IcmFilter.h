// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_ICM_FILTER_HEADER
#define CC_ICM_FILTER_HEADER

#include "FileIOFilter.h"

//! Calibrated images and cloud meta-file I/O filter
class /*ECV_IO_LIB_API*/ IcmFilter : public FileIOFilter {
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

#endif  // CC_ICM_FILTER_HEADER
