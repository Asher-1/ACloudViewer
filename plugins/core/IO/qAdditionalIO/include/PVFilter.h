// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_PV_FILTER_HEADER
#define ECV_PV_FILTER_HEADER

#include "FileIOFilter.h"

//! Point+Scalar value binary point cloud file I/O filter
class /*ECV_IO_LIB_API*/ PVFilter : public FileIOFilter {
public:
    PVFilter();

    // inherited from FileIOFilter
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) override;

    virtual bool canSave(CV_CLASS_ENUM type,
                         bool& multiple,
                         bool& exclusive) const override;
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;
};

#endif  // ECV_PV_FILTER_HEADER
