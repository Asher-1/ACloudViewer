// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_MA_FILTER_HEADER
#define ECV_MA_FILTER_HEADER

#include "FileIOFilter.h"

//! Maya ASCII meshes file I/O filter
class MAFilter : public FileIOFilter {
public:
    MAFilter();

    // inherited from FileIOFilter
    virtual bool canSave(CV_CLASS_ENUM type,
                         bool& multiple,
                         bool& exclusive) const override;
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;
};

#endif  // ECV_MA_FILTER_HEADER
