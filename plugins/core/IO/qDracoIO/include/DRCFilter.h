// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_DRC_FILTER_HEADER
#define CC_DRC_FILTER_HEADER

#include <FileIOFilter.h>

//! Draco compressed cloud and mesh file I/O filter
class DRCFilter : public FileIOFilter {
public:
    DRCFilter();

    // inherited from FileIOFilter
    CC_FILE_ERROR loadFile(const QString& filename,
                           ccHObject& container,
                           LoadParameters& parameters) override;

    bool canSave(CV_CLASS_ENUM type,
                 bool& multiple,
                 bool& exclusive) const override;
    CC_FILE_ERROR saveToFile(ccHObject* entity,
                             const QString& filename,
                             const SaveParameters& parameters) override;
};

#endif  // CC_DRC_FILTER_HEADER
