// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef PLUGIN_LAS_FILTER_HEADER
#define PLUGIN_LAS_FILTER_HEADER

#include "FileIOFilter.h"

//! ASPRS LAS point cloud file I/O filter
class LASFilter : public FileIOFilter {
public:
    LASFilter();

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

#endif  // PLUGIN_LAS_FILTER_HEADER
