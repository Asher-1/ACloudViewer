// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

//! Mensi Soisic point cloud file I/O filter
class /*ECV_IO_LIB_API*/ SoiFilter : public FileIOFilter {
public:
    SoiFilter();

    // inherited from FileIOFilter
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) override;
};
