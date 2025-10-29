// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// qCC_IO
#include <FileIOFilter.h>

//! E57 filter (relies on E57format lib)
class E57Filter : public FileIOFilter {
public:
    E57Filter();

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
