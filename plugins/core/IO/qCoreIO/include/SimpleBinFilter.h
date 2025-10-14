// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_SIMPLE_BIN_FILTER_HEADER
#define ECV_SIMPLE_BIN_FILTER_HEADER

#include "FileIOFilter.h"

//! Simple binary file (with attached text meta-file)
class SimpleBinFilter : public FileIOFilter {
public:
    SimpleBinFilter();

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

protected:
};

#endif  // ECV_SIMPLE_BIN_FILTER_HEADER
