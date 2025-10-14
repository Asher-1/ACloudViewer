// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_OFF_FILTER_HEADER
#define ECV_OFF_FILTER_HEADER

#include "FileIOFilter.h"

//! OFF file I/O filter
/** See http://people.sc.fsu.edu/~jburkardt/data/off/off.html
 **/
class ECV_IO_LIB_API OFFFilter : public FileIOFilter {
public:
    OFFFilter();

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

#endif  // CC_OFF_FILTER_HEADER
