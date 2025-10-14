// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_PDMS_FILTER_HEADER
#define ECV_PDMS_FILTER_HEADER

#include "FileIOFilter.h"

//! PDMS .mac file I/O filter
class PDMSFilter : public FileIOFilter {
public:
    PDMSFilter();

    // inherited from FileIOFilter
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) override;
};

#endif  // ECV_PDMS_FILTER_HEADER
