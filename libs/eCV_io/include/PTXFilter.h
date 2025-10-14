// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_PTX_FILTER_HEADER
#define ECV_PTX_FILTER_HEADER

#include "FileIOFilter.h"

//! PTX point cloud I/O filter
class ECV_IO_LIB_API PTXFilter : public FileIOFilter {
public:
    PTXFilter();

    // inherited from FileIOFilter
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) override;
};

#endif  // CC_PTX_FILTER_HEADER
