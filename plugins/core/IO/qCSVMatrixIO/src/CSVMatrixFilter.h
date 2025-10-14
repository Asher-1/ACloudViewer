// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_CSV_MATRIX_FILTER_HEADER
#define ECV_CSV_MATRIX_FILTER_HEADER

// qCC_io
#include <FileIOFilter.h>

//! CSV matrix I/O filter
class /*ECV_IO_LIB_API*/ CSVMatrixFilter : public FileIOFilter {
public:
    CSVMatrixFilter();

    // inherited from FileIOFilter
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters);
};

#endif  // ECV_CSV_MATRIX_FILTER_HEADER
