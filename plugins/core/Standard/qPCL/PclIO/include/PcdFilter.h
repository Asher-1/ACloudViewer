// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ECV_DB_LIB
#include <FileIOFilter.h>

//! PCD point cloud I/O filter
class PcdFilter : public FileIOFilter {
public:
    PcdFilter();

    // inherited from FileIOFilter
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters);

    virtual bool canSave(CV_CLASS_ENUM type,
                         bool& multiple,
                         bool& exclusive) const;
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters);

    //! Output file format
    enum PCDOutputFileFormat {
        COMPRESSED_BINARY = 0,
        BINARY = 1,
        ASCII = 2,
        AUTO = 255
    };

    //! Set the output file format
    /** \param format output file format
     **/
    static void SetOutputFileFormat(PCDOutputFileFormat format);
};
