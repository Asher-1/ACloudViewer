// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

class ccMesh;
class ccPointCloud;
class ccGenericMesh;

//! StereoLithography file I/O filter
/** See http://www.ennex.com/~fabbers/StL.asp
 **/
class ECV_IO_LIB_API STLFilter : public FileIOFilter {
public:
    STLFilter();

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
    //! Custom save method
    CC_FILE_ERROR saveToASCIIFile(ccGenericMesh* mesh,
                                  FILE* theFile,
                                  QWidget* parentWidget = 0);
    CC_FILE_ERROR saveToBINFile(ccGenericMesh* mesh,
                                FILE* theFile,
                                QWidget* parentWidget = 0);

    //! Custom load method for ASCII files
    CC_FILE_ERROR loadASCIIFile(QFile& fp,
                                ccMesh* mesh,
                                ccPointCloud* vertices,
                                LoadParameters& parameters);

    //! Custom load method for binary files
    CC_FILE_ERROR loadBinaryFile(QFile& fp,
                                 ccMesh* mesh,
                                 ccPointCloud* vertices,
                                 LoadParameters& parameters);
};
