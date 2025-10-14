// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

//! Filter to load or save an image (all types supported by Qt)
class ECV_IO_LIB_API ImageFileFilter : public FileIOFilter {
public:
    ImageFileFilter();

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

    //! Helper: select an input image filename
    static QString GetLoadFilename(const QString& dialogTitle,
                                   const QString& imageLoadPath,
                                   QWidget* parentWidget = nullptr);

    //! Helper: select an output image filename
    static QString GetSaveFilename(const QString& dialogTitle,
                                   const QString& baseName,
                                   const QString& imageSavePath,
                                   QWidget* parentWidget = nullptr);
};
