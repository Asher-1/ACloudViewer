// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

class QWidget;

//! CLOUDVIEWER dedicated binary point cloud I/O filter
class ECV_IO_LIB_API BinFilter : public FileIOFilter {
public:
    BinFilter();

    // static accessors
    static inline QString GetFileFilter() {
        return "CloudViewer entities (*.bin)";
    }
    static inline QString GetDefaultExtension() { return "bin"; }

    //! Returns the last saved file version
    static short GetLastSavedFileVersion();

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

    //! old style BIN loading
    static CC_FILE_ERROR LoadFileV1(QFile& in,
                                    ccHObject& container,
                                    unsigned nbScansTotal,
                                    const LoadParameters& parameters);

    //! new style BIN loading
    /** \param in the file to read from
        \param container the container to load the entities into
        \param flags the deserialization flags
        \param parallel whether to use parallel loading
        \param parentWidget the parent widget for progress dialogs
        \return the error code
    */
    static CC_FILE_ERROR LoadFileV2(QFile& in,
                                    ccHObject& container,
                                    int flags,
                                    bool parallel,
                                    QWidget* parentWidget = nullptr);

    //! new style BIN saving
    static CC_FILE_ERROR SaveFileV2(QFile& out, ccHObject* object);
};
