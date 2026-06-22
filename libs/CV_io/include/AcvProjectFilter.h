// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

/**
 * @brief ACloudViewer composite project file filter (.acv)
 *
 * Saves and loads complete project state as a single .acv file,
 * bundling entities (BIN format), view layout, per-view camera/
 * representation state, and GUI configuration.
 *
 * File format (QDataStream, BigEndian):
 *   [QString]  magic     "ACV_PROJECT"
 *   [quint32]  version   1
 *   [QByteArray] metadata  JSON (manifest + layout + views + gui)
 *   [QByteArray] entities  BIN v2 entity data
 */
class CV_IO_LIB_API AcvProjectFilter : public FileIOFilter {
public:
    AcvProjectFilter();

    static inline QString GetFileFilter() {
        return "ACloudViewer Project (*.acv)";
    }
    static inline QString GetDefaultExtension() { return "acv"; }

    CC_FILE_ERROR loadFile(const QString& filename,
                           ccHObject& container,
                           LoadParameters& parameters) override;

    CC_FILE_ERROR saveToFile(ccHObject* entity,
                             const QString& filename,
                             const SaveParameters& parameters) override;

    bool canSave(CV_CLASS_ENUM type,
                 bool& multiple,
                 bool& exclusive) const override;

private:
    static constexpr quint32 ACV_FORMAT_VERSION = 1;
    static const QString ACV_MAGIC;
};
