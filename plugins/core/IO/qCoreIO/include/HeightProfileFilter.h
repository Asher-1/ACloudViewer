// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_HEIGHT_PROFILE_HEADER
#define ECV_HEIGHT_PROFILE_HEADER

#include "FileIOFilter.h"

//! Polyline height profile I/O filter
/** This file format contains a 2D series: (curvilinear absisca ; height)
 **/
class HeightProfileFilter : public FileIOFilter {
public:
    HeightProfileFilter();

    // inherited from FileIOFilter
    virtual bool canSave(CV_CLASS_ENUM type,
                         bool& multiple,
                         bool& exclusive) const override;
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;
};

#endif  // ECV_HEIGHT_PROFILE_HEADER
