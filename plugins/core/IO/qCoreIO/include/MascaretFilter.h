// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_MASCARET_FILTER_HEADER
#define ECV_MASCARET_FILTER_HEADER

#include "FileIOFilter.h"

//! Mascaret profile I/O filter
/** See http://www.opentelemac.org/
 **/
class MascaretFilter : public FileIOFilter {
public:
    MascaretFilter();

    // inherited from FileIOFilter
    virtual bool canSave(CV_CLASS_ENUM type,
                         bool& multiple,
                         bool& exclusive) const override;
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;
};

#endif  // ECV_MASCARET_FILTER_HEADER
