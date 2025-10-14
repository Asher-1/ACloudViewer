// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_SALOME_HYDRO_HEADER
#define ECV_SALOME_HYDRO_HEADER

#include "FileIOFilter.h"

//! SALOME hydro polylines I/O filter
/** See http://chercheurs.edf.com/logiciels/salome-41218.html
 **/
class SalomeHydroFilter : public FileIOFilter {
public:
    SalomeHydroFilter();

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
};

#endif  // ECV_SALOME_HYDRO_HEADER
