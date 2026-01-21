// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

#ifdef CV_GDAL_SUPPORT

//! Raster grid format file I/O filter
/** Multiple formats are handled: see GDAL (http://www.gdal.org/)
 **/
class CV_IO_LIB_API RasterGridFilter : public FileIOFilter {
public:
    RasterGridFilter();

    // inherited from FileIOFilter
    CC_FILE_ERROR loadFile(const QString& filename,
                           ccHObject& container,
                           LoadParameters& parameters) override;
};

#endif  // CV_GDAL_SUPPORT
