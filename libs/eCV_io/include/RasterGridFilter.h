// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_RASTER_FILTER_HEADER
#define ECV_RASTER_FILTER_HEADER

#include "FileIOFilter.h"

#ifdef CV_GDAL_SUPPORT

//! Raster grid format file I/O filter
/** Multiple formats are handled: see GDAL (http://www.gdal.org/)
 **/
class ECV_IO_LIB_API RasterGridFilter : public FileIOFilter {
public:
    RasterGridFilter();

    // inherited from FileIOFilter
    CC_FILE_ERROR loadFile(const QString& filename,
                           ccHObject& container,
                           LoadParameters& parameters) override;
};

#endif  // CV_GDAL_SUPPORT

#endif  // ECV_RASTER_FILTER_HEADER
