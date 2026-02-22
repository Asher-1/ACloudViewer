// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

#ifdef CV_GDAL_SUPPORT

/**
 * @class RasterGridFilter
 * @brief Raster grid format I/O filter (GDAL-based)
 *
 * Handles import of raster grid data using the GDAL (Geospatial Data
 * Abstraction Library). Supports a wide variety of raster formats including:
 * - GeoTIFF (.tif, .tiff)
 * - ESRI ASCII Grid (.asc)
 * - ERDAS Imagine (.img)
 * - NetCDF (.nc)
 * - HDF (.hdf, .h5)
 * - And many more (see GDAL documentation)
 *
 * Raster data is typically converted to point clouds or 2.5D surfaces,
 * with elevation values from the raster grid.
 *
 * @note Requires GDAL library support (CV_GDAL_SUPPORT)
 * @see http://www.gdal.org/
 * @see FileIOFilter
 * @see ccPointCloud
 */
class CV_IO_LIB_API RasterGridFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    RasterGridFilter();

    /**
     * @brief Load raster grid file
     * @param filename Input raster file path
     * @param container Container for loaded entities
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR loadFile(const QString& filename,
                           ccHObject& container,
                           LoadParameters& parameters) override;
};

#endif  // CV_GDAL_SUPPORT
