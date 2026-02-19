// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

/**
 * @class PTXFilter
 * @brief PTX (Leica Cyclone) point cloud I/O filter
 * 
 * Handles import of point clouds in PTX format, which is the native
 * ASCII format for Leica Cyclone laser scanner data. PTX files contain:
 * - Registered scan data with transformation matrices
 * - Point coordinates (X, Y, Z)
 * - Intensity values
 * - RGB color (optional)
 * - Scanner position and orientation
 * 
 * PTX format is commonly used for terrestrial laser scanning (TLS) data.
 * 
 * @note This filter currently supports loading only (no export)
 * @see FileIOFilter
 * @see ccPointCloud
 */
class CV_IO_LIB_API PTXFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    PTXFilter();

    /**
     * @brief Load PTX file
     * @param filename Input PTX file path
     * @param container Container for loaded point cloud
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) override;
};
