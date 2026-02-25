// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

class ccGBLSensor;

/**
 * @class DepthMapFileFilter
 * @brief Depth map file I/O filter (ASCII format)
 *
 * Handles export of depth map data from ground-based laser (GBL) sensors
 * to ASCII text files. Depth maps represent range/distance measurements
 * in a 2D grid format, typically from terrestrial laser scanners.
 *
 * Output format:
 * - ASCII text (.txt, .asc)
 * - Grid-based layout
 * - Contains depth/range values
 *
 * @note This filter currently supports saving only (no import)
 * @see FileIOFilter
 * @see ccGBLSensor
 */
class CV_IO_LIB_API DepthMapFileFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    DepthMapFileFilter();

    /**
     * @brief Get file filter string for dialogs
     * @return Filter string "Depth Map [ascii] (*.txt *.asc)"
     */
    static inline QString GetFileFilter() {
        return "Depth Map [ascii] (*.txt *.asc)";
    }

    /**
     * @brief Check if entity type can be saved
     * @param type Entity type
     * @param multiple Output: whether multiple entities can be saved
     * @param exclusive Output: whether only this type can be saved
     * @return true if type can be saved
     */
    virtual bool canSave(CV_CLASS_ENUM type,
                         bool& multiple,
                         bool& exclusive) const override;

    /**
     * @brief Save entity to depth map file
     * @param entity Entity to save (must be GBL sensor)
     * @param filename Output file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;

    /**
     * @brief Direct method to save GBL sensor depth map
     * @param filename Output file path
     * @param sensor GBL sensor with depth map data
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR saveToFile(const QString& filename, ccGBLSensor* sensor);
};
