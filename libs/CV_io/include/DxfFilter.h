// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

/**
 * @class DxfFilter
 * @brief AutoCAD DXF (Drawing Exchange Format) file I/O filter
 * 
 * Handles import/export of CAD entities in AutoCAD DXF format (.dxf).
 * DXF is a text-based or binary format for exchanging CAD data between
 * applications. Supports:
 * - 2D/3D polylines
 * - Points
 * - Lines and arcs
 * - Text entities
 * - Layers and colors
 * 
 * This filter is primarily used for importing/exporting polyline data
 * and simple geometric entities.
 * 
 * @see FileIOFilter
 * @see ccPolyline
 */
class CV_IO_LIB_API DxfFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    DxfFilter();

    /**
     * @brief Load DXF file
     * @param filename Input DXF file path
     * @param container Container for loaded entities
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) override;

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
     * @brief Save entity to DXF file
     * @param entity Entity to save (polylines, etc.)
     * @param filename Output DXF file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;
};
