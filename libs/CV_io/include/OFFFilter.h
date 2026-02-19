// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

/**
 * @class OFFFilter
 * @brief Object File Format (OFF) mesh I/O filter
 * 
 * Handles import/export of meshes in OFF format, a simple ASCII format
 * for representing polygon meshes. OFF format features:
 * - Vertex positions (X, Y, Z)
 * - Face definitions (vertex indices)
 * - Optional vertex colors
 * - Simple, human-readable text format
 * 
 * OFF is commonly used in computational geometry and academic research.
 * 
 * @see http://people.sc.fsu.edu/~jburkardt/data/off/off.html
 * @see FileIOFilter
 * @see ccMesh
 */
class CV_IO_LIB_API OFFFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    OFFFilter();

    /**
     * @brief Load OFF file
     * @param filename Input OFF file path
     * @param container Container for loaded mesh
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
     * @brief Save entity to OFF file
     * @param entity Entity to save (mesh)
     * @param filename Output OFF file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;
};
