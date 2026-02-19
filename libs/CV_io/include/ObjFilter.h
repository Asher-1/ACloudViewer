// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

/**
 * @class ObjFilter
 * @brief Wavefront OBJ mesh file I/O filter
 * 
 * Handles import/export of meshes in Wavefront OBJ format (.obj).
 * This is a widely-used text-based 3D mesh format that supports:
 * - Vertex positions
 * - Texture coordinates
 * - Vertex normals
 * - Face definitions (triangles, quads, polygons)
 * - Material definitions (via .mtl files)
 * 
 * The OBJ format is human-readable and widely supported by 3D software.
 * 
 * @see FileIOFilter
 * @see ccMesh
 */
class CV_IO_LIB_API ObjFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    ObjFilter();

    /**
     * @brief Load OBJ file
     * @param filename Input OBJ file path
     * @param container Container for loaded mesh entities
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
     * @brief Save entity to OBJ file
     * @param entity Entity to save (mesh)
     * @param filename Output OBJ file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;
};
