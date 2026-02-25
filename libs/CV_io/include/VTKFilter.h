// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

/**
 * @class VTKFilter
 * @brief VTK (Visualization Toolkit) file I/O filter
 *
 * Handles import/export of point clouds and meshes in VTK legacy format (.vtk).
 * VTK is a widely-used format in scientific visualization that supports:
 * - Point clouds (POLYDATA with vertices)
 * - Meshes (POLYDATA with polygons)
 * - Scalar fields and vector fields
 * - Both ASCII and binary formats
 *
 * This filter supports the legacy VTK file format (not XML-based .vtp/.vtu).
 *
 * @see FileIOFilter
 * @see ccPointCloud
 * @see ccMesh
 */
class CV_IO_LIB_API VTKFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    VTKFilter();

    /**
     * @brief Load VTK file
     * @param filename Input VTK file path
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
     * @brief Save entity to VTK file
     * @param entity Entity to save (point cloud or mesh)
     * @param filename Output VTK file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;
};
