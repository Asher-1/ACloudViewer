// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "GenericCloud.h"

namespace cloudViewer {

/**
 * @class GenericIndexedCloud
 * @brief Generic 3D point cloud with index-based access
 * 
 * Extends GenericCloud to provide direct indexed access to points.
 * This interface allows efficient random access to individual points
 * by their index.
 * 
 * @warning Some methods may return non-persistent pointers
 * @warning Some methods may not be compatible with parallel strategies
 */
class CV_CORE_LIB_API GenericIndexedCloud : virtual public GenericCloud {
public:
    /**
     * @brief Virtual destructor
     */
    ~GenericIndexedCloud() override = default;

    /**
     * @brief Get point by index (pointer version)
     * 
     * Returns a pointer to the point at the specified index.
     * 
     * @param index Point index (must be < size())
     * @return Pointer to the point
     * @warning The returned pointer may not be persistent
     * @warning This method may not be compatible with parallel strategies
     * @see DgmOctree::executeFunctionForAllCellsAtLevel_MT
     * @see GenericIndexedCloudPersist for persistent access
     * @note Undefined behavior if index is invalid
     */
    virtual const CCVector3* getPoint(unsigned index) const = 0;

    /**
     * @brief Get point by index (copy version)
     * 
     * Copies the point at the specified index to the output parameter.
     * @param index Point index (must be < size())
     * @param P Output point (will be filled with point coordinates)
     * @note Undefined behavior if index is invalid
     */
    virtual void getPoint(unsigned index, CCVector3& P) const = 0;
    
    /**
     * @brief Get point by index (array version)
     * 
     * Copies the point coordinates to a double array.
     * @param index Point index (must be < size())
     * @param P Output array [x, y, z]
     */
    virtual void getPoint(unsigned index, double P[3]) const {
        const CCVector3* pt = getPoint(index);
        P[0] = pt->x;
        P[1] = pt->y;
        P[2] = pt->z;
    };

    /**
     * @brief Check if normals are available
     * @return true if per-point normals are available
     */
    virtual bool normalsAvailable() const { return false; }

    /**
     * @brief Get normal by index
     * 
     * If normals are available, returns the normal at the specified index.
     * @param index Point index
     * @return Pointer to normal (nullptr if normals not available)
     * @warning If overridden, should return valid normals for all points
     */
    virtual const CCVector3* getNormal(unsigned index) const {
        (void)index;
        return nullptr;
    }
};

}  // namespace cloudViewer
