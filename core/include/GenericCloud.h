// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>

// Local
#include "CVConst.h"
#include "CVGeom.h"

namespace cloudViewer {

/**
 * @class GenericCloud
 * @brief Generic 3D point cloud interface
 * 
 * Provides an abstract interface for 3D point cloud data communication
 * between library components and client applications. This interface
 * defines basic operations such as iteration, bounding box computation,
 * and scalar field access.
 */
class CV_CORE_LIB_API GenericCloud {
public:
    /**
     * @brief Default constructor
     */
    GenericCloud() = default;

    /**
     * @brief Default destructor
     */
    virtual ~GenericCloud() = default;

    /**
     * @brief Function type for point operations
     * 
     * Used with forEach() to apply operations to each point.
     */
    using genericPointAction =
            std::function<void(const CCVector3&, ScalarType&)>;

    /**
     * @brief Get the number of points in the cloud
     * @return Number of points
     */
    virtual unsigned size() const = 0;
    
    /**
     * @brief Check if cloud has any points
     * @return true if cloud contains at least one point
     */
    inline virtual bool hasPoints() const { return size() != 0; }

    /**
     * @brief Apply a function to all points
     * 
     * Fast iteration mechanism that applies the given function to each point.
     * @param action Function to apply (see genericPointAction)
     */
    virtual void forEach(genericPointAction action) = 0;

    /**
     * @brief Get the bounding box of the cloud
     * @param bbMin Output: minimum bounds (Xmin, Ymin, Zmin)
     * @param bbMax Output: maximum bounds (Xmax, Ymax, Zmax)
     */
    virtual void getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) = 0;

    /**
     * @brief Test point visibility
     * 
     * Returns the visibility state of a point relative to a sensor.
     * The visibility definition follows Daniel Girardeau-Montaut's PhD
     * manuscript (Chapter 2, section 2-3-3). This method is called before
     * performing point-to-cloud comparisons. If the result is not POINT_VISIBLE,
     * the comparison is skipped and the scalar field value is set to this
     * visibility value.
     * 
     * @param P 3D point to test
     * @return Visibility state (default: POINT_VISIBLE)
     * @note Should be overloaded if this functionality is required
     */
    virtual inline unsigned char testVisibility(const CCVector3& P) const {
        return POINT_VISIBLE;
    }

    /**
     * @brief Reset cloud iterator to the beginning
     */
    virtual void placeIteratorAtBeginning() = 0;

    /**
     * @brief Get the next point from the iterator
     * 
     * Returns the next point and advances the global iterator position.
     * 
     * @return Pointer to next point (nullptr if no more points)
     * @warning The returned object may not be persistent
     * @warning This method may not be compatible with parallel strategies
     * @see DgmOctree::executeFunctionForAllCellsAtLevel_MT
     * @see DgmOctree::executeFunctionForAllCellsAtStartingLevel_MT
     */
    virtual const CCVector3* getNextPoint() = 0;

    /**
     * @brief Enable the scalar field
     * 
     * If the scalar field structure is not initialized, this method triggers
     * its creation. The structure size should be pre-reserved to match the
     * number of points in the cloud.
     * 
     * @return true if successful
     */
    virtual bool enableScalarField() = 0;

    /**
     * @brief Check if scalar field is enabled
     * @return true if scalar field is enabled
     */
    virtual bool isScalarFieldEnabled() const = 0;

    /**
     * @brief Set scalar value for a point
     * @param pointIndex Index of the point
     * @param value Scalar value to set
     */
    virtual void setPointScalarValue(unsigned pointIndex, ScalarType value) = 0;

    /**
     * @brief Get scalar value for a point
     * @param pointIndex Index of the point
     * @return Scalar value
     */
    virtual ScalarType getPointScalarValue(unsigned pointIndex) const = 0;
};

}  // namespace cloudViewer
