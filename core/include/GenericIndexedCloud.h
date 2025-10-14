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

//! A generic 3D point cloud with index-based point access
/** Implements the GenericCloud interface.
 **/
class CV_CORE_LIB_API GenericIndexedCloud : virtual public GenericCloud {
public:
    //! Default destructor
    ~GenericIndexedCloud() override = default;

    //! Returns the ith point
    /**	Virtual method to request a point with a specific index.
            WARNINGS:
            - the returned object may not be persistent!
            - THIS METHOD MAY NOT BE COMPATIBLE WITH PARALLEL STRATEGIES
            (see the DgmOctree::executeFunctionForAllCellsAtLevel_MT and
            DgmOctree::executeFunctionForAllCellsAtStartingLevel_MT methods).
            Consider the other version of getPoint instead or the
            GenericIndexedCloudPersist class.
            \param index of the requested point (between 0 and the cloud size
    minus 1) \return the requested point (undefined behavior if index is
    invalid)
    **/
    virtual const CCVector3* getPoint(unsigned index) const = 0;

    //! Returns the ith point
    /**	Virtual method to request a point with a specific index.
            Index must be valid (undefined behavior if index is invalid)
            \param index of the requested point (between 0 and the cloud size
    minus 1) \param P output point
    **/
    virtual void getPoint(unsigned index, CCVector3& P) const = 0;
    virtual void getPoint(unsigned index, double P[3]) const {
        const CCVector3* pt = getPoint(index);
        P[0] = pt->x;
        P[1] = pt->y;
        P[2] = pt->z;
    };

    //! Returns whether normals are available
    virtual bool normalsAvailable() const { return false; }

    //! If per-point normals are available, returns the one at a specific index
    /** \warning If overriden, this method should return a valid normal for all
     *points
     **/
    virtual const CCVector3* getNormal(unsigned index) const {
        (void)index;
        return nullptr;
    }
};

}  // namespace cloudViewer
