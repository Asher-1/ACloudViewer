// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "GenericIndexedCloud.h"

namespace cloudViewer {

//! A generic 3D point cloud with index-based and presistent access to points
/** Implements the GenericIndexedCloud interface.
 **/
class CV_CORE_LIB_API GenericIndexedCloudPersist
    : virtual public GenericIndexedCloud {
public:
    //! Default destructor
    ~GenericIndexedCloudPersist() override = default;

    //! Returns the ith point as a persistent pointer
    /**	Virtual method to request a point with a specific index.
            WARNING: the returned object MUST be persistent in order
            to be compatible with parallel strategies!
            \param index of the requested point (between 0 and the cloud size
    minus 1) \return the requested point (or 0 if index is invalid)
    **/
    virtual const CCVector3* getPointPersistentPtr(unsigned index) = 0;
};

}  // namespace cloudViewer
