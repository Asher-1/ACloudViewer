// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef GENERIC_OCTREE_HEADER
#define GENERIC_OCTREE_HEADER

// Local
#include "CVGeom.h"

namespace cloudViewer {

//! A generic octree interface for data communication between library and client
//! applications
class CV_CORE_LIB_API GenericOctree {
public:
    //! Default destructor
    virtual ~GenericOctree() = default;
};

}  // namespace cloudViewer

#endif  // GENERIC_OCTREE_HEADER
