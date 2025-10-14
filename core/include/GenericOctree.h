// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

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
