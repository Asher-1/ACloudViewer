// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef GENERIC_TRIANGLE_HEADER
#define GENERIC_TRIANGLE_HEADER

// Local
#include "CVGeom.h"

namespace cloudViewer {

//! A generic triangle interface
/** Returns (temporary) references to each vertex.
 **/
class CV_CORE_LIB_API GenericTriangle {
public:
    //! Default destructor
    virtual ~GenericTriangle() = default;

    //! Returns the first vertex (A)
    virtual const CCVector3* _getA() const = 0;

    //! Returns the second vertex (B)
    virtual const CCVector3* _getB() const = 0;

    //! Returns the third vertex (C)
    virtual const CCVector3* _getC() const = 0;
};

}  // namespace cloudViewer

#endif  // GENERIC_TRIANGLE_HEADER
