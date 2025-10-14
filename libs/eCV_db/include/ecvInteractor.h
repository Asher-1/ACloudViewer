// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_INTERACTOR_HEADER
#define ECV_INTERACTOR_HEADER

// Local
#include "eCV_db.h"

// CV_CORE_LIB
#include "CVGeom.h"

// Qt
#include <Qt>

//! Interactor interface (entity that can be dragged or clicked in a 3D view)
class ECV_DB_LIB_API ccInteractor {
public:
    virtual ~ccInteractor() = default;

    //! Called on mouse click
    virtual bool acceptClick(int x, int y, Qt::MouseButton button) {
        return false;
    }

    //! Called on mouse move (for 2D interactors)
    /** \return true if a movement occurs
     **/
    virtual bool move2D(
            int x, int y, int dx, int dy, int screenWidth, int screenHeight) {
        return false;
    }

    //! Called on mouse move (for 3D interactors)
    /** \return true if a movement occurs
     **/
    virtual bool move3D(const CCVector3d& u) { return false; }
};

#endif  // ECV_INTERACTOR_HEADER
