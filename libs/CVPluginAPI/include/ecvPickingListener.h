// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CVPluginAPI.h"

// cloudViewer
#include <CVGeom.h>

// Qt
#include <QPoint>

class ccHObject;

//! Point/triangle picking listener interface
class CVPLUGIN_LIB_API ccPickingListener {
public:
    virtual ~ccPickingListener() = default;

    //! Picked item
    struct PickedItem {
        PickedItem() : entity(nullptr), itemIndex(0), entityCenter(false) {}

        QPoint clickPoint;   // position of the user click
        ccHObject* entity;   // picked entity (if any)
        unsigned itemIndex;  // e.g. point or triangle index
        CCVector3 P3D;       // picked point in 3D (if any)
        CCVector3d uvw;  // picked point barycentric coordinates (if picked on a
                         // triangle)
        bool entityCenter;  // the point doesn't correspond to a real 'item' but
                            // to the entity center
    };

    //! Method called whenever an item is picked
    virtual void onItemPicked(const PickedItem& pi) = 0;
};
