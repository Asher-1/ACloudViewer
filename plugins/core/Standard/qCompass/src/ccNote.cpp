// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccNote.h"

// pass ctors straight to PointPair
ccNote::ccNote(ccPointCloud* associatedCloud) : ccPointPair(associatedCloud) {
    updateMetadata();
}

ccNote::ccNote(ccPolyline* obj) : ccPointPair(obj) { updateMetadata(); }

void ccNote::updateMetadata() {
    // add metadata tag defining the ccCompass class type
    QVariantMap* map = new QVariantMap();
    map->insert("ccCompassType", "Note");
    setMetaData(*map, true);

    // update drawing stuff
    showNameIn3D(true);
    setDefaultColor(ecvColor::cyan);
    setActiveColor(ecvColor::red);
}

// returns true if object is a lineation
bool ccNote::isNote(ccHObject* object) {
    if (object->hasMetaData("ccCompassType")) {
        return object->getMetaData("ccCompassType").toString().contains("Note");
    }
    return false;
}
