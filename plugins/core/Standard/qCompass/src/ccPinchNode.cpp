// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccPinchNode.h"

// pass ctors straight to PointPair
ccPinchNode::ccPinchNode(ccPointCloud* associatedCloud)
    : ccPointPair(associatedCloud) {
    updateMetadata();
}

ccPinchNode::ccPinchNode(ccPolyline* obj) : ccPointPair(obj) {
    updateMetadata();
}

void ccPinchNode::updateMetadata() {
    // add metadata tag defining the ccCompass class type
    QVariantMap* map = new QVariantMap();
    map->insert("ccCompassType", "PinchNode");
    setMetaData(*map, true);

    // set drawing stuff (not really metadata, but hey!)
    setDefaultColor(ecvColor::blue);
    setActiveColor(ecvColor::orange);
    setHighlightColor(ecvColor::orange);
    setAlternateColor(ecvColor::orange);
}

// returns true if object is a lineation
bool ccPinchNode::isPinchNode(ccHObject* object) {
    if (object->hasMetaData("ccCompassType")) {
        return object->getMetaData("ccCompassType")
                .toString()
                .contains("PinchNode");
    }
    return false;
}
