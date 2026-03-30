// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccThickness.h"

// pass ctors straight to PointPair
ccThickness::ccThickness(ccPointCloud* associatedCloud)
    : ccPointPair(associatedCloud) {
    updateMetadata();
}

ccThickness::ccThickness(ccPolyline* obj) : ccPointPair(obj) {
    updateMetadata();
}

void ccThickness::updateMetadata() {
    setMetaData("ccCompassType", "Thickness");

    if (size() == 2) {
        CCVector3f dir = getDirection();
        dir.normalize();
        float trend = 0.0f;
        float plunge = 0.0f;

        if (dir.x + dir.y + dir.z == 0) {
            trend = 0;
            plunge = 0;
        } else if (dir.z > 0.9999999f || dir.z < -0.9999999f) {
            trend = 0;
            if (dir.z < 0)
                plunge = 90;
            else
                plunge = -90;
        } else {
            CCVector3f hzComp = CCVector3f(dir.x, dir.y, 0);
            hzComp.normalize();

            plunge = std::acos(dir.dot(hzComp)) * (180.0f / M_PI);
            if (dir.z > 0) plunge *= -1;

            CCVector3f N(0, 1, 0);
            float dot = hzComp.dot(N);
            float det = CCVector3f(0, 0, 1).dot(hzComp.cross(N));
            trend = std::atan2(det, dot) * (180.0f / M_PI);
            if (trend < 0) trend += 360;
        }

        CCVector3 s = *getPoint(0);
        CCVector3 e = *getPoint(1);
        float length = static_cast<float>((s - e).norm());

        QVariantMap map;
        map.insert("Sx", s.x);
        map.insert("Sy", s.y);
        map.insert("Sz", s.z);
        map.insert("Ex", e.x);
        map.insert("Ey", e.y);
        map.insert("Ez", e.z);
        map.insert("Trend", trend);
        map.insert("Plunge", plunge);
        map.insert("Length", length);
        setMetaData(map, true);

        setName(QString::asprintf("%.3fT", length));
    }
}

// returns true if object is a lineation
bool ccThickness::isThickness(ccHObject* object) {
    if (object->hasMetaData("ccCompassType")) {
        return object->getMetaData("ccCompassType")
                .toString()
                .contains("Thickness");
    }
    return false;
}
