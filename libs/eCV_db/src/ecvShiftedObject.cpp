// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvShiftedObject.h"

#include <CVLog.h>

// LOCAL
#include "ecvBBox.h"
#include "ecvSerializableObject.h"

ccShiftedObject::ccShiftedObject(QString name)
    : ccHObject(name), m_globalShift(0, 0, 0), m_globalScale(1.0) {}

ccShiftedObject::ccShiftedObject(const ccShiftedObject& s)
    : ccHObject(s),
      m_globalShift(s.m_globalShift),
      m_globalScale(s.m_globalScale) {}

void ccShiftedObject::copyGlobalShiftAndScale(const ccShiftedObject& s) {
    setGlobalShift(s.getGlobalShift());
    setGlobalScale(s.getGlobalScale());
}

void ccShiftedObject::setGlobalShift(const CCVector3d& shift) {
    m_globalShift = shift;
}

void ccShiftedObject::setGlobalScale(double scale) {
    if (scale == 0) {
        CVLog::Warning("[setGlobalScale] Invalid scale (zero)!");
        m_globalScale = 1.0;
    } else {
        m_globalScale = scale;
    }
}

bool ccShiftedObject::saveShiftInfoToFile(QFile& out) const {
    //'coordinates shift'
    if (out.write((const char*)m_globalShift.u, sizeof(double) * 3) < 0)
        return ccSerializableObject::WriteError();
    //'global scale'
    if (out.write((const char*)&m_globalScale, sizeof(double)) < 0)
        return ccSerializableObject::WriteError();

    return true;
}

bool ccShiftedObject::loadShiftInfoFromFile(QFile& in) {
    //'coordinates shift'
    if (in.read((char*)m_globalShift.u, sizeof(double) * 3) < 0)
        return ccSerializableObject::ReadError();
    //'global scale'
    if (in.read((char*)&m_globalScale, sizeof(double)) < 0)
        return ccSerializableObject::ReadError();

    return true;
}

bool ccShiftedObject::getOwnGlobalBB(CCVector3d& minCorner,
                                     CCVector3d& maxCorner) {
    ccBBox box = getOwnBB(false);
    minCorner = toGlobal3d(box.minCorner());
    maxCorner = toGlobal3d(box.maxCorner());
    return box.isValid();
}

ccHObject::GlobalBoundingBox ccShiftedObject::getOwnGlobalBB(
        bool withGLFeatures /*=false*/) {
    ccBBox box = getOwnBB(false);
    CCVector3d minCorner = toGlobal3d(box.minCorner());
    CCVector3d maxCorner = toGlobal3d(box.maxCorner());
    return GlobalBoundingBox(minCorner, maxCorner);
}
