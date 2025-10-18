// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericPointCloud.h"

// Local dependencies
#include "ccMPlanePoint.h"

unsigned int ccMPlanePoint::getIndex() const {
    return m_label->getPickedPoint(0).index;
}

const CCVector3& ccMPlanePoint::getCoordinates() const {
    return *m_label->getPickedPoint(0).cloudOrVertices()->getPoint(getIndex());
}

cc2DLabel* ccMPlanePoint::getLabel() { return m_label; }

QString ccMPlanePoint::getName() const { return m_label->getName(); }

void ccMPlanePoint::setName(const QString& newName) {
    m_label->setName(newName);
}

float ccMPlanePoint::getDistance() const { return m_distance; }

void ccMPlanePoint::setDistance(float distance) { m_distance = distance; }
