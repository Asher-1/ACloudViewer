// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvCylinder.h"

ccCylinder::ccCylinder(PointCoordinateType radius,
                       PointCoordinateType height,
                       const ccGLMatrix* transMat /*=0*/,
                       QString name /*=QString("Cylinder")*/,
                       unsigned precision /*=DEFAULT_DRAWING_PRECISION*/)
    : ccCone(radius, radius, height, 0, 0, transMat, name, precision) {}

ccCylinder::ccCylinder(QString name /*=QString("Cylinder")*/) : ccCone(name) {}

ccGenericPrimitive* ccCylinder::clone() const {
    return finishCloneJob(new ccCylinder(m_bottomRadius, m_height,
                                         &m_transformation, getName(),
                                         m_drawPrecision));
}

void ccCylinder::setBottomRadius(PointCoordinateType radius) {
    // we set the top radius as well!
    m_topRadius = radius;
    ccCone::setBottomRadius(radius);
}
