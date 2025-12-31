// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvDisc.h"

// Local
#include <CVTools.h>
#include <Logging.h>

#include "ecvNormalVectors.h"
#include "ecvPointCloud.h"

ccDisc::ccDisc(PointCoordinateType radius,
               const ccGLMatrix* transMat /*= nullptr*/,
               QString name /*= QString("Disc")*/,
               unsigned precision /*= DEFAULT_DRAWING_PRECISION*/)
    : ccGenericPrimitive(name, transMat), m_radius(std::abs(radius)) {
    setDrawingPrecision(std::max<unsigned>(
            precision,
            MIN_DRAWING_PRECISION));  // automatically calls buildUp &
                                      // applyTransformationToVertices
}

ccDisc::ccDisc(QString name /*="Disc"*/)
    : ccGenericPrimitive(name), m_radius(0) {}

ccGenericPrimitive* ccDisc::clone() const {
    return finishCloneJob(new ccDisc(m_radius, &m_transformation, getName(),
                                     m_drawPrecision));
}

bool ccDisc::buildUp() {
    if (m_drawPrecision < MIN_DRAWING_PRECISION) return false;

    // invalid dimensions?
    if (cloudViewer::LessThanEpsilon(m_radius)) {
        return false;
    }

    unsigned steps = m_drawPrecision;

    // vertices
    unsigned vertCount = steps + 1;  // At least the center

    // normals
    unsigned faceNormCounts = 1;
    // faces
    unsigned facesCount = steps;

    // allocate (& clear) structures
    if (!init(vertCount, false, facesCount, faceNormCounts)) {
        CVLog::Error("[ccDisc::buildUp] Not enough memory");
        return false;
    }

    ccPointCloud* verts = vertices();
    assert(verts);
    assert(m_triNormals);

    // first point: center of the disc
    CCVector3 center = CCVector3(0, 0, 0);
    // add center to the vertices
    verts->addPoint(center);
    CompressedNormType nIndex =
            ccNormalVectors::GetNormIndex(CCVector3(0, 0, 1).u);
    m_triNormals->addElement(nIndex);

    // then, angular sweep for the surface
    PointCoordinateType angle_rad_step =
            static_cast<PointCoordinateType>(2.0 * M_PI) / steps;
    // bottom surface
    for (unsigned i = 0; i < steps; ++i) {
        CCVector3 P(center.x + cos(angle_rad_step * i) * m_radius,
                    center.y + sin(angle_rad_step * i) * m_radius, center.z);
        verts->addPoint(P);
    }

    // mesh faces
    assert(m_triVertIndexes);

    // surface
    for (unsigned i = 0; i < steps; ++i) {
        unsigned i2 = 1 + i;
        unsigned i3 = 1 + (i + 1) % steps;
        addTriangle(0, i2, i3);
        addTriangleNormalIndexes(0, 0, 0);
    }

    notifyGeometryUpdate();
    showTriNorms(true);

    return true;
}

void ccDisc::setRadius(PointCoordinateType radius) {
    if (m_radius == radius) return;

    assert(radius > 0);
    m_radius = radius;

    buildUp();
    applyTransformationToVertices();
}

bool ccDisc::toFile_MeOnly(QFile& out) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));
    if (!ccGenericPrimitive::toFile_MeOnly(out)) {
        return false;
    }

    // parameters (dataVersion>=57)
    QDataStream outStream(&out);
    outStream << m_radius;

    return true;
}

bool ccDisc::fromFile_MeOnly(QFile& in,
                             short dataVersion,
                             int flags,
                             LoadedIDMap& oldToNewIDMap) {
    if (!ccGenericPrimitive::fromFile_MeOnly(in, dataVersion, flags,
                                             oldToNewIDMap))
        return false;

    if (dataVersion < 57) {
        return false;
    }

    // parameters (dataVersion>=57)
    QDataStream inStream(&in);
    ccSerializationHelper::CoordsFromDataStream(inStream, flags, &m_radius);

    return true;
}

ccBBox ccDisc::getOwnFitBB(ccGLMatrix& trans) {
    trans = m_transformation;
    // Disc is a 2D circle in XY plane, so bbox is a square centered at origin
    // with side length = 2 * radius, and height = 0
    return ccBBox(CCVector3(-m_radius, -m_radius, 0),
                  CCVector3(m_radius, m_radius, 0));
}
