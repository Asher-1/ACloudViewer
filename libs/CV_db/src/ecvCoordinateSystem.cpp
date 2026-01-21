// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvCoordinateSystem.h"

// CV_DB_LIB
#include "ecvDisplayTools.h"
#include "ecvPlane.h"
#include "ecvPointCloud.h"

ccCoordinateSystem::ccCoordinateSystem(
        PointCoordinateType displayScale,
        PointCoordinateType axisWidth,
        const ccGLMatrix* transMat /*= 0*/,
        QString name /*=QString("CoordinateSystem")*/)
    : ccGenericPrimitive(name, transMat),
      m_DisplayScale(displayScale),
      m_width(axisWidth),
      m_showAxisPlanes(true),
      m_showAxisLines(true) {
    updateRepresentation();
    showColors(true);
}

ccCoordinateSystem::ccCoordinateSystem(
        const ccGLMatrix* transMat /*= 0*/,
        QString name /*=QString("CoordinateSystem")*/)
    : ccGenericPrimitive(name, transMat),
      m_DisplayScale(DEFAULT_DISPLAY_SCALE),
      m_width(AXIS_DEFAULT_WIDTH),
      m_showAxisPlanes(true),
      m_showAxisLines(true) {
    updateRepresentation();
    showColors(true);
}

ccCoordinateSystem::ccCoordinateSystem(
        QString name /*=QString("CoordinateSystem")*/)
    : ccGenericPrimitive(name),
      m_DisplayScale(DEFAULT_DISPLAY_SCALE),
      m_width(AXIS_DEFAULT_WIDTH),
      m_showAxisPlanes(true),
      m_showAxisLines(true) {
    updateRepresentation();
    showColors(true);
}

void ccCoordinateSystem::ShowAxisPlanes(bool show) { m_showAxisPlanes = show; }

void ccCoordinateSystem::ShowAxisLines(bool show) { m_showAxisLines = show; }

void ccCoordinateSystem::setAxisWidth(PointCoordinateType width) {
    if (width == 0.0f) {
        m_width = AXIS_DEFAULT_WIDTH;
        return;
    }
    if (width >= MIN_AXIS_WIDTH_F && width <= MAX_AXIS_WIDTH_F) {
        m_width = width;
    }
}

void ccCoordinateSystem::setDisplayScale(PointCoordinateType scale) {
    if (scale >= MIN_DISPLAY_SCALE_F) {
        m_DisplayScale = scale;
        updateRepresentation();
    }
}

std::shared_ptr<ccPlane> ccCoordinateSystem::getXYplane() const {
    auto plane = std::make_shared<ccPlane>(createXYplane(&m_transformation));
    plane->clearTriNormals();
    plane->ComputeTriangleNormals();
    return plane;
}
std::shared_ptr<ccPlane> ccCoordinateSystem::getYZplane() const {
    auto plane = std::make_shared<ccPlane>(createYZplane(&m_transformation));
    plane->clearTriNormals();
    plane->ComputeTriangleNormals();
    return plane;
}
std::shared_ptr<ccPlane> ccCoordinateSystem::getZXplane() const {
    auto plane = std::make_shared<ccPlane>(createZXplane(&m_transformation));
    plane->clearTriNormals();
    plane->ComputeTriangleNormals();
    return plane;
}

void ccCoordinateSystem::clearDrawings() {
    ccGenericPrimitive::clearDrawings();
    ecvDisplayTools::RemoveEntities(&m_axis);
}

void ccCoordinateSystem::hideShowDrawings(CC_DRAW_CONTEXT& context) {
    ecvDisplayTools::HideShowEntities(this,
                                      context.visible && m_showAxisPlanes);
    ecvDisplayTools::HideShowEntities(&m_axis,
                                      context.visible && m_showAxisLines);
}

ccPlane ccCoordinateSystem::createXYplane(const ccGLMatrix* transMat) const {
    ccGLMatrix xyPlane_mtrx;
    xyPlane_mtrx.toIdentity();
    xyPlane_mtrx.setTranslation(
            CCVector3(m_DisplayScale / 2, m_DisplayScale / 2, 0.0));
    if (transMat) {
        xyPlane_mtrx = *transMat * xyPlane_mtrx;
    }
    ccPlane xyPlane(m_DisplayScale, m_DisplayScale, &xyPlane_mtrx);
    xyPlane.setColor(ecvColor::red);
    return xyPlane;
}

ccPlane ccCoordinateSystem::createYZplane(const ccGLMatrix* transMat) const {
    ccGLMatrix yzPlane_mtrx;
    yzPlane_mtrx.initFromParameters(
            static_cast<PointCoordinateType>(1.57079633),
            static_cast<PointCoordinateType>(0),
            static_cast<PointCoordinateType>(1.57079633),
            CCVector3(0.0, m_DisplayScale / 2, m_DisplayScale / 2));
    if (transMat) {
        yzPlane_mtrx = *transMat * yzPlane_mtrx;
    }
    ccPlane yzPlane(m_DisplayScale, m_DisplayScale, &yzPlane_mtrx);
    yzPlane.setColor(ecvColor::yellow);
    return yzPlane;
}

ccPlane ccCoordinateSystem::createZXplane(const ccGLMatrix* transMat) const {
    ccGLMatrix zxPlane_mtrx;
    zxPlane_mtrx.initFromParameters(
            static_cast<PointCoordinateType>(0),
            static_cast<PointCoordinateType>(-1.57079633),
            static_cast<PointCoordinateType>(-1.57079633),
            CCVector3(m_DisplayScale / 2, 0, m_DisplayScale / 2));
    if (transMat) {
        zxPlane_mtrx = *transMat * zxPlane_mtrx;
    }
    ccPlane zxPlane(m_DisplayScale, m_DisplayScale, &zxPlane_mtrx);
    //    zxPlane.setColor(ecvColor::FromRgbfToRgb(ecvColor::Rgbf(0.0f,
    //    0.7f, 1.0f)));
    zxPlane.setColor(ecvColor::green);
    return zxPlane;
}

bool ccCoordinateSystem::buildUp() {
    // clear triangles indexes
    if (m_triVertIndexes) {
        m_triVertIndexes->clear();
    }
    // clear per triangle normals
    removePerTriangleNormalIndexes();
    if (m_triNormals) {
        m_triNormals->clear();
    }
    // clear vertices
    ccPointCloud* verts = vertices();
    if (verts) {
        verts->clear();
    }

    *this += createXYplane();
    *this += createYZplane();
    *this += createZXplane();

    return (vertices() && vertices()->size() == 12 && this->size() == 6);
}

ccGenericPrimitive* ccCoordinateSystem::clone() const {
    return finishCloneJob(new ccCoordinateSystem(m_DisplayScale, m_width,
                                                 &m_transformation, getName()));
}

bool ccCoordinateSystem::toFile_MeOnly(QFile& out, short dataVersion) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));
    if (dataVersion < 52) {
        assert(false);
        return false;
    }

    if (!ccGenericPrimitive::toFile_MeOnly(out, dataVersion)) return false;

    // parameters (dataVersion>=52)
    QDataStream outStream(&out);
    outStream << m_DisplayScale;
    outStream << m_width;

    return true;
}

short ccCoordinateSystem::minimumFileVersion_MeOnly() const {
    return std::max(static_cast<short>(52),
                    ccGenericPrimitive::minimumFileVersion_MeOnly());
}

bool ccCoordinateSystem::fromFile_MeOnly(QFile& in,
                                         short dataVersion,
                                         int flags,
                                         LoadedIDMap& oldToNewIDMap) {
    if (!ccGenericPrimitive::fromFile_MeOnly(in, dataVersion, flags,
                                             oldToNewIDMap))
        return false;

    // parameters (dataVersion>=52)
    QDataStream inStream(&in);
    ccSerializationHelper::CoordsFromDataStream(inStream, flags,
                                                &m_DisplayScale, 1);
    ccSerializationHelper::CoordsFromDataStream(inStream, flags, &m_width, 1);
    return true;
}

void ccCoordinateSystem::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (m_showAxisPlanes) {
        // call parent method to draw the Planes
        context.viewID = this->getViewId();
        ccGenericPrimitive::drawMeOnly(context);
    } else {
        ccGenericPrimitive::clearDrawings();
    }

    // show axis
    if (MACRO_Draw3D(context)) {
        if (m_showAxisLines) {
            if (ecvDisplayTools::GetCurrentScreen() == nullptr) return;

            // build-up the normal representation own 'context'
            CC_DRAW_CONTEXT tempContext = context;
            if (m_width != 0.0f) {
                tempContext.currentLineWidth =
                        static_cast<unsigned char>(m_width);
            }

            // coordinate system axis
            m_axis.clear();
            m_axis.points_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
            m_axis.points_.push_back(Eigen::Vector3d(
                    static_cast<double>(m_DisplayScale * 2), 0.0, 0.0));
            m_axis.points_.push_back(Eigen::Vector3d(
                    0.0, static_cast<double>(m_DisplayScale * 2), 0.0));
            m_axis.points_.push_back(Eigen::Vector3d(
                    0.0, 0.0, static_cast<double>(m_DisplayScale * 2)));

            // x axis
            m_axis.lines_.push_back(Eigen::Vector2i(0, 1));
            m_axis.colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));

            // y axis
            m_axis.lines_.push_back(Eigen::Vector2i(0, 2));
            m_axis.colors_.push_back(Eigen::Vector3d(1.0, 1.0, 0.0));

            // z axis
            m_axis.lines_.push_back(Eigen::Vector2i(0, 3));
            m_axis.colors_.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
            m_axis.setRedrawFlagRecursive(true);

            // transformation
            {
                Eigen::Matrix4d transformation =
                        ccGLMatrixd::ToEigenMatrix4(m_transformation);
                m_axis.Transform(transformation);
            }

            tempContext.viewID = m_axis.getViewId();
            ecvDisplayTools::Draw(tempContext, &m_axis);
        } else {
            ecvDisplayTools::RemoveEntities(&m_axis);
        }
    }
}
