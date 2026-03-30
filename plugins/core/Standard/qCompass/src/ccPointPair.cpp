// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccPointPair.h"

#include <ecvDisplayTools.h>

// static sphere for drawing with
static QSharedPointer<ccSphere> c_unitPointMarker(nullptr);
static QSharedPointer<ccCylinder> c_bodyMarker(nullptr);
static QSharedPointer<ccCone> c_headMarker(nullptr);

// ctor
ccPointPair::ccPointPair(ccPointCloud* associatedCloud)
    : ccPolyline(associatedCloud) {
    // do nothing
}

ccPointPair::ccPointPair(ccPolyline* obj)
    : ccPolyline(obj->getAssociatedCloud()) {
    // load points
    for (unsigned i = 0; i < obj->size(); i++) {
        int pId = obj->getPointGlobalIndex(i);  // get global point ID
        addPointIndex(pId);                     // add point to this polyline
    }

    // copy name
    setName(obj->getName());
}

CCVector3 ccPointPair::getDirection() {
    if (size() != 2) {
        return CCVector3();  // null vector
    } else {
        const CCVector3 start = *getPoint(0);
        const CCVector3 end = *getPoint(1);
        return end - start;
    }
}

// overidden from ccHObject
void ccPointPair::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (!MACRO_Foreground(context))  // 2D foreground only
        return;                      // do nothing

    if (MACRO_Draw3D(context)) {
        if (size() == 0)  // no points -> bail!
            return;

        // get the set of OpenGL functions (version 2.1)
        if (ecvDisplayTools::GetCurrentScreen() == nullptr) {
            assert(false);
            return;
        }

        bool entityPickingMode = MACRO_EntityPicking(context);
        if (entityPickingMode) {
            if (MACRO_FastEntityPicking(context)) {
                return;
            }
        }

        if (!c_unitPointMarker) {
            c_unitPointMarker.reset(
                    new ccSphere(1.0f, nullptr, "PointMarker", 6));
            c_unitPointMarker->showColors(true);
            c_unitPointMarker->setVisible(true);
            c_unitPointMarker->setEnabled(true);
            c_unitPointMarker->showNormals(true);
        }

        if (!c_bodyMarker) {
            c_bodyMarker.reset(
                    new ccCylinder(1.0f, 0.9f, nullptr, "UnitNormal", 12));
            c_bodyMarker->showColors(true);
            c_bodyMarker->setVisible(true);
            c_bodyMarker->setEnabled(true);
            c_bodyMarker->setTempColor(ecvColor::green);
            c_bodyMarker->showNormals(false);
        }
        if (!c_headMarker) {
            c_headMarker.reset(new ccCone(
                    2.5f, 0.0f, 0.1f, 0, 0, nullptr, "UnitNormalHead", 12));
            c_headMarker->showColors(true);
            c_headMarker->setVisible(true);
            c_headMarker->setEnabled(true);
            c_headMarker->setTempColor(ecvColor::green);
            c_headMarker->showNormals(false);
        }

        CC_DRAW_CONTEXT markerContext = context;
        markerContext.drawingFlags &= (~CC_ENTITY_PICKING);

        ccGLCameraParameters camera;
        ecvDisplayTools::GetGLCameraParameters(camera);

        ecvColor::Rgb color = entityPickingMode
                                      ? ecvColor::Rgb(255, 255, 255)
                                      : getMeasurementColour();
        c_unitPointMarker->setTempColor(color);

        float pSize = 1.0f;

        const ecvViewportParameters& viewportParams =
                ecvDisplayTools::GetViewportParameters();
        for (unsigned i = 0; i < size(); i++) {
            const CCVector3* P = getPoint(i);
            markerContext.transformInfo.setTranslationStart(
                    CCVector3(P->x, P->y, P->z));
            float scale = context.labelMarkerSize * m_relMarkerScale * 0.2f *
                          fmin(pSize, 4.0f);
            if (viewportParams.perspectiveView && viewportParams.zFar > 0) {
                double d = (camera.modelViewMat * (*P)).norm();
                double unitD = viewportParams.zFar / 2;
                scale = static_cast<float>(scale * sqrt(d / unitD));
            }
            markerContext.transformInfo.setScale(
                    CCVector3(scale, scale, scale));
            c_unitPointMarker->showNormals(!entityPickingMode);
            c_unitPointMarker->draw(markerContext);
        }

        c_bodyMarker->setTempColor(color);
        c_headMarker->setTempColor(color);
        if (size() == 2)  // two points
        {
            const CCVector3 start = *getPoint(0);
            const CCVector3 end = *getPoint(1);

            CCVector3 disp = end - start;
            float length = disp.norm();
            float width = context.labelMarkerSize * m_relMarkerScale * 0.05 *
                          std::fmin(pSize, 5);
            CCVector3 dir = disp / length;

            // transform into coord space with origin at start and arrow head at
            // 0,0,1 (unashamedly pilfered from
            // ccPlanarEntityInterface::glDrawNormal(...)
            // glFunc->glMatrixMode(GL_MODELVIEW);
            markerContext.transformInfo.setTranslationStart(
                    CCVector3(start.x, start.y, start.z));

            ccGLMatrix mat = ccGLMatrix::FromToRotation(
                    CCVector3(0, 0, PC_ONE),
                    CCVector3(dir.x, dir.y, dir.z));  // end = 0,0,1
            markerContext.transformInfo.setTransformation(mat, false);
            markerContext.transformInfo.setScale(
                    CCVector3(width, width, length));

            markerContext.transformInfo.setTranslationEnd(
                    CCVector3(0, 0, 0.45f));
            c_bodyMarker->draw(markerContext);

            markerContext.transformInfo.setTranslationEnd(
                    CCVector3(0, 0, 0.9f));
            c_headMarker->draw(markerContext);
        }

        // finish picking name
    }
}

// returns true if object is a pointPair
bool ccPointPair::isPointPair(ccHObject* object) {
    if (object->hasMetaData("ccCompassType")) {
        return object->getMetaData("ccCompassType")
                       .toString()
                       .contains("PointPair") |
               object->getMetaData("ccCompassType")
                       .toString()
                       .contains("Lineation") |
               object->getMetaData("ccCompassType")
                       .toString()
                       .contains("Thickness") |
               object->getMetaData("ccCompassType")
                       .toString()
                       .contains("PinchNode") |
               object->getMetaData("ccCompassType")
                       .toString()
                       .contains("Relationship");
    }
    return false;
}
