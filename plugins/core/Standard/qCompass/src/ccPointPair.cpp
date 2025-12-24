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

        // push name for picking
        bool entityPickingMode = MACRO_EntityPicking(context);

        // check sphere exists
        if (!c_unitPointMarker) {
            c_unitPointMarker = QSharedPointer<ccSphere>(
                    new ccSphere(1.0f, 0, "PointMarker", 6));

            c_unitPointMarker->showColors(true);
            c_unitPointMarker->setVisible(true);
            c_unitPointMarker->setEnabled(true);
        }

        // check arrow parts exist
        if (!c_bodyMarker) {
            c_bodyMarker = QSharedPointer<ccCylinder>(
                    new ccCylinder(1.0f, 0.9f, 0, "UnitNormal", 12));
            c_bodyMarker->showColors(true);
            c_bodyMarker->setVisible(true);
            c_bodyMarker->setEnabled(true);
            c_bodyMarker->setTempColor(ecvColor::green);
            c_bodyMarker->showNormals(false);
        }
        if (!c_headMarker) {
            c_headMarker = QSharedPointer<ccCone>(new ccCone(
                    2.5f, 0.0f, 0.1f, 0, 0, 0, "UnitNormalHead", 12));
            c_headMarker->showColors(true);
            c_headMarker->setVisible(true);
            c_headMarker->setEnabled(true);
            c_headMarker->setTempColor(ecvColor::green);
            c_headMarker->showNormals(false);
        }

        // not sure what this does, but it looks like fun
        CC_DRAW_CONTEXT markerContext =
                context;  // build-up point maker own 'context'
        markerContext.drawingFlags &=
                (~CC_ENTITY_PICKING);  // we must remove the 'push name flag'
                                       // so that the sphere doesn't push its
                                       // own!

        // get camera info
        ccGLCameraParameters camera;
        ecvDisplayTools::GetGLCameraParameters(camera);

        // set draw colour
        c_unitPointMarker->setTempColor(getMeasurementColour());

        // get point size for drawing
        float pSize = markerContext.defaultPointSize;

        // draw points
        const ecvViewportParameters& viewportParams =
                ecvDisplayTools::GetViewportParameters();
        for (unsigned i = 0; i < size(); i++) {
            const CCVector3* P = getPoint(i);
            // glFunc->glMatrixMode(GL_MODELVIEW);
            // glFunc->glPushMatrix();
            // ccGL::Translate(glFunc, P->x, P->y, P->z);
            markerContext.transformInfo.setTranslationStart(
                    CCVector3(P->x, P->y, P->z));
            float scale = context.labelMarkerSize * m_relMarkerScale * 0.2 *
                          fmin(pSize, 4);
            if (viewportParams.perspectiveView && viewportParams.zFar > 0) {
                // in perspective view, the actual scale depends on the distance
                // to the camera!
                const double* M = camera.modelViewMat.data();
                double d = (camera.modelViewMat * CCVector3d::fromArray(P->u))
                                   .norm();
                double unitD = viewportParams.zFar /
                               2;  // we consider that the 'standard' scale is
                                   // at half the depth
                scale = static_cast<float>(
                        scale *
                        sqrt(d /
                             unitD));  // sqrt = empirical (probably because the
                                       // marker size is already partly
                                       // compensated by
                                       // ecvDisplayTools::computeActualPixelSize())
            }
            // glFunc->glScalef(scale, scale, scale);
            markerContext.transformInfo.setScale(
                    CCVector3(scale, scale, scale));
            c_unitPointMarker->draw(markerContext);
            // glFunc->glPopMatrix();
        }

        // draw arrow
        c_bodyMarker->setTempColor(getMeasurementColour());
        c_headMarker->setTempColor(getMeasurementColour());
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

            // draw arrow body
            markerContext.transformInfo.setTranslationEnd(
                    CCVector3(0, 0, 0.45f));
            c_bodyMarker->draw(markerContext);

            // draw arrow head
            markerContext.transformInfo.setTranslationEnd(
                    CCVector3(0, 0, 0.45f));
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
