// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccSNECloud.h"

#include <ecvDisplayTools.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>

ccSNECloud::ccSNECloud() : ccPointCloud() { updateMetadata(); }

ccSNECloud::ccSNECloud(ccPointCloud* obj) : ccPointCloud() {
    if (obj) {
        *this += obj;
        copyGlobalShiftAndScale(*obj);
        setName(obj->getName());
    } else {
        assert(false);
    }

    updateMetadata();
}

void ccSNECloud::updateMetadata() {
    setMetaData("ccCompassType", "SNECloud");
}

bool ccSNECloud::isSNECloud(ccHObject* object) {
    if (object->hasMetaData("ccCompassType")) {
        return object->getMetaData("ccCompassType")
                .toString()
                .contains("SNECloud");
    }
    return false;
}

void ccSNECloud::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (!MACRO_Foreground(context)) {
        return;
    }

    ccPointCloud::drawMeOnly(context);

    if (MACRO_Draw3D(context)) {
        if (size() == 0) {
            return;
        }

        if (ecvDisplayTools::GetCurrentScreen() == nullptr) {
            assert(false);
            return;
        }

        int thickID = getScalarFieldIndexByName("Thickness");
        if (thickID == -1) {
            thickID = getScalarFieldIndexByName("Spacing");
        }

        float lineWidth = static_cast<float>(context.defaultPointSize);

        ccPointCloud lineCloud;
        lineCloud.reserve(size() * 2);
        lineCloud.reserveTheRGBTable();

        for (unsigned p = 0; p < size(); p++) {
            if (m_currentDisplayedScalarField != nullptr) {
                if (!m_currentDisplayedScalarField->areNaNValuesShownInGrey()) {
                    if (!m_currentDisplayedScalarField->displayRange()
                                 .isInRange(m_currentDisplayedScalarField
                                                    ->getValue(p))) {
                        continue;
                    }
                }
            }

            if (isVisibilityTableInstantiated()) {
                if (!m_pointsVisibility.empty() &&
                    m_pointsVisibility[p] != POINT_VISIBLE) {
                    continue;
                }
            }

            float length = 1.0f;
            if (thickID != -1) {
                length = getScalarField(thickID)->getValue(p);
            }

            const CCVector3 start = *getPoint(p);
            CCVector3 end = start + (getPointNormal(p) * length);

            ecvColor::Rgb lineColor(200, 200, 200);
            if (m_currentDisplayedScalarField != nullptr) {
                const ecvColor::Rgb* col =
                        m_currentDisplayedScalarField->getColor(
                                m_currentDisplayedScalarField->getValue(p));
                if (col) {
                    lineColor = *col;
                }
            }

            lineCloud.addPoint(start);
            lineCloud.addRGBColor(lineColor);
            lineCloud.addPoint(end);
            lineCloud.addRGBColor(lineColor);
        }

        unsigned nPts = lineCloud.size();
        if (nPts < 2) {
            return;
        }

        for (unsigned i = 0; i < nPts; i += 2) {
            ccPolyline segment(&lineCloud);
            segment.addPointIndex(i);
            segment.addPointIndex(i + 1);
            segment.setVisible(true);
            segment.showColors(true);
            segment.setWidth(static_cast<PointCoordinateType>(lineWidth));
            segment.draw(context);
        }
    }
}

