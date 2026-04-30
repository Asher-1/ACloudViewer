// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccSNECloud.h"

#include <ecvPolyline.h>
#include <ecvScalarField.h>
#include <ecvViewManager.h>

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

ccSNECloud::~ccSNECloud() { removeNormalActors(); }

void ccSNECloud::removeNormalActors() {
    ecvGenericGLDisplay* eff = ecvViewManager::instance().getEffectiveView();
    if (!eff) {
        m_normalViewIds.clear();
        return;
    }
    for (const QString& viewId : m_normalViewIds) {
        CC_DRAW_CONTEXT ctx;
        ctx.display = eff;
        ctx.defaultViewPort = 0;
        ctx.removeViewID = viewId;
        ctx.removeEntityType = ENTITY_TYPE::ECV_LINES_3D;
        if (ctx.display) ctx.display->removeEntities(ctx);
    }
    m_normalViewIds.clear();
}

void ccSNECloud::updateMetadata() { setMetaData("ccCompassType", "SNECloud"); }

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

        if (ecvViewManager::instance().activeWidget() == nullptr) {
            assert(false);
            return;
        }

        removeNormalActors();

        int thickID = getScalarFieldIndexByName("Thickness");
        if (thickID == -1) {
            thickID = getScalarFieldIndexByName("Spacing");
        }

        float lineWidth = 1.0f;
        QString baseViewId = getViewId();

        ccPointCloud lineCloud;
        lineCloud.reserve(size() * 2);
        lineCloud.reserveTheRGBTable();

        struct SegmentInfo {
            unsigned startIdx;
            unsigned endIdx;
        };
        std::vector<SegmentInfo> segments;
        segments.reserve(size());

        unsigned lineIdx = 0;
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

            unsigned si = lineCloud.size();
            lineCloud.addPoint(start);
            lineCloud.addRGBColor(lineColor);
            lineCloud.addPoint(end);
            lineCloud.addRGBColor(lineColor);
            segments.push_back({si, si + 1});
            lineIdx++;
        }

        if (segments.empty()) {
            return;
        }

        CC_DRAW_CONTEXT lineContext = context;
        lineContext.opacity = 0.78;

        for (size_t i = 0; i < segments.size(); i++) {
            ccPolyline segment(&lineCloud);
            segment.addPointIndex(segments[i].startIdx);
            segment.addPointIndex(segments[i].endIdx);
            segment.setVisible(true);
            segment.showColors(true);
            segment.setWidth(static_cast<PointCoordinateType>(lineWidth));
            segment.setFixedId(true);

            QString viewId = baseViewId + "-sne" + QString::number(i);
            lineContext.viewID = viewId;
            segment.draw(lineContext);
            m_normalViewIds.append(viewId);
        }
    }
}
