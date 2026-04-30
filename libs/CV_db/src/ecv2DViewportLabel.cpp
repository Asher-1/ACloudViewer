// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecv2DViewportLabel.h"

#include <ecvGenericGLDisplay.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

// CV_CORE_LIB
#include <CVConst.h>

// Qt
#include <QDataStream>
#include <QFontMetrics>

// system
#include <string.h>

cc2DViewportLabel::cc2DViewportLabel(QString name /*=QString()*/)
    : cc2DViewportObject(name) {
    // label rectangle
    memset(m_roi, 0, sizeof(float) * 4);
    setVisible(false);
}

void cc2DViewportLabel::setRoi(const float* roi) {
    memcpy(m_roi, roi, sizeof(float) * 4);
}

bool cc2DViewportLabel::toFile_MeOnly(QFile& out, short dataVersion) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));
    if (dataVersion < 21) {
        assert(false);
        return false;
    }

    if (!cc2DViewportObject::toFile_MeOnly(out, dataVersion)) return false;

    // ROI (dataVersion>=21)
    QDataStream outStream(&out);
    for (int i = 0; i < 4; ++i) outStream << m_roi[i];

    return true;
}

short cc2DViewportLabel::minimumFileVersion_MeOnly() const {
    return std::max(static_cast<short>(21),
                    cc2DViewportObject::minimumFileVersion_MeOnly());
}

bool cc2DViewportLabel::fromFile_MeOnly(QFile& in,
                                        short dataVersion,
                                        int flags,
                                        LoadedIDMap& oldToNewIDMap) {
    if (!cc2DViewportObject::fromFile_MeOnly(in, dataVersion, flags,
                                             oldToNewIDMap))
        return false;

    if (dataVersion < 21) return false;

    // ROI (dataVersion>=21)
    QDataStream inStream(&in);
    for (int i = 0; i < 4; ++i) inStream >> m_roi[i];

    return true;
}

void cc2DViewportLabel::clear2Dviews() {
    ecvGenericGLDisplay* view = ecvViewManager::instance().getEffectiveView();
    if (!view || !view->asWidget()) return;

    CC_DRAW_CONTEXT ctx;
    ctx.display = view;
    ctx.defaultViewPort = 0;
    ctx.removeEntityType = ENTITY_TYPE::ECV_TRIANGLE_2D;
    ctx.removeViewID = this->getViewId();

    if (ctx.display) ctx.display->removeEntities(ctx);
}

void cc2DViewportLabel::updateLabel() {
    CC_DRAW_CONTEXT context;
    ecvGenericGLDisplay* view = ecvViewManager::instance().getEffectiveView();
    if (view) {
        view->getContext(context);
    } else {
        return;
    }

    update2DLabelView(context, true);
}

void cc2DViewportLabel::update2DLabelView(CC_DRAW_CONTEXT& context,
                                          bool updateScreen /* = true */) {
    context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
    drawMeOnly(context);
    if (updateScreen) {
        {
            ecvRedrawScope scope;
        }
    }
}

void cc2DViewportLabel::drawMeOnly(CC_DRAW_CONTEXT& context) {
    // 2D foreground only
    if (!MACRO_Foreground(context) || !MACRO_Draw2D(context)) return;

    if (!context.display || !context.display->asWidget()) return;

    // clear history (widgets for this viewport label in this view only)
    {
        CC_DRAW_CONTEXT clearCtx;
        clearCtx.display = context.display;
        clearCtx.defaultViewPort = 0;
        clearCtx.removeEntityType = ENTITY_TYPE::ECV_TRIANGLE_2D;
        clearCtx.removeViewID = getViewId();
        if (clearCtx.display) clearCtx.display->removeEntities(clearCtx);
    }
    if (!isVisible() || !isEnabled()) {
        return;
    }

    // test viewport parameters
    const ecvViewportParameters& params =
            context.display->getViewportParameters();

    // general parameters
    if (params.perspectiveView != m_params.perspectiveView ||
        params.objectCenteredView != m_params.objectCenteredView ||
        params.pixelSize != m_params.pixelSize) {
        return;
    }

    // test base view matrix
    for (unsigned i = 0; i < 12; ++i) {
        if (cloudViewer::GreaterThanEpsilon(fabs(params.viewMat.data()[i] -
                                                 m_params.viewMat.data()[i]))) {
            return;
        }
    }

    if (m_params.perspectiveView) {
        if (params.fov_deg != m_params.fov_deg ||
            params.cameraAspectRatio != m_params.cameraAspectRatio)
            return;

        if (cloudViewer::GreaterThanEpsilon(
                    (params.getPivotPoint() - m_params.getPivotPoint())
                            .norm()) ||
            cloudViewer::GreaterThanEpsilon(
                    (params.getCameraCenter() - m_params.getCameraCenter())
                            .norm())) {
            return;
        }
    }

    float relativeZoom = 1.0f;
    float dx = 0, dy = 0;
    if (!m_params.perspectiveView)  // ortho mode
    {
        // Screen pan & pivot compensation
        float totalZoom = m_params.zoom / m_params.pixelSize;
        float winTotalZoom = params.zoom / params.pixelSize;
        relativeZoom = winTotalZoom / totalZoom;

        CCVector3d dC = m_params.getCameraCenter() - params.getCameraCenter();

        CCVector3d P = m_params.getPivotPoint() - params.getPivotPoint();
        m_params.viewMat.apply(P);

        dx = static_cast<float>(dC.x + P.x);
        dy = static_cast<float>(dC.y + P.y);

        dx *= winTotalZoom;
        dy *= winTotalZoom;
    }

    const ecvColor::Rgb* defaultColor =
            m_selected ? &ecvColor::red : &context.textDefaultCol;

    WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_TRIANGLE_2D,
                            this->getViewId());
    param.context.display = context.display;
    const ecvColor::Rgbf& tempColor = ecvColor::FromRgb(*defaultColor);
    param.color.r = tempColor.r;
    param.color.g = tempColor.g;
    param.color.b = tempColor.b;
    param.color.a = 1.0f;
    param.p1 =
            QPoint(dx + m_roi[0] * relativeZoom, dy + m_roi[1] * relativeZoom);
    param.p2 =
            QPoint(dx + m_roi[2] * relativeZoom, dy + m_roi[1] * relativeZoom);
    param.p3 =
            QPoint(dx + m_roi[2] * relativeZoom, dy + m_roi[3] * relativeZoom);
    param.p4 =
            QPoint(dx + m_roi[0] * relativeZoom, dy + m_roi[3] * relativeZoom);

    if (context.display) context.display->drawWidgets(param);

    // title
    QString title(getName());
    if (!title.isEmpty()) {
        QFont titleFont(context.display->textDisplayFont());
        titleFont.setBold(true);
        QFontMetrics titleFontMetrics(titleFont);
        int titleHeight = titleFontMetrics.height();

        int xStart = static_cast<int>(dx + std::min<float>(m_roi[0], m_roi[2]) *
                                                   relativeZoom);
        int yStart = static_cast<int>(dy + std::min<float>(m_roi[1], m_roi[3]) *
                                                   relativeZoom);

        context.display->display2DText(
                title, xStart, yStart - 5 - titleHeight,
                static_cast<unsigned char>(
                        ecvGenericDisplayTools::ALIGN_DEFAULT),
                0.0f, defaultColor->rgb, &titleFont, this->getViewId());
    }
}
