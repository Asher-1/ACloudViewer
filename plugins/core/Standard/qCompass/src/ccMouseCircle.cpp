// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccMouseCircle.h"

#include <CVLog.h>
#include <ecvDisplayTools.h>

#include <QWheelEvent>
#include <cmath>

ccMouseCircle::ccMouseCircle(QWidget* owner, QString name)
    : cc2DViewportObject(name.isEmpty() ? "label" : name),
      m_owner(nullptr),
      m_pixelSize(0.0f),
      m_radius(50),
      m_radiusStep(4) {
    setVisible(true);
    setEnabled(false);

    assert(owner);
    m_owner = owner;
    m_owner->installEventFilter(this);
    ecvDisplayTools::AddToOwnDB(this, true);
}

ccMouseCircle::~ccMouseCircle() {
    if (m_owner) {
        m_owner->removeEventFilter(this);

        // Remove the circle overlay widget
        WIDGETS_PARAMETER removeParam(WIDGETS_TYPE::WIDGET_CIRCLE_2D,
                                      getViewId());
        ecvDisplayTools::RemoveWidgets(removeParam);

        ecvDisplayTools::RemoveFromOwnDB(this);
    }
}

float ccMouseCircle::computeOrthoPixelSize(int viewportHeight) {
    double parallelScale = ecvDisplayTools::GetParallelScale(0);
    if (parallelScale <= 0 || viewportHeight <= 0) {
        return 0.0f;
    }
    return static_cast<float>(2.0 * parallelScale / viewportHeight);
}

float ccMouseCircle::getRadiusWorld() {
    if (m_pixelSize == 0) {
        const ecvViewportParameters& params =
                ecvDisplayTools::GetViewportParameters();
        QWidget* screen = ecvDisplayTools::GetCurrentScreen();
        if (screen) {
            if (params.perspectiveView) {
                m_pixelSize = static_cast<float>(
                        std::abs(params.computePixelSize(screen->width())));
            } else {
                m_pixelSize = computeOrthoPixelSize(screen->height());
            }
        }
    }
    float r = static_cast<float>(getRadiusPx()) * m_pixelSize;
    return r;
}

void ccMouseCircle::draw(CC_DRAW_CONTEXT& context) {
    if (!m_owner) {
        assert(false);
        return;
    }

    if (!ccMouseCircle::isVisible()) {
        return;
    }

    if (!MACRO_Foreground(context) || !MACRO_Draw2D(context)) {
        return;
    }

    if (ecvDisplayTools::GetCurrentScreen() == nullptr) {
        return;
    }

    const ecvViewportParameters& params =
            ecvDisplayTools::GetViewportParameters();
    if (params.perspectiveView) {
        m_pixelSize = static_cast<float>(
                std::abs(params.computePixelSize(context.glW)));
    } else {
        m_pixelSize = computeOrthoPixelSize(context.glH);
    }

    {
        WIDGETS_PARAMETER removeParam(WIDGETS_TYPE::WIDGET_CIRCLE_2D,
                                      getViewId());
        ecvDisplayTools::RemoveWidgets(removeParam);
    }

    QPoint p =
            m_owner->mapFromGlobal(QCursor::pos()) * context.devicePixelRatio;
    int mx = p.x();
    int my = context.glH - 1 - p.y();

    WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_CIRCLE_2D, getViewId());
    param.rect = QRect(mx, my, 0, 0);
    param.radius = static_cast<float>(m_radius);
    param.color = ecvColor::Rgbaf(1.0f, 0.0f, 0.0f, 0.8f);
    ecvDisplayTools::DrawWidgets(param, false);
}

bool ccMouseCircle::eventFilter(QObject* obj, QEvent* event) {
    if (!ccMouseCircle::isVisible()) {
        return false;
    }

    if (event->type() == QEvent::MouseMove) {
        if (m_owner) {
            ecvDisplayTools::RedrawDisplay(true, true);
        }
    }

    if (event->type() == QEvent::Wheel) {
        QWheelEvent* wheelEvent = static_cast<QWheelEvent*>(event);

        if (wheelEvent->modifiers().testFlag(Qt::ControlModifier)) {
            m_radius = std::max(
                    m_radiusStep,
                    m_radius - static_cast<int>(
                                       m_radiusStep *
                                       (wheelEvent->angleDelta().y() / 100.0)));
            ecvDisplayTools::RedrawDisplay(true, true);
        }
    }
    return false;
}
