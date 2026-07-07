// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccMouseCircle.h"

#include <CVLog.h>
#include <ecvGenericGLDisplay.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

#include <QWheelEvent>
#include <algorithm>
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
    if (ecvGenericGLDisplay* eff =
                ecvViewManager::instance().getEffectiveView()) {
        eff->addToOwnDB(this, true);
    }
}

ccMouseCircle::~ccMouseCircle() {
    if (m_owner) {
        m_owner->removeEventFilter(this);

        if (ecvGenericGLDisplay* eff =
                    ecvViewManager::instance().getEffectiveView()) {
            CC_DRAW_CONTEXT ctx;
            ctx.display = eff;
            ctx.defaultViewPort = 0;
            ctx.removeEntityType = ENTITY_TYPE::ECV_CIRCLE_2D;
            ctx.removeViewID = getViewId();
            if (ctx.display) {
                ctx.display->removeEntities(ctx);
            }

            eff->removeFromOwnDB(this);
        }
    }
}

float ccMouseCircle::getRadiusWorld() {
    ecvGenericGLDisplay* eff = ecvViewManager::instance().getEffectiveView();
    if (eff) {
        m_pixelSize =
                static_cast<float>(std::abs(eff->computeActualPixelSize()));
    }
    if (m_pixelSize <= 0) {
        return 0.0f;
    }
    float r = static_cast<float>(getRadiusPx()) * m_pixelSize;
    CVLog::Print(QString("Radius_w = %1 (= %2 x %3)")
                         .arg(r)
                         .arg(getRadiusPx())
                         .arg(m_pixelSize));
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

    if (!ecvViewManager::instance().activeWidget()) {
        return;
    }

    ecvGenericGLDisplay* effView =
            ecvViewManager::instance().getEffectiveView();
    if (!effView) {
        return;
    }

    m_pixelSize =
            static_cast<float>(std::abs(effView->computeActualPixelSize()));

    {
        CC_DRAW_CONTEXT clr;
        clr.display = effView;
        clr.defaultViewPort = 0;
        clr.removeEntityType = ENTITY_TYPE::ECV_CIRCLE_2D;
        clr.removeViewID = getViewId();
        if (clr.display) {
            clr.display->removeEntities(clr);
        }
    }

    QPoint p =
            m_owner->mapFromGlobal(QCursor::pos()) * context.devicePixelRatio;
    int mx = p.x();
    int my = context.glH - 1 - p.y();

    WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_CIRCLE_2D, getViewId());
    param.context.display = effView;
    param.rect = QRect(mx, my, 0, 0);
    param.radius = static_cast<float>(m_radius);
    param.color = ecvColor::Rgbaf(1.0f, 0.0f, 0.0f, 0.8f);
    if (param.context.display) {
        param.context.display->drawWidgets(param);
    }
}

bool ccMouseCircle::eventFilter(QObject* obj, QEvent* event) {
    if (!ccMouseCircle::isVisible()) {
        return false;
    }

    if (event->type() == QEvent::MouseMove) {
        if (m_owner) {
            {
                ecvRedrawScope redrawScope(true, true);
            }
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
            { ecvRedrawScope redrawScope(true, true); }
        }
    }
    return false;
}
