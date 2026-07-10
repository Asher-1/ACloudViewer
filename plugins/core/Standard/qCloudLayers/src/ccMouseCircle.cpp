// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../include/ccMouseCircle.h"

#include <ecvGenericGLDisplay.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

// Qt
#include <QWheelEvent>

// System
#include <algorithm>
#include <cmath>

ccMouseCircle::ccMouseCircle(ecvMainAppInterface* appInterface,
                             QWidget* owner,
                             QString name)
    : cc2DViewportObject(name.isEmpty() ? "label" : name),
      m_app(appInterface),
      m_radius(50),
      m_radiusStep(4),
      m_allowScroll(true) {
    setVisible(true);
    setEnabled(false);

    // attach to owner
    assert(owner);  // check valid pointer
    ccMouseCircle::m_owner = owner;
    m_owner->installEventFilter(this);
    m_bindView = ecvViewManager::instance().getEffectiveView();
    if (m_bindView) {
        m_bindView->addToOwnDB(this, true);
    }
}

ccMouseCircle::~ccMouseCircle() {
    if (m_owner) {
        m_owner->removeEventFilter(this);
        if (m_bindView) {
            CC_DRAW_CONTEXT clr;
            clr.display = m_bindView;
            clr.defaultViewPort = 0;
            clr.removeEntityType = ENTITY_TYPE::ECV_CIRCLE_2D;
            clr.removeViewID = getViewId();
            m_bindView->removeEntities(clr);
            m_bindView->removeFromOwnDB(this);
        }
    }
}

void ccMouseCircle::setOwner(QWidget* newOwner) {
    if (m_owner == newOwner) return;
    if (m_owner) {
        m_owner->removeEventFilter(this);
    }
    m_owner = newOwner;
    if (m_owner) {
        m_owner->installEventFilter(this);
    }
}

void ccMouseCircle::setBindView(ecvGenericGLDisplay* view) {
    if (m_bindView == view) return;
    if (m_bindView) {
        CC_DRAW_CONTEXT clr;
        clr.display = m_bindView;
        clr.defaultViewPort = 0;
        clr.removeEntityType = ENTITY_TYPE::ECV_CIRCLE_2D;
        clr.removeViewID = getViewId();
        m_bindView->removeEntities(clr);
        m_bindView->removeFromOwnDB(this);
    }
    m_bindView = view;
    if (m_bindView) {
        m_bindView->addToOwnDB(this, true);
    }
}

// get the circle radius in world coordinates
float ccMouseCircle::getRadiusWorld() { return getRadiusPx() * m_pixelSize; }

// override draw function
void ccMouseCircle::draw(CC_DRAW_CONTEXT& context) {
    if (!m_owner) {
        assert(false);
        return;
    }

    // only draw when visible
    if (!ccMouseCircle::isVisible()) {
        return;
    }

    // only draw in 2D foreground mode
    if (!MACRO_Foreground(context) || !MACRO_Draw2D(context)) {
        return;
    }

    ecvGenericGLDisplay* drawView =
            context.display ? context.display : m_bindView;
    if (!drawView) {
        return;
    }

    m_pixelSize =
            static_cast<float>(std::abs(drawView->computeActualPixelSize()));
    if (m_pixelSize <= 0) {
        return;
    }

    {
        CC_DRAW_CONTEXT clr;
        clr.display = drawView;
        clr.defaultViewPort = 0;
        clr.removeEntityType = ENTITY_TYPE::ECV_CIRCLE_2D;
        clr.removeViewID = getViewId();
        drawView->removeEntities(clr);
    }

    // get mouse position
    QPoint p =
            m_owner->mapFromGlobal(QCursor::pos()) * context.devicePixelRatio;
    int mx = p.x();
    int my = context.glH - 1 - p.y();

    WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_CIRCLE_2D, getViewId());
    param.context.display = drawView;
    param.rect = QRect(mx, my, 0, 0);
    param.radius = static_cast<float>(m_radius);
    param.color = ecvColor::Rgbaf(1.0f, 0.0f, 0.0f, 0.8f);
    if (param.context.display) {
        param.context.display->drawWidgets(param);
    }
}

bool ccMouseCircle::eventFilter(QObject* obj, QEvent* event) {
    if (!ccMouseCircle::isVisible()) return false;

    if (event->type() == QEvent::MouseMove) {
        if (m_bindView) {
            m_bindView->redraw(true, false);
        }
    } else if (event->type() == QEvent::Wheel && m_allowScroll) {
        QWheelEvent* wheelEvent = static_cast<QWheelEvent*>(event);

        double delta = qtCompatWheelEventDelta(wheelEvent);
        m_radius = std::max(
                m_radiusStep,
                m_radius - static_cast<int>(m_radiusStep * (delta / 100.0)));

        if (m_bindView) {
            m_bindView->redraw(true, false);
        }
    }
    return false;
}
