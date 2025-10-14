// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_MOUSE_CIRCLE_HEADER
#define ECV_MOUSE_CIRCLE_HEADER

/**
This is a custom 2DViewportLabel which takes up the entire viewport but is
entirely transparent, except for a circle with radius r around the mouse.
*/
#include <ecv2DViewportObject.h>
#include <qevent.h>

#include <QObject>
#include <QPoint>

#include "ecvStdPluginInterface.h"

class ccMouseCircle : public cc2DViewportObject, public QObject {
public:
    // constructor
    explicit ccMouseCircle(QWidget* owner,
                           QString name = QString("MouseCircle"));

    // deconstructor
    virtual ~ccMouseCircle() override;

    // get the circle radius in px
    int getRadiusPx();

    // get the circle radius in world coordinates
    float getRadiusWorld();

    // removes the link with the owner (no cleanup)
    void ownerIsDead() { m_owner = nullptr; }

protected:
    // draws a circle of radius r around the mouse
    void draw(CC_DRAW_CONTEXT& context) override;

private:
    // QWidget this overlay is attached to -> used to get mouse position &
    // events
    QWidget* m_owner;
    float m_winTotalZoom;

    // event to get mouse-move updates & trigger repaint
    bool eventFilter(QObject* obj, QEvent* event) override;

public:
    static const int RESOLUTION = 100;
    int RADIUS = 50;
    int RADIUS_STEP = 4;
    float UNIT_CIRCLE[RESOLUTION][2];
};

#endif  // ECV_MOUSE_CIRCLE_HEADER
