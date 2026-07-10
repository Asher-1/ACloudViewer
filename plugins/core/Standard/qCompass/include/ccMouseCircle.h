// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
This is a custom 2DViewportLabel which takes up the entire viewport but is
entirely transparent, except for a circle with radius r around the mouse.
*/
#include <ecv2DViewportObject.h>

#include <QObject>

class ecvGenericGLDisplay;

class ccMouseCircle : public cc2DViewportObject, public QObject {
public:
    explicit ccMouseCircle(QWidget* owner,
                           QString name = QString("MouseCircle"));

    virtual ~ccMouseCircle() override;

    //! Get the circle radius in pixels
    int getRadiusPx() { return m_radius; }

    //! Get the circle radius in world coordinates
    float getRadiusWorld();

    //! Removes the link with the owner (no cleanup)
    void ownerIsDead() { m_owner = nullptr; }

protected:
    void draw(CC_DRAW_CONTEXT& context) override;

private:
    QWidget* m_owner;
    ecvGenericGLDisplay* m_bindView = nullptr;
    float m_pixelSize;
    int m_radius;
    int m_radiusStep;

    bool eventFilter(QObject* obj, QEvent* event) override;
};
