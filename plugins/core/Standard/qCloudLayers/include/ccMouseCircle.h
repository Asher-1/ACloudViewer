//##########################################################################
//#                                                                        #
//#                    CloudViewer PLUGIN: ccCompass                      #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                     COPYRIGHT: Sam Thiele  2017                        #
//#                                                                        #
//##########################################################################

#pragma once

/**
This is a custom 2DViewportLabel which takes up the entire viewport but is
entirely transparent, except for a circle with radius r around the mouse.
*/

#include <ecv2DViewportObject.h>
#include <ecvStdPluginInterface.h>

#include <ecvDisplayTools.h>

// Qt
#include <QEvent>
#include <QObject>
#include <QPoint>

class ccMouseCircle : public cc2DViewportObject, public QObject {
public:
    // constructor
    explicit ccMouseCircle(ecvMainAppInterface* appInterface,
                           QWidget* owner,
                           QString name = QString("MouseCircle"));

    // deconstructor
    ~ccMouseCircle();

    // get the circle radius in px
    inline int getRadiusPx() const { return m_radius; }

    // get the circle radius in world coordinates
    float getRadiusWorld();

    // removes the link with the owner (no cleanup)
    inline void ownerIsDead() { m_owner = nullptr; }

    // sets whether scroll is allowed or not
    inline void setAllowScroll(bool state) { m_allowScroll = state; }

protected:
    // draws a circle of radius r around the mouse
    void draw(CC_DRAW_CONTEXT& context) override;

private:
    ecvMainAppInterface* m_app;

    QWidget* m_owner;

    float m_pixelSize;

    // event to get mouse-move updates & trigger repaint
    bool eventFilter(QObject* obj, QEvent* event) override;

    int m_radius;
    int m_radiusStep;
    bool m_allowScroll;
};
