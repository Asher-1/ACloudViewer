//##########################################################################
//#                                                                        #
//#                    CLOUDVIEWER  PLUGIN: ccCompass                      #
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

#ifndef ECV_MOUSE_CIRCLE_HEADER
#define ECV_MOUSE_CIRCLE_HEADER

/**
This is a custom 2DViewportLabel which takes up the entire viewport but is entirely transparent,
except for a circle with radius r around the mouse. 
*/
#include "ecvStdPluginInterface.h"
#include <ecv2DViewportObject.h>
#include <qevent.h>
#include <QPoint>
#include <QObject>

class ccMouseCircle : public cc2DViewportObject, public QObject
{
public:
	//constructor
	explicit ccMouseCircle(QWidget* owner, QString name = QString("MouseCircle"));

	//deconstructor
    virtual ~ccMouseCircle() override;

	//get the circle radius in px
	int getRadiusPx();

	//get the circle radius in world coordinates
	float getRadiusWorld();

	//removes the link with the owner (no cleanup)
    void ownerIsDead() { m_owner = nullptr; }

protected:
	//draws a circle of radius r around the mouse
    void draw(CC_DRAW_CONTEXT& context) override;

private:
	// QWidget this overlay is attached to -> used to get mouse position & events
	QWidget* m_owner;
	float m_winTotalZoom;

	//event to get mouse-move updates & trigger repaint
    bool eventFilter(QObject* obj, QEvent* event) override;

public:
	static const int RESOLUTION = 100;
	int RADIUS = 50;
	int RADIUS_STEP = 4;
	float UNIT_CIRCLE[ RESOLUTION ][2];
};

#endif // ECV_MOUSE_CIRCLE_HEADER
