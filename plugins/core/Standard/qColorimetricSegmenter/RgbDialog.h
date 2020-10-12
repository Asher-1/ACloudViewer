#pragma once

//##########################################################################
//#                                                                        #
//#            CLOUDVIEWER PLUGIN: ColorimetricSegmenter                   #
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
//#    COPYRIGHT:	Tri-Thien TRUONG, Ronan COLLIER, Mathieu LETRONE       #
//#                                                                        #
//##########################################################################

#include <ui_RgbDialog.h>

#include <ecvPickingListener.h>

//Qt
#include <QDialog>

class ccPickingHub;
/*
	Get the values of the RGB interface, and interactions
*/
class RgbDialog : public QDialog, public ccPickingListener, public Ui::RgbDialog
{
	Q_OBJECT
public:
	explicit RgbDialog(ccPickingHub* pickingHub, QWidget* parent = nullptr);

	//! Inherited from ccPickingListener
	virtual void onItemPicked(const PickedItem& pi);

public slots:
	void pickPoint_first(bool);
	void pickPoint_second(bool);

protected: //members

	//! Picking window (if any)
	QWidget* m_pickingWin;

	//! Picking hub
	ccPickingHub* m_pickingHub;
};
