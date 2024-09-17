#pragma once

//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: ACloudViewer project                            #
//#                                                                        #
//##########################################################################

#include "CVAppCommon.h"

#include <QObject>

class QAction;
class QMenu;

class ecvMainAppInterface;
class Mouse3DInput;

class CVAPPCOMMON_LIB_API cc3DMouseManager : public QObject
{
	Q_OBJECT

public:
	cc3DMouseManager( ecvMainAppInterface *appInterface, QObject *parent );
	~cc3DMouseManager();

	//! Gets the menu associated with the 3D mouse
	QMenu	*menu() { return m_menu; }

private:
	void enableDevice(bool state, bool silent);
	void releaseDevice();

	void setupMenu();

	void on3DMouseKeyUp(int key);
	void on3DMouseCMDKeyUp(int cmd);
	void on3DMouseKeyDown(int key);
	void on3DMouseCMDKeyDown(int cmd);
	void on3DMouseMove(std::vector<float> &vec);
	void on3DMouseReleased();


	ecvMainAppInterface *m_appInterface;

	Mouse3DInput *m3dMouseInput;

	QMenu *m_menu;
	QAction *m_actionEnable;
};

