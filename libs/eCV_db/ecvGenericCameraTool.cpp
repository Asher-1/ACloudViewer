//##########################################################################
//#                                                                        #
//#                               ECV_DB                                   #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU Library General Public License as       #
//#  published by the Free Software Foundation; version 2 or later of the  #
//#  License.                                                              #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#include "ecvGenericCameraTool.h"
#include "ecvDisplayTools.h"

// Qt includes.
#include <QDebug>
#include <QPointer>
#include <QString>
#include <QToolButton>
#include <QSettings>

// STL
#include <sstream>
#include <string>

ecvGenericCameraTool::CameraInfo ecvGenericCameraTool::OldCameraParam = ecvGenericCameraTool::CameraInfo();
ecvGenericCameraTool::CameraInfo ecvGenericCameraTool::CurrentCameraParam = ecvGenericCameraTool::CameraInfo();

//-----------------------------------------------------------------------------
ecvGenericCameraTool::ecvGenericCameraTool()
{
}

//-----------------------------------------------------------------------------
ecvGenericCameraTool::~ecvGenericCameraTool()
{
}

//-----------------------------------------------------------------------------
void ecvGenericCameraTool::setAutoPickPivotAtCenter(bool state)
{
	ecvDisplayTools::SetAutoPickPivotAtCenter(state);
}

//-----------------------------------------------------------------------------
void ecvGenericCameraTool::saveCameraConfiguration(const std::string& file)
{
}

//-----------------------------------------------------------------------------
void ecvGenericCameraTool::loadCameraConfiguration(const std::string& file)
{
}