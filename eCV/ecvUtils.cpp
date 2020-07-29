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
//#          COPYRIGHT: CLOUDVIEWER  project                               #
//#                                                                        #
//##########################################################################

#include "ecvUtils.h"
#include "ecvConsole.h"

namespace ecvUtils
{
	void DisplayLockedVerticesWarning(const QString &meshName, bool displayAsError)
	{
		QString message = QString("Vertices of mesh '%1' are locked (they may be shared by multiple entities for instance).\nYou should call this method directly on the vertices cloud.\n(warning: all entities depending on this cloud will be impacted!)").arg(meshName);
		
		if (displayAsError)
			ecvConsole::Error(message);
		else
			ecvConsole::Warning(message);
	}
}
