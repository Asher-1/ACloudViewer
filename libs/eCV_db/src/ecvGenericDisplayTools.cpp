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

#include "ecvGenericDisplayTools.h"

static ecvGenericDisplayTools* s_genericTools = nullptr;

ecvGenericDisplayTools::ecvGenericDisplayTools()
{
}

void ecvGenericDisplayTools::SetInstance(ecvGenericDisplayTools * tool)
{
	s_genericTools = tool;
}

ecvGenericDisplayTools* ecvGenericDisplayTools::GetInstance()
{
	return s_genericTools;
}





