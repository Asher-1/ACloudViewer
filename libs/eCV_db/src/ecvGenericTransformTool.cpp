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
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#include "ecvGenericTransformTool.h"
#include "ecvHObject.h"

ecvGenericTransformTool::ecvGenericTransformTool()
	: m_associatedEntity(nullptr)
{
}

bool ecvGenericTransformTool::setInputData(ccHObject * entity, int viewport)
{
	if (!entity)
	{
		return false;
	}
	m_associatedEntity = entity;
	return true;
}

