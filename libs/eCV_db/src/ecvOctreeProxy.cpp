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

//Always first
//#include "ccIncludeGL.h"

#include "ecvOctreeProxy.h"
#include "ecvDisplayTools.h"
#include "ecvBBox.h"

//Local
//#include "ccCameraSensor.h"
//#include "ccNormalVectors.h"
//#include "ccBox.h"

//cloudViewer
//#include <ScalarFieldTools.h>
//#include <RayAndBox.h>

ccOctreeProxy::ccOctreeProxy(	ccOctree::Shared octree/*=ccOctree::Shared(0)*/,
								QString name/*="Octree"*/)
	: ccHObject(name)
	, m_octree(octree)
{
	setVisible(false);
	lockVisibility(false);
}

ccOctreeProxy::~ccOctreeProxy()
{
}

ccBBox ccOctreeProxy::getOwnBB(bool withGLFeatures/*=false*/)
{
	if (!m_octree)
	{
		assert(false);
		return ccBBox();
	}
	
	return withGLFeatures ? m_octree->getSquareBB() : m_octree->getPointsBB();
}

void ccOctreeProxy::drawMeOnly(CC_DRAW_CONTEXT& context)
{
	if (!m_octree)
	{
		assert(false);
		return;
	}

	if (!MACRO_Draw3D(context))
		return;

	if (ecvDisplayTools::GetMainWindow() == nullptr)
		return;

	bool pushName = MACRO_DrawEntityNames(context);

	if (pushName)
	{
		//not fast at all!
		if (MACRO_DrawFastNamesOnly(context))
			return;
		//glFunc->glPushName(getUniqueIDForDisplay());
	}

	setOctreeVisibale(isEnabled());
	m_octree->draw(context);

	if (pushName)
	{
		//glFunc->glPopName();
	}
}
