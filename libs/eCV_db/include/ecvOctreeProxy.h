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

#ifndef ECV_OCTREE_PROXY_HEADER
#define ECV_OCTREE_PROXY_HEADER

//Local
#include "ecvHObject.h"
#include "ecvOctree.h"

//! Octree structure proxy
/** Implements ccHObject while holding a (shared) pointer on the octree instance (--> safer)
**/
class ECV_DB_LIB_API ccOctreeProxy : public ccHObject
{
public:

	//! Default constructor
	ccOctreeProxy(ccOctree::Shared octree = ccOctree::Shared(0), QString name = "Octree");

	//! Destructor
	virtual ~ccOctreeProxy();

	//! Sets the associated octree
	inline void setOctree(ccOctree::Shared octree) { m_octree = octree; }

	//! Returns the associated octree
	inline ccOctree::Shared getOctree() const { return m_octree; }

	inline void setOctreeVisibale(bool state) { m_octree->setVisible(state); }

	//Inherited from ccHObject
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::POINT_OCTREE; }
	virtual ccBBox getOwnBB(bool withGLFeatures = false) override;

protected:

	//Inherited from ccHObject
	virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;

protected: //members

	//! Associated octree
	ccOctree::Shared m_octree;
};

#endif // ECV_OCTREE_PROXY_HEADER
