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

#include "ecvHObjectCaster.h"

//types
#include "ecv2DLabel.h"
#include "ecv2DViewportLabel.h"
#include "ecv2DViewportObject.h"
#include "ecvCameraSensor.h"
#include "ecvCone.h"
#include "ecvCylinder.h"
#include "ecvDish.h"
#include "ecvExtru.h"
#include "ecvFacet.h"
#include "ecvBox.h"
#include "ecvQuadric.h"
#include "ecvGBLSensor.h"
#include "ecvGenericMesh.h"
#include "ecvGenericPointCloud.h"
#include "ecvGenericPrimitive.h"
#include "ecvHObject.h"
#include "ecvImage.h"
#include "ecvKdTree.h"
#include "ecvMesh.h"
#include "ecvOctree.h"
#include "ecvOctreeProxy.h"
#include "ecvPlane.h"
#include "ecvPointCloud.h"
#include "ecvPolyline.h"
#include "ecvShiftedObject.h"
#include "ecvSphere.h"
#include "ecvSubMesh.h"
#include "ecvTorus.h"
#include "ecvIndexedTransformationBuffer.h"
#include "ecvSensor.h"

#include "Image.h"
#include "Octree.h"
#include "LineSet.h"
#include "RGBDImage.h"
#include "VoxelGrid.h"
#include "ecvBBox.h"
#include "ecvOrientedBBox.h"

/*** helpers ***/

ccPointCloud* ccHObjectCaster::ToPointCloud(ccHObject* obj, bool* lockedVertices /*= 0*/)
{
	if (lockedVertices)
	{
		*lockedVertices = false;
	}

	if (obj)
	{
		if (obj->isA(CV_TYPES::POINT_CLOUD))
		{
			return static_cast<ccPointCloud*>(obj);
		}
		else if (obj->isKindOf(CV_TYPES::MESH))
		{
			ccGenericPointCloud* vertices = static_cast<ccGenericMesh*>(obj)->getAssociatedCloud();
			if (vertices)
			{
				if (!obj->isA(CV_TYPES::MESH) && lockedVertices) //no need to 'lock' the vertices if the user works on the parent mesh
				{
					*lockedVertices = vertices->isLocked();
				}
				return ccHObjectCaster::ToPointCloud(vertices);
			}
		}
	}

	return nullptr;
}

ccGenericPointCloud* ccHObjectCaster::ToGenericPointCloud(ccHObject* obj, bool* lockedVertices /*= 0*/)
{
	if (lockedVertices)
	{
		*lockedVertices = false;
	}

	if (obj)
	{
		if (obj->isKindOf(CV_TYPES::POINT_CLOUD))
		{
			return static_cast<ccGenericPointCloud*>(obj);
		}
		else if (obj->isKindOf(CV_TYPES::MESH))
		{
			ccGenericPointCloud* vertices = static_cast<ccGenericMesh*>(obj)->getAssociatedCloud();
			if (vertices)
			{
				if (!obj->isA(CV_TYPES::MESH) && lockedVertices) //no need to 'lock' the vertices if the user works on the parent mesh
				{
					*lockedVertices = vertices->isLocked();
				}
				return vertices;
			}
		}
	}

	return nullptr;
}

ccShiftedObject* ccHObjectCaster::ToShifted(ccHObject* obj, bool* lockedVertices /*= 0*/)
{
	ccGenericPointCloud* cloud = ToGenericPointCloud(obj, lockedVertices);
	if (cloud)
		return cloud;

	if (obj && obj->isKindOf(CV_TYPES::POLY_LINE))
	{
		if (lockedVertices)
		{
			*lockedVertices = false;
		}
		return static_cast<ccPolyline*>(obj);
	}

	return nullptr;
}

ccGenericMesh* ccHObjectCaster::ToGenericMesh(ccHObject* obj)
{
	return (obj && obj->isKindOf(CV_TYPES::MESH) ? static_cast<ccGenericMesh*>(obj) : 0);
}

ccMesh* ccHObjectCaster::ToMesh(ccHObject* obj)
{
	return (obj && (obj->isA(CV_TYPES::MESH) || obj->isKindOf(CV_TYPES::PRIMITIVE)) ? static_cast<ccMesh*>(obj) : 0);
}

ccSubMesh* ccHObjectCaster::ToSubMesh(ccHObject* obj)
{
	return (obj && obj->isA(CV_TYPES::SUB_MESH) ? static_cast<ccSubMesh*>(obj) : 0);
}

ccPolyline* ccHObjectCaster::ToPolyline(ccHObject* obj)
{
	return (obj && obj->isA(CV_TYPES::POLY_LINE) ? static_cast<ccPolyline*>(obj) : 0);
}

ccFacet* ccHObjectCaster::ToFacet(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::FACET) ? static_cast<ccFacet*>(obj) : 0;
}

ccPlanarEntityInterface* ccHObjectCaster::ToPlanarEntity(ccHObject* obj)
{
	if (obj)
	{
		if (obj->isA(CV_TYPES::FACET))
		{
			return static_cast<ccFacet*>(obj);
		}
		else if (obj->isA(CV_TYPES::PLANE))
		{
			return static_cast<ccPlane*>(obj);
		}
	}
	return nullptr;
}

ccGenericPrimitive* ccHObjectCaster::ToPrimitive(ccHObject* obj)
{
	return obj && obj->isKindOf(CV_TYPES::PRIMITIVE) ? static_cast<ccGenericPrimitive*>(obj) : nullptr;
}

ccSphere*	ccHObjectCaster::ToSphere(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::SPHERE) ? static_cast<ccSphere*>(obj) : nullptr;
}

ccCylinder*	ccHObjectCaster::ToCylinder(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::CYLINDER) ? static_cast<ccCylinder*>(obj) : nullptr;
}

ccCone*		ccHObjectCaster::ToCone(ccHObject* obj)
{
	return obj && obj->isKindOf(CV_TYPES::CONE) ? static_cast<ccCone*>(obj) : nullptr;
}

ccQuadric*	ccHObjectCaster::ToQuadric(ccHObject* obj)
{
	return obj && obj->isKindOf(CV_TYPES::QUADRIC) ? static_cast<ccQuadric*>(obj) : nullptr;
}

ccBox*		ccHObjectCaster::ToBox(ccHObject* obj)
{
	return obj && obj->isKindOf(CV_TYPES::BOX) ? static_cast<ccBox*>(obj) : nullptr;
}

ccPlane*	ccHObjectCaster::ToPlane(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::PLANE) ? static_cast<ccPlane*>(obj) : nullptr;
}

ccDish*		ccHObjectCaster::ToDish(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::DISH) ? static_cast<ccDish*>(obj) : nullptr;
}

ccExtru*	ccHObjectCaster::ToExtru(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::EXTRU) ? static_cast<ccExtru*>(obj) : nullptr;
}

ccTorus*	ccHObjectCaster::ToTorus(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::TORUS) ? static_cast<ccTorus*>(obj) : nullptr;
}

ccOctreeProxy* ccHObjectCaster::ToOctreeProxy(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::POINT_OCTREE) ? static_cast<ccOctreeProxy*>(obj) : nullptr;
}

ccOctree* ccHObjectCaster::ToOctree(ccHObject* obj)
{
	ccOctreeProxy* proxy = ToOctreeProxy(obj);
	return proxy ? proxy->getOctree().data() : nullptr;
}

ccKdTree* ccHObjectCaster::ToKdTree(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::POINT_KDTREE) ? static_cast<ccKdTree*>(obj) : nullptr;
}

ccSensor* ccHObjectCaster::ToSensor(ccHObject* obj)
{
	return obj && obj->isKindOf(CV_TYPES::SENSOR) ? static_cast<ccSensor*>(obj) : nullptr;
}

ccGBLSensor* ccHObjectCaster::ToGBLSensor(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::GBL_SENSOR) ? static_cast<ccGBLSensor*>(obj) : nullptr;
}

ccCameraSensor* ccHObjectCaster::ToCameraSensor(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::CAMERA_SENSOR) ? static_cast<ccCameraSensor*>(obj) : nullptr;
}

ccImage* ccHObjectCaster::ToImage(ccHObject* obj)
{
	return obj && obj->isKindOf(CV_TYPES::IMAGE) ? static_cast<ccImage*>(obj) : nullptr;
}

cc2DLabel* ccHObjectCaster::To2DLabel(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::LABEL_2D) ? static_cast<cc2DLabel*>(obj) : nullptr;
}

cc2DViewportLabel* ccHObjectCaster::To2DViewportLabel(ccHObject* obj)
{
	return obj && obj->isA(CV_TYPES::VIEWPORT_2D_LABEL) ? static_cast<cc2DViewportLabel*>(obj) : nullptr;
}

cc2DViewportObject* ccHObjectCaster::To2DViewportObject(ccHObject* obj)
{
	return obj && obj->isKindOf(CV_TYPES::VIEWPORT_2D_OBJECT) ? static_cast<cc2DViewportObject*>(obj) : nullptr;
}

ccIndexedTransformationBuffer* ccHObjectCaster::ToTransBuffer(ccHObject* obj)
{
	return obj && obj->isKindOf(CV_TYPES::TRANS_BUFFER) ? static_cast<ccIndexedTransformationBuffer*>(obj) : nullptr;
}

using namespace cloudViewer;

geometry::Image* ccHObjectCaster::ToImage2(ccHObject* obj)
{
	return obj && obj->isKindOf(CV_TYPES::IMAGE2) ? static_cast<geometry::Image*>(obj) : nullptr;
}

geometry::RGBDImage * ccHObjectCaster::ToRGBDImage(ccHObject * obj)
{
	return obj && obj->isKindOf(CV_TYPES::RGBD_IMAGE) ? static_cast<geometry::RGBDImage*>(obj) : nullptr;
}

geometry::VoxelGrid * ccHObjectCaster::ToVoxelGrid(ccHObject * obj)
{
	return obj && obj->isKindOf(CV_TYPES::VOXEL_GRID) ? static_cast<geometry::VoxelGrid*>(obj) : nullptr;
}

geometry::LineSet * ccHObjectCaster::ToLineSet(ccHObject * obj)
{
	return obj && obj->isKindOf(CV_TYPES::LINESET) ? static_cast<geometry::LineSet*>(obj) : nullptr;
}

geometry::Octree * ccHObjectCaster::ToOctree2(ccHObject * obj)
{
	return obj && obj->isKindOf(CV_TYPES::POINT_OCTREE2) ? static_cast<geometry::Octree*>(obj) : nullptr;
}

ccBBox * ccHObjectCaster::ToBBox(ccHObject * obj)
{
	return obj && obj->isKindOf(CV_TYPES::BBOX) ? static_cast<ccBBox*>(obj) : nullptr;
}

ecvOrientedBBox * ccHObjectCaster::ToOrientedBBox(ccHObject * obj)
{
	return obj && obj->isKindOf(CV_TYPES::ORIENTED_BBOX) ? static_cast<ecvOrientedBBox*>(obj) : nullptr;
}
