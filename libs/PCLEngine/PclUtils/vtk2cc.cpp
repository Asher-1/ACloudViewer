//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER BACKEND : qPCL                       #
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
//#                         COPYRIGHT: DAHAI LU                         #
//#                                                                        #
//##########################################################################
//
#include "vtk2cc.h"

//Local
#include "my_point_types.h"
#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"
#include "PclUtils/PCLConv.h"
#include "PclUtils/PCLCloud.h"

// PCL
#include <pcl/common/io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/impl/vtk_lib_io.hpp>

// VTK
#include <vtkPolyData.h>
#include <vtkFloatArray.h>

// CV_CORE_LIB
#include <CVGeom.h>

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvHObject.h>
#include <ecvPolyline.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

vtk2ccConverter::vtk2ccConverter()
{
}

ccPointCloud* vtk2ccConverter::getPointCloudFromPolyData(vtkPolyData* polydata, bool silent)
{
	if (!polydata) return nullptr;

	PCLCloud::Ptr smCloud = cc2smReader().getVtkPolyDataAsSM(polydata);
	if (!smCloud)
	{
		if (!silent)
		{
			CVLog::Error(QString("[getPointCloudFromPolyData] failed to convert vtkPolyData to pcl::PCLPointCloud2"));
		}
		return nullptr;
	}

	if (smCloud->width * smCloud->height == 0)
	{
		if (!silent)
		{
			CVLog::Warning(QString("[getPointCloudFromPolyData] pcl::PCLPointCloud2 is empty!"));
		}
		
		return nullptr;
	}

	return sm2ccConverter(smCloud).getCloud();
}

ccMesh* vtk2ccConverter::getMeshFromPolyData(vtkPolyData* polydata, bool silent)
{
	PCLMesh::Ptr pclMesh = cc2smReader().getVtkPolyDataAsPclMesh(polydata);
	if (!pclMesh)
	{
		if (!silent)
		{
			CVLog::Warning(QString("[getMeshFromPolyData] failed to convert vtkPolyData to pcl::PolygonMesh"));
		}
		
		return nullptr;
	}

	if (pclMesh->cloud.width * pclMesh->cloud.height == 0)
	{
		if (!silent)
		{
			CVLog::Warning(QString("[getMeshFromPolyData] pcl::polygonMesh is empty!"));
		}
		return nullptr;
	}

	return sm2ccConverter(pclMesh->cloud).getMesh(pclMesh->polygons);
}

ccPolyline* vtk2ccConverter::getPolylineFromPolyData(vtkPolyData* polydata, bool silent)
{
	if (!polydata) return nullptr;

	ccPointCloud* obj = getPointCloudFromPolyData(polydata, silent);
	if (!obj)
	{
		CVLog::Error(QString("[getPolylineFromPolyData] failed to convert vtkPolyData to ccPointCloud"));
		return nullptr;
	}

	if (obj->size() == 0)
	{
		CVLog::Warning(QString("[getPolylineFromPolyData] polyline vertices is empty!"));
		return nullptr;
	}

	return getPolylineFromCC(obj);
}

ccHObject::Container vtk2ccConverter::getMultiPolylinesFromPolyData(vtkPolyData* polydata, QString baseName, const ecvColor::Rgb &color)
{
	// initialize output
	ccHObject::Container container;

	int iCells = polydata->GetNumberOfCells();
	for (int i = 0; i < iCells; i++)
	{
		ccPointCloud* vertices = 0;
		vtkCell* cell = polydata->GetCell(i);
		vtkIdType ptsCount = cell->GetNumberOfPoints();
		if (ptsCount > 1)
		{
			vertices = new ccPointCloud("vertices");
			if (!vertices->reserve(ptsCount))
			{
				CVLog::Error("not enough memory to allocate vertices...");
				return container;
			}

			for (vtkIdType iPt = 0; iPt < ptsCount; ++iPt)
			{
                CCVector3 P = CCVector3::fromArray(cell->GetPoints()->GetPoint(iPt));
				vertices->addPoint(P);
			}
			// end POINTS
		}

		if (vertices && vertices->size() == 0)
		{
			delete vertices;
			vertices = 0;
		}

		if (vertices)
		{
			vertices->setName("vertices");
			vertices->setEnabled(false);
			vertices->setPointSize(4);
			vertices->showColors(true);
			vertices->setTempColor(ecvColor::red);
			if (vertices->hasNormals())
				vertices->showNormals(true);
			if (vertices->hasScalarFields())
			{
				vertices->setCurrentDisplayedScalarField(0);
				vertices->showSF(true);
			}

			ccPolyline* poly = getPolylineFromCC(vertices);
			if (!poly)
			{
				delete vertices;
				vertices = nullptr;
				continue;
			}

			// update global scale and shift by m_entity
			QString contourName = baseName;
			if (poly->size() > 1)
			{
				contourName += QString(" (part %1)").arg(i + 1);
			}
			poly->setName(contourName);
			poly->showColors(true);
			poly->setTempColor(color);
			poly->set2DMode(false);

			container.push_back(poly);
		}

	}
	return container;
}

ccPolyline* vtk2ccConverter::getPolylineFromCC(ccPointCloud* vertices)
{
	if (!vertices || !vertices->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		return nullptr;
	}

	ccPointCloud* polyVertices = ccHObjectCaster::ToPointCloud(vertices);
	if (!polyVertices)
	{
		return nullptr;
	}

	ccPolyline* curvePoly = new ccPolyline(polyVertices);
	{
		if (!curvePoly)
		{
			return nullptr;
		}

		int verticesCount = polyVertices->size();
		if (curvePoly->reserve(verticesCount))
		{
			curvePoly->addPointIndex(0, verticesCount);
			curvePoly->setVisible(true);

			bool closed = false;
			CCVector3 start = CCVector3::fromArray(polyVertices->getPoint(0)->u);
			CCVector3 end = CCVector3::fromArray(polyVertices->getPoint(verticesCount - 1)->u);
			if ((end - start).norm() < ZERO_TOLERANCE)
			{
				closed = true;
			}
			else
			{
				closed = false;
			}

			curvePoly->setClosed(closed);
			curvePoly->setName("polyline");

			curvePoly->addChild(polyVertices);
			curvePoly->showColors(true);
			curvePoly->setTempColor(ecvColor::green);
			curvePoly->set2DMode(false);
		}
		else
		{
			delete curvePoly;
			curvePoly = nullptr;
		}
	}

	return curvePoly;
}
