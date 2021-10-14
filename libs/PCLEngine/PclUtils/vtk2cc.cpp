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

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

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

// CV_CORE_LIB
#include <CVGeom.h>

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvHObject.h>
#include <ecvPolyline.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

// VTK
#include <vtkPolyData.h>
#include <vtkFloatArray.h>

// Support for VTK 7.1 upwards
#ifdef vtkGenericDataArray_h
#define SetTupleValue SetTypedTuple
#define InsertNextTupleValue InsertNextTypedTuple
#define GetTupleValue GetTypedTuple
#endif

ccPointCloud* vtk2cc::ConvertToPointCloud(vtkPolyData* polydata, bool silent)
{
	if (!polydata) return nullptr;

	// Set the colors of the pcl::PointCloud (if the pcl::PointCloud supports colors and the input vtkPolyData has colors)
	vtkUnsignedCharArray* colors = vtkUnsignedCharArray::SafeDownCast(polydata->GetPointData()->GetScalars());

	// Set the normals of the pcl::PointCloud (if the pcl::PointCloud supports normals and the input vtkPolyData has normals)
	vtkFloatArray* normals = vtkFloatArray::SafeDownCast(polydata->GetPointData()->GetNormals());

	//create cloud
	ccPointCloud* cloud = new ccPointCloud("vertices");

    vtkIdType pointCount = polydata->GetNumberOfPoints();
	if (!cloud->resize(static_cast<unsigned>(pointCount)))
	{
		if (!silent)
		{
			CVLog::Warning(QString("[getPointCloudFromPolyData] not enough memory!"));
		}
		delete cloud;
		cloud = nullptr;
		return nullptr;
	}

	if (normals && !cloud->reserveTheNormsTable())
	{
		if (!silent)
		{
			CVLog::Warning(QString("[getPointCloudFromPolyData] not enough memory!"));
		}
		delete cloud;
		cloud = nullptr;
		return nullptr;
	}

	if (colors && !cloud->reserveTheRGBTable())
	{
		if (!silent)
		{
			CVLog::Warning(QString("[getPointCloudFromPolyData] not enough memory!"));
		}
		delete cloud;
		cloud = nullptr;
		return nullptr;
	}
	
    for (vtkIdType i = 0; i < pointCount; ++i)
	{
		double coordinate[3];
		polydata->GetPoint(i, coordinate);
        cloud->setPoint(static_cast<std::size_t>(i), CCVector3::fromArray(coordinate));
		if (normals)
		{
			float normal[3];
			normals->GetTupleValue(i, normal);
			CCVector3 N(static_cast<PointCoordinateType>(normal[0]),
						static_cast<PointCoordinateType>(normal[1]),
						static_cast<PointCoordinateType>(normal[2]));
			cloud->addNorm(N);
		}
		if (colors)
		{
			unsigned char color[3];
			colors->GetTupleValue(i, color);
			ecvColor::Rgb C(static_cast<ColorCompType>(color[0]),
							static_cast<ColorCompType>(color[1]),
							static_cast<ColorCompType>(color[2]));
			cloud->addRGBColor(C);
		}
	}

	if (normals)
	{
		cloud->showNormals(true);
	}
	if (colors)
	{
		cloud->showColors(true);
	}

	return cloud;
}

ccMesh* vtk2cc::ConvertToMesh(vtkPolyData* polydata, bool silent)
{
	vtkSmartPointer<vtkPoints> mesh_points = polydata->GetPoints();
	unsigned nr_points = static_cast<unsigned>(mesh_points->GetNumberOfPoints());
	unsigned nr_polygons = static_cast<unsigned>(polydata->GetNumberOfPolys());
	if (nr_points == 0)
	{
		if (!silent)
		{
			CVLog::Warning(QString("[getMeshFromPolyData] cannot find points data!"));
		}
		return nullptr;
	}

    ccPointCloud* vertices = ConvertToPointCloud(polydata, silent);
	if (!vertices)
	{
		return nullptr;
	}
	vertices->setEnabled(false);
	// DGM: no need to lock it as it is only used by one mesh!
	vertices->setLocked(false);

	// mesh
	ccMesh* mesh = new ccMesh(vertices);
	mesh->setName("Mesh");
	mesh->addChild(vertices);

	if (!mesh->reserve(nr_polygons))
	{
		if (!silent)
		{
			CVLog::Warning(QString("[getMeshFromPolyData] not enough memory!"));
		}
		return nullptr;
	}

	vtkIdType* cell_points;
	vtkIdType nr_cell_points;
	vtkCellArray * mesh_polygons = polydata->GetPolys();
	mesh_polygons->InitTraversal();
	int id_poly = 0;
	while (mesh_polygons->GetNextCell(nr_cell_points, cell_points))
	{
		if (nr_cell_points != 3)
		{
			if (!silent)
			{
				CVLog::Warning(QString("[getMeshFromPolyData] only support triangles!"));
			}
			break;
		}

        mesh->addTriangle(static_cast<unsigned>(cell_points[0]),
                          static_cast<unsigned>(cell_points[1]),
                          static_cast<unsigned>(cell_points[2]));
		++id_poly;
	}

	//do some cleaning
	{
		vertices->shrinkToFit();
		mesh->shrinkToFit();
		NormsIndexesTableType* normals = mesh->getTriNormsTable();
		if (normals)
		{
			normals->shrink_to_fit();
		}
	}

	return mesh;
}

ccPolyline* vtk2cc::ConvertToPolyline(vtkPolyData* polydata, bool silent)
{
	if (!polydata) return nullptr;

    ccPointCloud* obj = ConvertToPointCloud(polydata, silent);
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

    return ConvertToPolyline(obj);
}

ccPolyline* vtk2cc::ConvertToPolyline(ccPointCloud* vertices)
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

        unsigned verticesCount = polyVertices->size();
		if (curvePoly->reserve(verticesCount))
		{
			curvePoly->addPointIndex(0, verticesCount);
			curvePoly->setVisible(true);

			bool closed = false;
			CCVector3 start = CCVector3::fromArray(polyVertices->getPoint(0)->u);
			CCVector3 end = CCVector3::fromArray(polyVertices->getPoint(verticesCount - 1)->u);
            if (cloudViewer::LessThanEpsilon((end - start).norm()))
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

ccHObject::Container vtk2cc::ConvertToMultiPolylines(vtkPolyData* polydata, QString baseName, const ecvColor::Rgb &color)
{
    // initialize output
    ccHObject::Container container;

    vtkIdType iCells = polydata->GetNumberOfCells();
    for (vtkIdType i = 0; i < iCells; i++)
    {
        ccPointCloud* vertices = nullptr;
        vtkCell* cell = polydata->GetCell(i);
        vtkIdType ptsCount = cell->GetNumberOfPoints();
        if (ptsCount > 1)
        {
            vertices = new ccPointCloud("vertices");
            if (!vertices->reserve(static_cast<unsigned>(ptsCount)))
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
            vertices = nullptr;
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

            ccPolyline* poly = ConvertToPolyline(vertices);
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
