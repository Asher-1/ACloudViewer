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
#ifndef Q_PCL_VTK2CC_H
#define Q_PCL_VTK2CC_H

//Local
#include "../qPCL.h"

// ECV_DB_LIB
#include <ecvColorTypes.h>

class ccHObject;
class ccMesh;
class ccPointCloud;
class ccPolyline;
class vtkPolyData;

//! CC to PCL cloud converter
class QPCL_ENGINE_LIB_API vtk2ccConverter
{
public:
	explicit vtk2ccConverter();

	ccPointCloud* getPointCloudFromPolyData(vtkPolyData* polydata, bool silent = false);
	ccMesh* getMeshFromPolyData(vtkPolyData* polydata, bool silent = false);
	ccPolyline* getPolylineFromPolyData(vtkPolyData* polydata, bool silent = false);

	ccPolyline* getPolylineFromCC(ccPointCloud* vertices);
	std::vector<ccHObject*> getMultiPolylinesFromPolyData(vtkPolyData* polydata, QString baseName = "Slice", const ecvColor::Rgb &color = ecvColor::green);
};

#endif // Q_PCL_VTK2CC_H
