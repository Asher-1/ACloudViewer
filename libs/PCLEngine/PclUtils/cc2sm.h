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
#ifndef Q_PCL_CC2SM_H
#define Q_PCL_CC2SM_H

// Local
#include "../qPCL.h"
#include "PCLCloud.h"

// ECV_DB_LIB
#include <ecvMaterial.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// VTK
#include <vtkSmartPointer.h>

// system
#include <list>
#include <string>

class ccPointCloud;
class ccGenericMesh;
class ccPolyline;

class vtkPolyData;
class vtkDataArray;

//! CC to PCL cloud converter
class QPCL_ENGINE_LIB_API cc2smReader {
public:
    explicit cc2smReader(bool showMode = false);
    explicit cc2smReader(const ccPointCloud* cc_cloud, bool showMode = false);

    PCLCloud::Ptr getGenericField(std::string field_name) const;

    unsigned getvisibilityNum() const;

    PCLCloud::Ptr getXYZ() const;
    pcl::PointCloud<pcl::PointXYZ>::Ptr getXYZ2() const;

    PCLCloud::Ptr getNormals() const;
    PCLCloud::Ptr getPointNormals() const;

    PCLCloud::Ptr getColors() const;

    bool getvtkScalars(vtkSmartPointer<vtkDataArray>& scalars,
                       bool sfColors) const;

    enum Fields { COORD_X, COORD_Y, COORD_Z, NORM_X, NORM_Y, NORM_Z };
    PCLCloud::Ptr getOneOf(Fields field) const;

    PCLCloud::Ptr getFloatScalarField(const std::string& field_name) const;

    PCLCloud::Ptr getAsSM(std::list<std::string>& requested_fields) const;

    //! Converts all the data in a ccPointCloud to a sesor_msgs::PointCloud2
    /** This is useful for saving a ccPointCloud into a PCD file.
            For pcl filters other methods are suggested (to get only the
    necessary bits of data)
    **/
    PCLCloud::Ptr getAsSM(bool ignoreScalars = false) const;

    PCLCloud::Ptr getVtkPolyDataAsSM(vtkPolyData* const polydata) const;
    PCLMesh::Ptr getVtkPolyDataAsPclMesh(vtkPolyData* const polydata) const;

    PCLMesh::Ptr getPclMesh(ccGenericMesh* mesh);
    PCLTextureMesh::Ptr getPclTextureMesh(ccGenericMesh* mesh);

    PCLPolygon::Ptr getPclPolygon(ccPolyline* polyline) const;

    bool getPclCloud2(ccGenericMesh* mesh, PCLCloud& cloud) const;

    static std::string GetSimplifiedSFName(const std::string& ccSfName);

    static void ConVertToPCLMaterial(ccMaterial::CShared inMaterial,
                                     PCLMaterial& outMaterial);

protected:
    bool checkIfFieldExists(const std::string& field_name) const;

    //! Associated cloud
    const ccPointCloud* m_cc_cloud;
    bool m_showMode;
    bool m_partialVisibility;
    unsigned m_visibilityNum;
};

#endif  // Q_PCL_CC2SM_H
