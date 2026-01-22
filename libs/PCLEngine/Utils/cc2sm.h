// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include <Utils/PCLCloud.h>

#include "qPCL.h"

// CV_DB_LIB
#include <ecvMaterial.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Eigen (for texture coordinates)
#include <Eigen/Dense>

// VTK
#include <vtkSmartPointer.h>

// system
#include <list>
#include <string>
#include <vector>

class ccPointCloud;
class ccGenericMesh;
class ccPolyline;

class vtkPolyData;
class vtkDataArray;
class vtkMatrix4x4;
class vtkPolyData;

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
    PCLCloud::Ptr getAsSM(bool xyz,
                          bool normals,
                          bool rgbColors,
                          const QStringList& scalarFields) const;

    //! Converts the ccPointCloud to a 'pcl::PointXYZ' cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr getRawXYZ() const;

    //! Converts the ccPointCloud to a 'pcl::PointNormal' cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr getAsPointNormal() const;

    PCLCloud::Ptr getVtkPolyDataAsSM(vtkPolyData* const polydata) const;
    PCLMesh::Ptr getVtkPolyDataAsPclMesh(vtkPolyData* const polydata) const;

    PCLMesh::Ptr getPclMesh(ccGenericMesh* mesh);
    PCLTextureMesh::Ptr getPclTextureMesh(ccGenericMesh* mesh);

    PCLPolygon::Ptr getPclPolygon(ccPolyline* polyline) const;

    bool getPclCloud2(ccGenericMesh* mesh, PCLCloud& cloud) const;

    /**
     * @brief Convert ccGenericMesh to vtkPolyData using same logic as
     * getPclCloud2 Direct conversion without PCL intermediate format for
     * efficiency
     * @param mesh Input mesh
     * @param polydata Output VTK polydata (will be created)
     * @return true on success
     * @note This method uses the same point indexing as getPclCloud2:
     *       pointIndex = n * dimension + vertexIndex
     *       This ensures consistency with getPclTextureMesh
     */
    bool getVtkPolyDataFromMeshCloud(
            ccGenericMesh* mesh, vtkSmartPointer<vtkPolyData>& polydata) const;

    /**
     * @brief Convert ccGenericMesh to vtkPolyData with texture coordinates
     *        Reuses getPclTextureMesh logic to ensure consistency
     * @param mesh Input mesh
     * @param polydata Output VTK polydata (will be created)
     * @param transformation Output transformation matrix (will be created)
     * @param tex_coordinates Output texture coordinates grouped by material
     * index
     * @return true on success
     * @note This method reuses getPclTextureMesh logic to ensure texture
     * coordinate mapping is consistent with addTextureMesh interface
     */
    bool getVtkPolyDataWithTextures(
            ccGenericMesh* mesh,
            vtkSmartPointer<vtkPolyData>& polydata,
            vtkSmartPointer<vtkMatrix4x4>& transformation,
            std::vector<std::vector<Eigen::Vector2f>>& tex_coordinates);

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
