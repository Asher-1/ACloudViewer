// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file cc2sm.h
 *  @brief CloudViewer (cc) to PCL/sensor_msgs (sm) conversion utilities
 */

// Local
#include <PclUtils/PCLCloud.h>

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

/** @class cc2smReader
 *  @brief Converts CloudViewer (cc) point clouds and meshes to PCL/sensor_msgs
 * format
 */
class cc2smReader {
public:
    /** @param showMode if true, scalar fields are converted to RGB for
     * visualization
     */
    explicit cc2smReader(bool showMode = false);
    /** @param cc_cloud the CloudViewer point cloud to convert
     *  @param showMode if true, scalar fields are converted to RGB for
     * visualization
     */
    explicit cc2smReader(const ccPointCloud* cc_cloud, bool showMode = false);

    /** @param field_name name of field (x, y, z, normal_x, rgb, etc.) or scalar
     * field name
     *  @return PCLCloud containing the requested field, or empty on failure
     */
    PCLCloud::Ptr getGenericField(std::string field_name) const;

    /** @return number of visible points (or total if no visibility filtering)
     */
    unsigned getvisibilityNum() const;

    /** @return XYZ coordinates as PCLCloud
     */
    PCLCloud::Ptr getXYZ() const;
    /** @return XYZ coordinates as pcl::PointCloud<PointXYZ>
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr getXYZ2() const;

    /** @return normals as PCLCloud
     */
    PCLCloud::Ptr getNormals() const;
    /** @return XYZ concatenated with normals as PCLCloud
     */
    PCLCloud::Ptr getPointNormals() const;

    /** @return RGB colors as PCLCloud
     */
    PCLCloud::Ptr getColors() const;

    /** @param scalars output VTK array for color scalars (created if null)
     *  @param sfColors if true, use scalar field colormap; else use RGB colors
     *  @return true on success
     */
    bool getvtkScalars(vtkSmartPointer<vtkDataArray>& scalars,
                       bool sfColors) const;

    enum Fields { COORD_X, COORD_Y, COORD_Z, NORM_X, NORM_Y, NORM_Z };
    /** @param field which coordinate or normal component to extract
     *  @return PCLCloud with single scalar field
     */
    PCLCloud::Ptr getOneOf(Fields field) const;

    /** @param field_name name of the scalar field
     *  @return PCLCloud containing the scalar field, or empty if not found
     */
    PCLCloud::Ptr getFloatScalarField(const std::string& field_name) const;

    /** @param requested_fields list of field names to include (xyz, normal_xyz,
     * rgb, scalar names)
     *  @return PCLCloud with concatenated requested fields, or empty if any
     * field missing
     */
    PCLCloud::Ptr getAsSM(std::list<std::string>& requested_fields) const;

    //! Converts all the data in a ccPointCloud to a sesor_msgs::PointCloud2
    /** This is useful for saving a ccPointCloud into a PCD file.
            For pcl filters other methods are suggested (to get only the
    necessary bits of data)
     *  @param ignoreScalars if true, scalar fields are excluded
     *  @return PCLCloud with all available data
     **/
    PCLCloud::Ptr getAsSM(bool ignoreScalars = false) const;
    /** @param xyz include XYZ coordinates
     *  @param normals include normals
     *  @param rgbColors include RGB colors
     *  @param scalarFields list of scalar field names to include
     *  @return PCLCloud with requested fields concatenated
     */
    PCLCloud::Ptr getAsSM(bool xyz,
                          bool normals,
                          bool rgbColors,
                          const QStringList& scalarFields) const;

    //! Converts the ccPointCloud to a 'pcl::PointXYZ' cloud
    /** @return point cloud with XYZ only (all points, no visibility filter)
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr getRawXYZ() const;

    //! Converts the ccPointCloud to a 'pcl::PointNormal' cloud
    /** @return point cloud with XYZ and normals
     */
    pcl::PointCloud<pcl::PointNormal>::Ptr getAsPointNormal() const;

    /** @param polydata input VTK polydata
     *  @return PCLCloud converted from polydata (with colors/normals if
     * present)
     */
    PCLCloud::Ptr getVtkPolyDataAsSM(vtkPolyData* const polydata) const;
    /** @param polydata input VTK polydata
     *  @return PCL polygon mesh
     */
    PCLMesh::Ptr getVtkPolyDataAsPclMesh(vtkPolyData* const polydata) const;

    /** @param mesh input CloudViewer mesh
     *  @return PCL polygon mesh with vertices and triangles
     */
    PCLMesh::Ptr getPclMesh(ccGenericMesh* mesh);
    /** @param mesh input CloudViewer mesh with materials/textures
     *  @return PCL texture mesh with materials and UV coordinates
     */
    PCLTextureMesh::Ptr getPclTextureMesh(ccGenericMesh* mesh);

    /** @param polyline input CloudViewer polyline
     *  @return PCL planar polygon
     */
    PCLPolygon::Ptr getPclPolygon(ccPolyline* polyline) const;

    /** @param mesh input CloudViewer mesh
     *  @param cloud output PCL cloud (points, colors, normals per vertex)
     *  @return true on success
     */
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

    /** @param ccSfName CloudViewer scalar field name (may contain spaces)
     *  @return simplified name with spaces replaced by underscores
     */
    static std::string GetSimplifiedSFName(const std::string& ccSfName);

    /** @param inMaterial CloudViewer material
     *  @param outMaterial output PCL material (tex_name, tex_file, colors,
     * etc.)
     */
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
