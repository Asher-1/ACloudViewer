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

/**
 * @brief CloudViewer to PCL cloud converter
 *
 * This class provides comprehensive conversion between CloudViewer's
 * ccPointCloud format and PCL's various point cloud and mesh formats. It
 * handles:
 * - Point coordinates (XYZ)
 * - Normals
 * - Colors (RGB/RGBA)
 * - Scalar fields
 * - Meshes (with and without textures)
 * - Polylines and polygons
 *
 * The converter respects point visibility and can operate in different modes
 * for visualization vs. processing.
 */
class QPCL_ENGINE_LIB_API cc2smReader {
public:
    /**
     * @brief Default constructor
     * @param showMode If true, respects point visibility for display (default:
     * false)
     */
    explicit cc2smReader(bool showMode = false);

    /**
     * @brief Constructor with point cloud
     * @param cc_cloud CloudViewer point cloud to convert
     * @param showMode If true, respects point visibility for display (default:
     * false)
     */
    explicit cc2smReader(const ccPointCloud* cc_cloud, bool showMode = false);

    /**
     * @brief Get generic field by name
     * @param field_name Name of the field to extract
     * @return PCL cloud containing the specified field
     */
    PCLCloud::Ptr getGenericField(std::string field_name) const;

    /**
     * @brief Get number of visible points
     * @return Count of visible points (respects visibility table if
     * showMode=true)
     */
    unsigned getvisibilityNum() const;

    /**
     * @brief Get XYZ coordinates as generic PCL cloud
     * @return PCL cloud with XYZ fields
     */
    PCLCloud::Ptr getXYZ() const;

    /**
     * @brief Get XYZ coordinates as typed point cloud
     * @return Typed PCL cloud (pcl::PointXYZ)
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr getXYZ2() const;

    /**
     * @brief Get normal vectors
     * @return PCL cloud with normal XYZ components
     */
    PCLCloud::Ptr getNormals() const;

    /**
     * @brief Get points with normals
     * @return PCL cloud with both positions and normals
     */
    PCLCloud::Ptr getPointNormals() const;

    /**
     * @brief Get RGB colors
     * @return PCL cloud with RGB color fields
     */
    PCLCloud::Ptr getColors() const;

    /**
     * @brief Get scalar values as VTK data array
     * @param scalars Output VTK array
     * @param sfColors If true, use scalar field colors; otherwise use RGB
     * @return true on success
     */
    bool getvtkScalars(vtkSmartPointer<vtkDataArray>& scalars,
                       bool sfColors) const;

    /**
     * @brief Field identifiers for single-field extraction
     */
    enum Fields {
        COORD_X,  ///< X coordinate
        COORD_Y,  ///< Y coordinate
        COORD_Z,  ///< Z coordinate
        NORM_X,   ///< Normal X component
        NORM_Y,   ///< Normal Y component
        NORM_Z    ///< Normal Z component
    };

    /**
     * @brief Get single coordinate or normal component
     * @param field Field identifier
     * @return PCL cloud with single field
     */
    PCLCloud::Ptr getOneOf(Fields field) const;

    /**
     * @brief Get scalar field by name
     * @param field_name Scalar field name
     * @return PCL cloud with scalar field values
     */
    PCLCloud::Ptr getFloatScalarField(const std::string& field_name) const;

    /**
     * @brief Get PCL cloud with specific fields
     * @param requested_fields List of field names to include
     * @return PCL cloud with requested fields
     */
    PCLCloud::Ptr getAsSM(std::list<std::string>& requested_fields) const;

    /**
     * @brief Convert entire ccPointCloud to PCL cloud
     * @param ignoreScalars If true, exclude scalar fields (default: false)
     * @return Complete PCL cloud suitable for PCD file saving
     *
     * Converts all data including XYZ, normals, colors, and scalar fields.
     * Useful for saving to PCD files. For filters, use more specific methods.
     */
    PCLCloud::Ptr getAsSM(bool ignoreScalars = false) const;

    /**
     * @brief Convert with selective data inclusion
     * @param xyz Include XYZ coordinates
     * @param normals Include normals
     * @param rgbColors Include RGB colors
     * @param scalarFields List of scalar field names to include
     * @return PCL cloud with selected data
     */
    PCLCloud::Ptr getAsSM(bool xyz,
                          bool normals,
                          bool rgbColors,
                          const QStringList& scalarFields) const;

    /**
     * @brief Convert to typed XYZ cloud
     * @return Typed pcl::PointXYZ cloud
     *
     * Provides direct access to typed cloud for PCL algorithms.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr getRawXYZ() const;

    /**
     * @brief Convert to PointNormal cloud
     * @return Typed cloud with positions and normals
     *
     * Suitable for algorithms requiring surface normals.
     */
    pcl::PointCloud<pcl::PointNormal>::Ptr getAsPointNormal() const;

    /**
     * @brief Convert VTK polydata to PCL cloud
     * @param polydata Input VTK polydata
     * @return PCL cloud representation
     */
    PCLCloud::Ptr getVtkPolyDataAsSM(vtkPolyData* const polydata) const;

    /**
     * @brief Convert VTK polydata to PCL mesh
     * @param polydata Input VTK polydata
     * @return PCL polygon mesh
     */
    PCLMesh::Ptr getVtkPolyDataAsPclMesh(vtkPolyData* const polydata) const;

    /**
     * @brief Convert ccGenericMesh to PCL mesh
     * @param mesh Input CloudViewer mesh
     * @return PCL polygon mesh
     */
    PCLMesh::Ptr getPclMesh(ccGenericMesh* mesh);

    /**
     * @brief Convert ccGenericMesh to PCL textured mesh
     * @param mesh Input CloudViewer mesh with materials
     * @return PCL texture mesh with material definitions
     */
    PCLTextureMesh::Ptr getPclTextureMesh(ccGenericMesh* mesh);

    /**
     * @brief Convert ccPolyline to PCL polygon
     * @param polyline Input CloudViewer polyline
     * @return PCL planar polygon
     */
    PCLPolygon::Ptr getPclPolygon(ccPolyline* polyline) const;

    /**
     * @brief Get mesh vertices as PCL cloud
     * @param mesh Input mesh
     * @param cloud Output PCL cloud with mesh vertices
     * @return true on success
     */
    bool getPclCloud2(ccGenericMesh* mesh, PCLCloud& cloud) const;

    /**
     * @brief Convert ccPointCloud directly to VTK polydata (bypasses PCL)
     * @param polydata Output VTK polydata (will be created)
     * @param showColors Whether to include RGB vertex colors
     * @param showSF Whether to use scalar field colors instead of RGB
     * @return true on success
     *
     * High-performance direct conversion that bypasses PCL format completely.
     * Creates VTK polydata with points, vertex cells, colors, and normals.
     * Automatically handles visibility filtering in showMode.
     *
     * @note This is the preferred method for visualization as it's more
     * efficient than going through PCL intermediate format.
     */
    bool getVtkPolyDataFromPointCloud(vtkSmartPointer<vtkPolyData>& polydata,
                                      bool showColors,
                                      bool showSF) const;

    /**
     * @brief Convert ccGenericMesh to VTK polydata (bypasses PCL)
     * @param mesh Input CloudViewer mesh
     * @param polydata Output VTK polydata (will be created)
     * @return true on success
     *
     * Direct conversion without PCL intermediate format for efficiency.
     * Uses same point indexing as getPclCloud2 for consistency:
     * pointIndex = n * dimension + vertexIndex
     *
     * @note Ensures consistency with getPclTextureMesh for textured meshes.
     */
    bool getVtkPolyDataFromMeshCloud(
            ccGenericMesh* mesh, vtkSmartPointer<vtkPolyData>& polydata) const;

    /**
     * @brief Convert mesh to VTK polydata with texture coordinates
     * @param mesh Input CloudViewer mesh with materials
     * @param polydata Output VTK polydata (will be created)
     * @param transformation Output transformation matrix (will be created)
     * @param tex_coordinates Output texture coordinates grouped by material
     * index
     * @return true on success
     *
     * Reuses getPclTextureMesh logic to ensure texture coordinate mapping
     * is consistent with addTextureMesh interface. Handles multiple materials
     * and per-face texture coordinates.
     */
    bool getVtkPolyDataWithTextures(
            ccGenericMesh* mesh,
            vtkSmartPointer<vtkPolyData>& polydata,
            vtkSmartPointer<vtkMatrix4x4>& transformation,
            std::vector<std::vector<Eigen::Vector2f>>& tex_coordinates);

    /**
     * @brief Get simplified scalar field name
     * @param ccSfName CloudViewer scalar field name
     * @return Simplified name suitable for PCL
     * @static
     *
     * Converts CloudViewer scalar field names to PCL-compatible names
     * by removing special characters and spaces.
     */
    static std::string GetSimplifiedSFName(const std::string& ccSfName);

    /**
     * @brief Convert CloudViewer material to PCL material
     * @param inMaterial Input CloudViewer material
     * @param outMaterial Output PCL material
     * @static
     *
     * Converts material properties including colors, textures, and
     * rendering parameters from CloudViewer to PCL format.
     */
    static void ConVertToPCLMaterial(ccMaterial::CShared inMaterial,
                                     PCLMaterial& outMaterial);

protected:
    /**
     * @brief Check if a field exists in the cloud
     * @param field_name Field name to check
     * @return true if field exists
     */
    bool checkIfFieldExists(const std::string& field_name) const;

    /// Associated CloudViewer point cloud
    const ccPointCloud* m_cc_cloud;
    /// Show mode flag (respects visibility)
    bool m_showMode;
    /// Whether partial visibility is active
    bool m_partialVisibility;
    /// Number of visible points
    unsigned m_visibilityNum;
};
