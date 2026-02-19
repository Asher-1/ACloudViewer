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

// system
#include <list>

class ccMesh;
class ccPointCloud;

/**
 * @brief PCL to CloudViewer cloud converter
 * 
 * Provides static methods for converting PCL point clouds, meshes, and
 * materials to CloudViewer's native ccPointCloud and ccMesh formats.
 * 
 * Supports conversion of:
 * - Point coordinates (XYZ)
 * - Normals
 * - RGB/RGBA colors
 * - Scalar fields
 * - Polygon meshes
 * - Textured meshes with materials
 */
class QPCL_ENGINE_LIB_API pcl2cc {
public:
    // =====================================================================
    // Main Conversion Methods
    // =====================================================================
    
    /**
     * @brief Convert PCL textured mesh to CloudViewer mesh
     * @param textureMesh Input PCL textured mesh with materials
     * @return CloudViewer mesh with texture materials, or nullptr on failure
     * @static
     * 
     * Converts complete textured mesh including geometry and material definitions.
     */
    static ccMesh* Convert(PCLTextureMesh::ConstPtr textureMesh);
    
    /**
     * @brief Convert PCL cloud to CloudViewer point cloud
     * @param pclCloud Input PCL cloud (generic format)
     * @param ignoreScalars If true, skip scalar field conversion (default: false)
     * @param ignoreRgb If true, skip RGB color conversion (default: false)
     * @return CloudViewer point cloud, or nullptr on failure
     * @static
     * 
     * Converts PCL cloud with optional data filtering.
     */
    static ccPointCloud* Convert(const PCLCloud& pclCloud,
                                 bool ignoreScalars = false,
                                 bool ignoreRgb = false);
    
    /**
     * @brief Convert PCL cloud with polygons to CloudViewer mesh
     * @param pclCloud Input PCL cloud with point data
     * @param polygons Vector of polygon vertex indices
     * @param ignoreScalars If true, skip scalar field conversion (default: false)
     * @param ignoreRgb If true, skip RGB color conversion (default: false)
     * @return CloudViewer mesh, or nullptr on failure
     * @static
     * 
     * Creates mesh from point cloud and polygon connectivity.
     */
    static ccMesh* Convert(const PCLCloud& pclCloud,
                           const std::vector<pcl::Vertices>& polygons,
                           bool ignoreScalars = false,
                           bool ignoreRgb = false);

public:  
    // =====================================================================
    // Data Copying Utilities
    // =====================================================================
    
    /**
     * @brief Copy XYZ coordinates from PCL to CloudViewer cloud
     * @param pclCloud Source PCL cloud
     * @param ccCloud Target CloudViewer cloud (must be pre-allocated)
     * @param coordinateType Coordinate data type (double/float)
     * @return true on success
     * @static
     * 
     * Copies point positions with optional type conversion.
     */
    static bool CopyXYZ(const PCLCloud& pclCloud,
                        ccPointCloud& ccCloud,
                        uint8_t coordinateType);
    
    /**
     * @brief Copy normals from PCL to CloudViewer cloud
     * @param pclCloud Source PCL cloud
     * @param ccCloud Target CloudViewer cloud
     * @return true on success
     * @static
     * 
     * Extracts and copies normal vectors if present in PCL cloud.
     */
    static bool CopyNormals(const PCLCloud& pclCloud, ccPointCloud& ccCloud);
    
    /**
     * @brief Copy RGB colors from PCL to CloudViewer cloud
     * @param pclCloud Source PCL cloud
     * @param ccCloud Target CloudViewer cloud
     * @return true on success
     * @static
     * 
     * Extracts and copies RGB or RGBA colors.
     */
    static bool CopyRGB(const PCLCloud& pclCloud, ccPointCloud& ccCloud);
    
    /**
     * @brief Copy specific scalar field from PCL to CloudViewer cloud
     * @param pclCloud Source PCL cloud
     * @param sfName Scalar field name to copy
     * @param ccCloud Target CloudViewer cloud
     * @param overwriteIfExist Overwrite existing field with same name (default: true)
     * @return true on success
     * @static
     * 
     * Copies a named scalar field from PCL to CloudViewer format.
     */
    static bool CopyScalarField(const PCLCloud& pclCloud,
                                const std::string& sfName,
                                ccPointCloud& ccCloud,
                                bool overwriteIfExist = true);

    /**
     * @brief Convert PCL material to CloudViewer material
     * @param inMaterial Input PCL material
     * @param outMaterial Output CloudViewer material
     * @static
     * 
     * Converts material properties including textures, colors,
     * and rendering parameters from PCL to CloudViewer format.
     */
    static void FromPCLMaterial(const PCLMaterial& inMaterial,
                                ccMaterial::Shared& outMaterial);
};
