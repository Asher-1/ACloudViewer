// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file sm2cc.h
 * @brief PCL/sensor_msgs (sm) to CloudViewer (cc) conversion utilities.
 *
 * Provides the pcl2cc class for converting PCL point clouds, meshes,
 * and materials into native CloudViewer data structures.
 *
 * @see cc2sm.h for the reverse conversion direction.
 */

// Local
#include <PclUtils/PCLCloud.h>

// CV_DB_LIB
#include <ecvMaterial.h>

// system
#include <list>

class ccMesh;
class ccPointCloud;

/**
 * @class pcl2cc
 * @brief Converts PCL point clouds and meshes to CloudViewer entities.
 *
 * Static utility class that converts PCL sensor_msgs data (PCLCloud,
 * PCLTextureMesh) into CloudViewer-native types (ccPointCloud, ccMesh).
 * Supports copying of XYZ coordinates, normals, RGB colors, scalar fields,
 * and PBR materials.
 *
 * @see cc2smReader for the reverse conversion direction.
 */
class pcl2cc {
public:
    /**
     * @brief Convert a PCL texture mesh to a ccMesh with materials.
     * @param textureMesh Source PCL texture mesh.
     * @return New ccMesh instance (caller owns), or nullptr on failure.
     */
    static ccMesh* Convert(PCLTextureMesh::ConstPtr textureMesh);

    /**
     * @brief Convert a PCL cloud to a ccPointCloud.
     * @param pclCloud      Source PCL cloud (sensor_msgs format).
     * @param ignoreScalars If true, skip scalar field conversion.
     * @param ignoreRgb     If true, skip RGB color conversion.
     * @return New ccPointCloud instance (caller owns), or nullptr on failure.
     */
    static ccPointCloud* Convert(const PCLCloud& pclCloud,
                                 bool ignoreScalars = false,
                                 bool ignoreRgb = false);

    /**
     * @brief Convert a PCL cloud with polygon indices to a ccMesh.
     * @param pclCloud      Source PCL cloud (sensor_msgs format).
     * @param polygons      Triangle vertex index lists.
     * @param ignoreScalars If true, skip scalar field conversion.
     * @param ignoreRgb     If true, skip RGB color conversion.
     * @return New ccMesh instance (caller owns), or nullptr on failure.
     */
    static ccMesh* Convert(const PCLCloud& pclCloud,
                           const std::vector<pcl::Vertices>& polygons,
                           bool ignoreScalars = false,
                           bool ignoreRgb = false);

public:
    /**
     * @brief Copy XYZ coordinates from a PCL cloud to a ccPointCloud.
     * @param pclCloud       Source PCL cloud.
     * @param ccCloud        Destination CloudViewer point cloud.
     * @param coordinateType Coordinate data type (4=float, 8=double).
     * @return True on success.
     */
    static bool CopyXYZ(const PCLCloud& pclCloud,
                        ccPointCloud& ccCloud,
                        uint8_t coordinateType);

    /**
     * @brief Copy normals from a PCL cloud to a ccPointCloud.
     * @param pclCloud Source PCL cloud.
     * @param ccCloud  Destination CloudViewer point cloud.
     * @return True on success, false if normals not found.
     */
    static bool CopyNormals(const PCLCloud& pclCloud, ccPointCloud& ccCloud);

    /**
     * @brief Copy RGB colors from a PCL cloud to a ccPointCloud.
     * @param pclCloud Source PCL cloud.
     * @param ccCloud  Destination CloudViewer point cloud.
     * @return True on success, false if RGB data not found.
     */
    static bool CopyRGB(const PCLCloud& pclCloud, ccPointCloud& ccCloud);

    /**
     * @brief Copy a named scalar field from a PCL cloud to a ccPointCloud.
     * @param pclCloud         Source PCL cloud.
     * @param sfName           Name of the scalar field to copy.
     * @param ccCloud          Destination CloudViewer point cloud.
     * @param overwriteIfExist If true, overwrite existing scalar field with
     * same name.
     * @return True on success.
     */
    static bool CopyScalarField(const PCLCloud& pclCloud,
                                const std::string& sfName,
                                ccPointCloud& ccCloud,
                                bool overwriteIfExist = true);

    /**
     * @brief Convert a PCL material to a CloudViewer material.
     * @param inMaterial  Source PCL material.
     * @param outMaterial Destination CloudViewer shared material (output).
     */
    static void FromPCLMaterial(const PCLMaterial& inMaterial,
                                ccMaterial::Shared& outMaterial);
};
