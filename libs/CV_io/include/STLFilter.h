// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

class ccMesh;
class ccPointCloud;
class ccGenericMesh;

/**
 * @class STLFilter
 * @brief StereoLithography (STL) mesh file I/O filter
 * 
 * Handles import/export of meshes in STL format (.stl), a widely-used
 * format for 3D printing and CAD applications. Supports both ASCII and
 * binary STL formats.
 * 
 * STL format characteristics:
 * - Stores triangular mesh geometry only (vertices and normals)
 * - No color, texture, or material information
 * - Binary format is more compact and faster to process
 * - ASCII format is human-readable but larger
 * 
 * @see http://www.ennex.com/~fabbers/StL.asp
 * @see FileIOFilter
 * @see ccMesh
 */
class CV_IO_LIB_API STLFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    STLFilter();

    /**
     * @brief Load STL file (auto-detects ASCII/binary)
     * @param filename Input STL file path
     * @param container Container for loaded mesh
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) override;

    /**
     * @brief Check if entity type can be saved
     * @param type Entity type
     * @param multiple Output: whether multiple entities can be saved
     * @param exclusive Output: whether only this type can be saved
     * @return true if type can be saved
     */
    virtual bool canSave(CV_CLASS_ENUM type,
                         bool& multiple,
                         bool& exclusive) const override;
    
    /**
     * @brief Save entity to STL file
     * @param entity Entity to save (mesh)
     * @param filename Output STL file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;

protected:
    /**
     * @brief Save mesh to ASCII STL file
     * @param mesh Mesh to save
     * @param theFile Output file stream
     * @param parentWidget Parent widget for progress (optional)
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR saveToASCIIFile(ccGenericMesh* mesh,
                                  FILE* theFile,
                                  QWidget* parentWidget = 0);
    
    /**
     * @brief Save mesh to binary STL file
     * @param mesh Mesh to save
     * @param theFile Output file stream
     * @param parentWidget Parent widget for progress (optional)
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR saveToBINFile(ccGenericMesh* mesh,
                                FILE* theFile,
                                QWidget* parentWidget = 0);

    /**
     * @brief Load ASCII STL file
     * @param fp Input file stream
     * @param mesh Output mesh
     * @param vertices Output vertex cloud
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR loadASCIIFile(QFile& fp,
                                ccMesh* mesh,
                                ccPointCloud* vertices,
                                LoadParameters& parameters);

    /**
     * @brief Load binary STL file
     * @param fp Input file stream
     * @param mesh Output mesh
     * @param vertices Output vertex cloud
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR loadBinaryFile(QFile& fp,
                                 ccMesh* mesh,
                                 ccPointCloud* vertices,
                                 LoadParameters& parameters);
};
