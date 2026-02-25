// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_PLY_FILTER_HEADER
#define CC_PLY_FILTER_HEADER

#include "FileIOFilter.h"
#include "rply.h"

/**
 * @brief PLY format type names
 *
 * String representations of all supported PLY data types.
 */
static const char e_ply_type_names[][12] = {
        "PLY_INT8",   "PLY_UINT8",   "PLY_INT16",   "PLY_UINT16", "PLY_INT32",
        "PLY_UIN32",  "PLY_FLOAT32", "PLY_FLOAT64", "PLY_CHAR",   "PLY_UCHAR",
        "PLY_SHORT",  "PLY_USHORT",  "PLY_INT",     "PLY_UINT",   "PLY_FLOAT",
        "PLY_DOUBLE", "PLY_LIST"};

/**
 * @brief PLY format storage mode names
 *
 * String representations of PLY storage/encoding modes.
 */
static const char e_ply_storage_mode_names[][24] = {
        "PLY_BIG_ENDIAN", "PLY_LITTLE_ENDIAN", "PLY_ASCII", "PLY_DEFAULT"};

/**
 * @struct plyProperty
 * @brief PLY file property descriptor
 *
 * Describes a single property (attribute) within a PLY element.
 */
struct plyProperty {
    p_ply_property prop;     ///< RPly property handle
    const char* propName;    ///< Property name
    e_ply_type type;         ///< Property data type
    e_ply_type length_type;  ///< List length type (for list properties)
    e_ply_type value_type;   ///< List value type (for list properties)
    int elemIndex;           ///< Element index
};

/**
 * @struct plyElement
 * @brief PLY file element descriptor
 *
 * Describes a PLY element (e.g., vertex, face) and its properties.
 */
struct plyElement {
    p_ply_element elem;                   ///< RPly element handle
    const char* elementName;              ///< Element name
    long elementInstances;                ///< Number of instances
    std::vector<plyProperty> properties;  ///< Element properties
    int propertiesCount;                  ///< Number of properties
    bool isFace;                          ///< Whether this is a face element
};

/**
 * @class PlyFilter
 * @brief Stanford PLY file I/O filter
 *
 * Handles import/export of point clouds and meshes in Stanford PLY format.
 * Supports both ASCII and binary (little/big endian) encodings.
 *
 * PLY format can store:
 * - Point positions
 * - Colors (RGB/RGBA)
 * - Normals
 * - Texture coordinates
 * - Mesh faces
 * - Custom properties
 *
 * @see FileIOFilter
 */
class CV_IO_LIB_API PlyFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    PlyFilter();

    /**
     * @brief Set default output format
     * @param format Storage mode (ASCII, binary little/big endian)
     */
    static void SetDefaultOutputFormat(e_ply_storage_mode format);

    /**
     * @brief Load PLY file
     * @param filename Input file path
     * @param container Container for loaded entities
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR loadFile(const QString& filename,
                           ccHObject& container,
                           LoadParameters& parameters) override;

    /**
     * @brief Check if entity type can be saved
     * @param type Entity type
     * @param multiple Output: whether multiple entities can be saved
     * @param exclusive Output: whether only this type can be saved
     * @return true if type can be saved
     */
    bool canSave(CV_CLASS_ENUM type,
                 bool& multiple,
                 bool& exclusive) const override;

    /**
     * @brief Save entity to PLY file
     * @param entity Entity to save
     * @param filename Output file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR saveToFile(ccHObject* entity,
                             const QString& filename,
                             const SaveParameters& parameters) override;

    /**
     * @brief Load PLY file with texture
     *
     * Custom loading method that also loads an associated texture file.
     * @param filename Input PLY file path
     * @param textureFilename Texture image file path
     * @param container Container for loaded entities
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR loadFile(const QString& filename,
                           const QString& textureFilename,
                           ccHObject& container,
                           LoadParameters& parameters);

private:
    //! Internal method
    CC_FILE_ERROR saveToFile(ccHObject* entity,
                             QString filename,
                             e_ply_storage_mode storageType);
};

#endif  // CC_PLY_FILTER_HEADER