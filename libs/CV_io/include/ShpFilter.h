// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#ifdef CV_SHP_SUPPORT

// qCC_io
#include <FileIOFilter.h>

// Qt
#include <QString>

// system
#include <vector>

class GenericDBFField;

/**
 * @class ShpFilter
 * @brief ESRI Shapefile (.shp) I/O filter
 *
 * Handles import/export of geospatial vector data in ESRI Shapefile format.
 * Shapefiles are widely used in GIS applications and consist of multiple files:
 * - .shp: geometry data
 * - .shx: shape index
 * - .dbf: attribute data (dBase format)
 * - .prj: coordinate system (optional)
 *
 * Supported geometry types:
 * - Points (2D/3D)
 * - Polylines (2D/3D)
 * - Polygons (2D/3D)
 *
 * @see http://www.esri.com/library/whitepapers/pdfs/shapefile.pdf
 * @see FileIOFilter
 * @see ccPolyline
 */
class CV_IO_LIB_API ShpFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    ShpFilter();

    /**
     * @brief Load Shapefile
     * @param filename Input .shp file path
     * @param container Container for loaded entities
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
     * @brief Save entity to Shapefile
     * @param entity Entity to save (polylines, points)
     * @param filename Output .shp file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;

    /**
     * @brief Save entity with custom attribute fields
     *
     * Extended save method that allows specifying custom DBF attribute fields.
     * @param entity Entity to save
     * @param fields Vector of DBF field definitions
     * @param filename Output .shp file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(
            ccHObject* entity,
            const std::vector<GenericDBFField*>& fields,
            const QString& filename,
            const SaveParameters& parameters);

    /**
     * @brief Set whether closed polylines are treated as polygons
     * @param state true to treat closed polylines as polygons
     */
    void treatClosedPolylinesAsPolygons(bool state) {
        m_closedPolylinesAsPolygons = state;
    }

    /**
     * @brief Check if closed polylines are treated as polygons
     * @return true if closed polylines are polygons
     */
    bool areClosedPolylinesAsPolygons() const {
        return m_closedPolylinesAsPolygons;
    }

    /**
     * @brief Set whether to save 3D polylines as 2D
     * @param state true to save as 2D (Z coordinate ignored)
     */
    void save3DPolyAs2D(bool state) { m_save3DPolyAs2D = state; }

    /**
     * @brief Set whether to save 3D polyline height in DBF
     * @param state true to save Z coordinate in attribute table
     */
    void save3DPolyHeightInDBF(bool state) { m_save3DPolyHeightInDBF = state; }

protected:
    //! Whether to consider closed polylines as polygons or not
    bool m_closedPolylinesAsPolygons = true;

    //! Whether to save 3D poly as 2D (note: all polylines from shapefiles are
    //! loaded as 3D)
    bool m_save3DPolyAs2D = false;

    //! Whether to save the 3D height in .dbf file
    bool m_save3DPolyHeightInDBF = false;
};

#endif  // CV_SHP_SUPPORT
