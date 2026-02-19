// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#ifdef CV_SHP_SUPPORT

// local
#include "CV_io.h"

// cloudViewer
#include <CVGeom.h>

// Qt
#include <QString>

// system
#include <vector>

// Shapelib
#include <shapefil.h>

/**
 * @class GenericDBFField
 * @brief Base class for Shapefile DBF attribute fields
 *
 * Abstract base class representing a field (column) in a Shapefile's
 * DBF attribute table. Each field contains one value per record/primitive.
 *
 * DBF (dBase format) is used by Shapefiles to store tabular attribute data
 * associated with geometric features.
 *
 * @see IntegerDBFField
 * @see DoubleDBFField
 * @see DoubleDBFField3D
 */
class GenericDBFField {
public:
    /**
     * @brief Constructor with field name
     * @param name Field name/identifier
     */
    GenericDBFField(QString name) : m_name(name) {}

    /**
     * @brief Get field name
     * @return Field name string
     */
    const QString& name() const { return m_name; }

    /**
     * @brief Check if field is 3D (stores vectors)
     * @return true if field stores 3D vectors, false otherwise
     */
    virtual bool is3D() const { return false; }

    /**
     * @brief Get DBF field type
     * @return Field type (integer, double, etc.)
     */
    virtual DBFFieldType type() const = 0;

    /**
     * @brief Get field width (column width in characters)
     * @return Field width
     */
    virtual int width() const = 0;

    /**
     * @brief Get decimal precision for numeric fields
     * @return Number of decimal places
     */
    virtual int decimal() const = 0;

    /**
     * @brief Save 1D field to DBF file
     * @param handle DBF file handle
     * @param fieldIndex Field index in DBF table
     * @return true if successful, false otherwise
     */
    virtual bool save(DBFHandle handle, int fieldIndex) const { return false; }

    /**
     * @brief Save 3D field to DBF file (X, Y, Z columns)
     * @param handle DBF file handle
     * @param xFieldIndex X component field index
     * @param yFieldIndex Y component field index
     * @param zFieldIndex Z component field index
     * @return true if successful, false otherwise
     */
    virtual bool save(DBFHandle handle,
                      int xFieldIndex,
                      int yFieldIndex,
                      int zFieldIndex) const {
        return false;
    }

protected:
    QString m_name;  ///< Field name
};

/**
 * @class IntegerDBFField
 * @brief Integer-valued Shapefile DBF field
 *
 * Stores integer attribute values for Shapefile features.
 * Common uses include feature IDs, classification codes, etc.
 */
class CV_IO_LIB_API IntegerDBFField : public GenericDBFField {
public:
    /**
     * @brief Constructor with field name
     * @param name Field name
     */
    IntegerDBFField(QString name) : GenericDBFField(name) {}

    /**
     * @brief Get field type (FTInteger)
     * @return DBF integer type
     */
    virtual DBFFieldType type() const { return FTInteger; }

    /**
     * @brief Get field width (6 characters)
     * @return Field width
     */
    virtual int width() const { return 6; }

    /**
     * @brief Get decimal precision (0 for integers)
     * @return Decimal places
     */
    virtual int decimal() const { return 0; }

    /**
     * @brief Save integer values to DBF file
     * @param handle DBF file handle
     * @param fieldIndex Field index
     * @return true if successful
     */
    virtual bool save(DBFHandle handle, int fieldIndex) const;

    std::vector<int> values;  ///< Integer field values (one per feature)
};

/**
 * @class DoubleDBFField
 * @brief Double-valued Shapefile DBF field
 *
 * Stores floating-point attribute values for Shapefile features.
 * Common uses include measurements, distances, areas, etc.
 */
class CV_IO_LIB_API DoubleDBFField : public GenericDBFField {
public:
    /**
     * @brief Constructor with field name
     * @param name Field name
     */
    DoubleDBFField(QString name) : GenericDBFField(name) {}

    /**
     * @brief Get field type (FTDouble)
     * @return DBF double type
     */
    virtual DBFFieldType type() const { return FTDouble; }

    /**
     * @brief Get field width (8 characters)
     * @return Field width
     */
    virtual int width() const { return 8; }

    /**
     * @brief Get decimal precision (8 decimal places)
     * @return Decimal places
     */
    virtual int decimal() const { return 8; }

    /**
     * @brief Save double values to DBF file
     * @param handle DBF file handle
     * @param fieldIndex Field index
     * @return true if successful
     */
    virtual bool save(DBFHandle handle, int fieldIndex) const;

    std::vector<double> values;  ///< Double field values (one per feature)
};

/**
 * @class DoubleDBFField3D
 * @brief 3D vector Shapefile DBF field
 *
 * Stores 3D vector (X, Y, Z) attribute values for Shapefile features.
 * Data is stored in three separate DBF columns.
 * Common uses include normals, directions, 3D coordinates, etc.
 */
class CV_IO_LIB_API DoubleDBFField3D : public GenericDBFField {
public:
    /**
     * @brief Constructor with field name
     * @param name Field name (used as base for X, Y, Z columns)
     */
    DoubleDBFField3D(QString name) : GenericDBFField(name) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~DoubleDBFField3D() {}

    /**
     * @brief Check if field is 3D
     * @return Always returns true
     */
    virtual bool is3D() const { return true; }

    /**
     * @brief Get field type (FTDouble)
     * @return DBF double type
     */
    virtual DBFFieldType type() const { return FTDouble; }

    /**
     * @brief Get field width (8 characters)
     * @return Field width
     */
    virtual int width() const { return 8; }

    /**
     * @brief Get decimal precision (8 decimal places)
     * @return Decimal places
     */
    virtual int decimal() const { return 8; }

    /**
     * @brief Save 3D vectors to DBF file (3 columns)
     * @param handle DBF file handle
     * @param xFieldIndex X component field index
     * @param yFieldIndex Y component field index
     * @param zFieldIndex Z component field index
     * @return true if successful
     */
    virtual bool save(DBFHandle handle,
                      int xFieldIndex,
                      int yFieldIndex,
                      int zFieldIndex) const;

    std::vector<CCVector3d>
            values;  ///< 3D vector field values (one per feature)
};

#endif  // CV_SHP_SUPPORT
