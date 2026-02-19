// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

// dialogs
#include "AsciiOpenDlg.h"
#include "AsciiSaveDlg.h"

// Qt
#include <QByteArray>
#include <QTextStream>

/**
 * @class AsciiFilter
 * @brief ASCII point cloud I/O filter
 * 
 * Handles import/export of point clouds in various ASCII text formats
 * including TXT, ASC, NEU, XYZ, XYZRGB, XYZN, PTS, and CSV.
 * 
 * Supports:
 * - Multiple column formats
 * - Custom separators (space, comma, semicolon, tab)
 * - Optional headers
 * - Color and scalar field data
 * - Normal vectors
 * 
 * @see FileIOFilter
 */
class CV_IO_LIB_API AsciiFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    AsciiFilter();

    /**
     * @brief Get file filter string
     * @return Filter string for file dialogs
     */
    static inline QString GetFileFilter() {
        return "ASCII cloud (*.txt *.asc *.neu *.xyz *.xyzrgb *.xyzn *.pts "
               "*.csv)";
    }

    /**
     * @brief Load point cloud from file
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
     * @brief Save entity to file
     * @param entity Entity to save
     * @param filename Output file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR saveToFile(ccHObject* entity,
                             const QString& filename,
                             const SaveParameters& parameters) override;

    /**
     * @brief Load point cloud from byte array
     * 
     * Loads ASCII point cloud data directly from memory.
     * @param data ASCII data as byte array
     * @param sourceName Name for the loaded cloud
     * @param container Container for loaded entities
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    CC_FILE_ERROR loadAsciiData(const QByteArray& data,
                                QString sourceName,
                                ccHObject& container,
                                LoadParameters& parameters);

public:  // Default / persistent settings
    /**
     * @brief Set default number of lines to skip when loading
     * @param count Number of lines to skip (e.g., for headers)
     */
    static void SetDefaultSkippedLineCount(int count);

    /**
     * @brief Set coordinate precision for output
     * @param prec Decimal precision for coordinates
     */
    static void SetOutputCoordsPrecision(int prec);
    
    /**
     * @brief Set scalar field precision for output
     * @param prec Decimal precision for scalar values
     */
    static void SetOutputSFPrecision(int prec);
    
    /**
     * @brief Set output separator type
     * @param separatorIndex Separator index:
     *        - 0: space
     *        - 1: comma
     *        - 2: semicolon
     *        - 3: tab
     */
    static void SetOutputSeparatorIndex(int separatorIndex);
    
    /**
     * @brief Set scalar field output order
     * @param state true to save SF before color (default: color then SF)
     */
    static void SaveSFBeforeColor(bool state);
    
    /**
     * @brief Set whether to save column names header
     * @param state true to save column names (default: false)
     */
    static void SaveColumnsNamesHeader(bool state);
    
    /**
     * @brief Set whether to save point count header
     * @param state true to save point count on first line (default: false)
     */
    static void SavePointCountHeader(bool state);

protected:
    //! Loads an ASCII stream
    CC_FILE_ERROR loadStream(QTextStream& stream,
                             QString filenameOrTitle,
                             qint64 dataSize,
                             ccHObject& container,
                             LoadParameters& parameters);

    //! Loads an ASCII stream with a predefined format
    CC_FILE_ERROR loadCloudFromFormatedAsciiStream(
            QTextStream& stream,
            QString filenameOrTitle,
            ccHObject& container,
            const AsciiOpenDlg::Sequence& openSequence,
            char separator,
            bool commaAsDecimal,
            unsigned approximateNumberOfLines,
            qint64 fileSize,
            unsigned maxCloudSize,
            unsigned skipLines,
            LoadParameters& parameters,
            bool showLabelsIn2D = false);
};
