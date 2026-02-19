// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

class QWidget;

/**
 * @class BinFilter
 * @brief CloudViewer native binary format I/O filter
 * 
 * Handles import/export of CloudViewer entities in the native .bin format.
 * This is the most efficient format for saving/loading CloudViewer data
 * as it preserves all entity properties, metadata, and relationships.
 * 
 * Supports two format versions:
 * - V1: Legacy format (older CloudViewer versions)
 * - V2: Current format with improved serialization and parallel loading
 * 
 * @see FileIOFilter
 */
class CV_IO_LIB_API BinFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    BinFilter();

    /**
     * @brief Get file filter string for dialogs
     * @return Filter string "CloudViewer entities (*.bin)"
     */
    static inline QString GetFileFilter() {
        return "CloudViewer entities (*.bin)";
    }
    
    /**
     * @brief Get default file extension
     * @return Extension string "bin"
     */
    static inline QString GetDefaultExtension() { return "bin"; }

    /**
     * @brief Get the last saved file format version
     * @return Version number
     */
    static short GetLastSavedFileVersion();

    /**
     * @brief Load BIN file
     * @param filename Input file path
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
     * @brief Save entity to BIN file
     * @param entity Entity to save
     * @param filename Output file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;

    /**
     * @brief Load legacy V1 format BIN file
     * @param in Input file stream
     * @param container Container for loaded entities
     * @param nbScansTotal Total number of scans
     * @param parameters Loading parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    static CC_FILE_ERROR LoadFileV1(QFile& in,
                                    ccHObject& container,
                                    unsigned nbScansTotal,
                                    const LoadParameters& parameters);

    /**
     * @brief Load current V2 format BIN file
     * 
     * Supports parallel loading for improved performance on large files.
     * @param in Input file stream
     * @param container Container for loaded entities
     * @param flags Deserialization flags
     * @param parallel Enable parallel loading
     * @param parentWidget Parent widget for progress dialogs (optional)
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    static CC_FILE_ERROR LoadFileV2(QFile& in,
                                    ccHObject& container,
                                    int flags,
                                    bool parallel,
                                    QWidget* parentWidget = nullptr);

    /**
     * @brief Save to V2 format BIN file
     * @param out Output file stream
     * @param object Object to save
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    static CC_FILE_ERROR SaveFileV2(QFile& out, ccHObject* object);
};
