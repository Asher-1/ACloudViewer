// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

/**
 * @class ImageFileFilter
 * @brief Image file I/O filter (Qt-supported formats)
 * 
 * Handles loading and saving of image files in all formats supported by Qt,
 * including PNG, JPEG, BMP, TIFF, GIF, etc. Images are loaded as ccImage
 * entities that can be displayed in the 3D view.
 * 
 * Supported formats depend on Qt's QImage capabilities and installed plugins.
 * 
 * @see FileIOFilter
 * @see ccImage
 */
class CV_IO_LIB_API ImageFileFilter : public FileIOFilter {
public:
    /**
     * @brief Constructor
     */
    ImageFileFilter();

    /**
     * @brief Load image file
     * @param filename Input image file path
     * @param container Container for loaded image entity
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
     * @brief Save entity to image file
     * @param entity Entity to save (must be image type)
     * @param filename Output image file path
     * @param parameters Saving parameters
     * @return Error code (CC_FERR_NO_ERROR on success)
     */
    virtual CC_FILE_ERROR saveToFile(ccHObject* entity,
                                     const QString& filename,
                                     const SaveParameters& parameters) override;

    /**
     * @brief Show file dialog to select input image
     * 
     * Helper method to display a file open dialog for image selection.
     * @param dialogTitle Dialog window title
     * @param imageLoadPath Initial directory path
     * @param parentWidget Parent widget for dialog (optional)
     * @return Selected filename (empty if cancelled)
     */
    static QString GetLoadFilename(const QString& dialogTitle,
                                   const QString& imageLoadPath,
                                   QWidget* parentWidget = nullptr);

    /**
     * @brief Show file dialog to select output image
     * 
     * Helper method to display a file save dialog for image export.
     * @param dialogTitle Dialog window title
     * @param baseName Default base filename
     * @param imageSavePath Initial directory path
     * @param parentWidget Parent widget for dialog (optional)
     * @return Selected filename (empty if cancelled)
     */
    static QString GetSaveFilename(const QString& dialogTitle,
                                   const QString& baseName,
                                   const QString& imageSavePath,
                                   QWidget* parentWidget = nullptr);
};
