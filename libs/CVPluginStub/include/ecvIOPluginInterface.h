// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QVector>

// qCC_io
#include <FileIOFilter.h>

#include "ecvDefaultPluginInterface.h"

/**
 * @class ccIOPluginInterface
 * @brief Plugin interface for file I/O filters
 * 
 * Interface for plugins that add support for additional file formats.
 * I/O plugins provide FileIOFilter instances that handle loading and
 * saving of specific file types.
 * 
 * **Plugin Interface Version: 1.3**
 * 
 * Typical I/O plugin workflow:
 * 1. Plugin creates FileIOFilter instances in getFilters()
 * 2. Main app registers these filters with the file I/O system
 * 3. Filters appear in Open/Save dialogs automatically
 * 4. User selects file → appropriate filter is invoked
 * 
 * Supported operations:
 * - Loading point clouds, meshes, and other entities from files
 * - Saving entities to files in custom formats
 * - Format detection and validation
 * 
 * @see ccPluginInterface
 * @see FileIOFilter
 */
class ccIOPluginInterface : public ccDefaultPluginInterface {
public:
    /// List of file I/O filters
    using FilterList = QVector<FileIOFilter::Shared>;

public:
    /**
     * @brief Constructor
     * @param resourcePath Path to plugin resources (optional)
     */
    ccIOPluginInterface(const QString &resourcePath = QString())
        : ccDefaultPluginInterface(resourcePath) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~ccIOPluginInterface() override = default;

    /**
     * @brief Get plugin type
     * @return Always returns ECV_IO_FILTER_PLUGIN
     */
    virtual CC_PLUGIN_TYPE getType() const override {
        return ECV_IO_FILTER_PLUGIN;
    }

    /**
     * @brief Get list of I/O filters provided by this plugin
     * 
     * Returns file format filters that will be integrated into
     * CloudViewer's file I/O system. Each filter handles a specific
     * file format (or family of formats).
     * 
     * @return List of FileIOFilter instances (empty if none)
     */
    virtual FilterList getFilters() { return FilterList{}; }
};

Q_DECLARE_INTERFACE(ccIOPluginInterface,
                    "edf.rd.cloudviewer.ccIOFilterPluginInterface/1.3")
