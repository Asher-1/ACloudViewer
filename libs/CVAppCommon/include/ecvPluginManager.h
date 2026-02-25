// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QObject>
#include <QVector>

#include "CVAppCommon.h"

class ccPluginInterface;

/**
 * @typedef ccPluginInterfaceList
 * @brief List of plugin interfaces
 *
 * Container type for managing multiple plugin interface pointers.
 * @see ccPluginInterface
 */
using ccPluginInterfaceList = QVector<ccPluginInterface*>;

/**
 * @class ccPluginManager
 * @brief Manager for CloudViewer plugins
 *
 * Singleton class responsible for plugin lifecycle management including:
 * - Plugin discovery and loading from specified directories
 * - Plugin enable/disable state management
 * - Plugin interface access
 * - Plugin path configuration
 *
 * Plugins extend CloudViewer functionality through well-defined interfaces.
 * The manager handles loading plugins at application startup and provides
 * access to loaded plugin instances.
 *
 * @see ccPluginInterface
 */
class CVAPPCOMMON_LIB_API ccPluginManager : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Destructor
     */
    ~ccPluginManager() override = default;

    /**
     * @brief Get singleton instance
     * @return Reference to the plugin manager singleton
     */
    static ccPluginManager& get();

    /**
     * @brief Set plugin search paths
     *
     * Configures directories where the manager will search for plugins.
     * @param paths List of directory paths to search for plugins
     */
    void setPaths(const QStringList& paths);

    /**
     * @brief Get current plugin search paths
     * @return List of configured plugin search paths
     */
    QStringList pluginPaths();

    /**
     * @brief Load all plugins from configured paths
     *
     * Scans plugin directories and loads all valid plugin libraries.
     * Already-loaded plugins are skipped. Disabled plugins are not loaded.
     */
    void loadPlugins();

    /**
     * @brief Get list of loaded plugins
     * @return Reference to vector of plugin interface pointers
     */
    ccPluginInterfaceList& pluginList();

    /**
     * @brief Check if plugin is enabled
     * @param plugin Plugin interface to check
     * @return true if plugin is enabled, false if disabled
     */
    bool isEnabled(const ccPluginInterface* plugin) const;

    /**
     * @brief Enable or disable a plugin
     *
     * Disabled plugins are not loaded at application startup.
     * Changes take effect on next application restart.
     *
     * @param plugin Plugin interface to modify
     * @param enabled New enabled state
     */
    void setPluginEnabled(const ccPluginInterface* plugin, bool enabled);

protected:
    /**
     * @brief Constructor (protected for singleton)
     * @param parent Parent QObject (optional)
     */
    explicit ccPluginManager(QObject* parent = nullptr);

private:
    /**
     * @brief Load plugins from paths and add to list
     *
     * Internal method to scan directories and load plugin libraries.
     */
    void loadFromPathsAndAddToList();

    /**
     * @brief Get list of disabled plugin IIDs
     * @return List of interface IDs for disabled plugins
     */
    QStringList disabledPluginIIDs() const;

    QStringList m_pluginPaths;           ///< Plugin search paths
    ccPluginInterfaceList m_pluginList;  ///< Loaded plugin interfaces
};
