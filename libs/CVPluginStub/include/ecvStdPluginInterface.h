// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// QT
#include <QActionGroup>
#include <QWidget>

// CV_DB_LIB
#include <ecvHObject.h>

// LOCAL
#include "ecvDefaultPluginInterface.h"
#include "ecvMainAppInterface.h"

/**
 * @defgroup PluginUIFlags UI Modification Flags
 * @brief Flags controlling UI updates after plugin operations
 * @{
 */
/// Refresh entity browser after operation
#define CC_PLUGIN_REFRESH_ENTITY_BROWSER 0x00000002
/// Auto-expand database tree after operation
#define CC_PLUGIN_EXPAND_DB_TREE 0x00000004
/** @} */

/**
 * @class ccStdPluginInterface
 * @brief Standard plugin interface for action-based plugins
 *
 * Interface for standard CloudViewer plugins that provide custom actions
 * (menu items, toolbar buttons, etc.). This is the most common plugin type.
 *
 * Features:
 * - Plugin manages its own QAction objects for UI integration
 * - Full access to main application through ecvMainAppInterface
 * - Selection change notifications via onNewSelection()
 * - Console output through dispToConsole()
 *
 * **Plugin Interface Version: 1.5**
 *
 * Typical plugin workflow:
 * 1. Plugin creates QAction objects (in getActions())
 * 2. Main app adds actions to menus/toolbars
 * 3. User triggers action → plugin executes functionality
 * 4. Plugin can modify entities, update UI, etc.
 *
 * @see ccPluginInterface
 * @see ecvMainAppInterface
 */
class ccStdPluginInterface : public ccDefaultPluginInterface {
public:
    /**
     * @brief Constructor
     * @param resourcePath Path to plugin resources (icons, etc.)
     */
    ccStdPluginInterface(const QString& resourcePath = QString())
        : ccDefaultPluginInterface(resourcePath), m_app(nullptr) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~ccStdPluginInterface() override = default;

    /**
     * @brief Get plugin type
     * @return Always returns ECV_STD_PLUGIN
     */
    virtual CC_PLUGIN_TYPE getType() const override { return ECV_STD_PLUGIN; }

    /**
     * @brief Set main application interface
     *
     * Called by CloudViewer immediately after plugin creation.
     * Sets up communication channel between plugin and application.
     *
     * Also synchronizes the unique ID generator to ensure entity IDs
     * are globally unique across plugin and application.
     *
     * @param app Pointer to main application interface
     */
    virtual void setMainAppInterface(ecvMainAppInterface* app) {
        m_app = app;

        if (m_app) {
            // we use the same 'unique ID' generator in plugins as in the main
            // application (otherwise we'll have issues with 'unique IDs'!)
            ccObject::SetUniqueIDGenerator(m_app->getUniqueIDGenerator());
        }
    }

    /**
     * @brief Get main application interface
     *
     * Provides access to application functionality for plugin tools.
     * @return Pointer to main application interface
     */
    virtual ecvMainAppInterface* getMainAppInterface() { return m_app; }

    /**
     * @brief Get plugin actions
     *
     * Returns list of QAction objects that will be integrated into
     * the main application's menus and toolbars.
     *
     * @return List of actions provided by this plugin
     */
    virtual QList<QAction*> getActions() = 0;

    /**
     * @brief Notification callback for selection changes
     *
     * Called by main application whenever entity selection changes.
     * Plugins can override this to update their UI state, enable/disable
     * actions, etc. based on current selection.
     *
     * Default implementation does nothing.
     *
     * @param selectedEntities Currently selected entities
     */
    virtual void onNewSelection(
            const ccHObject::Container&
                    selectedEntities) { /*ignored by default*/ }

    /**
     * @brief Display message in console (convenience method)
     *
     * Shortcut to ecvMainAppInterface::dispToConsole().
     * Allows plugins to easily output messages to the console.
     *
     * @param message Message text
     * @param level Message severity level (default: standard)
     */
    inline virtual void dispToConsole(
            QString message,
            ecvMainAppInterface::ConsoleMessageLevel level =
                    ecvMainAppInterface::STD_CONSOLE_MESSAGE) {
        if (m_app) {
            m_app->dispToConsole(message, level);
        }
    }

protected:
    ecvMainAppInterface* m_app;  ///< Main application interface pointer
};

Q_DECLARE_METATYPE(const ccStdPluginInterface*);

Q_DECLARE_INTERFACE(ccStdPluginInterface,
                    "edf.rd.cloudviewer.ccStdPluginInterface/1.5")
