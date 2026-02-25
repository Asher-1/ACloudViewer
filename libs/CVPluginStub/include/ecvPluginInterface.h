// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QIcon>
#include <QList>
#include <QObject>
#include <QPair>
#include <QString>

// Qt version
#include <qglobal.h>

class ccExternalFactory;
class ccCommandLineInterface;

/**
 * @brief Plugin type enumeration
 *
 * Defines the type/category of CloudViewer plugin.
 * Multiple types can be combined using bitwise OR.
 */
enum CC_PLUGIN_TYPE {
    ECV_STD_PLUGIN = 1,            ///< Standard plugin with custom actions
    ECV_PCL_ALGORITHM_PLUGIN = 2,  ///< PCL algorithm plugin
    ECV_IO_FILTER_PLUGIN = 4,      ///< File I/O filter plugin
};

/**
 * @class ccPluginInterface
 * @brief Base interface for all CloudViewer plugins
 *
 * Abstract interface that all CloudViewer plugins must implement.
 * Defines the plugin contract including:
 *
 * - Plugin identification (name, description, type, IID)
 * - Metadata (authors, maintainers, references)
 * - Plugin lifecycle (start/stop)
 * - Custom object factory support
 * - Command-line integration
 *
 * Plugins are dynamically loaded at runtime and interact with the
 * main application through well-defined interfaces. This design
 * allows extending CloudViewer functionality without modifying
 * the core application.
 *
 * **Plugin Interface Version: 3.2**
 *
 * @see ecvMainAppInterface
 * @see ccPluginManager
 */
class ccPluginInterface {
public:
    /**
     * @struct Contact
     * @brief Contact information for a person
     *
     * Represents a person with name and email, used for
     * author and maintainer lists.
     */
    struct Contact {
        QString name;   ///< Person's name
        QString email;  ///< Email address
    };

    /// List of contacts
    typedef QList<Contact> ContactList;

    /**
     * @struct Reference
     * @brief Reference to publication or online resource
     *
     * Represents a journal article or website where users
     * can find more information about the plugin.
     */
    struct Reference {
        QString article;  ///< Article title or description
        QString url;      ///< URL to resource
    };

    /// List of references
    using ReferenceList = QList<Reference>;

public:
    /**
     * @brief Virtual destructor
     */
    virtual ~ccPluginInterface() = default;

    /**
     * @brief Get plugin type
     * @return Plugin type (standard, PCL algorithm, I/O filter, etc.)
     */
    virtual CC_PLUGIN_TYPE getType() const = 0;

    /**
     * @brief Check if this is a core plugin
     *
     * Core plugins are essential plugins shipped with CloudViewer.
     * @return true if core plugin, false if third-party
     */
    virtual bool isCore() const = 0;

    /**
     * @brief Get plugin short name
     *
     * Short name displayed in menus and plugin lists.
     * @return Plugin name (e.g., "My Plugin")
     */
    virtual QString getName() const = 0;

    /**
     * @brief Get plugin description
     *
     * Detailed description shown in tooltips and about dialogs.
     * @return Plugin description
     */
    virtual QString getDescription() const = 0;

    /**
     * @brief Get plugin icon
     *
     * Optional icon displayed in menus and toolbars.
     * Default implementation returns empty icon.
     * @return Plugin icon (or empty icon if not provided)
     */
    virtual QIcon getIcon() const { return QIcon(); }

    /**
     * @brief Get list of references
     *
     * Optional list of publications or online resources about the plugin.
     * Users can consult these for more detailed information.
     *
     * @return List of references (empty if none)
     * @note Added in plugin interface v3.1
     */
    virtual ReferenceList getReferences() const { return ReferenceList{}; }

    /**
     * @brief Get list of authors
     *
     * Optional list of plugin authors with contact information.
     * @return List of author contacts (empty if not provided)
     * @note Added in plugin interface v3.1
     */
    virtual ContactList getAuthors() const { return ContactList{}; }

    /**
     * @brief Get list of maintainers
     *
     * Optional list of current plugin maintainers with contact info.
     * @return List of maintainer contacts (empty if not provided)
     * @note Added in plugin interface v3.1
     */
    virtual ContactList getMaintainers() const { return ContactList{}; }

    /**
     * @brief Start the plugin
     *
     * Called when plugin is started from command line or by the application.
     * Can be used to initialize background services, threads, etc.
     *
     * Default implementation returns true (success).
     * @return true if started successfully, false on failure
     */
    virtual bool start() { return true; }

    /**
     * @brief Stop the plugin
     *
     * Called to stop a previously started plugin. Should clean up
     * resources, stop threads, etc.
     *
     * Default implementation does nothing.
     * @see start()
     */
    virtual void stop() {}

    /**
     * @brief Get custom object factory
     *
     * Plugins can provide a factory to create custom object types.
     * This enables proper serialization of custom objects in BIN files.
     *
     * Custom objects must inherit ccCustomHObject or ccCustomLeafObject.
     *
     * @return Pointer to custom factory (or nullptr if not provided)
     */
    virtual ccExternalFactory* getCustomObjectsFactory() const {
        return nullptr;
    }

    /**
     * @brief Register command-line commands
     *
     * Optional method to register custom commands for command-line mode.
     * Allows plugins to be controlled via command-line interface.
     *
     * Default implementation does nothing.
     *
     * @param cmd Command-line interface to register commands with
     * @warning Use unique command prefixes to avoid conflicts with
     *          other plugins and the main application
     */
    virtual void registerCommands(ccCommandLineInterface* cmd) {
        Q_UNUSED(cmd);
    }

protected:
    friend class ccPluginManager;

    /**
     * @brief Set plugin interface ID
     *
     * Internal method called by plugin manager to set the unique
     * interface ID (from Q_PLUGIN_METADATA).
     * @param iid Interface ID string
     */
    virtual void setIID(const QString& iid) = 0;

    /**
     * @brief Get plugin interface ID
     *
     * Returns the unique interface ID used to identify the plugin.
     * @return Interface ID string
     */
    virtual const QString& IID() const = 0;
};

Q_DECLARE_METATYPE(const ccPluginInterface*);

Q_DECLARE_INTERFACE(ccPluginInterface,
                    "edf.rd.cloudviewer.ccPluginInterface/3.2")
