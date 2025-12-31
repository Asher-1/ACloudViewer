// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_LAYOUT_MANAGER_H
#define ECV_LAYOUT_MANAGER_H

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <QDockWidget>
#include <QGuiApplication>
#include <QMainWindow>
#include <QScreen>
#include <QSet>
#include <QSettings>
#include <QToolBar>

// Forward declarations
class ccPluginUIManager;

//! Layout manager for MainWindow
/**
 * This class handles all layout-related operations including:
 * - Setting up default layout
 * - Managing toolbar positions and sizes
 * - Handling plugin toolbar unification
 * - Saving and restoring layout state
 */
class ecvLayoutManager : public QObject {
    Q_OBJECT

public:
    //! Constructor
    explicit ecvLayoutManager(QMainWindow* mainWindow,
                              ccPluginUIManager* pluginManager);

    //! Destructor
    virtual ~ecvLayoutManager();

public:  // Layout operations
    //! Setup default layout for the main window
    /** This includes toolbar positioning, dock widget layout, and window sizing
     * based on screen resolution */
    void setupDefaultLayout();

    //! Reposition the unified plugin toolbar to the end of the first row
    void repositionUnifiedPluginToolbar();

    //! Hide additional plugin toolbars that have been merged into
    //! UnifiedPluginToolbar
    void hideAdditionalPluginToolbars();

    //! Save current GUI layout to settings
    void saveGUILayout();

    //! Restore GUI layout from settings
    /** @param forceDefault If true, ignore saved settings and use default
     * layout */
    void restoreGUILayout(bool forceDefault = false);

    //! Save current layout as custom layout
    void saveCustomLayout();

    //! Restore the default layout
    void restoreDefaultLayout();

    //! Restore previously saved custom layout
    /** @return true if custom layout exists and was restored, false otherwise
     */
    bool restoreCustomLayout();

public:  // Configuration
    //! Set whether to auto-save layout on close
    void setAutoSaveEnabled(bool enabled) { m_autoSaveEnabled = enabled; }

    //! Get whether auto-save is enabled
    bool isAutoSaveEnabled() const { return m_autoSaveEnabled; }

    //! Register a toolbar to be placed on the right side
    void registerRightSideToolBar(QToolBar* toolbar);

    //! Register a toolbar to be placed on the left side
    void registerLeftSideToolBar(QToolBar* toolbar);

    //! Register a dock widget to be placed at the bottom
    void registerBottomDockWidget(QDockWidget* dockWidget);

    //! Set icon size and style for a toolbar based on screen resolution
    //! This is the unified method for setting toolbar icon size across all
    //! toolbars
    void setToolbarIconSize(QToolBar* toolbar, int screenWidth);

private:  // Helper methods
    //! Get screen resolution
    QRect getScreenGeometry() const;

    //! Calculate appropriate icon size based on screen width
    QSize getIconSizeForScreen(int screenWidth) const;

    //! Check if auto-restore is enabled
    /** @return true if auto-restore is enabled, false otherwise
     *  Default value is true (restore enabled) to match UI action's default
     * checked state
     */
    bool isAutoRestoreEnabled() const;

    //! Setup toolbar layout (two-row configuration)
    void setupToolbarLayout(int screenWidth);

    //! Setup dock widget layout
    void setupDockWidgetLayout(int screenWidth, int screenHeight);

    //! Setup main window size and position
    void setupMainWindowGeometry(int screenWidth, int screenHeight);

private:
    QMainWindow* m_mainWindow;           //! Main window reference
    ccPluginUIManager* m_pluginManager;  //! Plugin manager reference
    bool m_autoSaveEnabled;              //! Whether to auto-save layout

    // Toolbar sets for categorization
    QSet<QToolBar*> m_rightSideToolBars;     //! Toolbars on the right side
    QSet<QToolBar*> m_leftSideToolBars;      //! Toolbars on the left side
    QSet<QDockWidget*> m_bottomDockWidgets;  //! Dock widgets at the bottom
};

#endif  // ECV_LAYOUT_MANAGER_H
