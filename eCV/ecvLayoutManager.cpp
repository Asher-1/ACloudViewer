// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvLayoutManager.h"

// Local
#include "MainWindow.h"
#include "ecvPersistentSettings.h"
#include "ecvSettingManager.h"
#include "pluginManager/ecvPluginUIManager.h"

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <QDockWidget>
#include <QMessageBox>
#include <QSettings>
#include <QToolBar>

ecvLayoutManager::ecvLayoutManager(QMainWindow* mainWindow,
                                   ccPluginUIManager* pluginManager)
    : QObject(mainWindow),
      m_mainWindow(mainWindow),
      m_pluginManager(pluginManager),
      m_autoSaveEnabled(true) {
    Q_ASSERT(m_mainWindow);
    Q_ASSERT(m_pluginManager);
}

ecvLayoutManager::~ecvLayoutManager() {}

QRect ecvLayoutManager::getScreenGeometry() const {
    QScreen* screen = QGuiApplication::primaryScreen();
    // Use geometry() instead of availableGeometry() to match MainWindow::updateAllToolbarIconSizes()
    // This ensures consistent screen width calculation
    return screen ? screen->geometry() : QRect(0, 0, 1920, 1080);
}

QSize ecvLayoutManager::getIconSizeForScreen(int screenWidth) const {
    // Icon size scaling based on physical screen resolution
    // Consider devicePixelRatio for High DPI displays (especially macOS Retina)
    // This ensures proper icon sizing across Windows, Linux, and macOS platforms
    
    QScreen* screen = QGuiApplication::primaryScreen();
    qreal dpr = screen ? screen->devicePixelRatio() : 1.0;
    
    // Calculate physical resolution (logical resolution × devicePixelRatio)
    // This gives us the actual pixel density of the display
    // Examples:
    // - macOS Retina: logical 1920 × dpr 2.0 = physical 3840 (4K)
    // - Standard 2K: logical 2560 × dpr 1.0 = physical 2560 (2K)
    // - 8K display: logical 7680 × dpr 1.0 = physical 7680 (8K)
    int physicalWidth = static_cast<int>(screenWidth * dpr);
    
    // Icon size calculation based on physical resolution
    // Supports: 8K, 4K, 2K, 1080p, HD+, HD, and lower resolutions
    int baseSize;
    
    if (physicalWidth >= 7680) {
        // 8K physical resolution (7680x4320)
        baseSize = 40;
    } else if (physicalWidth >= 3840) {
        // 4K physical resolution (3840x2160)
        // Includes macOS Retina displays where logical 1920 × dpr 2.0 = 3840
        baseSize = 32;
    } else if (physicalWidth >= 2560) {
        // 2K physical resolution (2560x1440)
        // Scale icon size based on DPR for better visual consistency
        // For standard 2K (dpr=1.0): 22px
        // For scaled displays (dpr>1.0): scale up proportionally
        baseSize = static_cast<int>(22 * (1.0 + (dpr - 1.0) * 0.3));
        baseSize = qBound(22, baseSize, 28);  // Clamp between 22 and 28
    } else if (physicalWidth >= 1920) {
        // Full HD physical resolution (1920x1080)
        // For standard 1080p (dpr=1.0): 18px
        // For slightly scaled displays (1.0 < dpr < 2.0): scale up slightly
        baseSize = static_cast<int>(18 * (1.0 + (dpr - 1.0) * 0.2));
        baseSize = qBound(18, baseSize, 22);  // Clamp between 18 and 22
    } else {
        // HD+ (1600x900), HD (1280x720), and lower resolutions - minimum size
        baseSize = 16;
    }

    return QSize(baseSize, baseSize);
}

void ecvLayoutManager::setToolbarIconSize(QToolBar* toolbar, int screenWidth) {
    if (!toolbar) return;

    QSize iconSize = getIconSizeForScreen(screenWidth);
    toolbar->setIconSize(iconSize);

    // Set stylesheet to ensure buttons have proper size and icons fill the
    // button Use consistent padding across all toolbars
    // Reduce padding for smaller icons to make buttons more compact
    int buttonSize = iconSize.width() + (iconSize.width() <= 16 ? 2 : 4);  // Less padding for small icons
    toolbar->setStyleSheet(QString("QToolBar#%1 QToolButton { "
                                   "    min-width: %2px; "
                                   "    min-height: %2px; "
                                   "    max-width: %2px; "
                                   "    max-height: %2px; "
                                   "    padding: 1px; "
                                   "} "
                                   "QToolBar#%1 QToolButton::menu-indicator { "
                                   "    image: none; "
                                   "}")
                                   .arg(toolbar->objectName())
                                   .arg(buttonSize));

    // Scale icons to match the toolbar icon size and ensure they fill the
    // button
    QList<QAction*> actions = toolbar->actions();
    for (QAction* action : actions) {
        if (!action->icon().isNull()) {
            QIcon originalIcon = action->icon();
            // Get the pixmap at the desired size
            QPixmap pixmap = originalIcon.pixmap(iconSize);
            if (pixmap.isNull()) {
                // If pixmap is null, try to get any available size
                QList<QSize> availableSizes = originalIcon.availableSizes();
                if (!availableSizes.isEmpty()) {
                    pixmap = originalIcon.pixmap(availableSizes.first());
                }
            }
            if (!pixmap.isNull()) {
                // Scale to exact size while keeping aspect ratio
                if (pixmap.size() != iconSize) {
                    pixmap = pixmap.scaled(iconSize, Qt::KeepAspectRatio,
                                           Qt::SmoothTransformation);
                }
                // Create new icon with properly sized pixmap
                QIcon scaledIcon(pixmap);
                action->setIcon(scaledIcon);
            }
        }
    }

    // Force toolbar to update
    toolbar->adjustSize();
    toolbar->updateGeometry();
    toolbar->update();
}

void ecvLayoutManager::hideAdditionalPluginToolbars() {
    // Hide all additional plugin toolbars that were unified
    QList<QToolBar*> additionalToolbars =
            m_pluginManager->additionalPluginToolbars();

    for (QToolBar* toolbar : additionalToolbars) {
        if (toolbar) {
            // Remove from main window if attached
            if (toolbar->parent() == m_mainWindow) {
                m_mainWindow->removeToolBar(toolbar);
            }
            // Set parent to nullptr to prevent restoreState from re-adding it
            toolbar->setParent(nullptr);
            toolbar->setVisible(false);
            toolbar->hide();
        }
    }
}

void ecvLayoutManager::repositionUnifiedPluginToolbar() {
    // Find UnifiedPluginToolbar - search all children
    QList<QToolBar*> allToolbars = m_mainWindow->findChildren<QToolBar*>();
    QToolBar* unifiedPluginToolbar = nullptr;

    for (QToolBar* tb : allToolbars) {
        if (tb->objectName() == "UnifiedPluginToolbar") {
            unifiedPluginToolbar = tb;
            break;
        }
    }

    if (!unifiedPluginToolbar) {
        CVLog::Warning("[LayoutManager] UnifiedPluginToolbar not found!");
        return;
    }

    if (unifiedPluginToolbar->actions().isEmpty()) {
        CVLog::Warning("[LayoutManager] UnifiedPluginToolbar has no actions!");
        return;
    }

    // Ensure it's attached to main window
    if (unifiedPluginToolbar->parent() != m_mainWindow) {
        unifiedPluginToolbar->setParent(m_mainWindow);
    }

    // Simply ensure it's visible and in the top area
    Qt::ToolBarArea area = m_mainWindow->toolBarArea(unifiedPluginToolbar);
    if (area == Qt::NoToolBarArea) {
        // Not added yet, add it to top
        m_mainWindow->addToolBar(Qt::TopToolBarArea, unifiedPluginToolbar);
    } else if (area != Qt::TopToolBarArea) {
        // In wrong area, move it to top
        m_mainWindow->removeToolBar(unifiedPluginToolbar);
        m_mainWindow->addToolBar(Qt::TopToolBarArea, unifiedPluginToolbar);
    }

    unifiedPluginToolbar->setVisible(true);
    unifiedPluginToolbar->show();
}

void ecvLayoutManager::setupToolbarLayout(int screenWidth) {
    // Get all toolbars and categorize them
    QList<QToolBar*> toolBars = m_mainWindow->findChildren<QToolBar*>();
    QSet<QToolBar*> processedToolbars;

    QList<QToolBar*> topToolbars;
    QList<QToolBar*> rightToolbars;
    QList<QToolBar*> leftToolbars;

    // Get list of additional plugin toolbars that were unified (to skip them)
    QList<QToolBar*> additionalPluginToolbars =
            m_pluginManager->additionalPluginToolbars();
    QSet<QToolBar*> additionalToolbarsSet(additionalPluginToolbars.begin(),
                                          additionalPluginToolbars.end());

    // Categorize toolbars by area
    for (QToolBar* toolbar : toolBars) {
        // Skip hidden toolbars
        if (!toolbar->isVisible() && toolbar->parent() == m_mainWindow &&
            toolbar->objectName() != "UnifiedPluginToolbar") {
            continue;
        }

        // Skip unified additional plugin toolbars
        if (additionalToolbarsSet.contains(toolbar)) {
            continue;
        }

        // Handle UnifiedPluginToolbar
        if (toolbar->objectName() == "UnifiedPluginToolbar") {
            if (toolbar->actions().isEmpty() ||
                processedToolbars.contains(toolbar)) {
                continue;
            }
            processedToolbars.insert(toolbar);
            m_mainWindow->removeToolBar(toolbar);
            topToolbars.append(toolbar);
            continue;
        }

        // Categorize by side
        if (m_rightSideToolBars.contains(toolbar)) {
            m_mainWindow->removeToolBar(toolbar);
            rightToolbars.append(toolbar);
            continue;
        }

        m_mainWindow->removeToolBar(toolbar);

        if (m_leftSideToolBars.contains(toolbar)) {
            leftToolbars.append(toolbar);
        } else {
            topToolbars.append(toolbar);
        }
    }

    // Add right-side toolbars
    for (QToolBar* toolbar : rightToolbars) {
        m_mainWindow->addToolBar(Qt::RightToolBarArea, toolbar);
        toolbar->setVisible(true);
        setToolbarIconSize(toolbar, screenWidth);
    }

    // Add left-side toolbars
    for (QToolBar* toolbar : leftToolbars) {
        m_mainWindow->addToolBar(Qt::LeftToolBarArea, toolbar);
        toolbar->setVisible(true);
        setToolbarIconSize(toolbar, screenWidth);
    }

    // Setup top toolbars in two-row configuration
    // First row: mainToolBar, SFToolBar, FilterToolBar, [UnifiedPluginToolbar]
    // Second row: [Reconstruction], AnnotationToolBar, MeasurementsToolBar,
    // SelectionToolBar, [Others]

    QMap<QString, QToolBar*> toolbarMap;
    QList<QToolBar*> reconstructionToolbars;
    QToolBar* unifiedPluginToolbar = nullptr;

    for (QToolBar* toolbar : topToolbars) {
        QString toolbarName = toolbar->objectName();

        if (toolbarName == "UnifiedPluginToolbar") {
            if (!unifiedPluginToolbar && !toolbar->actions().isEmpty()) {
                unifiedPluginToolbar = toolbar;
            }
        } else if (toolbarName == "Reconstruction") {
            reconstructionToolbars.append(toolbar);
        } else {
            toolbarMap[toolbarName] = toolbar;
        }
    }

    // Add first row toolbars
    QStringList firstRowOrder = {"mainToolBar", "SFToolBar", "FilterToolBar"};
    for (const QString& name : firstRowOrder) {
        if (toolbarMap.contains(name)) {
            QToolBar* toolbar = toolbarMap[name];
            m_mainWindow->addToolBar(Qt::TopToolBarArea, toolbar);
            toolbar->setVisible(true);
            setToolbarIconSize(toolbar, screenWidth);
            toolbarMap.remove(name);
        }
    }

    // Add UnifiedPluginToolbar at end of first row
    if (unifiedPluginToolbar && !unifiedPluginToolbar->actions().isEmpty()) {
        m_mainWindow->addToolBar(Qt::TopToolBarArea, unifiedPluginToolbar);
        unifiedPluginToolbar->setVisible(true);
        setToolbarIconSize(unifiedPluginToolbar, screenWidth);
    }

    // Add toolbar break for second row
    m_mainWindow->addToolBarBreak(Qt::TopToolBarArea);

    // Add reconstruction toolbars at beginning of second row
    for (QToolBar* toolbar : reconstructionToolbars) {
        m_mainWindow->addToolBar(Qt::TopToolBarArea, toolbar);
        toolbar->setVisible(true);
        setToolbarIconSize(toolbar, screenWidth);
    }

    // Add second row toolbars
    QStringList secondRowOrder = {"AnnotationToolBar", "MeasurementsToolBar",
                                  "SelectionToolBar"};
    for (const QString& name : secondRowOrder) {
        if (toolbarMap.contains(name)) {
            QToolBar* toolbar = toolbarMap[name];
            m_mainWindow->addToolBar(Qt::TopToolBarArea, toolbar);
            toolbar->setVisible(true);
            setToolbarIconSize(toolbar, screenWidth);
            toolbarMap.remove(name);
        }
    }

    // Add remaining toolbars
    for (QToolBar* toolbar : toolbarMap.values()) {
        m_mainWindow->addToolBar(Qt::TopToolBarArea, toolbar);
        toolbar->setVisible(true);
        setToolbarIconSize(toolbar, screenWidth);
    }

    CVLog::Print("[LayoutManager] Toolbar layout configured");
}

void ecvLayoutManager::setupDockWidgetLayout(int screenWidth,
                                             int screenHeight) {
    // Get all dock widgets
    QList<QDockWidget*> dockWidgets =
            m_mainWindow->findChildren<QDockWidget*>();

    // Setup dock widgets based on screen resolution
    for (QDockWidget* dw : dockWidgets) {
        if (m_bottomDockWidgets.contains(dw)) {
            m_mainWindow->addDockWidget(Qt::BottomDockWidgetArea, dw);
        }
        // Other dock widgets are handled by MainWindow
    }

    CVLog::Print("[LayoutManager] Dock widget layout configured");
}

void ecvLayoutManager::setupMainWindowGeometry(int screenWidth,
                                               int screenHeight) {
    // Adjust main window size based on screen resolution
    int windowWidth;
    int windowHeight;

    if (screenWidth >= 3840) {
        // 4K resolution: use 90% of screen
        windowWidth = static_cast<int>(screenWidth * 0.9);
        windowHeight = static_cast<int>(screenHeight * 0.9);
    } else if (screenWidth >= 1920) {
        // Full HD: use 95% of screen
        windowWidth = static_cast<int>(screenWidth * 0.95);
        windowHeight = static_cast<int>(screenHeight * 0.95);
    } else {
        // Lower resolution: maximize
        m_mainWindow->showMaximized();
        return;
    }

    // Center the window
    int x = (screenWidth - windowWidth) / 2;
    int y = (screenHeight - windowHeight) / 2;

    m_mainWindow->setGeometry(x, y, windowWidth, windowHeight);

    CVLog::Print(QString("[LayoutManager] Main window geometry set: %1x%2 at "
                         "(%3, %4)")
                         .arg(windowWidth)
                         .arg(windowHeight)
                         .arg(x)
                         .arg(y));
}

void ecvLayoutManager::setupDefaultLayout() {
    // Get screen resolution
    QRect screenGeometry = getScreenGeometry();
    int screenWidth = screenGeometry.width();
    int screenHeight = screenGeometry.height();

    CVLog::Print(QString("[LayoutManager] Screen resolution: %1x%2")
                         .arg(screenWidth)
                         .arg(screenHeight));

    // Setup components
    setupToolbarLayout(screenWidth);
    setupDockWidgetLayout(screenWidth, screenHeight);
    setupMainWindowGeometry(screenWidth, screenHeight);

    CVLog::Print("[LayoutManager] GUI Default layout setup complete");
}

void ecvLayoutManager::saveGUILayout() {
    if (!m_autoSaveEnabled) {
        return;
    }

    // Ensure additional plugin toolbars remain hidden before saving
    hideAdditionalPluginToolbars();

    // Save the state as settings
    QSettings settings;
    settings.setValue(ecvPS::MainWinGeom(), m_mainWindow->saveGeometry());
    settings.setValue(ecvPS::MainWinState(), m_mainWindow->saveState());

    CVLog::Print("[LayoutManager] GUI layout saved");
}

void ecvLayoutManager::restoreGUILayout(bool forceDefault) {
    QSettings settings;
    QVariant geometry = settings.value(ecvPS::MainWinGeom());

    // Get screen resolution for icon size calculation
    QRect screenGeometry = getScreenGeometry();
    int screenWidth = screenGeometry.width();

    if (!forceDefault && geometry.isValid()) {
        // Restore saved layout
        m_mainWindow->restoreGeometry(geometry.toByteArray());
        m_mainWindow->restoreState(
                settings.value(ecvPS::MainWinState()).toByteArray());

        // After restoring, hide additional plugin toolbars
        hideAdditionalPluginToolbars();

        // Ensure UnifiedPluginToolbar is visible
        repositionUnifiedPluginToolbar();

        // Update icon sizes for all toolbars after restoring layout
        // This ensures icons are properly sized even when restoring from saved state
        QList<QToolBar*> allToolbars = m_mainWindow->findChildren<QToolBar*>();
        for (QToolBar* toolbar : allToolbars) {
            if (toolbar && toolbar->parent() == m_mainWindow) {
                setToolbarIconSize(toolbar, screenWidth);
            }
        }

        CVLog::Print("[LayoutManager] GUI layout restored from settings");
    } else {
        // Use default layout
        setupDefaultLayout();
        CVLog::Print("[LayoutManager] GUI Using default layout");
    }
}

void ecvLayoutManager::registerRightSideToolBar(QToolBar* toolbar) {
    if (toolbar) {
        m_rightSideToolBars.insert(toolbar);
    }
}

void ecvLayoutManager::registerLeftSideToolBar(QToolBar* toolbar) {
    if (toolbar) {
        m_leftSideToolBars.insert(toolbar);
    }
}

void ecvLayoutManager::registerBottomDockWidget(QDockWidget* dockWidget) {
    if (dockWidget) {
        m_bottomDockWidgets.insert(dockWidget);
    }
}

void ecvLayoutManager::saveCustomLayout() {
    if (!m_mainWindow) {
        CVLog::Error("[ecvLayoutManager] Main window is null!");
        return;
    }

    QSettings settings;
    settings.setValue(ecvPS::CustomLayoutGeom(), m_mainWindow->saveGeometry());
    settings.setValue(ecvPS::CustomLayoutState(), m_mainWindow->saveState());

    CVLog::Print("[ecvLayoutManager] Custom layout saved successfully");
}

void ecvLayoutManager::restoreDefaultLayout() {
    if (!m_mainWindow) {
        CVLog::Error("[ecvLayoutManager] Main window is null!");
        return;
    }

    // Clear main window state to restore default layout
    QSettings settings;
    settings.remove(ecvPS::MainWinGeom());
    settings.remove(ecvPS::MainWinState());

    // Setup default layout
    setupDefaultLayout();
    saveGUILayout();

    CVLog::Print("[ecvLayoutManager] Default layout restored successfully");
}

bool ecvLayoutManager::restoreCustomLayout() {
    if (!m_mainWindow) {
        CVLog::Error("[ecvLayoutManager] Main window is null!");
        return false;
    }

    QSettings settings;
    QVariant geometry = settings.value(ecvPS::CustomLayoutGeom());
    QVariant state = settings.value(ecvPS::CustomLayoutState());

    if (!geometry.isValid() || !state.isValid()) {
        CVLog::Warning("[ecvLayoutManager] No saved custom layout found");
        return false;
    }

    // Restore custom layout
    m_mainWindow->restoreGeometry(geometry.toByteArray());
    m_mainWindow->restoreState(state.toByteArray());

    CVLog::Print("[ecvLayoutManager] Custom layout restored successfully");
    return true;
}
