// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CVAppCommon.h"

// Qt
#include <QApplication>
#include <QFont>
#include <QPalette>

/**
 * @def ecvApp
 * @brief Global pointer to the application instance
 *
 * Mimics Qt's qApp macro for easy access to the CloudViewer application
 * instance. Provides quick access to application-wide functionality.
 *
 * @see ecvApplicationBase
 */
#define ecvApp (static_cast<ecvApplicationBase *>(QCoreApplication::instance()))

/**
 * @class ecvApplicationBase
 * @brief Base class for CloudViewer applications
 *
 * Provides core application functionality including:
 * - OpenGL initialization and setup
 * - Version management
 * - Translation/localization support
 * - Path management (shaders, translations, plugins)
 * - Application styling (including dark/light themes)
 * - Command-line vs GUI mode handling
 *
 * This class extends QApplication and should be instantiated once at
 * application startup. The global ecvApp macro provides convenient access.
 *
 * @see QApplication
 */
class CVAPPCOMMON_LIB_API ecvApplicationBase : public QApplication {
    Q_OBJECT
public:
    // One effective state for startup, menus and Display Options. An empty
    // styleKey selects the platform default; only "Custom" enables our chrome.
    struct AppearanceState {
        QString styleKey;
        QString themeId = QStringLiteral(":/qss/paraview.css");
    };
    /**
     * @brief Initialize OpenGL context
     *
     * Must be called before instantiating the application class.
     * Sets up OpenGL surface format and other OpenGL-related initialization.
     *
     * @warning Call this before creating the application instance!
     */
    static void InitOpenGL();

    /**
     * @brief Constructor
     * @param argc Argument count (from main)
     * @param argv Argument vector (from main)
     * @param isCommandLine Whether running in command-line mode
     * @param version Application version string
     */
    ecvApplicationBase(int &argc,
                       char **argv,
                       bool isCommandLine,
                       const QString &version);

    /**
     * @brief Check if running in command-line mode
     * @return true if command-line mode, false if GUI mode
     */
    bool isCommandLine() const { return c_CommandLine; }

    /**
     * @brief Get short version string
     * @return Version string (e.g., "2.12.4")
     */
    QString versionStr() const;

    /**
     * @brief Get detailed version string
     * @param includeOS Include OS information in version string
     * @return Detailed version string with build info and optionally OS
     */
    QString versionLongStr(bool includeOS) const;

    /**
     * @brief Get translation files path
     * @return Path to translation (.qm) files directory
     */
    const QString &translationPath() const;

    /**
     * @brief Set the application style/theme
     *
     * Changes the visual appearance of the application. Supports both
     * Qt native styles and custom themes.
     *
     * @param styleKey Style name, supported values:
     *   - Qt native: empty (platform default), or a QStyleFactory key
     *   - Custom: "Custom" (theme presets are selectable only in this mode)
     * Legacy QDarkStyleSheet keys are interpreted only by restoreAppearance().
     * @param persist Commit only after successful application; false previews
     *        the state for Display Options, whose Cancel restores its snapshot.
     * @return true if style was applied successfully, false otherwise
     *
     * @note On macOS, native "macintosh" or "macOS" style provides
     *       platform-native appearance
     */
    bool setAppStyle(const QString &styleKey, bool persist = true);
    const AppearanceState &appearance() const { return m_appearance; }
    bool isCustomAppearance() const {
        return m_appearance.styleKey == QStringLiteral("Custom");
    }
    bool applyAppearance(const AppearanceState &state, bool persist = true);
    bool setCustomTheme(const QString &themeId);
    bool restoreAppearance();
    void commitAppearance();

signals:
    void appearanceChanged();

private:
    /**
     * @brief Setup application paths
     *
     * Initializes paths for shaders, translations, and plugins
     * based on application location and platform conventions.
     */
    void setupPaths();

    const QString c_VersionStr;  ///< Application version string

    QString m_ShaderPath;       ///< Path to shader files
    QString m_TranslationPath;  ///< Path to translation files
    QStringList m_PluginPaths;  ///< Paths to plugin directories

    const bool c_CommandLine;  ///< Command-line mode flag
    AppearanceState m_appearance;
    QString m_systemStyleKey;
    QPalette m_systemPalette;
    QFont m_systemFont;
};
