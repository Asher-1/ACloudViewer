// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QDir>
#include <QFile>
#include <QSettings>
#include <QStandardPaths>
#include <QString>
#include <QStyle>
#include <QStyleFactory>
#include <QSurfaceFormat>
#include <QTextStream>
#include <QTranslator>
#include <QtGlobal>

// CV_CORE_LIB
#include <CVPlatform.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>

// Common
#include "ecvApplicationBase.h"
#include "ecvPersistentSettings.h"
#include "ecvPluginManager.h"
#include "ecvSettingManager.h"
#include "ecvTranslationManager.h"

// CV_CORE_LIB
#include <CVLog.h>

#if (QT_VERSION < QT_VERSION_CHECK(5, 5, 0))
#error ACloudViewer does not support versions of Qt prior to 5.5
#endif

void ecvApplicationBase::InitOpenGL() {
    // See
    // http://doc.qt.io/qt-5/qopenglwidget.html#opengl-function-calls-headers-and-qopenglfunctions
    /** Calling QSurfaceFormat::setDefaultFormat() before constructing the
    QApplication instance is mandatory on some platforms (for example, OS X)
    when an OpenGL core profile context is requested. This is to ensure that
    resource sharing between contexts stays functional as all internal contexts
    are created using the correct version and profile.
    **/
    //
    // CRITICAL (Qt6 + VTK 9.x):
    // VTK documentation requires that the application-wide default format is
    // compatible with QVTKOpenGLNativeWidget::defaultFormat().  In Qt6, the
    // global shared OpenGL context is created from the application-wide default
    // format.  All QOpenGLWidget instances (including QVTKOpenGLNativeWidget)
    // share resources with this global context.  If the global context lacks
    // depth/stencil/alpha buffers, VTK falls back to slow rendering paths,
    // causes visual artifacts and stuttering.
    //
    // We replicate VTK's recommended format values here rather than including
    // VTK headers, so that CVAppCommon stays independent of VTK.
    {
        QSurfaceFormat format;

        // Rendering type & swap
        format.setRenderableType(QSurfaceFormat::OpenGL);
        format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
        format.setSwapInterval(1);  // VSync: prevents tearing & reduces idle
                                    // GPU spinning

        // Buffer sizes matching VTK's QVTKRenderWindowAdapter::defaultFormat()
        format.setDepthBufferSize(24);   // VTK default is 8; 24 gives better
                                         // depth precision for 3D scenes
        format.setStencilBufferSize(8);  // VTK needs stencil for depth peeling,
                                         // selection highlights, etc.
        format.setRedBufferSize(8);
        format.setGreenBufferSize(8);
        format.setBlueBufferSize(8);
        format.setAlphaBufferSize(8);    // needed for VTK compositing / OIT

        // No MSAA at the surface level — VTK handles anti-aliasing internally
        format.setSamples(0);

#ifdef CV_GL_WINDOW_USE_QWINDOW
        format.setStereo(true);
#endif

#ifdef Q_OS_MAC
        // macOS requires explicit Core Profile to get OpenGL 3.3+
        format.setVersion(3, 3);
        format.setProfile(QSurfaceFormat::CoreProfile);
#else
        // Other platforms: request GL 3.2 Core (VTK minimum)
        format.setVersion(3, 2);
        format.setProfile(QSurfaceFormat::CoreProfile);
#endif

#if (QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)) && \
        (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
        // These attributes are deprecated in Qt6 (high DPI is enabled by
        // default)
        QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
        QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

#ifdef QT_DEBUG
        format.setOption(QSurfaceFormat::DebugContext, true);
#endif
        QSurfaceFormat::setDefaultFormat(format);
    }

    // The 'AA_ShareOpenGLContexts' attribute must be defined BEFORE the
    // creation of the Q(Gui)Application DGM: this is mandatory to enable
    // exclusive full screen for ccGLWidget (at least on Windows)
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
}

ecvApplicationBase::ecvApplicationBase(int &argc,
                                       char **argv,
                                       bool isCommandLine,
                                       const QString &version)
    : QApplication(argc, argv),
      c_VersionStr(version),
      c_CommandLine(isCommandLine) {
    setOrganizationName("ECVCorp");

    setupPaths();

#ifdef Q_OS_MAC
    // Mac OS X apps don't show icons in menus
    setAttribute(Qt::AA_DontShowIconsInMenus);
#endif

    // Force 'english' locale so as to get a consistent behavior everywhere
    QLocale::setDefault(QLocale::English);

#ifdef Q_OS_UNIX
    // We reset the numeric locale for POSIX functions
    // See https://doc.qt.io/qt-5/qcoreapplication.html#locale-settings
    setlocale(LC_NUMERIC, "C");
#endif

    // Restore the style from persistent settings
    // (matching CloudCompare's approach - using QSettings directly)
    // Note: We use Qt API directly here instead of setAppStyle() because
    // CVLog and ecvSettingManager are not yet initialized in the constructor
    QSettings settings;
    settings.beginGroup(ecvPS::AppStyle());
    {
        QString styleKey = settings.value("style", QString()).toString();

        // Apply platform-appropriate default if no saved style
        if (styleKey.isEmpty()) {
#ifdef Q_OS_MAC
            // macOS: Use Fusion for consistent button borders (ParaView
            // approach)
            styleKey = "Fusion";
#endif
        }

        // Apply the style using Qt API directly (safe in constructor)
        if (!styleKey.isEmpty()) {
            if (styleKey == "QDarkStyleSheet::Dark") {
                QFile f(":/qdarkstyle/dark/darkstyle.qss");
                if (f.open(QFile::ReadOnly | QFile::Text)) {
                    QTextStream ts(&f);
                    setStyleSheet(ts.readAll());
                    f.close();
                }
            } else if (styleKey == "QDarkStyleSheet::Light") {
                QFile f(":/qdarkstyle/light/lightstyle.qss");
                if (f.open(QFile::ReadOnly | QFile::Text)) {
                    QTextStream ts(&f);
                    setStyleSheet(ts.readAll());
                    f.close();
                }
            } else {
                // Qt native style
                QStyle *style = QStyleFactory::create(styleKey);
                if (style) {
                    setStyle(style);
                }
            }
        }
    }
    settings.endGroup();

    ccPluginManager::get().setPaths(m_PluginPaths);

    ccTranslationManager::get().registerTranslatorFile(QStringLiteral("qt"),
                                                       m_TranslationPath);
    ccTranslationManager::get().registerTranslatorFile(
            QStringLiteral("ACloudViewer"), m_TranslationPath);
    ccTranslationManager::get().loadTranslations();

    connect(this, &ecvApplicationBase::aboutToQuit,
            [=]() { ccMaterial::ReleaseTextures(); });
}

QString ecvApplicationBase::versionStr() const { return c_VersionStr; }

QString ecvApplicationBase::versionLongStr(bool includeOS) const {
    QString verStr = c_VersionStr;

#if defined(CV_ENV_64)
    const QString arch("64-bit");
#elif defined(CV_ENV_32)
    const QString arch("32-bit");
#else
    const QString arch("\?\?-bit");
#endif

    if (includeOS) {
#if defined(CV_WINDOWS)
        const QString platform("Windows");
#elif defined(CV_MAC_OS)
        const QString platform("macOS");
#elif defined(CV_LINUX)
        const QString platform("Linux");
#else
        const QString platform("Unknown OS");
#endif
        verStr += QStringLiteral(" [%1 %2]").arg(platform, arch);
    } else {
        verStr += QStringLiteral(" [%1]").arg(arch);
    }

#ifdef QT_DEBUG
    verStr += QStringLiteral(" [DEBUG]");
#endif

    return verStr;
}

const QString &ecvApplicationBase::translationPath() const {
    return m_TranslationPath;
}

void ecvApplicationBase::setupPaths() {
    QDir appDir = QCoreApplication::applicationDirPath();

    // Set up our shader and plugin paths
#if defined(Q_OS_MAC)
    QDir bundleDir = appDir;

    if (bundleDir.dirName() == "MacOS") {
        bundleDir.cdUp();
    }
    m_PluginPaths << (bundleDir.absolutePath() + "/cvPlugins");
    m_PluginPaths << (bundleDir.absolutePath() + "/PlugIns/cvPlugins");

#if defined(CV_MAC_DEV_PATHS)
    // Used for development only - this is the path where the plugins are built
    // and the shaders are located.
    // This avoids having to install into the application bundle when
    // developing.
    bundleDir.cdUp();
    bundleDir.cdUp();
    bundleDir.cdUp();

    m_PluginPaths << (bundleDir.absolutePath() + "/cvPlugins");
    m_ShaderPath = (bundleDir.absolutePath() + "/shaders");
    m_TranslationPath = (bundleDir.absolutePath() + "/app/translations");
#else
    m_ShaderPath = (bundleDir.absolutePath() + "/Shaders");
    m_TranslationPath = (bundleDir.absolutePath() + "/translations");
#endif
#elif defined(Q_OS_WIN)
    m_PluginPaths << (appDir.absolutePath() + "/plugins");
    m_ShaderPath = (appDir.absolutePath() + "/shaders");
    m_TranslationPath = (appDir.absolutePath() + "/translations");
#elif defined(Q_OS_LINUX)  // Q_OS_LINUX
    // Shaders & plugins are relative to the bin directory where the executable
    // is found
    QDir theDir = appDir;

    if (theDir.dirName() == "bin") {
        theDir.cdUp();
        m_PluginPaths << (theDir.absolutePath() + "/plugins");
        m_PluginPaths << (theDir.absolutePath() + "/bin/plugins");
        m_PluginPaths << (theDir.absolutePath() + "/lib/ACloudViewer/plugins");
        m_ShaderPath = (theDir.absolutePath() + "/share/ACloudViewer/shaders");
        m_TranslationPath =
                (theDir.absolutePath() + "/share/ACloudViewer/translations");
    } else {
        // Choose a reasonable default to look in
        m_PluginPaths << "/usr/lib/ACloudViewer/plugins";
        m_PluginPaths << (theDir.absolutePath() + "/plugins");
        m_PluginPaths << (theDir.absolutePath() + "/bin/plugins");
        m_PluginPaths << (theDir.absolutePath() + "/lib/ACloudViewer/plugins");
        m_ShaderPath = "/usr/share/ACloudViewer/shaders";
        m_TranslationPath = "/usr/share/ACloudViewer/translations";
    }

    // check current application translations path whether exists or not
    // if exist and then overwrite above translation settings.
    // Priority: bin/translations > build_root/translations > standard paths

    // First check bin/translations/ (for development builds)
    QString binTransPath = (appDir.absolutePath() + "/translations");
    if (QDir(binTransPath).exists()) {
        m_TranslationPath = binTransPath;
    } else {
        // Then check build_root/translations/
        QString translationPath = (theDir.absolutePath() + "/translations");
        if (QDir(translationPath).exists()) {
            m_TranslationPath = translationPath;
        }
    }

#else
#warning Need to specify the shader path for this OS.
#endif

    // Add any app data paths to plugin paths
    // Plugins in these directories take precendence over the included ones
    // This allows users to put plugins outside of the install directories.
    const QStringList appDataPaths =
            QStandardPaths::standardLocations(QStandardPaths::AppDataLocation);

    for (const QString &appDataPath : appDataPaths) {
        QString path = appDataPath + "/plugins";

        if (!m_PluginPaths.contains(path))  // avoid duplicate entries (can
                                            // happen, at least on Windows)
        {
            m_PluginPaths << path;
        }
    }
}

bool ecvApplicationBase::setAppStyle(const QString &styleKey) {
    // Helper lambda to load stylesheet from resources
    const auto loadStyleSheet = [this](const QString &resourcePath) -> bool {
        QFile f(resourcePath);
        if (!f.exists()) {
            return false;
        }

        if (!f.open(QFile::ReadOnly | QFile::Text)) {
            return false;
        }

        QTextStream ts(&f);
        setStyleSheet(ts.readAll());
        f.close();
        return true;
    };

    // Handle custom stylesheets
    if (styleKey == "QDarkStyleSheet::Dark") {
        // Load dark stylesheet from resources
        if (!loadStyleSheet(":/qdarkstyle/dark/darkstyle.qss")) {
            return false;
        }
    } else if (styleKey == "QDarkStyleSheet::Light") {
        // Load light stylesheet from resources
        if (!loadStyleSheet(":/qdarkstyle/light/lightstyle.qss")) {
            return false;
        }
    } else {
        // Use Qt native styles (Fusion, Windows, macOS, etc.)
        QStyle *style = QStyleFactory::create(styleKey);
        if (!style) {
            CVLog::Warning(QStringLiteral("Invalid style key or style couldn't "
                                          "be created: %1")
                                   .arg(styleKey));
            return false;
        }

        // Clear any existing stylesheet
        setStyleSheet({});
        CVLog::Print(
                QStringLiteral("Applying application style: %1").arg(styleKey));
        setStyle(style);
    }

    // Save to persistent settings (must be after successful style application)
    ecvSettingManager::setValue(ecvPS::AppStyle(), "style", styleKey);

    return true;
}
