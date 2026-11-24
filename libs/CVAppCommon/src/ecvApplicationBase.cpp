// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSettings>
#include <QStandardPaths>
#include <QString>
#include <QStyle>
#include <QStyleFactory>
#include <QSurfaceFormat>
#include <QTextStream>
#include <QTranslator>
#include <QWidget>
#include <QtGlobal>

// CV_CORE_LIB
#include <CVPlatform.h>

// CV_DB_LIB
#include <ecvMaterial.h>

// Common
#include <oclero/qlementine/resources/ResourceInitialization.hpp>
#include <oclero/qlementine/style/QlementineStyle.hpp>
#include <oclero/qlementine/style/Theme.hpp>

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
    {
        QSurfaceFormat format = QSurfaceFormat::defaultFormat();
        format.setStencilBufferSize(0);
#ifdef CV_GL_WINDOW_USE_QWINDOW
        format.setStereo(true);
#endif
        format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
        format.setSwapInterval(
                0);  // Disable VSync globally for responsive interaction

#ifdef Q_OS_MAC
        format.setVersion(3, 3);  // must be 3.3
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
    setOrganizationName("CVCorp");

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

    // Capture the platform appearance before any Custom palette/style is used.
    // Persistent restore happens once after applicationName and the settings
    // manager are initialized, not here under a different QSettings identity.
    m_systemStyleKey = style()->objectName();
    m_systemPalette = palette();
    m_systemFont = font();

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

bool ecvApplicationBase::setAppStyle(const QString &styleKey, bool persist) {
    AppearanceState state = m_appearance;
    state.styleKey = styleKey;
    return applyAppearance(state, persist);
}

namespace {
std::optional<oclero::qlementine::Theme> loadCustomTheme(const QString &id) {
    QString name;
    if (id == QStringLiteral("QDarkStyleSheet::Dark"))
        name = QStringLiteral("qdarkstyle-dark");
    else if (id == QStringLiteral("QDarkStyleSheet::Light"))
        name = QStringLiteral("qdarkstyle-light");
    else {
        const QRegularExpression preset(
                QStringLiteral("^:/qss/([A-Za-z0-9_-]+)\\.css$"));
        const auto match = preset.match(id);
        if (!match.hasMatch()) return std::nullopt;
        name = match.captured(1);
    }
    QFile file(QStringLiteral(":/ecv/qlementine/themes/") + name +
               QStringLiteral(".json"));
    if (!file.open(QFile::ReadOnly)) return std::nullopt;
    QJsonParseError error;
    const auto document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject())
        return std::nullopt;
    const auto preset = document.object();
    const auto base = preset.value(QStringLiteral("base")).toString();
    if (base != QStringLiteral("light") && base != QStringLiteral("dark"))
        return std::nullopt;
    if (!preset.value(QStringLiteral("overrides")).isObject())
        return std::nullopt;

    oclero::qlementine::resources::initializeResources();
    QFile defaults(QStringLiteral(":/qlementine/resources/themes/") + base +
                   QStringLiteral(".json"));
    if (!defaults.open(QFile::ReadOnly)) return std::nullopt;
    auto merged = QJsonDocument::fromJson(defaults.readAll(), &error).object();
    if (error.error != QJsonParseError::NoError || merged.isEmpty())
        return std::nullopt;
    const auto overrides = preset.value(QStringLiteral("overrides")).toObject();
    for (auto it = overrides.begin(); it != overrides.end(); ++it)
        merged.insert(it.key(), it.value());
    // Dense scientific windows should keep the platform's readable font and
    // fallback families rather than silently shrinking everything to Inter12px.
    // Qlementine still owns geometry, control states and animation.
    merged.insert(QStringLiteral("useSystemFonts"), true);
    return oclero::qlementine::Theme::fromJsonDoc(QJsonDocument(merged));
}
}  // namespace

bool ecvApplicationBase::applyAppearance(const AppearanceState &requested,
                                         bool persist) {
    AppearanceState state = requested;
    const bool custom = state.styleKey == QStringLiteral("Custom");
    std::optional<oclero::qlementine::Theme> theme;
    if (custom) {
        if (state.themeId.isEmpty())
            state.themeId = QStringLiteral(":/qss/paraview.css");
        theme = loadCustomTheme(state.themeId);
        if (!theme) return false;
    }

    QStyle *nextStyle = nullptr;
    if (custom) {
        auto *backend = new oclero::qlementine::QlementineStyle;
        // Approved multicolor application/plugin art is not monochrome chrome.
        backend->setAutoIconColor(oclero::qlementine::AutoIconColor::None);
        backend->setTheme(*theme);
        nextStyle = backend;
    } else {
        const auto key =
                state.styleKey.isEmpty() ? m_systemStyleKey : state.styleKey;
        nextStyle = QStyleFactory::create(key);
        if (!nextStyle) return false;
        if (!state.styleKey.isEmpty()) state.styleKey = nextStyle->objectName();
    }

    // Delete the previous custom style through QApplication ownership. Native
    // modes never pass through the component proxy or its global stylesheet.
    setStyleSheet(QString());
    // Let the old style unpolish against the values it actually applied before
    // changing inherited fonts/palettes; otherwise conditional ownership checks
    // can mistake our reset for a business override and leave local bold fonts.
    setStyle(nextStyle);
    setProperty("ecvCustomAppearance", custom);
    if (custom) {
        setPalette(theme->palette);
        setFont(theme->fontRegular);
    } else {
        setPalette(m_systemPalette);
        setFont(m_systemFont);
        // Qt5 can retain stylesheet-derived Disabled brushes on already-open
        // children even after app QSS removal (without WA_SetPalette). Resolve
        // inherited palettes again, parent first; preserve deliberate local
        // palettes such as data colors, image canvases and plugin overrides.
        const auto inheritPalette = [](auto &&self, QWidget *widget) -> void {
            if (!widget->testAttribute(Qt::WA_SetPalette))
                widget->setPalette(QPalette());
            for (QObject *child : widget->children()) {
                if (QWidget *descendant = qobject_cast<QWidget *>(child))
                    self(self, descendant);
            }
        };
        for (QWidget *window : topLevelWidgets())
            inheritPalette(inheritPalette, window);
    }
    m_appearance = state;
    if (persist) commitAppearance();
    emit appearanceChanged();
    return true;
}

bool ecvApplicationBase::setCustomTheme(const QString &themeId) {
    // A legacy theme menu must not silently turn Fusion/Windows into Custom.
    if (!isCustomAppearance()) return false;
    AppearanceState state = m_appearance;
    state.themeId = themeId;
    return applyAppearance(state);
}

void ecvApplicationBase::commitAppearance() {
    QVariantMap state;
    state.insert(QStringLiteral("version"), 1);
    state.insert(QStringLiteral("style"), m_appearance.styleKey);
    state.insert(QStringLiteral("theme"), m_appearance.themeId);
    // One map/key is the authoritative state. Legacy keys are migration input
    // only; no competing writers remain in the menu or Display Options.
    ecvSettingManager::setValue(QStringLiteral("Appearance"),
                                QStringLiteral("state"), state);
}

bool ecvApplicationBase::restoreAppearance() {
    const QVariant saved = ecvSettingManager::getValue(
            QStringLiteral("Appearance"), QStringLiteral("state"));
    AppearanceState state;
    if (saved.isValid()) {
        const QVariantMap map = saved.toMap();
        if (map.value(QStringLiteral("version")).toInt() != 1) return false;
        state.styleKey = map.value(QStringLiteral("style")).toString();
        state.themeId =
                map.value(QStringLiteral("theme"), state.themeId).toString();
        return applyAppearance(state, false);
    }

    const QString oldStyle = ecvSettingManager::getValue(
                                     ecvPS::AppStyle(), QStringLiteral("style"))
                                     .toString();
    const QString oldTheme = ecvSettingManager::getValue(ecvPS::ThemeSettings(),
                                                         ecvPS::CurrentTheme())
                                     .toString();
    // Respect an explicitly saved native selection over an old CSS menu value.
    // A saved legacy custom style/theme maps to the explicit Custom mode once.
    if (oldStyle.startsWith(QStringLiteral("QDarkStyleSheet::"))) {
        state.styleKey = QStringLiteral("Custom");
        state.themeId = oldStyle;
    } else if (!oldStyle.isEmpty()) {
        state.styleKey = oldStyle;
        if (!oldTheme.isEmpty()) state.themeId = oldTheme;
    } else if (!oldTheme.isEmpty()) {
        state.styleKey = QStringLiteral("Custom");
        state.themeId = oldTheme;
    }
    return applyAppearance(state, true);
}
