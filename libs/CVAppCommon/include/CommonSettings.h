// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// core
#include <CloudViewerConfig.h>

// QT
#include <QCoreApplication>
#include <QMainWindow>
#include <QSettings>
#include <QString>

// LOCAL
#include "ecvVersion.h"

namespace Themes {
static const QString THEME_DEFAULT = "";
static const QString THEME_PARAVIEW = ":/qss/paraview.css";
static const QString THEME_BLUE = ":/qss/blue.css";
static const QString THEME_LIGHTBLUE = ":/qss/lightblue.css";
static const QString THEME_DARKBLUE = ":/qss/darkblue.css";
static const QString THEME_BLACK = ":/qss/black.css";
static const QString THEME_LIGHTBLACK = ":/qss/lightblack.css";
static const QString THEME_FLATBLACK = ":/qss/flatblack.css";
static const QString THEME_DarkBLACK = ":/qss/darkblack.css";
static const QString THEME_GRAY = ":/qss/gray.css";
static const QString THEME_LIGHTGRAY = ":/qss/lightgray.css";
static const QString THEME_DarkGRAY = ":/qss/darkgray.css";
static const QString THEME_FLATWHITE = ":/qss/flatwhite.css";
static const QString THEME_PSBLACK = ":/qss/psblack.css";
static const QString THEME_SILVER = ":/qss/silvery.css";
static const QString THEME_BF = ":/qss/bf.css";
static const QString THEME_TEST = ":/qss/test.css";
}  // namespace Themes

namespace Settings {
// settings
static QString CONFIG_PATH = "configuration.ini";
static QString LOGFILE = "log.log";

// Define the URL of the Update Definitions file
static const QString UPDATE_RUL =
        "https://raw.githubusercontent.com/Asher-1/CloudViewerUpdate/main/"
        "updates.json";
//    static const QString UPDATE_RUL     =
//    "https://github.com/Asher-1/Asher-1.github.io/blob/main/files/ACloudViewer/updates.json?raw=true";

// logos
static const QString APP_LOGO = ":/Resources/images/icon/logo_256.png";
static const QString APP_START_LOGO = ":/Resources/images/icon/logo.png";
static const QString CLOUDFILE_LOGO = ":/Resources/images/dbCloudSymbol.png";
static const QString THEME_LOGO = ":/Resources/images/theme.png";
static const QString MINIMUM_LOGO = ":/Resources/images/mini.png";
static const QString MAXIMUM_LOGO = ":/Resources/images/max.png";
static const QString CLOSE_LOGO = ":/Resources/images/close.png";

// coding
static const char* CODING = "UTF8";

// application information
static const QString APP_VERSION =
        QString::fromStdString(cloudViewer::GetCloudViewerVersion());
static const QString TITLE = QObject::tr("ACloudViewer");
static const QString APP_TITLE =
        TITLE + " " + versionLongStr(true, APP_VERSION);

// theme style
static bool UI_WRAPPER = false;
static const QString DEFAULT_STYLE = Themes::THEME_DEFAULT;
}  // namespace Settings
