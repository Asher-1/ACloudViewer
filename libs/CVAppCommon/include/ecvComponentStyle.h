// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QProxyStyle>

#include "CVAppCommon.h"

// Global theme-aware vector art for Qt standard component icons.
class CVAPPCOMMON_LIB_API ecvComponentStyle : public QProxyStyle {
    Q_OBJECT
public:
    explicit ecvComponentStyle(const QString& nativeStyleKey);
    QIcon standardIcon(StandardPixmap icon,
                       const QStyleOption* option = nullptr,
                       const QWidget* widget = nullptr) const override;

    static bool install(QApplication* app,
                        const QString& nativeStyleKey = QString());
    static QString componentRules(bool dark);
    static void applyTheme(QApplication* app,
                           const QString& baseStyleSheet,
                           const QString& themeKey);

private:
    static bool useDarkIcons(QApplication* app,
                             const QString& styleSheet,
                             const QString& themeKey);
};
