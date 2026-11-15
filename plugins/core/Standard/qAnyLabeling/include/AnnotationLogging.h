// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <aicore/inference_log.h>

#include <QSettings>
#include <QString>

// Plugin-side logger for qAnyLabeling: upstream parity for the
// logger_level setting ("info" | "warning" | "error" — a standard
// logging level filter). Lines carry their severity in the
// [Error] / [Warning] prefix convention of aicore_inference_log; the
// level gate drops lines below the configured floor before routing.
// The level is cached after the first read and refreshed by the
// settings dialog when the user changes it.

namespace qany_log {

inline int& levelCache() {
    static int level = [] {
        QSettings settings;
        const QString value =
                settings.value(QStringLiteral("qAnyLabeling/loggerLevel"),
                               QStringLiteral("info"))
                        .toString();
        if (value == QLatin1String("warning")) return 1;
        if (value == QLatin1String("error")) return 2;
        return 0;
    }();
    return level;
}

inline void refreshLevel() {
    QSettings settings;
    const QString value =
            settings.value(QStringLiteral("qAnyLabeling/loggerLevel"),
                           QStringLiteral("info"))
                    .toString();
    levelCache() = value == QLatin1String("warning") ? 1
                   : value == QLatin1String("error") ? 2
                                                     : 0;
}

inline void log(const QString& line) {
    const int level = levelCache();
    if (level >= 1) {
        const bool isError =
                line.contains(QStringLiteral("[Error]"), Qt::CaseInsensitive) ||
                line.startsWith(QStringLiteral("Error"), Qt::CaseInsensitive);
        if (level >= 2 && !isError) return;
        const bool isWarning =
                line.contains(QStringLiteral("[Warning]"),
                              Qt::CaseInsensitive) ||
                line.startsWith(QStringLiteral("Warning"), Qt::CaseInsensitive);
        if (level >= 1 && !isError && !isWarning) return;
    }
    aicore_inference_log::log(line);
}

}  // namespace qany_log
