// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVLog.h>

#include <QCoreApplication>
#include <QString>
#include <QStringList>

#include "aicore/backend_capi.h"

namespace aicore_inference_log {

inline QString trText(const char* text,
                      const char* disambiguation = nullptr,
                      int n = -1) {
    return QCoreApplication::translate("aicore_inference_log", text,
                                       disambiguation, n);
}

/** Route to CVLog (console + log file via ecvConsole registration). */
inline void log(const QString& line) {
    if (line.contains(QStringLiteral("[Error]"), Qt::CaseInsensitive) ||
        line.startsWith(QStringLiteral("Error"), Qt::CaseInsensitive)) {
        CVLog::Error(line);
    } else if (line.contains(QStringLiteral("[Warning]"),
                             Qt::CaseInsensitive) ||
               line.startsWith(QStringLiteral("Warning"),
                               Qt::CaseInsensitive)) {
        CVLog::Warning(line);
    } else {
        CVLog::Print(line);
    }
}

inline QString format_device_request(const QString& device) {
    const QString req = device.trimmed();
    if (!req.isEmpty() &&
        req.compare(QLatin1String("auto"), Qt::CaseInsensitive) != 0) {
        return req;
    }
    return QStringLiteral("auto (%1)")
            .arg(QString::fromUtf8(aicore_auto_device_order()));
}

inline QString backend_probe_line(const QString& tag) {
    const int n = aicore_device_count();
    QStringList names;
    names.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (const aicore_device_info* d = aicore_device_at(i)) {
            names << QString::fromUtf8(d->id);
        }
    }
    return trText("[%1] Backends available (%2): %3")
            .arg(tag)
            .arg(n)
            .arg(names.isEmpty() ? trText("(none)")
                                 : names.join(QStringLiteral(", ")));
}

inline void log_backend_probe(const QString& tag) {
    log(backend_probe_line(tag));
}

inline void log_device_request(const QString& tag, const QString& device) {
    log(trText("[%1] Using device: %2")
                .arg(tag, format_device_request(device)));
}

inline void log_device_resolved(const QString& tag,
                                const QString& resolved_device) {
    if (resolved_device.trimmed().isEmpty()) return;
    log(trText("[%1] ggml backend ready on device: %2")
                .arg(tag, resolved_device));
}

inline void log_inference_done(const QString& tag,
                               const QString& resolved_device,
                               double runtime_ms,
                               const QString& summary) {
    QString line =
            trText("[%1] Done in %2 ms").arg(tag).arg(runtime_ms, 0, 'f', 1);
    if (!resolved_device.trimmed().isEmpty()) {
        line += trText(" [%1]").arg(resolved_device);
    }
    if (!summary.trimmed().isEmpty()) {
        line += trText(" — %1").arg(summary);
    }
    log(line);
}

}  // namespace aicore_inference_log
