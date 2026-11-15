// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeySequence>
#include <QVariant>

// UI defaults only. Model facts and inference defaults belong to AICore's
// public task catalogs; this is shared by the workspace and standalone canvas.
namespace annotationConfig {
inline const QJsonObject& defaults() {
    static const QJsonObject object = QJsonDocument::fromJson(
#include "AnnotationDefaults.inc"
                                              )
                                              .object();
    return object;
}

inline QJsonValue jsonValue(const QString& path) {
    QJsonValue result(defaults());
    for (const QString& key : path.split(QLatin1Char('/')))
        result = result.toObject().value(key);
    return result;
}

inline QVariant value(const char* path) {
    return jsonValue(QString::fromLatin1(path)).toVariant();
}

inline QColor color(const char* path) {
    const QJsonValue item = jsonValue(QString::fromLatin1(path));
    if (item.isString()) return QColor(item.toString());
    const QJsonArray rgba = item.toArray();
    return QColor(rgba.at(0).toInt(), rgba.at(1).toInt(), rgba.at(2).toInt(),
                  rgba.size() > 3 ? rgba.at(3).toInt() : 255);
}

inline QList<QKeySequence> shortcuts(const QString& key) {
    const QJsonValue item = jsonValue(QStringLiteral("shortcuts/") + key);
    QList<QKeySequence> result;
    if (item.isString()) result.append(QKeySequence(item.toString()));
    for (const QJsonValue& sequence : item.toArray())
        result.append(QKeySequence(sequence.toString()));
    return result;
}
}  // namespace annotationConfig
