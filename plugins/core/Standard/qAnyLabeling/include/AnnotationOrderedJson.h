// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <QByteArray>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPair>
#include <QString>
#include <QVector>

namespace AnnotationOrderedJson {
// JSON is validated by QJsonDocument before these views are used. Keep raw
// member order and integer/float spelling: Python str(json.loads(...)) uses
// both, whereas QJsonObject sorts keys and merges numeric types.
inline QVector<QByteArray> items(const QByteArray& raw) {
    QVector<QByteArray> items;
    int depth = 0, start = 1;
    bool quoted = false, escaped = false;
    for (int i = 1; i < raw.size() - 1; ++i) {
        const char c = raw[i];
        if (quoted) {
            if (escaped)
                escaped = false;
            else if (c == '\\')
                escaped = true;
            else if (c == '"')
                quoted = false;
        } else if (c == '"')
            quoted = true;
        else if (c == '[' || c == '{')
            ++depth;
        else if (c == ']' || c == '}')
            --depth;
        else if (c == ',' && depth == 0) {
            items.append(raw.mid(start, i - start).trimmed());
            start = i + 1;
        }
    }
    const auto last = raw.mid(start, raw.size() - start - 1).trimmed();
    if (!last.isEmpty()) items.append(last);
    return items;
}

inline QVector<QPair<QString, QByteArray>> members(const QByteArray& raw) {
    QVector<QPair<QString, QByteArray>> members;
    QHash<QString, int> positions;
    for (const auto& item : items(raw)) {
        int end = 1;
        bool escaped = false;
        for (; end < item.size(); ++end) {
            if (escaped)
                escaped = false;
            else if (item[end] == '\\')
                escaped = true;
            else if (item[end] == '"')
                break;
        }
        const QString key = QJsonDocument::fromJson(QByteArray("[") +
                                                    item.left(end + 1) + "]")
                                    .array()
                                    .first()
                                    .toString();
        const auto value = item.mid(item.indexOf(':', end) + 1).trimmed();
        // json.loads keeps the first position but last value for duplicate
        // keys.
        const auto found = positions.constFind(key);
        if (found == positions.constEnd()) {
            positions.insert(key, members.size());
            members.append({key, value});
        } else {
            members[*found].second = value;
        }
    }
    return members;
}

}  // namespace AnnotationOrderedJson
