// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFile>
#include <QHash>
#include <QJsonArray>
#include <QSet>
#include <QStringList>

// Presentation vocabulary only: inference labels/thresholds still come from
// the public AICore tagging result. The paired source files retain their GPL
// provenance in resources/ram/README.md.
inline const QHash<QString, QString>& annotationRamTranslations() {
    static const auto translations = []() {
        QHash<QString, QString> result;
        QFile en(QStringLiteral(
                ":/CC/plugin/qAnyLabeling/resources/ram/ram_tag_list.txt"));
        QFile zh(
                QStringLiteral(":/CC/plugin/qAnyLabeling/resources/ram/"
                               "ram_tag_list_chinese.txt"));
        if (!en.open(QIODevice::ReadOnly) || !zh.open(QIODevice::ReadOnly))
            return result;
        const auto lines = [](QFile& file) {
            auto values =
                    QString::fromUtf8(file.readAll()).split(QLatin1Char('\n'));
            if (!values.isEmpty() && values.last().isEmpty())
                values.removeLast();
            return values;
        };
        const auto english = lines(en);
        const auto chinese = lines(zh);
        if (english.size() != chinese.size()) return result;
        for (int i = 0; i < english.size(); ++i)
            if (!english[i].isEmpty()) result.insert(english[i], chinese[i]);
        return result;
    }();
    return translations;
}

inline QString annotationRamTag(const QString& english, const QString& mode) {
    return mode == QStringLiteral("zh")
                   ? annotationRamTranslations().value(english, english)
                   : english;
}

inline bool annotationKeepRamTag(const QString& name,
                                 const QSet<QString>& deleted,
                                 const QSet<QString>& retained) {
    // Source __base__/ram.py: delete_tags takes precedence over filter_tags.
    return !name.isEmpty() &&
           (!deleted.isEmpty() ? !deleted.contains(name)
                               : retained.isEmpty() || retained.contains(name));
}

inline QJsonArray annotationNormalizedImageTags(const QStringList& values) {
    QJsonArray tags;
    QSet<QString> seen;
    for (const auto& value : values) {
        const QString tag = value.trimmed();
        if (tag.isEmpty() || tag.contains(QLatin1Char('\r')) ||
            tag.contains(QLatin1Char('\n')) || seen.contains(tag))
            continue;
        seen.insert(tag);
        tags.append(tag);
    }
    return tags;
}
