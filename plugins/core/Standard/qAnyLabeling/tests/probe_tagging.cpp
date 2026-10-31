// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <cstdio>

#include "AnnotationTagging.h"

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QFile vocabulary(QStringLiteral(
            ":/CC/plugin/qAnyLabeling/resources/ram/ram_tag_list.txt"));
    if (!vocabulary.open(QIODevice::ReadOnly)) return 1;
    const auto names =
            QString::fromUtf8(vocabulary.readAll()).split(QLatin1Char('\n'));
    if (names.size() != 4585 || annotationRamTranslations().size() != 4585 ||
        annotationRamTag(names[0], QStringLiteral("zh")) !=
                QStringLiteral("三维CG渲染 ") ||
        annotationRamTag(QStringLiteral("unregistered"),
                         QStringLiteral("zh")) !=
                QStringLiteral("unregistered"))
        return 1;
    QJsonArray scenarios;
    for (const QString mode : {QStringLiteral("en"), QStringLiteral("zh"),
                               QStringLiteral("invalid"), QString()}) {
        for (int filter = 0; filter < 4; ++filter) {
            const QStringList deleted = filter == 1 || filter == 3
                                                ? names.mid(0, 2)
                                                : QStringList();
            const QStringList retained = filter == 2 || filter == 3
                                                 ? names.mid(0, 5)
                                                 : QStringList();
            const QSet<QString> removed(deleted.cbegin(), deleted.cend());
            const QSet<QString> kept(retained.cbegin(), retained.cend());
            QJsonArray tags;
            QStringList values;
            for (const auto& name : names)
                if (annotationKeepRamTag(name, removed, kept)) {
                    const auto tag = annotationRamTag(name, mode);
                    tags.append(tag);
                    values.append(tag);
                }
            const int expected = filter == 1 || filter == 3 ? 4583
                                 : filter == 2              ? 5
                                                            : 4585;
            if (tags.size() != expected) return 1;
            scenarios.append(QJsonObject{
                    {"mode", mode},
                    {"delete_tags", QJsonArray::fromStringList(deleted)},
                    {"filter_tags", QJsonArray::fromStringList(retained)},
                    {"tags", tags},
                    {"normalized_tags",
                     annotationNormalizedImageTags(values)}});
        }
    }
    if (argc == 2) {
        QFile output(QString::fromLocal8Bit(argv[1]));
        if (!output.open(QIODevice::WriteOnly) ||
            output.write(QJsonDocument(scenarios).toJson()) < 0)
            return 1;
    }
    std::printf(
            "RAM presentation: 4585 bilingual labels, 16 language/filter "
            "scenarios PASS\n");
    return 0;
}
