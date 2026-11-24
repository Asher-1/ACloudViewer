// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <cstdio>

#include "AnnotationTagging.h"
#include "QtCompat.h"

namespace {
int probeArgc = 0;
char** probeArgv = nullptr;
}  // namespace

TEST(AnnotationTaggingContractTest, VocabularyAndFilterScenarios) {
    int argc = probeArgc;
    char** argv = probeArgv;
    QCoreApplication app(argc, argv);
    QFile vocabulary(QStringLiteral(
            ":/CC/plugin/qAnyLabeling/resources/ram/ram_tag_list.txt"));
    ASSERT_TRUE(vocabulary.open(QIODevice::ReadOnly));
    const auto names =
            QString::fromUtf8(vocabulary.readAll()).split(QLatin1Char('\n'));
    ASSERT_EQ(names.size(), 4585);
    ASSERT_EQ(annotationRamTranslations().size(), 4585);
    EXPECT_EQ(annotationRamTag(names[0], QStringLiteral("zh")),
              QStringLiteral("三维CG渲染 "));
    EXPECT_EQ(annotationRamTag(QStringLiteral("unregistered"),
                               QStringLiteral("zh")),
              QStringLiteral("unregistered"));
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
            const QSet<QString> removed = qtCompatQSetFromList(deleted);
            const QSet<QString> kept = qtCompatQSetFromList(retained);
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
            ASSERT_EQ(tags.size(), expected);
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
        ASSERT_TRUE(output.open(QIODevice::WriteOnly));
        const auto bytes = QJsonDocument(scenarios).toJson();
        ASSERT_EQ(output.write(bytes), bytes.size());
    }
    std::printf(
            "RAM presentation: 4585 bilingual labels, 16 language/filter "
            "scenarios PASS\n");
}

int main(int argc, char** argv) {
    probeArgc = argc;
    probeArgv = argv;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
