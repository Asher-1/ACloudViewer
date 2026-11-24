// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QtCompat.h>
#include <gtest/gtest.h>

#include <QApplication>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QLineEdit>
#include <QTableWidget>
#include <QTemporaryDir>
#include <memory>

#include "RelationFilterWidget.h"

namespace {
QJsonObject edge(int from,
                 int to,
                 const QString& subject,
                 const QString& predicate,
                 const QString& object,
                 double score,
                 const QString& type) {
    return QJsonObject{{QStringLiteral("subject_index"), from},
                       {QStringLiteral("object_index"), to},
                       {QStringLiteral("subject"), subject},
                       {QStringLiteral("predicate"), predicate},
                       {QStringLiteral("object"), object},
                       {QStringLiteral("score"), score},
                       {QStringLiteral("type"), type}};
}

class RelationFilterTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // QApplication must outlive every widget and every test in the suite.
        static int argc = 1;
        static char name[] = "test_qanylabeling_relation_filter";
        static char* argv[] = {name, nullptr};
        static QApplication app(argc, argv);
    }

    void SetUp() override {
        widget = std::make_unique<RelationFilterWidget>();
        widget->resize(600, 300);
        table = widget->findChild<QTableWidget*>(
                QStringLiteral("AnnotationRelationTable"));
        type = widget->findChild<QComboBox*>();
        minScore = widget->findChild<QDoubleSpinBox*>();
        search = widget->findChild<QLineEdit*>();
        ASSERT_NE(table, nullptr);
        ASSERT_NE(type, nullptr);
        ASSERT_NE(minScore, nullptr);
        ASSERT_NE(search, nullptr);

        edges = QJsonArray{
                edge(0, 1, QStringLiteral("person"), QStringLiteral("riding"),
                     QStringLiteral("horse"), 0.59, QStringLiteral("semantic")),
                edge(1, 2, QStringLiteral("horse"), QStringLiteral("on"),
                     QStringLiteral("grass"), 0.80, QStringLiteral("spatial")),
                // Comma, quote, and spreadsheet-formula injection characters.
                edge(2, 0, QStringLiteral("=cmd"),
                     QStringLiteral("next to, left"), QStringLiteral("a\"b"),
                     0.52, QStringLiteral("spatial"))};
        QObject::connect(widget.get(), &RelationFilterWidget::filterChanged,
                         widget.get(), [this](const QJsonArray& visible) {
                             ++filterSignals;
                             lastVisible = visible;
                         });
        widget->loadRelations(edges);
    }

    std::unique_ptr<RelationFilterWidget> widget;
    QTableWidget* table = nullptr;
    QComboBox* type = nullptr;
    QDoubleSpinBox* minScore = nullptr;
    QLineEdit* search = nullptr;
    QJsonArray edges;
    int filterSignals = 0;
    QJsonArray lastVisible;
};

TEST_F(RelationFilterTest, LoadingShowsEveryRelationAndPublishesVisibleSet) {
    EXPECT_EQ(table->rowCount(), 3);
    EXPECT_EQ(widget->visibleRelations(), edges);
    EXPECT_EQ(lastVisible, edges);
    EXPECT_GT(filterSignals, 0);
}

TEST_F(RelationFilterTest, TypeAndScoreFiltersIntersectAndCanBeReset) {
    const int spatial = type->findData(QStringLiteral("spatial"));
    ASSERT_GE(spatial, 0);
    type->setCurrentIndex(spatial);
    EXPECT_EQ(table->rowCount(), 2);
    EXPECT_EQ(lastVisible, (QJsonArray{edges[1], edges[2]}));

    minScore->setValue(0.6);
    ASSERT_EQ(table->rowCount(), 1);
    ASSERT_NE(table->item(0, 1), nullptr);
    EXPECT_EQ(table->item(0, 1)->text(), QStringLiteral("on"));
    EXPECT_EQ(widget->visibleRelations(), (QJsonArray{edges[1]}));
    EXPECT_EQ(lastVisible, widget->visibleRelations());

    minScore->setValue(0.0);
    const int all = type->findData(QStringLiteral("all"));
    ASSERT_GE(all, 0);
    type->setCurrentIndex(all);
    EXPECT_EQ(table->rowCount(), 3);
    EXPECT_EQ(widget->visibleRelations(), edges);
}

TEST_F(RelationFilterTest, ScoreFloorIncludesAnExactBoundaryScore) {
    minScore->setValue(0.59);
    EXPECT_EQ(widget->visibleRelations(), (QJsonArray{edges[0], edges[1]}));
    minScore->setValue(0.60);
    EXPECT_EQ(widget->visibleRelations(), (QJsonArray{edges[1]}));
}

TEST_F(RelationFilterTest, SearchMatchesSubjectObjectAndPredicateIgnoringCase) {
    search->setText(QStringLiteral("HORSE"));
    EXPECT_EQ(table->rowCount(), 2);
    EXPECT_EQ(widget->visibleRelations(), (QJsonArray{edges[0], edges[1]}));
    search->setText(QStringLiteral("RIDING"));
    EXPECT_EQ(widget->visibleRelations(), (QJsonArray{edges[0]}));
    search->clear();
    EXPECT_EQ(table->rowCount(), 3);
    EXPECT_EQ(widget->visibleRelations(), edges);
}

TEST_F(RelationFilterTest, RowClickReportsVisibleRowAndOriginalShapeIndices) {
    QJsonObject selected;
    int selections = 0;
    QObject::connect(widget.get(), &RelationFilterWidget::relationSelected,
                     widget.get(), [&](const QJsonObject& relation) {
                         ++selections;
                         selected = relation;
                     });
    const int spatial = type->findData(QStringLiteral("spatial"));
    ASSERT_GE(spatial, 0);
    type->setCurrentIndex(spatial);
    ASSERT_TRUE(QMetaObject::invokeMethod(table, "cellClicked", Q_ARG(int, 1),
                                          Q_ARG(int, 0)));
    EXPECT_EQ(selections, 1);
    EXPECT_EQ(selected.value(QStringLiteral("row")).toInt(-1), 1);
    EXPECT_EQ(selected.value(QStringLiteral("subject_index")).toInt(-1), 2);
    EXPECT_EQ(selected.value(QStringLiteral("object_index")).toInt(-1), 0);
}

TEST_F(RelationFilterTest, CsvFormatsScoresEscapesTextAndDefusesFormulas) {
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("relations.csv"));
    QString error;
    ASSERT_TRUE(widget->saveCsv(path, &error)) << error.toStdString();
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    const QStringList lines =
            QString::fromUtf8(file.readAll())
                    .split(QLatin1Char('\n'), QtCompat::SkipEmptyParts);
    ASSERT_EQ(lines.size(), 4);
    EXPECT_EQ(lines[0], QStringLiteral("Subject,Predicate,Object,Type,Score"));
    EXPECT_EQ(lines[1], QStringLiteral("person,riding,horse,semantic,0.590"));
    EXPECT_EQ(lines[3], QStringLiteral("'=cmd,\"next to, left\",\"a\"\"b\","
                                       "spatial,0.520"));
}

TEST_F(RelationFilterTest, CsvExportsOnlyVisibleRelations) {
    search->setText(QStringLiteral("RIDING"));
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("filtered.csv"));
    QString error;
    ASSERT_TRUE(widget->saveCsv(path, &error)) << error.toStdString();
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_EQ(file.readAll(),
              QByteArray("Subject,Predicate,Object,Type,Score\n"
                         "person,riding,horse,semantic,0.590\n"));
}

TEST_F(RelationFilterTest, CsvWriteFailureReportsAReason) {
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    QString failure;
    EXPECT_FALSE(widget->saveCsv(
            directory.filePath(QStringLiteral("missing/x.csv")), &failure));
    EXPECT_FALSE(failure.isEmpty());
}

TEST_F(RelationFilterTest, ClearEmptiesTableAndVisibleRelations) {
    widget->clear();
    EXPECT_EQ(table->rowCount(), 0);
    EXPECT_TRUE(widget->visibleRelations().isEmpty());
}
}  // namespace
