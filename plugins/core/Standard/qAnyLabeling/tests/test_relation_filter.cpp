// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QLineEdit>
#include <QTableWidget>
#include <QTemporaryDir>
#include <cstdio>

#include "RelationFilterWidget.h"

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

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
}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    RelationFilterWidget widget;
    widget.resize(600, 300);
    auto* table = widget.findChild<QTableWidget*>(
            QStringLiteral("AnnotationRelationTable"));
    auto* type = widget.findChild<QComboBox*>();
    auto* minScore = widget.findChild<QDoubleSpinBox*>();
    auto* search = widget.findChild<QLineEdit*>();
    if (!check(table && type && minScore && search,
               "Relation filter controls are missing"))
        return 1;

    const QJsonArray edges{
            edge(0, 1, QStringLiteral("person"), QStringLiteral("riding"),
                 QStringLiteral("horse"), 0.59, QStringLiteral("semantic")),
            edge(1, 2, QStringLiteral("horse"), QStringLiteral("on"),
                 QStringLiteral("grass"), 0.80, QStringLiteral("spatial")),
            // Hostile text: a comma, a quote and a leading formula character.
            edge(2, 0, QStringLiteral("=cmd"), QStringLiteral("next to, left"),
                 QStringLiteral("a\"b"), 0.52, QStringLiteral("spatial"))};

    int filterSignals = 0;
    int lastVisible = -1;
    QObject::connect(&widget, &RelationFilterWidget::filterChanged,
                     [&](const QJsonArray& visible) {
                         ++filterSignals;
                         lastVisible = visible.size();
                     });
    QJsonObject selected;
    int selections = 0;
    QObject::connect(&widget, &RelationFilterWidget::relationSelected,
                     [&](const QJsonObject& relation) {
                         ++selections;
                         selected = relation;
                     });

    widget.loadRelations(edges);
    if (!check(table->rowCount() == 3 &&
                       widget.visibleRelations().size() == 3 &&
                       lastVisible == 3,
               "Loading must show every relation and publish the visible set"))
        return 1;

    type->setCurrentIndex(type->findData(QStringLiteral("spatial")));
    if (!check(table->rowCount() == 2 && lastVisible == 2,
               "The type filter must keep only spatial relations"))
        return 1;
    minScore->setValue(0.6);
    if (!check(table->rowCount() == 1 &&
                       table->item(0, 1)->text() == QStringLiteral("on"),
               "The score floor must drop relations below it"))
        return 1;
    minScore->setValue(0.0);
    type->setCurrentIndex(type->findData(QStringLiteral("all")));
    search->setText(QStringLiteral("HORSE"));
    if (!check(table->rowCount() == 2,
               "Search must match subject or object, ignoring case"))
        return 1;
    search->clear();
    if (!check(table->rowCount() == 3, "Clearing the search restores rows"))
        return 1;

    // A click reports the row in the visible list, which is also the index of
    // the edge in the canvas overlay, and keeps the shape indices.
    type->setCurrentIndex(type->findData(QStringLiteral("spatial")));
    QMetaObject::invokeMethod(table, "cellClicked", Q_ARG(int, 1),
                              Q_ARG(int, 0));
    if (!check(selections == 1 &&
                       selected.value(QStringLiteral("row")).toInt(-1) == 1 &&
                       selected.value(QStringLiteral("subject_index"))
                                       .toInt(-1) == 2 &&
                       selected.value(QStringLiteral("object_index"))
                                       .toInt(-1) == 0,
               "A row click must report its visible row and shape indices"))
        return 1;
    type->setCurrentIndex(type->findData(QStringLiteral("all")));

    QTemporaryDir directory;
    const QString path = directory.filePath(QStringLiteral("relations.csv"));
    QString error;
    if (!check(widget.saveCsv(path, &error), "CSV export must succeed")) {
        std::fprintf(stderr, "%s\n", error.toLocal8Bit().constData());
        return 1;
    }
    QFile file(path);
    if (!check(file.open(QIODevice::ReadOnly), "CSV must be readable"))
        return 1;
    const QStringList lines =
            QString::fromUtf8(file.readAll())
                    .split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    if (!check(lines.size() == 4 &&
                       lines[0] ==
                               QStringLiteral(
                                       "Subject,Predicate,Object,Type,Score") &&
                       lines[1] ==
                               QStringLiteral("person,riding,horse,semantic,"
                                              "0.590"),
               "CSV header and first row are wrong"))
        return 1;
    if (!check(lines[3] == QStringLiteral("'=cmd,\"next to, left\",\"a\"\"b\","
                                          "spatial,0.520"),
               "CSV must quote commas and quotes and defuse formulas"))
        return 1;

    // A failed write must be reported, never silently accepted.
    QString failure;
    if (!check(!widget.saveCsv(
                       directory.filePath(QStringLiteral("missing/x.csv")),
                       &failure) &&
                       !failure.isEmpty(),
               "CSV export to an unwritable path must fail with a reason"))
        return 1;

    widget.clear();
    if (!check(table->rowCount() == 0 && widget.visibleRelations().isEmpty(),
               "clear() must empty the table"))
        return 1;
    if (!check(filterSignals > 0, "filterChanged was never emitted")) return 1;
    return 0;
}
