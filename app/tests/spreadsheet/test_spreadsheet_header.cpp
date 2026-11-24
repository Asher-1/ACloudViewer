// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Regression gate for the spreadsheet header word-wrap optimization:
// compound field names (Normals_Magnitude / Points_Magnitude) are displayed
// on two lines and their columns narrowed, while the underlying field name
// (model headerData, TSV export, grouping) stays byte-for-byte unchanged.

#include <ecvPointCloud.h>
#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QCoreApplication>
#include <QFontMetrics>
#include <QHeaderView>
#include <QMenu>
#include <QMouseEvent>
#include <QTableView>

#include "ecvSpreadSheetView.h"

namespace {

void pump() {
    for (int i = 0; i < 5; ++i) {
        QApplication::processEvents();
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    }
}

void fillCloudWithNormals(ccPointCloud& cloud) {
    cloud.reserveThePointsTable(3);
    cloud.reserveTheNormsTable();
    for (int i = 0; i < 3; ++i) {
        cloud.addPoint(CCVector3(1.0f + i, 2.0f, 3.0f));
        cloud.addNorm(CCVector3(0.0f, 0.0f, 1.0f));
    }
}

int columnByName(const QAbstractItemModel* model, const QString& name) {
    for (int c = 0; c < model->columnCount(); ++c) {
        if (model->headerData(c, Qt::Horizontal, Qt::DisplayRole).toString() ==
            name) {
            return c;
        }
    }
    return -1;
}

}  // namespace

// The model and TSV export must keep the exact underscore-joined field name.
TEST(SpreadSheetHeader, FieldNameConsistencyPreserved) {
    ccPointCloud cloud("test");
    fillCloudWithNormals(cloud);
    ecvSpreadSheetModel model;
    model.setEntity(&cloud);

    const int nrm = columnByName(&model, "Normals_Magnitude");
    const int pts = columnByName(&model, "Points_Magnitude");
    ASSERT_GE(nrm, 0) << "Normals_Magnitude column exists";
    ASSERT_GE(pts, 0) << "Points_Magnitude column exists";

    EXPECT_EQ(model.headerData(nrm, Qt::Horizontal, Qt::DisplayRole).toString(),
              "Normals_Magnitude");
    EXPECT_EQ(model.headerData(pts, Qt::Horizontal, Qt::DisplayRole).toString(),
              "Points_Magnitude");

    const QString tsv = model.getRowsAsString();
    const QString headerLine = tsv.section('\n', 0, 0);
    EXPECT_TRUE(headerLine.contains("Normals_Magnitude"));
    EXPECT_TRUE(headerLine.contains("Points_Magnitude"));
    EXPECT_FALSE(headerLine.contains("Normals\nMagnitude"));
}

// The rendered column must be narrower than the full unwrapped name yet wide
// enough for the longest wrapped word.
TEST(SpreadSheetHeader, CompoundColumnNarrowedToWrappedWidth) {
    ccPointCloud cloud("test");
    fillCloudWithNormals(cloud);
    ecvSpreadSheetView view;
    view.setEntity(&cloud);
    pump();

    auto* table = view.findChild<QTableView*>();
    ASSERT_TRUE(table != nullptr);
    auto* header = table->horizontalHeader();
    ASSERT_TRUE(header != nullptr);

    const int nrm = columnByName(table->model(), "Normals_Magnitude");
    ASSERT_GE(nrm, 0);

    QFont boldFont = header->font();
    boldFont.setBold(true);
    QFontMetrics fm(boldFont);
    const int fullWidth = fm.horizontalAdvance("Normals_Magnitude") + 12;
    const int wrappedWidth = fm.horizontalAdvance("Magnitude") + 12;

    const int actual = header->sectionSize(nrm);
    EXPECT_LT(actual, fullWidth)
            << "column no longer reserves the full underscore-joined name";
    EXPECT_GE(actual, wrappedWidth)
            << "column still fits the longest wrapped word";
}

// The column-visibility menu hosts real QCheckBox widgets; a mouse click on
// one must toggle the column (independent of QMenu action-trigger mediation).
TEST(SpreadSheetHeader, ColumnVisibilityCheckboxClickable) {
    ccPointCloud cloud("test");
    fillCloudWithNormals(cloud);
    ecvSpreadSheetView view;
    view.setEntity(&cloud);
    pump();
    QMenu* menu = nullptr;
    for (auto* m : view.findChildren<QMenu*>()) {
        if (m->parentWidget() == &view) menu = m;
    }
    ASSERT_TRUE(menu != nullptr);
    menu->popup(QPoint(100, 100));
    pump();
    QCheckBox* target = nullptr;
    for (auto* c : menu->findChildren<QCheckBox*>()) {
        if (c->text() == "Normals_Magnitude") target = c;
    }
    ASSERT_TRUE(target != nullptr) << "menu hosts a real checkbox";
    const int col = columnByName(view.getModel(), "Normals_Magnitude");
    ASSERT_GE(col, 0);
    const bool before = view.getModel()->isColumnVisible(col);
    const QRect r = target->rect();
    const QPoint c = r.center();
    const QPoint g = target->mapToGlobal(c);
    QMouseEvent press(QEvent::MouseButtonPress, c, g, Qt::LeftButton,
                      Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(target, &press);
    QApplication::processEvents();
    QMouseEvent release(QEvent::MouseButtonRelease, c, g, Qt::LeftButton,
                        Qt::NoButton, Qt::NoModifier);
    QCoreApplication::sendEvent(target, &release);
    pump();
    const bool after = view.getModel()->isColumnVisible(col);
    EXPECT_NE(before, after) << "clicking the checkbox must toggle visibility";

    // Re-showing must not accumulate stale checkbox widgets.
    menu->popup(QPoint(120, 120));
    pump();
    int count = 0;
    for (auto* c : menu->findChildren<QCheckBox*>()) {
        if (c->text() == "Normals_Magnitude") ++count;
    }
    EXPECT_EQ(count, 1) << "rebuild drops stale checkboxes";
}

TEST(SpreadSheetModel, SelectedOnlyKeepsSourceIdsAndEmptyMeansEmpty) {
    ccPointCloud cloud("source");
    fillCloudWithNormals(cloud);
    ecvSpreadSheetModel model;
    model.setEntity(&cloud);
    model.setSelectionOnly(true);
    EXPECT_EQ(model.rowCount(), 0);
    model.setSelectedIndices({2, 0, 999});
    ASSERT_EQ(model.rowCount(), 2);
    EXPECT_EQ(model.sourceRow(0), 0u);
    EXPECT_EQ(model.sourceRow(1), 2u);
    model.setSelectedIndices({});
    EXPECT_EQ(model.rowCount(), 0);
    model.setSelectedIndices({1});
    model.setAttributeType(ecvSpreadSheetModel::CELL_DATA);
    EXPECT_EQ(model.rowCount(), 0);
    model.setAttributeType(ecvSpreadSheetModel::POINT_DATA);
    model.setSelectedIndices({1});
    model.setEntity(&cloud);
    EXPECT_EQ(model.rowCount(), 0);
    model.setSelectionOnly(false);
    model.setDecimalPrecision(1);
    int column = columnByName(&model, "Points");
    ASSERT_GE(column, 0);
    EXPECT_DOUBLE_EQ(
            model.data(model.index(2, column), Qt::UserRole).toDouble(), 3.0);
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}