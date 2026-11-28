// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Link real application owners: no coordinator, publisher or receiver stubs.
#include <QVTKOpenGLNativeWidget.h>
#include <Tools/SelectionTools/cvRenderViewSelectionReaction.h>
#include <Tools/SelectionTools/cvSelectionData.h>
#include <Tools/SelectionTools/cvSelectionPipeline.h>
#include <Tools/SelectionTools/cvSelectionToolController.h>
#include <Tools/SelectionTools/cvViewSelectionManager.h>
#include <VTKExtensions/Views/vtkChartView.h>
#include <VTKExtensions/Views/vtkComparativeViewWidget.h>
#include <VTKExtensions/Views/vtkOrthoSliceViewWidget.h>
#include <Visualization/VtkVis.h>
#include <Visualization/vtkGLView.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvSettingManager.h>
#include <ecvViewManager.h>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkAnnotationLink.h>
#include <vtkChart.h>
#include <vtkCommand.h>
#include <vtkContextActor.h>
#include <vtkContextScene.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkInteractorStyle.h>
#include <vtkMapper.h>
#include <vtkNew.h>
#include <vtkPlot.h>
#include <vtkPolyData.h>
#include <vtkPropCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkScatterPlotMatrix.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkTable.h>

#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QItemSelectionModel>
#include <QListWidget>
#include <QLocale>
#include <QPointer>
#include <QPushButton>
#include <QSettings>
#include <QSortFilterProxyModel>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableView>
#include <QTemporaryDir>
#include <algorithm>
#include <cmath>
#include <set>

#include "MainWindow.h"
#include "ecvApplication.h"
#include "ecvMultiViewWidget.h"
#include "ecvPythonView.h"
#include "ecvSpreadSheetView.h"
#include "ecvTabbedMultiViewWidget.h"

namespace {
void pump() {
    for (int i = 0; i < 5; ++i) {
        QApplication::processEvents();
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    }
}

struct Scene {
    ccPointCloud* a = nullptr;
    ccPointCloud* b = nullptr;
};

const Scene& scene() {
    static Scene result;
    if (!result.a) {
        auto* window = MainWindow::TheInstance();
        for (bool second : {false, true}) {
            auto* cloud = new ccPointCloud(second ? "Linkage B" : "Linkage A");
            EXPECT_TRUE(cloud->reserve(13));
            for (int i = 0; i < 13; ++i)
                cloud->addPoint(CCVector3((second ? 30 : 0) + i % 4, i / 4,
                                          (i % 3) * 0.125f));
            // Deliberately reverse scalar-field storage order in source B.
            for (const auto& name : second ? QStringList{"group", "signal"}
                                           : QStringList{"signal", "group"}) {
                const int field =
                        cloud->addScalarField(name.toUtf8().constData());
                EXPECT_GE(field, 0);
                for (int i = 0; i < 13; ++i)
                    cloud->getScalarField(field)->setValue(
                            i, name == "group"
                                       ? i % 3
                                       : (second ? 30.25 : 10.25) + 2 * i);
                cloud->getScalarField(field)->computeMinAndMax();
            }
            cloud->setVisible(true);
            window->addToDB(cloud, true, true, false, true);
            (second ? result.b : result.a) = cloud;
        }
        window->setSelectedInDB(result.b, false);
        window->setSelectedInDB(result.a, true);
        pump();
    }
    return result;
}

// Invoke the actual view-choice button connection, not a test factory.
struct Cell {
    ecvTabbedMultiViewWidget* tabs = nullptr;
    QPointer<QWidget> page;
    QPointer<ecvMultiViewWidget> multi;
    bool created = false;

    explicit Cell(const QString& type) {
        tabs = MainWindow::TheInstance()
                       ->findChild<ecvTabbedMultiViewWidget*>();
        if (!tabs) return;
        const int index = tabs->createTab();
        tabs->setCurrentTab(index);
        page = tabs->tabWidget()->widget(index);
        multi = tabs->currentMultiView();
        if (!multi) return;
        for (auto* button : multi->findChildren<QPushButton*>()) {
            if (button->property("VIEW_TYPE_ID").toString() == type &&
                button->isEnabled()) {
                button->click();
                created = true;
                break;
            }
        }
        pump();
    }
    ~Cell() {
        if (page) {
            const int index = tabs->tabWidget()->indexOf(page);
            if (index >= 0) tabs->closeTab(index);
            pump();
        }
    }
    Cell(const Cell&) = delete;
    Cell& operator=(const Cell&) = delete;
};

vtkAbstractContextItem* chartItem(vtkChartView* view) {
    auto* widget = view->findChild<QVTKOpenGLNativeWidget*>();
    if (!widget) return nullptr;
    auto* renderers = widget->renderWindow()->GetRenderers();
    renderers->InitTraversal();
    while (auto* renderer = renderers->GetNextItem()) {
        auto* props = renderer->GetViewProps();
        props->InitTraversal();
        while (auto* prop = props->GetNextProp())
            if (auto* actor = vtkContextActor::SafeDownCast(prop))
                return actor->GetScene()->GetItem(0);
    }
    return nullptr;
}

vtkChart* chart(vtkChartView* view) {
    auto* item = chartItem(view);
    auto* matrix = vtkScatterPlotMatrix::SafeDownCast(item);
    return matrix ? matrix->GetMainChart() : vtkChart::SafeDownCast(item);
}

void fields(vtkChartView* view, const QStringList& names) {
    auto* list = view->findChild<QListWidget*>();
    ASSERT_NE(list, nullptr);
    for (int i = 0; i < list->count(); ++i) {
        auto* item = list->item(i);
        item->setSelected(names.contains(item->text()));
    }
    pump();
}

void nativeRows(vtkChartView* view, const QVector<qint64>& rows) {
    auto* rendered = chart(view);
    ASSERT_NE(rendered, nullptr);
    ASSERT_NE(rendered->GetAnnotationLink(), nullptr);
    vtkNew<vtkIdTypeArray> ids;
    for (auto row : rows) ids->InsertNextValue(row);
    vtkNew<vtkSelectionNode> node;
    node->SetContentType(vtkSelectionNode::INDICES);
    node->SetFieldType(view->chartType() == vtkChartView::PARALLEL_COORDINATES
                               ? vtkSelectionNode::POINT
                               : vtkSelectionNode::ROW);
    node->SetSelectionList(ids);
    vtkNew<vtkSelection> selection;
    selection->AddNode(node);
    // This setter itself emits the native event. Do not emit it twice.
    rendered->GetAnnotationLink()->SetCurrentSelection(selection);
    pump();
}

void expectChartRows(vtkChartView* view, const std::set<vtkIdType>& expected) {
    auto* tabs =
            MainWindow::TheInstance()->findChild<ecvTabbedMultiViewWidget*>();
    for (int i = 0; i < tabs->tabCount(); ++i) {
        auto* page = tabs->tabWidget()->widget(i);
        if (page->isAncestorOf(view)) tabs->setCurrentTab(i);
    }
    pump();
    auto* rendered = chart(view);
    ASSERT_NE(rendered, nullptr);
    ASSERT_GT(rendered->GetNumberOfPlots(), 0);
    view->findChild<QVTKOpenGLNativeWidget*>()->renderWindow()->Render();
    rendered->Update();
    std::set<vtkIdType> actual;
    if (auto* ids = rendered->GetPlot(0)->GetSelection())
        for (vtkIdType i = 0; i < ids->GetNumberOfValues(); ++i)
            actual.insert(ids->GetValue(i));
    EXPECT_EQ(actual, expected);
}

void expectSelection(ccHObject* source, const QVector<qint64>& ids) {
    const auto selected =
            cvViewSelectionManager::instance()->currentSelection();
    EXPECT_EQ(selected.fieldAssociation(), cvSelectionData::POINTS);
    ASSERT_EQ(selected.actorInfos().size(), 1);
    const auto& info = selected.actorInfos().front();
    EXPECT_EQ(info.sourceId, source->getViewId());
    EXPECT_EQ(std::set<qint64>(info.sourceIds.begin(), info.sourceIds.end()),
              std::set<qint64>(ids.begin(), ids.end()));
}

void expectOverlay(Visualization::VtkVis* vis,
                   ccPointCloud* source,
                   const QVector<qint64>& ids,
                   double dx = 0,
                   double dy = 0,
                   double dz = 0) {
    auto* actor = vis->getActorById("__highlight_selected_overlay__");
    if (ids.isEmpty()) {
        EXPECT_EQ(actor, nullptr);
        return;
    }
    ASSERT_NE(actor, nullptr);
    EXPECT_FALSE(actor->GetPickable());
    actor->GetMapper()->Update();
    auto* poly = vtkPolyData::SafeDownCast(actor->GetMapper()->GetInput());
    ASSERT_NE(poly, nullptr);
    ASSERT_EQ(poly->GetNumberOfPoints(), ids.size());
    for (auto id : ids) {
        const auto* point = source->getPoint(static_cast<unsigned>(id));
        bool found = false;
        for (vtkIdType i = 0; i < poly->GetNumberOfPoints(); ++i) {
            double p[3];
            poly->GetPoint(i, p);
            if (std::abs(p[0] - point->x - dx) < 1e-6 &&
                std::abs(p[1] - point->y - dy) < 1e-6 &&
                std::abs(p[2] - point->z - dz) < 1e-6)
                found = true;
        }
        EXPECT_TRUE(found) << "original source point " << id;
    }
}

int column(QAbstractItemModel* model, const QString& name) {
    for (int c = 0; c < model->columnCount(); ++c)
        if (model->headerData(c, Qt::Horizontal).toString() == name) return c;
    return -1;
}
}  // namespace

TEST(ViewLinkage, RealApplicationOwnsTheSharedSelectionCoordinator) {
    auto* window = MainWindow::TheInstance();
    ASSERT_NE(window, nullptr);
    window->resize(1200, 800);
    window->show();
    pump();
    ASSERT_NE(window->dbRootObject(), nullptr);
    EXPECT_EQ(window->getSelectionManager(),
              cvViewSelectionManager::instance());
    EXPECT_EQ(ecvViewManager::instance().getAllViews().size(), 1);
    ASSERT_NE(window->findChild<ecvTabbedMultiViewWidget*>(), nullptr);
    EXPECT_EQ(window->findChild<ecvTabbedMultiViewWidget*>()->tabCount(), 1);
}

class FactoryViews : public ::testing::TestWithParam<const char*> {};
TEST_P(FactoryViews, CreatesAndDestroysTheActualOwner) {
    scene();
    auto* manager = cvViewSelectionManager::instance();
    manager->clearSelection();
    const auto before = ecvViewManager::instance().getAllViews();
    {
        Cell cell(QString::fromLatin1(GetParam()));
        ASSERT_TRUE(cell.created) << GetParam();
        cell.tabs->setCurrentTab(0);
        pump();
        cell.tabs->setCurrentTab(cell.tabs->tabWidget()->indexOf(cell.page));
        pump();
        ASSERT_NE(cell.multi->activeFrame(), nullptr);
        EXPECT_FALSE(
                cell.multi->activeFrame()->property("IS_EMPTY_CELL").toBool());
        const QString name = QString::fromLatin1(GetParam());
        if (name.contains("(Comparative)")) {
            auto* view = cell.multi->findChild<vtkComparativeViewWidget*>();
            ASSERT_NE(view, nullptr);
            EXPECT_FALSE(view->subWidgets().isEmpty());
        } else if (name == "Python View") {
            EXPECT_NE(cell.multi->findChild<ecvPythonView*>(), nullptr);
        } else if (name == "SpreadSheet View") {
            auto* view = cell.multi->findChild<ecvSpreadSheetView*>();
            ASSERT_NE(view, nullptr);
            EXPECT_EQ(view->getModel()->entity(), scene().a);
            EXPECT_EQ(view->getModel()->rowCount(), 13);
        } else if (name == "Orthographic Slice View") {
            EXPECT_NE(cell.multi->findChild<vtkOrthoSliceViewWidget*>(),
                      nullptr);
        } else if (name == "Render View" || name == "Eye Dome Lighting" ||
                   name == "Slice View") {
            vtkGLView* added = nullptr;
            for (auto* display : ecvViewManager::instance().getAllViews())
                if (!before.contains(display))
                    added = dynamic_cast<vtkGLView*>(display);
            ASSERT_NE(added, nullptr);
            if (name == "Eye Dome Lighting") EXPECT_TRUE(added->isEDLEnabled());
            if (name == "Slice View") EXPECT_TRUE(added->isSliceModeEnabled());
        } else {
            auto* view = cell.multi->findChild<vtkChartView*>();
            ASSERT_NE(view, nullptr);
            const QMap<QString, vtkChartView::ChartType> types = {
                    {"Line Chart View", vtkChartView::LINE_CHART},
                    {"Bar Chart View", vtkChartView::BAR_CHART},
                    {"Point Chart View", vtkChartView::POINT_CHART},
                    {"Histogram View", vtkChartView::HISTOGRAM},
                    {"Box Chart View", vtkChartView::BOX_CHART},
                    {"Quartile Chart View", vtkChartView::QUARTILE_CHART},
                    {"Image Chart View", vtkChartView::IMAGE_CHART},
                    {"Parallel Coordinates View",
                     vtkChartView::PARALLEL_COORDINATES},
                    {"Plot Matrix View", vtkChartView::PLOT_MATRIX}};
            EXPECT_EQ(view->chartType(), types.value(name));
        }
    }
    EXPECT_EQ(ecvViewManager::instance().getAllViews().size(), before.size());
}
INSTANTIATE_TEST_SUITE_P(All18,
                         FactoryViews,
                         ::testing::Values("Render View",
                                           "Render View (Comparative)",
                                           "Bar Chart View",
                                           "Bar Chart View (Comparative)",
                                           "Box Chart View",
                                           "Eye Dome Lighting",
                                           "Histogram View",
                                           "Image Chart View",
                                           "Line Chart View",
                                           "Line Chart View (Comparative)",
                                           "Orthographic Slice View",
                                           "Parallel Coordinates View",
                                           "Plot Matrix View",
                                           "Point Chart View",
                                           "Python View",
                                           "Quartile Chart View",
                                           "Slice View",
                                           "SpreadSheet View"));

class ChartLinkage : public ::testing::TestWithParam<const char*> {};
TEST_P(ChartLinkage, NativeChartAndFactorySpreadsheetRoundTrip) {
    auto* a = scene().a;
    auto* b = scene().b;
    auto* manager = cvViewSelectionManager::instance();
    manager->clearSelection();
    auto* primary = dynamic_cast<vtkGLView*>(
            ecvViewManager::instance().getAllViews().front());
    ASSERT_NE(primary, nullptr);
    Cell render("Render View");
    ASSERT_TRUE(render.created);
    auto* secondary = dynamic_cast<vtkGLView*>(
            ecvViewManager::instance().getActiveView());
    ASSERT_NE(secondary, nullptr);
    ASSERT_NE(secondary, primary);
    // A new layout does not implicitly bind sources from another layout.
    a->addToDisplay(secondary);
    b->addToDisplay(secondary);
    primary->redraw(false, true);
    secondary->redraw(false, true);
    auto* primaryVis = primary->getVisualizer3D();
    auto* secondaryVis = secondary->getVisualizer3D();
    auto* secondaryA = secondaryVis->getActorById(a->getViewId().toStdString());
    ASSERT_NE(secondaryA, nullptr);
    secondaryA->SetPosition(7, -2, 1);

    Cell publisher(QString::fromLatin1(GetParam()));
    ASSERT_TRUE(publisher.created);
    auto* source = publisher.multi->findChild<vtkChartView*>();
    ASSERT_NE(source, nullptr);
    source->setMaxChartPoints(5);  // Original IDs 0, 3, 6, 9, 12.
    fields(source, {"Points_X", "signal"});
    ASSERT_NE(chart(source), nullptr);
    ASSERT_EQ(chart(source)->GetPlot(0)->GetInput()->GetNumberOfRows(), 5);
    Cell receiver("Point Chart View");
    ASSERT_TRUE(receiver.created);
    auto* target = receiver.multi->findChild<vtkChartView*>();
    ASSERT_NE(target, nullptr);
    target->setMaxChartPoints(13);
    fields(target, {"signal"});
    Cell tableCell("SpreadSheet View");
    ASSERT_TRUE(tableCell.created);
    auto* sheet = tableCell.multi->findChild<ecvSpreadSheetView*>();
    ASSERT_NE(sheet, nullptr);
    auto* model = sheet->getModel();
    ASSERT_EQ(model->entity(), a);
    model->setSelectionOnly(true);
    auto* table = sheet->findChild<QTableView*>();
    ASSERT_NE(table, nullptr);
    auto* proxy = qobject_cast<QSortFilterProxyModel*>(table->model());
    ASSERT_NE(proxy, nullptr);
    const int signal = column(model, "signal");
    ASSERT_GE(signal, 0);
    proxy->sort(signal, Qt::DescendingOrder);
    QObject observer;
    int publications = 0;
    QObject::connect(manager,
                     qOverload<const cvSelectionData&>(
                             &cvViewSelectionManager::selectionChanged),
                     &observer,
                     [&](const cvSelectionData&) { ++publications; });

    nativeRows(source, {1, 3});
    EXPECT_EQ(publications, 1);
    expectSelection(a, {3, 9});
    ASSERT_EQ(model->rowCount(), 2);
    EXPECT_EQ(model->sourceRow(0), 3u);
    EXPECT_EQ(model->sourceRow(1), 9u);
    EXPECT_DOUBLE_EQ(proxy->index(0, signal).data(Qt::UserRole).toDouble(),
                     28.25);
    expectChartRows(target, {3, 9});
    expectOverlay(primaryVis, a, {3, 9});
    expectOverlay(secondaryVis, a, {3, 9}, 7, -2, 1);
    EXPECT_EQ(publications, 1);

    proxy->setFilterKeyColumn(0);
    proxy->setFilterFixedString("9");
    ASSERT_EQ(proxy->rowCount(), 1);
    EXPECT_EQ(model->rowCount(), 2);
    EXPECT_EQ(publications, 1);
    proxy->setFilterFixedString({});
    table->selectionModel()->select(
            proxy->index(1, 0),
            QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows);
    pump();
    EXPECT_EQ(publications, 2);
    expectSelection(a, {3});
    ASSERT_EQ(model->rowCount(), 1);
    EXPECT_EQ(model->sourceRow(0), 3u);
    expectChartRows(source, {1});
    expectChartRows(target, {3});
    expectOverlay(primaryVis, a, {3});
    expectOverlay(secondaryVis, a, {3}, 7, -2, 1);
    EXPECT_EQ(publications, 2);

    // Source B has identical row numbers, but a different source ID/schema
    // order.
    target->setEntity(b);
    sheet->setEntity(b);
    expectChartRows(target, {});
    EXPECT_EQ(model->rowCount(), 0);
    EXPECT_EQ(publications, 2);
    source->setEntity(b);
    nativeRows(source, {1});
    EXPECT_EQ(publications, 3);
    expectSelection(b, {3});
    expectChartRows(target, {3});
    ASSERT_EQ(model->rowCount(), 1);
    EXPECT_EQ(model->sourceRow(0), 3u);
    expectOverlay(primaryVis, b, {3});
    expectOverlay(secondaryVis, b, {3});

    source->clearSelection();
    pump();
    EXPECT_EQ(publications, 4);
    EXPECT_TRUE(manager->currentSelection().isEmpty());
    EXPECT_EQ(model->rowCount(), 0);
    expectChartRows(target, {});
    expectOverlay(primaryVis, a, {});
    expectOverlay(secondaryVis, a, {});
    EXPECT_EQ(publications, 4);
}
INSTANTIATE_TEST_SUITE_P(NativePublishers,
                         ChartLinkage,
                         ::testing::Values("Line Chart View",
                                           "Point Chart View",
                                           "Bar Chart View",
                                           "Plot Matrix View",
                                           "Parallel Coordinates View"));

TEST(ViewLinkage, RealRenderReactionPublishesToFactoryReceivers) {
    auto* a = scene().a;
    auto* b = scene().b;
    auto* manager = cvViewSelectionManager::instance();
    manager->clearSelection();
    Cell receiver("Point Chart View");
    ASSERT_TRUE(receiver.created);
    auto* target = receiver.multi->findChild<vtkChartView*>();
    ASSERT_NE(target, nullptr);
    target->setMaxChartPoints(5);
    Cell tableCell("SpreadSheet View");
    ASSERT_TRUE(tableCell.created);
    auto* sheet = tableCell.multi->findChild<ecvSpreadSheetView*>();
    ASSERT_NE(sheet, nullptr);
    sheet->getModel()->setSelectionOnly(true);
    auto* window = MainWindow::TheInstance();
    auto* tabs = window->findChild<ecvTabbedMultiViewWidget*>();
    tabs->setCurrentTab(0);
    pump();
    auto* primary = dynamic_cast<vtkGLView*>(
            ecvViewManager::instance().getActiveView());
    ASSERT_NE(primary, nullptr);
    primary->redraw(false, true);
    primary->zoomGlobal();
    auto* vis = primary->getVisualizer3D();
    auto* selectedActor = vis->getActorById(a->getViewId().toStdString());
    ASSERT_NE(selectedActor, nullptr);
    ASSERT_TRUE(selectedActor->GetVisibility());
    ASSERT_TRUE(selectedActor->GetPickable());
    ASSERT_GT(vis->getCurrentRenderer()->GetViewProps()->IsItemPresent(
                      selectedActor),
              0);
    vis->getRenderWindow()->Render();
    cvRenderViewSelectionReaction* reaction = nullptr;
    for (auto* item : window->findChildren<cvRenderViewSelectionReaction*>())
        if (item->mode() == SelectionMode::SELECT_SURFACE_POINTS)
            reaction = item;
    ASSERT_NE(reaction, nullptr);
    QObject observer;
    int publications = 0, completions = 0;
    QObject::connect(manager,
                     qOverload<const cvSelectionData&>(
                             &cvViewSelectionManager::selectionChanged),
                     &observer,
                     [&](const cvSelectionData&) { ++publications; });
    QObject::connect(cvSelectionToolController::instance(),
                     &cvSelectionToolController::selectionFinished, &observer,
                     [&](const cvSelectionData&) { ++completions; });
    reaction->parentAction()->trigger();
    ASSERT_TRUE(reaction->isActive());
    const auto* size = vis->getRenderWindow()->GetSize();
    ASSERT_GT(size[0], 2);
    ASSERT_GT(size[1], 2);
    int region[] = {0, 0, size[0] - 1, size[1] - 1};
    auto* renderer = vis->getCurrentRenderer();
    for (unsigned i = 0; i < a->size(); ++i) {
        SCOPED_TRACE(i);
        const auto* point = a->getPoint(i);
        renderer->SetWorldPoint(point->x, point->y, point->z, 1);
        renderer->WorldToDisplay();
        const auto* projected = renderer->GetDisplayPoint();
        EXPECT_GE(projected[0], 0);
        EXPECT_LT(projected[0], size[0]);
        EXPECT_GE(projected[1], 0);
        EXPECT_LT(projected[1], size[1]);
        EXPECT_GE(projected[2], 0);
        EXPECT_LE(projected[2], 1);
    }
    auto* style = vis->getRenderWindow()->GetInteractor()->GetInteractorStyle();
    // Drive the real installed VTK observer, not a test publisher or OS input.
    style->InvokeEvent(vtkCommand::SelectionChangedEvent, region);
    pump();
    cvRenderViewSelectionReaction::endActiveSelection();
    QVector<qint64> ids;
    for (qint64 i = 0; i < 13; ++i) ids.push_back(i);
    EXPECT_EQ(publications, 1);
    EXPECT_EQ(completions, 1);
    const auto selected = manager->currentSelection();
    EXPECT_EQ(selected.fieldAssociation(), cvSelectionData::POINTS);
    ASSERT_EQ(selected.actorInfos().size(), 2);
    for (auto* cloud : {a, b}) {
        int matches = 0;
        for (const auto& info : selected.actorInfos()) {
            if (info.sourceId != cloud->getViewId()) continue;
            ++matches;
            EXPECT_EQ(info.sourceIds, ids);
        }
        EXPECT_EQ(matches, 1);
    }
    EXPECT_EQ(sheet->getModel()->entity(), a);
    EXPECT_EQ(sheet->getModel()->rowCount(), 13);
    expectChartRows(target, {0, 1, 2, 3, 4});
    auto* overlay = vis->getActorById("__highlight_selected_overlay__");
    ASSERT_NE(overlay, nullptr);
    EXPECT_FALSE(overlay->GetPickable());
    auto* poly = vtkPolyData::SafeDownCast(overlay->GetMapper()->GetInput());
    ASSERT_NE(poly, nullptr);
    ASSERT_EQ(poly->GetNumberOfPoints(), 26);
    for (auto* cloud : {a, b}) {
        for (qint64 i : ids) {
            const auto* point = cloud->getPoint(static_cast<unsigned>(i));
            double expected[] = {point->x, point->y, point->z};
            const auto closest = poly->FindPoint(expected);
            ASSERT_GE(closest, 0);
            const auto* actual = poly->GetPoint(closest);
            for (int axis = 0; axis < 3; ++axis)
                EXPECT_DOUBLE_EQ(actual[axis], expected[axis]);
        }
    }
    EXPECT_EQ(publications, 1);
    EXPECT_EQ(completions, 1);
    manager->clearSelection();
    EXPECT_EQ(publications, 2);
    EXPECT_EQ(completions, 2);
    EXPECT_EQ(sheet->getModel()->rowCount(), 0);
    expectChartRows(target, {});
    expectOverlay(vis, a, {});

    // Numerical boundary checks, not an alternate publication route.
    tabs->setCurrentTab(0);
    pump();
    primary->redraw(false, true);
    auto* pipeline = manager->getPipeline();
    int outside[] = {-100, -100, -70, -70};
    auto empty = pipeline->executeRectangleSelection(
            outside, cvSelectionPipeline::SURFACE_POINTS);
    ASSERT_NE(empty, nullptr);
    EXPECT_EQ(empty->GetNumberOfNodes(), 0u);
    const auto* viewport = vis->getRenderWindow()->GetSize();
    vtkNew<vtkIntArray> polygon;
    polygon->SetNumberOfComponents(2);
    for (int coordinate : {0, 0, viewport[0] - 1, 0, viewport[0] - 1,
                           viewport[1] - 1, 0, viewport[1] - 1})
        polygon->InsertNextValue(coordinate);
    auto picked = pipeline->executePolygonSelection(
            polygon, cvSelectionPipeline::POLYGON_POINTS);
    ASSERT_NE(picked, nullptr);
    const auto polygonData = cvSelectionPipeline::convertToCvSelectionData(
            picked, cvSelectionPipeline::FIELD_ASSOCIATION_POINTS, vis);
    ASSERT_EQ(polygonData.actorInfos().size(), 2);
    EXPECT_EQ(polygonData.count(), 26);
    for (const auto& info : polygonData.actorInfos()) {
        EXPECT_TRUE(info.sourceId == a->getViewId() ||
                    info.sourceId == b->getViewId());
        EXPECT_EQ(info.sourceIds, ids);
    }
    polygon->Reset();
    for (int coordinate : {-100, -100, -70, -100, -70, -70, -100, -70})
        polygon->InsertNextValue(coordinate);
    empty = pipeline->executePolygonSelection(
            polygon, cvSelectionPipeline::POLYGON_POINTS);
    ASSERT_NE(empty, nullptr);
    EXPECT_EQ(empty->GetNumberOfNodes(), 0u);
    EXPECT_TRUE(manager->currentSelection().isEmpty());
    EXPECT_EQ(publications, 2);
    EXPECT_EQ(completions, 2);
}

TEST(ViewLinkage, NativeHistogramBinAndFactorySpreadsheetKeepOriginalRows) {
    auto* a = scene().a;
    auto* manager = cvViewSelectionManager::instance();
    manager->clearSelection();
    Cell sourceCell("Histogram View");
    ASSERT_TRUE(sourceCell.created);
    auto* source = sourceCell.multi->findChild<vtkChartView*>();
    ASSERT_NE(source, nullptr);
    source->setMaxChartPoints(2);
    fields(source, {"signal"});
    for (auto* spin : source->findChildren<QSpinBox*>())
        if (spin->minimum() == 2 && spin->maximum() == 1000) spin->setValue(2);
    Cell receiver("Point Chart View");
    ASSERT_TRUE(receiver.created);
    auto* target = receiver.multi->findChild<vtkChartView*>();
    ASSERT_NE(target, nullptr);
    target->setMaxChartPoints(13);
    fields(target, {"signal"});
    Cell tableCell("SpreadSheet View");
    ASSERT_TRUE(tableCell.created);
    auto* sheet = tableCell.multi->findChild<ecvSpreadSheetView*>();
    ASSERT_NE(sheet, nullptr);
    auto* model = sheet->getModel();
    ASSERT_EQ(model->entity(), a);
    model->setSelectionOnly(true);
    auto* table = sheet->findChild<QTableView*>();
    ASSERT_NE(table, nullptr);
    QObject observer;
    int publications = 0;
    QObject::connect(manager,
                     qOverload<const cvSelectionData&>(
                             &cvViewSelectionManager::selectionChanged),
                     &observer,
                     [&](const cvSelectionData&) { ++publications; });
    nativeRows(source, {0});
    EXPECT_EQ(publications, 1);
    expectSelection(a, {0, 1, 2, 3, 4, 5});
    ASSERT_EQ(model->rowCount(), 6);
    for (int row = 0; row < 6; ++row)
        EXPECT_EQ(model->sourceRow(row), static_cast<unsigned>(row));
    expectChartRows(target, {0, 1, 2, 3, 4, 5});
    auto* proxy = qobject_cast<QSortFilterProxyModel*>(table->model());
    ASSERT_NE(proxy, nullptr);
    table->selectionModel()->select(
            proxy->mapFromSource(model->index(2, 0)),
            QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows);
    pump();
    EXPECT_EQ(publications, 2);
    expectSelection(a, {2});
    ASSERT_EQ(model->rowCount(), 1);
    EXPECT_EQ(model->sourceRow(0), 2u);
    expectChartRows(source, {0});
    expectChartRows(target, {2});
    EXPECT_EQ(publications, 2);
    source->clearSelection();
    pump();
    EXPECT_EQ(publications, 3);
    EXPECT_TRUE(manager->currentSelection().isEmpty());
    EXPECT_EQ(model->rowCount(), 0);
    expectChartRows(target, {});
}

TEST(ViewLinkage, NativeCellChartAndFactorySpreadsheetKeepCellAssociation) {
    auto* a = scene().a;
    auto* window = MainWindow::TheInstance();
    auto* manager = cvViewSelectionManager::instance();
    manager->clearSelection();
    auto* vertices = new ccPointCloud("Cell linkage vertices");
    ASSERT_TRUE(vertices->reserve(4));
    for (const auto& point : {CCVector3(0, 0, 0), CCVector3(3, 0, 0),
                              CCVector3(0, 3, 0), CCVector3(3, 3, 0)})
        vertices->addPoint(point);
    auto* mesh = new ccMesh(vertices);
    mesh->setName("Cell linkage mesh");
    mesh->addChild(vertices);
    vertices->setVisible(false);
    ASSERT_TRUE(mesh->reserve(2));
    mesh->addTriangle(0, 1, 2);
    mesh->addTriangle(1, 3, 2);
    mesh->setVisible(true);
    window->addToDB(mesh, true, true, false, true);
    struct RestoreSource {
        MainWindow* window;
        ccPointCloud* a;
        ccMesh* mesh;
        ~RestoreSource() {
            cvViewSelectionManager::instance()->clearSelection();
            window->setSelectedInDB(mesh, false);
            window->setSelectedInDB(a, true);
            mesh->setVisible(false);
        }
    } restore{window, a, mesh};
    Cell sourceCell("Point Chart View");
    ASSERT_TRUE(sourceCell.created);
    auto* source = sourceCell.multi->findChild<vtkChartView*>();
    ASSERT_NE(source, nullptr);
    source->setEntity(mesh);
    QComboBox* association = nullptr;
    for (auto* combo : source->findChildren<QComboBox*>())
        if (combo->findText("Cell Data") >= 0) association = combo;
    ASSERT_NE(association, nullptr);
    association->setCurrentIndex(association->findText("Cell Data"));
    ASSERT_TRUE(source->usesCellData());
    fields(source, {"Centroid_X"});
    Cell receiver("Point Chart View");
    ASSERT_TRUE(receiver.created);
    auto* target = receiver.multi->findChild<vtkChartView*>();
    ASSERT_NE(target, nullptr);
    target->setEntity(mesh);
    ASSERT_FALSE(target->usesCellData());
    fields(target, {"Points_X"});
    Cell tableCell("SpreadSheet View");
    ASSERT_TRUE(tableCell.created);
    auto* sheet = tableCell.multi->findChild<ecvSpreadSheetView*>();
    ASSERT_NE(sheet, nullptr);
    sheet->setEntity(mesh);
    auto* model = sheet->getModel();
    model->setAttributeType(ecvSpreadSheetModel::CELL_DATA);
    model->setSelectionOnly(true);
    auto* table = sheet->findChild<QTableView*>();
    ASSERT_NE(table, nullptr);
    QObject observer;
    int publications = 0;
    QObject::connect(manager,
                     qOverload<const cvSelectionData&>(
                             &cvViewSelectionManager::selectionChanged),
                     &observer,
                     [&](const cvSelectionData&) { ++publications; });
    nativeRows(source, {0, 1});
    EXPECT_EQ(publications, 1);
    auto selected = manager->currentSelection();
    EXPECT_EQ(selected.fieldAssociation(), cvSelectionData::CELLS);
    ASSERT_EQ(selected.actorInfos().size(), 1);
    EXPECT_EQ(selected.actorInfos().front().sourceId, mesh->getViewId());
    EXPECT_EQ(selected.actorInfos().front().sourceIds, QVector<qint64>({0, 1}));
    ASSERT_EQ(model->rowCount(), 2);
    EXPECT_EQ(model->sourceRow(0), 0u);
    EXPECT_EQ(model->sourceRow(1), 1u);
    expectChartRows(target, {});
    auto* proxy = qobject_cast<QSortFilterProxyModel*>(table->model());
    ASSERT_NE(proxy, nullptr);
    table->selectionModel()->select(
            proxy->mapFromSource(model->index(1, 0)),
            QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows);
    pump();
    EXPECT_EQ(publications, 2);
    selected = manager->currentSelection();
    EXPECT_EQ(selected.fieldAssociation(), cvSelectionData::CELLS);
    ASSERT_EQ(selected.actorInfos().size(), 1);
    EXPECT_EQ(selected.actorInfos().front().sourceId, mesh->getViewId());
    EXPECT_EQ(selected.actorInfos().front().sourceIds, QVector<qint64>({1}));
    ASSERT_EQ(model->rowCount(), 1);
    EXPECT_EQ(model->sourceRow(0), 1u);
    expectChartRows(source, {1});
    expectChartRows(target, {});
    EXPECT_EQ(publications, 2);
    source->clearSelection();
    pump();
    EXPECT_EQ(publications, 3);
    EXPECT_TRUE(manager->currentSelection().isEmpty());
    EXPECT_EQ(model->rowCount(), 0);
}

int main(int argc, char** argv) {
    QTemporaryDir settings;
    if (!settings.isValid()) return 2;
    // Isolate startup preferences before QApplication exists.
    for (const auto& key : {"CONFIG", "CACHE", "DATA"}) {
        const auto path = settings.filePath(QLatin1String(key));
        QDir().mkpath(path);
        const QByteArray variable = QByteArray("XDG_") + key + "_HOME";
        qputenv(variable.constData(), path.toLocal8Bit());
    }
    ecvApplication::InitOpenGL();
    ecvApplication app(argc, argv, false);
    app.setQuitOnLastWindowClosed(false);
    QLocale::setDefault(QLocale::c());
    QCoreApplication::setOrganizationName("ViewLinkageTest");
    QCoreApplication::setApplicationName("Isolated");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       settings.path());
    // Production InitEnvironment initializes this before MainWindow/recent
    // files.
    ecvSettingManager::Init(settings.filePath("application.ini"));
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MainWindow::DestroyInstance();
    return result;
}
