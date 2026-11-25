// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Regression gate for ParaView-style multi-view layout management:
// creating 3-4 render views via splits and closing them in ANY order must
// keep the layout proxy, the frame tree and the close buttons consistent.
// Links the production layout widgets; MainWindow is stubbed to the few
// symbols ecvMultiViewWidget.cpp references.

#include <QtCompat.h>
#include <Tools/SelectionTools/cvSelectionData.h>
#include <Tools/SelectionTools/cvViewSelectionManager.h>
#include <VTKExtensions/Views/vtkChartView.h>
#include <VTKExtensions/Views/vtkComparativeViewWidget.h>
#include <VTKExtensions/Views/vtkOrthoSliceRepresentation.h>
#include <VTKExtensions/Views/vtkOrthoSliceViewWidget.h>
#include <VTKExtensions/Widgets/QVTKWidgetCustom.h>
#include <Visualization/VtkDisplayTools.h>
#include <Visualization/VtkVis.h>
#include <Visualization/vtkGLView.h>
#include <ecvImage.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvRepresentationManager.h>
#include <ecvViewLayoutProxy.h>
#include <ecvViewManager.h>
#include <gtest/gtest.h>
#include <vtkAbstractArray.h>
#include <vtkAbstractContextItem.h>
#include <vtkActorCollection.h>
#include <vtkAnnotationLink.h>
#include <vtkAxis.h>
#include <vtkCellData.h>
#include <vtkChart.h>
#include <vtkCommand.h>
#include <vtkContextActor.h>
#include <vtkContextScene.h>
#include <vtkCubeSource.h>
#include <vtkDataSetMapper.h>
#include <vtkDoubleArray.h>
#include <vtkFieldData.h>
#include <vtkIdTypeArray.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkLookupTable.h>
#include <vtkMapper.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPen.h>
#include <vtkPlane.h>
#include <vtkPlaneCollection.h>
#include <vtkPlot.h>
#include <vtkPlotHistogram2D.h>
#include <vtkPlotPoints.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPropCollection.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkScalarsToColors.h>
#include <vtkScatterPlotMatrix.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkStringArray.h>
#include <vtkTable.h>

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QCoreApplication>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QFrame>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QListWidget>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QSortFilterProxyModel>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <limits>
#include <set>

#include "MainWindow.h"
#include "ecvMultiViewWidget.h"
#include "ecvPythonCodeEditor.h"
#include "ecvPythonView.h"
#include "ecvSpreadSheetView.h"
#include "ecvTabbedMultiViewWidget.h"

// --- Minimal MainWindow stubs (only referenced symbols are linked) ---------
MainWindow* MainWindow::TheInstance() { return nullptr; }
void MainWindow::copyPrimaryViewConfig(vtkGLView*, vtkGLView*) {}
void MainWindow::rebindToolsToActiveView(ecvGenericGLDisplay*) {}
ccHObject* MainWindow::dbRootObject() { return nullptr; }

namespace {

void pump() {
    QApplication::processEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    QApplication::processEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    QApplication::processEvents();
}

QWidget* frameOf(ecvGenericGLDisplay* view) {
    if (!view || !view->asWidget()) return nullptr;
    return view->asWidget()->parentWidget();
}

QToolButton* closeBtnOf(QWidget* frame) {
    return frame ? frame->findChild<QToolButton*>("closeViewBtn") : nullptr;
}

QList<QWidget*> emptyCellFrames(ecvMultiViewWidget* mvw) {
    QList<QWidget*> result;
    if (!mvw) return result;
    for (auto* f : mvw->findChildren<QFrame*>()) {
        if (f->property("IS_EMPTY_CELL").toBool()) result.append(f);
    }
    return result;
}

struct LayoutFixture {
    QMainWindow window;
    ecvTabbedMultiViewWidget* tabbed = nullptr;
    ecvMultiViewWidget* mvw = nullptr;
    QTemporaryDir settings;

    LayoutFixture() {
        QCoreApplication::setOrganizationName("LayoutCloseTest");
        QCoreApplication::setApplicationName("Isolated");
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                           settings.path());
        ecvViewManager::instance().initDisplayTools(
                new Visualization::VtkDisplayTools(), &window, false);
        tabbed = new ecvTabbedMultiViewWidget(&window);
        tabbed->setViewFactory([this]() -> vtkGLView* {
            return vtkGLView::Create(&window, false, false);
        });
        tabbed->setFrameFactory([this](QWidget* viewWidget,
                                       const QString& /*title*/) -> QWidget* {
            auto* frame = new QFrame();
            auto* lay = new QVBoxLayout(frame);
            lay->setContentsMargins(0, 0, 0, 0);
            auto* bar = new QWidget(frame);
            auto* bl = new QHBoxLayout(bar);
            bl->setContentsMargins(0, 0, 0, 0);
            auto* btn = new QToolButton(bar);
            btn->setObjectName("closeViewBtn");
            bl->addWidget(btn);
            lay->addWidget(bar);
            lay->addWidget(viewWidget);
            QPointer<QWidget> vw(viewWidget);
            QObject::connect(
                    btn, &QToolButton::clicked, frame, [this, frame, vw]() {
                        auto* mvw = tabbed->currentMultiView();
                        if (!mvw) return;
                        if (ecvViewManager::instance().viewCount() <= 1) return;
                        // Mirror MainWindow::prepareViewClose.
                        if (auto* d = ecvGenericGLDisplay::FromWidget(
                                    vw.data())) {
                            ecvViewManager::instance().unregisterView(d);
                        }
                        mvw->onCloseView(frame);
                    });
            return frame;
        });
        // Mirror MainWindow::onViewClosingFromLayout signal flow.
        QObject::connect(
                tabbed, &ecvTabbedMultiViewWidget::viewClosing,
                [](ecvGenericGLDisplay* d) {
                    if (!d) return;
                    ecvViewManager::instance().unregisterView(d);
                    if (d == ecvViewManager::instance().getActiveView()) {
                        for (auto* v :
                             ecvViewManager::instance().getAllViews()) {
                            if (v != d) {
                                ecvViewManager::instance().setActiveView(v);
                                break;
                            }
                        }
                    }
                });
        window.setCentralWidget(tabbed);
        window.resize(1200, 800);
        window.show();
        pump();
    }

    ~LayoutFixture() { ecvViewManager::instance().releaseDisplayTools(); }

    vtkGLView* newView() {
        auto* v = vtkGLView::Create(&window, false, false);
        return v;
    }

    void adoptFirst(vtkGLView* v) {
        int tab = tabbed->createTabWithView(v);
        mvw = qobject_cast<ecvMultiViewWidget*>(
                tabbed->tabWidget()->widget(tab));
        if (mvw && mvw->layoutManager()) {
            ecvViewManager::instance().registerLayout(mvw->layoutManager());
        }
        ecvViewManager::instance().setActiveView(v);
        pump();
    }

    void fillCell(int location, vtkGLView* v) {
        mvw->layoutManager()->assignView(location, v);
        pump();
    }

    void split(QWidget* frame, bool horizontal) {
        if (horizontal)
            mvw->onSplitHorizontal(frame);
        else
            mvw->onSplitVertical(frame);
        pump();
    }

    void clickClose(QWidget* frame) {
        auto* btn = closeBtnOf(frame);
        ASSERT_TRUE(btn != nullptr) << "frame has a close button";
        ASSERT_TRUE(btn->isEnabled()) << "close button must be enabled";
        btn->click();
        pump();
    }
};

// Build 4 views: v1@1, v2@5, v3@13, v4@14 (split chain, fill empty cells).
void buildFourViews(LayoutFixture& f,
                    vtkGLView*& v1,
                    vtkGLView*& v2,
                    vtkGLView*& v3,
                    vtkGLView*& v4) {
    v1 = f.newView();
    f.adoptFirst(v1);
    f.split(frameOf(v1), true);  // 1:v1, 2:empty
    v2 = f.newView();
    f.fillCell(2, v2);
    f.split(frameOf(v2), false);  // 5:v2, 6:empty
    v3 = f.newView();
    f.fillCell(6, v3);
    f.split(frameOf(v3), false);  // 13:v3, 14:empty
    v4 = f.newView();
    f.fillCell(14, v4);
    ASSERT_EQ(f.mvw->layoutManager()->viewCount(), 4);
}

}  // namespace

// Close views in middle-first order; the layout must stay consistent and
// every remaining close button must keep working.
TEST(LayoutClose, MiddleOrderCloseKeepsButtonsAlive) {
    LayoutFixture f;
    vtkGLView *v1, *v2, *v3, *v4;
    buildFourViews(f, v1, v2, v3, v4);
    auto* proxy = f.mvw->layoutManager();

    f.clickClose(frameOf(v2));  // not the last created
    EXPECT_EQ(proxy->viewCount(), 3);
    EXPECT_EQ(proxy->getViewLocation(v2), -1);
    EXPECT_EQ(proxy->getViewLocation(v3), 5);
    EXPECT_EQ(proxy->getViewLocation(v4), 6);

    f.clickClose(frameOf(v3));
    EXPECT_EQ(proxy->viewCount(), 2);
    EXPECT_EQ(proxy->getViewLocation(v4), 2);

    f.clickClose(frameOf(v1));
    EXPECT_EQ(proxy->viewCount(), 1);
    EXPECT_EQ(proxy->getViewLocation(v4), 0);
}

// Reverse creation order close.
TEST(LayoutClose, ReverseOrderClose) {
    LayoutFixture f;
    vtkGLView *v1, *v2, *v3, *v4;
    buildFourViews(f, v1, v2, v3, v4);
    auto* proxy = f.mvw->layoutManager();

    f.clickClose(frameOf(v4));
    EXPECT_EQ(proxy->viewCount(), 3);
    f.clickClose(frameOf(v3));
    EXPECT_EQ(proxy->viewCount(), 2);
    f.clickClose(frameOf(v2));
    EXPECT_EQ(proxy->viewCount(), 1);
    EXPECT_EQ(proxy->getViewLocation(v1), 0);
}

// After closing a view whose sibling is an empty cell, the promoted empty
// cell's close button must still collapse it (the reported dead-X scenario).
TEST(LayoutClose, PromotedEmptyCellCloseButtonWorks) {
    LayoutFixture f;
    vtkGLView *v1, *v2, *v3, *v4;
    buildFourViews(f, v1, v2, v3, v4);
    auto* proxy = f.mvw->layoutManager();

    f.clickClose(frameOf(v2));
    f.clickClose(frameOf(v3));
    // Now [v1@1, v4@2]. Split v4's cell and close v4: the empty sibling is
    // promoted, leaving [v1 | empty] like the reported screenshot.
    f.split(frameOf(v4), true);  // 5:v4, 6:empty
    ASSERT_EQ(proxy->viewCount(), 2);
    f.clickClose(frameOf(v4));
    ASSERT_EQ(proxy->viewCount(), 1);

    auto empties = emptyCellFrames(f.mvw);
    ASSERT_EQ(empties.size(), 1) << "one promoted empty cell remains";
    QWidget* emptyFrame = empties.first();
    int loc = emptyFrame->property("CELL_INDEX").toInt();
    EXPECT_NE(loc, 0) << "empty cell is not the root";
    auto* btn = closeBtnOf(emptyFrame);
    ASSERT_TRUE(btn != nullptr);
    EXPECT_TRUE(btn->isEnabled())
            << "non-root empty cell close must be enabled";
    btn->click();
    pump();
    EXPECT_EQ(proxy->viewCount(), 1);
    EXPECT_FALSE(proxy->isSplitCell(0));
    EXPECT_EQ(proxy->getView(0), v1) << "empty cell collapsed into root view";
    EXPECT_TRUE(emptyCellFrames(f.mvw).isEmpty());
}

TEST(LayoutRegistry, RemovesDestroyedLayout) {
    auto& manager = ecvViewManager::instance();
    const auto before = manager.allLayouts();
    auto* layout = new ecvViewLayoutProxy;
    manager.registerLayout(layout);
    ASSERT_TRUE(manager.allLayouts().contains(layout));
    delete layout;
    EXPECT_EQ(manager.allLayouts(), before);
}

TEST(SliceView, PreservesOtherPlanesAndCutsBothTransformedSources) {
    ccPointCloud sourceA("a"), sourceB("b");
    sourceA.setVisible(true);
    sourceB.setVisible(true);
    LayoutFixture fixture;
    auto* view = fixture.newView();
    fixture.adoptFirst(view);
    auto* vis = view->getVisualizer3D();
    ASSERT_NE(vis, nullptr);
    vtkNew<vtkCubeSource> cube;
    cube->Update();
    vtkSmartPointer<vtkPolyData> geometry = cube->GetOutput();
    ASSERT_TRUE(
            vis->addPointCloud(geometry, sourceA.getViewId().toStdString()));
    ASSERT_TRUE(
            vis->addPointCloud(geometry, sourceB.getViewId().toStdString()));
    auto* actorA = vis->getActorById(sourceA.getViewId().toStdString());
    auto* actorB = vis->getActorById(sourceB.getViewId().toStdString());
    ASSERT_NE(actorA, nullptr);
    ASSERT_NE(actorB, nullptr);
    actorB->SetPosition(5, 0, 0);
    ecvRepresentationManager::instance().ensureRepresentation(&sourceA, view);
    ecvRepresentationManager::instance().ensureRepresentation(&sourceB, view);
    vtkNew<vtkPlane> externalPlane;
    externalPlane->SetNormal(0, 1, 0);
    actorA->GetMapper()->AddClippingPlane(externalPlane);
    actorB->GetMapper()->AddClippingPlane(externalPlane);
    view->enableSliceMode(true);
    EXPECT_EQ(actorA->GetMapper()->GetNumberOfClippingPlanes(), 2);
    EXPECT_EQ(actorB->GetMapper()->GetNumberOfClippingPlanes(), 2);
    auto* renderer = vis->getCurrentRenderer();
    std::set<vtkActor*> previous;
    auto* actors = renderer->GetActors();
    actors->InitTraversal();
    while (auto* actor = actors->GetNextActor()) previous.insert(actor);
    ASSERT_TRUE(sourceA.isVisible());
    ASSERT_TRUE(sourceB.isVisible());
    ASSERT_TRUE(actorA->GetVisibility());
    ASSERT_TRUE(actorB->GetVisibility());
    view->setMultiSlicePositions(2, {0.0});
    int cuts = 0;
    bool translatedCut = false;
    actors = renderer->GetActors();
    actors->InitTraversal();
    while (auto* actor = actors->GetNextActor()) {
        if (previous.count(actor)) continue;
        ++cuts;
        if (actor->GetBounds()[0] > 4.0) translatedCut = true;
    }
    EXPECT_EQ(cuts, 2);
    EXPECT_TRUE(translatedCut);
    sourceB.setVisible(false);
    actorA->SetPosition(3, 0, 0);
    view->updateScene();
    actors = renderer->GetActors();
    actors->InitTraversal();
    cuts = 0;
    while (auto* actor = actors->GetNextActor()) {
        if (previous.count(actor)) continue;
        ++cuts;
        EXPECT_GT(actor->GetBounds()[0], 2.0);
    }
    EXPECT_EQ(cuts, 1);
    view->enableSliceMode(false);
    EXPECT_EQ(actorA->GetMapper()->GetNumberOfClippingPlanes(), 1);
    EXPECT_TRUE(actorA->GetMapper()->GetClippingPlanes()->IsItemPresent(
            externalPlane));
    EXPECT_EQ(actorB->GetMapper()->GetNumberOfClippingPlanes(), 1);
}

vtkAbstractContextItem* renderedChartItem(vtkChartView& view) {
    auto* widget = view.findChild<QVTKOpenGLNativeWidget*>();
    if (!widget) return nullptr;
    auto* renderers = widget->renderWindow()->GetRenderers();
    renderers->InitTraversal();
    while (auto* renderer = renderers->GetNextItem()) {
        auto* props = renderer->GetViewProps();
        props->InitTraversal();
        while (auto* prop = props->GetNextProp()) {
            if (auto* actor = vtkContextActor::SafeDownCast(prop))
                return actor->GetScene()->GetItem(0);
        }
    }
    return nullptr;
}

vtkChart* renderedChart(vtkChartView& view) {
    return vtkChart::SafeDownCast(renderedChartItem(view));
}

TEST(ChartView, MatrixAndParallelNativeSelectionsKeepSourceIdsAndAssociation) {
    ccPointCloud vertices("vertices"), unrelated("unrelated");
    ASSERT_TRUE(vertices.reserve(7));
    for (int i = 0; i < 7; ++i) vertices.addPoint(CCVector3(i, i % 2, i % 3));
    ccMesh mesh(&vertices);
    ASSERT_TRUE(mesh.reserve(7));
    for (int i = 0; i < 7; ++i) mesh.addTriangle(0, 1, 2);
    auto* manager = cvViewSelectionManager::instance();
    for (auto type :
         {vtkChartView::PLOT_MATRIX, vtkChartView::PARALLEL_COORDINATES}) {
        for (bool cells : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << "type=" << type << " cells=" << cells);
            manager->setSourceObject(nullptr);
            manager->setCurrentSelection(cvSelectionData());
            ccHObject* source = cells ? static_cast<ccHObject*>(&mesh)
                                      : static_cast<ccHObject*>(&vertices);
            const auto association =
                    cells ? cvSelectionData::CELLS : cvSelectionData::POINTS;
            vtkChartView view(type);
            view.setMaxChartPoints(3);  // source IDs 0, 3, 6 (stride 3)
            view.setEntity(source);
            view.setEntity(source);  // rebinding must not add observers
            if (cells) {
                QComboBox* attributes = nullptr;
                for (auto* combo : view.findChildren<QComboBox*>())
                    if (combo->findText("Cell Data") >= 0) attributes = combo;
                ASSERT_NE(attributes, nullptr);
                attributes->setCurrentIndex(attributes->findText("Cell Data"));
            }
            EXPECT_EQ(view.usesCellData(), cells);
            view.resize(900, 700);
            view.show();
            pump();
            auto* matrix =
                    vtkScatterPlotMatrix::SafeDownCast(renderedChartItem(view));
            EXPECT_EQ(matrix != nullptr, type == vtkChartView::PLOT_MATRIX);
            auto* chart = matrix ? matrix->GetMainChart() : renderedChart(view);
            ASSERT_NE(chart, nullptr);
            ASSERT_GT(chart->GetNumberOfPlots(), 0);
            auto* table = chart->GetPlot(0)->GetInput();
            ASSERT_NE(table, nullptr);
            ASSERT_EQ(table->GetNumberOfRows(), 3);
            const char* idField = cells ? "CellIndex" : "Points_X";
            auto* values = table->GetColumnByName(idField);
            ASSERT_NE(values, nullptr);
            for (int row = 0; row < 3; ++row)
                EXPECT_DOUBLE_EQ(values->GetVariantValue(row).ToDouble(),
                                 row * 3.0);
            auto* link = matrix ? matrix->GetAnnotationLink()
                                : chart->GetAnnotationLink();
            ASSERT_NE(link, nullptr);
            EXPECT_EQ(chart->GetAnnotationLink(), link);
            auto* widget = view.findChild<QVTKOpenGLNativeWidget*>();
            ASSERT_NE(widget, nullptr);
            auto expectRows = [chart, widget](std::set<vtkIdType> expected) {
                // Parallel coordinates consumes its link during Paint, not
                // Update.
                widget->renderWindow()->Render();
                chart->Update();
                std::set<vtkIdType> actual;
                if (auto* ids = chart->GetPlot(0)->GetSelection())
                    for (vtkIdType i = 0; i < ids->GetNumberOfValues(); ++i)
                        actual.insert(ids->GetValue(i));
                EXPECT_EQ(actual, expected);
            };
            QObject observer;
            int publications = 0, highlights = 0;
            QObject::connect(manager,
                             qOverload<const cvSelectionData&>(
                                     &cvViewSelectionManager::selectionChanged),
                             &observer,
                             [&](const cvSelectionData&) { ++publications; });
            QObject::connect(&view, &vtkChartView::pointsHighlighted, &observer,
                             [&](ccHObject* entity, const QVector<unsigned>&) {
                                 EXPECT_EQ(entity, source);
                                 ++highlights;
                             });
            vtkNew<vtkIdTypeArray> rows;
            for (vtkIdType row : {-1, 1, 2, 3}) rows->InsertNextValue(row);
            vtkNew<vtkSelectionNode> node;
            node->SetContentType(vtkSelectionNode::INDICES);
            node->SetFieldType(type == vtkChartView::PARALLEL_COORDINATES
                                       ? vtkSelectionNode::POINT
                                       : vtkSelectionNode::ROW);
            node->SetSelectionList(rows);
            vtkNew<vtkSelection> gesture;
            gesture->AddNode(node);
            // SetCurrentSelection itself emits the native annotation event.
            link->SetCurrentSelection(gesture);
            EXPECT_EQ(publications, 1);
            EXPECT_EQ(highlights, 1);
            auto selected = manager->currentSelection();
            EXPECT_EQ(selected.fieldAssociation(), association);
            ASSERT_EQ(selected.actorInfos().size(), 1);
            EXPECT_EQ(selected.actorInfos().front().sourceId,
                      source->getViewId());
            EXPECT_EQ(selected.actorInfos().front().sourceIds,
                      QVector<qint64>({3, 6}));
            expectRows({1, 2});
            EXPECT_EQ(publications, 1);

            auto incoming = [&](cvSelectionData::FieldAssociation field,
                                const QString& sourceId,
                                const QVector<qint64>& ids,
                                std::set<vtkIdType> expected) {
                cvSelectionData selection(ids, field);
                cvActorSelectionInfo info;
                info.sourceId = sourceId;
                info.sourceIds = ids;
                selection.addActorInfo(info);
                publications = highlights = 0;
                manager->setCurrentSelection(selection);
                expectRows(expected);
                EXPECT_EQ(publications, 1) << "no incoming-selection feedback";
                EXPECT_EQ(highlights, 0);
            };
            incoming(association, source->getViewId(), {-1, 0, 1, 6, 7},
                     {0, 2});
            incoming(cells ? cvSelectionData::POINTS : cvSelectionData::CELLS,
                     source->getViewId(), {3}, {});
            incoming(association, source->getViewId(), {3}, {1});
            incoming(association, unrelated.getViewId(), {3}, {});
            incoming(association, source->getViewId(), {}, {});
            incoming(association, source->getViewId(), {6}, {2});
            vtkNew<vtkSelection> empty;
            publications = highlights = 0;
            link->SetCurrentSelection(empty);
            EXPECT_EQ(publications, 1);
            EXPECT_EQ(highlights, 1);
            selected = manager->currentSelection();
            EXPECT_TRUE(selected.isEmpty());
            EXPECT_EQ(selected.fieldAssociation(), association);
            ASSERT_EQ(selected.actorInfos().size(), 1);
            EXPECT_EQ(selected.actorInfos().front().sourceId,
                      source->getViewId());
            EXPECT_TRUE(selected.actorInfos().front().sourceIds.isEmpty());
            expectRows({});

            incoming(association, source->getViewId(), {3}, {1});
            view.setSelectionModifier(vtkContextScene::SELECTION_ADDITION);
            EXPECT_EQ(matrix ? matrix->GetSelectionMode()
                             : chart->GetSelectionMode(),
                      vtkContextScene::SELECTION_ADDITION);
            publications = highlights = 0;
            view.clearSelection();
            expectRows({});
            EXPECT_EQ(publications, 1) << "public clear has one publication";
            EXPECT_EQ(highlights, 1);
            EXPECT_TRUE(manager->currentSelection().isEmpty());
            view.setEntity(nullptr);
            if (matrix) {
                EXPECT_EQ(matrix->GetSize()[0], 0);
                EXPECT_EQ(matrix->GetSize()[1], 0);
                EXPECT_FALSE(matrix->GetVisible());
            }
            manager->setSourceObject(nullptr);
        }
    }
}

void selectChartFields(vtkChartView& view, const QStringList& names) {
    auto* fields = view.findChild<QListWidget*>();
    ASSERT_NE(fields, nullptr);
    {
        QSignalBlocker blocker(fields);
        fields->clearSelection();
        for (const auto& name : names) {
            const auto items = fields->findItems(name, Qt::MatchExactly);
            ASSERT_EQ(items.size(), 1) << name.toStdString();
            items.front()->setSelected(true);
        }
    }
    ASSERT_TRUE(QMetaObject::invokeMethod(&view, "onSelectionChanged",
                                          Qt::DirectConnection));
}

TEST(ChartView, NamedMeshCellFieldsUseNormalsAndBothMaterialStoragePaths) {
    ccPointCloud vertices("vertices");
    ASSERT_TRUE(vertices.reserve(4));
    vertices.addPoint(CCVector3(0, 0, 0));
    vertices.addPoint(CCVector3(3, 0, 0));
    vertices.addPoint(CCVector3(0, 2, 0));
    vertices.addPoint(CCVector3(0, 0, 4));
    ccMesh mesh(&vertices);
    ASSERT_TRUE(mesh.reserve(2));
    mesh.addTriangle(0, 1, 2);
    mesh.addTriangle(0, 2, 3);
    ASSERT_TRUE(mesh.addTriangleNorm(CCVector3(1, 0, 0)));
    ASSERT_TRUE(mesh.addTriangleNorm(CCVector3(0, -1, 0)));
    ASSERT_TRUE(mesh.hasTriNormals());
    ASSERT_TRUE(mesh.reservePerTriangleMtlIndexes());
    mesh.addTriangleMtlIndex(7);
    mesh.addTriangleMtlIndex(-1);
    const QMap<QString, QVector<double>> expected{
            {"CellIndex", {0, 1}},
            {"Centroid_X", {1, 0}},
            {"Centroid_Y", {2.0 / 3.0, 2.0 / 3.0}},
            {"Centroid_Z", {0, 4.0 / 3.0}},
            {"Area", {3, 4}},
            {"TriNormal_X", {1, 0}},
            {"TriNormal_Y", {0, -1}},
            {"TriNormal_Z", {0, 0}},
            {"TriNormal_Mag", {1, 1}},
            {"MaterialId", {7, -1}}};
    for (bool eigenStorage : {false, true}) {
        if (eigenStorage) {
            mesh.removePerTriangleMtlIndexes();
            mesh.triangle_material_ids_ = {7, -1};
        }
        for (auto type : {vtkChartView::POINT_CHART, vtkChartView::PLOT_MATRIX,
                          vtkChartView::PARALLEL_COORDINATES}) {
            SCOPED_TRACE(::testing::Message()
                         << "type=" << type << " eigen=" << eigenStorage);
            vtkChartView view(type);
            view.setEntity(&mesh);
            QComboBox* attributes = nullptr;
            for (auto* combo : view.findChildren<QComboBox*>())
                if (combo->findText("Cell Data") >= 0) attributes = combo;
            ASSERT_NE(attributes, nullptr);
            attributes->setCurrentIndex(attributes->findText("Cell Data"));
            selectChartFields(view, expected.keys());
            auto* matrix =
                    vtkScatterPlotMatrix::SafeDownCast(renderedChartItem(view));
            auto* chart = matrix ? matrix->GetMainChart() : renderedChart(view);
            ASSERT_NE(chart, nullptr);
            ASSERT_GT(chart->GetNumberOfPlots(), 0);
            auto* table = chart->GetPlot(0)->GetInput();
            ASSERT_NE(table, nullptr);
            ASSERT_EQ(table->GetNumberOfRows(), 2);
            for (auto it = expected.cbegin(); it != expected.cend(); ++it) {
                auto* column =
                        table->GetColumnByName(it.key().toUtf8().constData());
                ASSERT_NE(column, nullptr) << it.key().toStdString();
                for (int row = 0; row < 2; ++row)
                    EXPECT_NEAR(column->GetVariantValue(row).ToDouble(),
                                it.value()[row],
                                it.key().startsWith("TriNormal") ? 0.005 : 1e-6)
                            << it.key().toStdString() << " row=" << row;
            }
            // Changing the named combination must remove the old arrays/axes.
            selectChartFields(view,
                              {"Centroid_Y", "TriNormal_Z", "MaterialId"});
            chart = matrix ? matrix->GetMainChart() : renderedChart(view);
            ASSERT_NE(chart, nullptr);
            table = chart->GetPlot(0)->GetInput();
            ASSERT_NE(table, nullptr);
            EXPECT_EQ(table->GetNumberOfColumns(),
                      type == vtkChartView::POINT_CHART ? 4 : 3);
            EXPECT_EQ(table->GetColumnByName("TriNormal_X"), nullptr);
            ASSERT_NE(table->GetColumnByName("MaterialId"), nullptr);
            EXPECT_DOUBLE_EQ(table->GetColumnByName("MaterialId")
                                     ->GetVariantValue(1)
                                     .ToDouble(),
                             -1.0);
        }
    }
}

TEST(ChartView,
     SamplingCapAndSelectionUseOriginalIdsWithoutDuplicateObservers) {
    ccPointCloud source("source");
    ASSERT_TRUE(source.reserve(10001));
    for (int i = 0; i < 10001; ++i) source.addPoint(CCVector3(i, 0, 0));
    vtkChartView view(vtkChartView::POINT_CHART);
    view.setEntity(&source);
    view.setEntity(&source);
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    ASSERT_GT(chart->GetNumberOfPlots(), 0);
    EXPECT_LE(chart->GetPlot(0)->GetInput()->GetNumberOfRows(),
              view.maxChartPoints());
    auto* bottom = chart->GetAxis(vtkAxis::BOTTOM);
    auto* left = chart->GetAxis(vtkAxis::LEFT);
    bottom->SetTitle("custom X");
    bottom->SetPrecision(7);
    bottom->SetNotation(1);
    bottom->SetLogScale(true);
    bottom->SetBehavior(vtkAxis::FIXED);
    bottom->SetUnscaledRange(1, 100);
    left->SetPrecision(3);
    left->SetLogScale(false);
    view.setMaxChartPoints(view.maxChartPoints());
    EXPECT_EQ(bottom->GetTitle(), "custom X");
    EXPECT_EQ(bottom->GetPrecision(), 7);
    EXPECT_EQ(bottom->GetNotation(), 1);
    EXPECT_TRUE(bottom->GetLogScale());
    EXPECT_EQ(bottom->GetBehavior(), vtkAxis::FIXED);
    EXPECT_DOUBLE_EQ(bottom->GetUnscaledMinimum(), 1);
    EXPECT_DOUBLE_EQ(bottom->GetUnscaledMaximum(), 100);
    EXPECT_EQ(left->GetPrecision(), 3);
    EXPECT_FALSE(left->GetLogScale());
    QVector<unsigned> selected;
    int callbacks = 0;
    QObject::connect(
            &view, &vtkChartView::pointsHighlighted,
            [&selected, &callbacks](ccHObject*, const QVector<unsigned>& ids) {
                selected = ids;
                ++callbacks;
            });
    vtkNew<vtkIdTypeArray> rows;
    rows->InsertNextValue(1);
    vtkNew<vtkSelectionNode> node;
    node->SetContentType(vtkSelectionNode::INDICES);
    node->SetSelectionList(rows);
    vtkNew<vtkSelection> selection;
    selection->AddNode(node);
    ASSERT_NE(chart->GetAnnotationLink(), nullptr);
    chart->GetAnnotationLink()->SetCurrentSelection(selection);
    callbacks = 0;
    chart->GetAnnotationLink()->InvokeEvent(vtkCommand::AnnotationChangedEvent);
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(selected, QVector<unsigned>({2}));
    cvSelectionData external({4}, cvSelectionData::POINTS);
    cvActorSelectionInfo info;
    info.sourceId = source.getViewId();
    info.sourceIds = {4};
    external.addActorInfo(info);
    auto* manager = cvViewSelectionManager::instance();
    const int gestures = callbacks;
    manager->setCurrentSelection(external);
    chart->Update();
    auto* highlighted = chart->GetPlot(0)->GetSelection();
    ASSERT_NE(highlighted, nullptr);
    ASSERT_EQ(highlighted->GetNumberOfValues(), 1);
    EXPECT_EQ(highlighted->GetValue(0), 2);
    EXPECT_EQ(callbacks, gestures)
            << "external selection must not republish as a gesture";
    manager->setCurrentSelection(cvSelectionData());
    chart->Update();
    highlighted = chart->GetPlot(0)->GetSelection();
    EXPECT_TRUE(!highlighted || highlighted->GetNumberOfValues() == 0);
}

TEST(ChartView, HistogramSelectionResolvesAllSourceRowsInTheBin) {
    ccPointCloud source("histogram-source");
    ASSERT_TRUE(source.reserve(4));
    for (float x : {0.f, 0.f, 9.f, 10.f}) source.addPoint(CCVector3(x, 0, 0));
    vtkChartView view(vtkChartView::HISTOGRAM);
    for (auto* spin : view.findChildren<QSpinBox*>())
        if (spin->minimum() == 2 && spin->maximum() == 1000) spin->setValue(2);
    view.setEntity(&source);
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    QVector<unsigned> selected;
    QObject::connect(&view, &vtkChartView::pointsHighlighted,
                     [&selected](ccHObject*, const QVector<unsigned>& ids) {
                         selected = ids;
                     });
    vtkNew<vtkIdTypeArray> rows;
    rows->InsertNextValue(0);
    vtkNew<vtkSelectionNode> node;
    node->SetContentType(vtkSelectionNode::INDICES);
    node->SetSelectionList(rows);
    vtkNew<vtkSelection> selection;
    selection->AddNode(node);
    ASSERT_NE(chart->GetAnnotationLink(), nullptr);
    chart->GetAnnotationLink()->SetCurrentSelection(selection);
    chart->GetAnnotationLink()->InvokeEvent(vtkCommand::AnnotationChangedEvent);
    EXPECT_EQ(selected, QVector<unsigned>({0, 1}));
}

TEST(ChartView, MeshCellSelectionAndCsvUseCellAssociationAndValues) {
    ccPointCloud vertices("vertices");
    ASSERT_TRUE(vertices.reserve(4));
    vertices.addPoint(CCVector3(0, 0, 0));
    vertices.addPoint(CCVector3(3, 0, 0));
    vertices.addPoint(CCVector3(0, 3, 0));
    vertices.addPoint(CCVector3(3, 3, 0));
    ccMesh mesh(&vertices);
    ASSERT_TRUE(mesh.reserve(2));
    mesh.addTriangle(0, 1, 2);
    mesh.addTriangle(1, 3, 2);
    vtkChartView view(vtkChartView::POINT_CHART);
    view.setEntity(&mesh);
    QComboBox* association = nullptr;
    for (auto* combo : view.findChildren<QComboBox*>())
        if (combo->findText("Cell Data") >= 0) association = combo;
    ASSERT_NE(association, nullptr);
    association->setCurrentIndex(association->findText("Cell Data"));
    ASSERT_TRUE(view.usesCellData());
    auto* fields = view.findChild<QListWidget*>();
    ASSERT_NE(fields, nullptr);
    fields->clearSelection();
    const auto centroid = fields->findItems("Centroid_X", Qt::MatchExactly);
    ASSERT_EQ(centroid.size(), 1);
    centroid.front()->setSelected(true);
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    ASSERT_EQ(chart->GetNumberOfPlots(), 1);
    auto* table = chart->GetPlot(0)->GetInput();
    ASSERT_NE(table, nullptr);
    ASSERT_EQ(table->GetNumberOfRows(), 2);
    auto* values = table->GetColumnByName("Centroid_X");
    ASSERT_NE(values, nullptr);
    EXPECT_DOUBLE_EQ(values->GetVariantValue(0).ToDouble(), 1.0);
    EXPECT_DOUBLE_EQ(values->GetVariantValue(1).ToDouble(), 2.0);

    vtkNew<vtkIdTypeArray> rows;
    rows->InsertNextValue(1);
    vtkNew<vtkSelectionNode> node;
    node->SetContentType(vtkSelectionNode::INDICES);
    node->SetFieldType(vtkSelectionNode::ROW);
    node->SetSelectionList(rows);
    vtkNew<vtkSelection> gesture;
    gesture->AddNode(node);
    auto* link = chart->GetAnnotationLink();
    ASSERT_NE(link, nullptr);
    link->SetCurrentSelection(gesture);
    auto* manager = cvViewSelectionManager::instance();
    const auto selected = manager->currentSelection();
    EXPECT_EQ(selected.fieldAssociation(), cvSelectionData::CELLS);
    ASSERT_EQ(selected.actorInfos().size(), 1);
    EXPECT_EQ(selected.actorInfos().front().sourceId, mesh.getViewId());
    EXPECT_EQ(selected.actorInfos().front().sourceIds, QVector<qint64>({1}));
    auto expectRows = [chart](int count) {
        chart->Update();
        auto* ids = chart->GetPlot(0)->GetSelection();
        ASSERT_NE(ids, nullptr);
        ASSERT_EQ(ids->GetNumberOfValues(), count);
        if (count) EXPECT_EQ(ids->GetValue(0), 1);
    };
    expectRows(1);
    cvSelectionData points({1}, cvSelectionData::POINTS);
    points.addActorInfo(selected.actorInfos().front());
    manager->setCurrentSelection(points);
    expectRows(0);
    cvSelectionData otherCells({1}, cvSelectionData::CELLS);
    cvActorSelectionInfo other = selected.actorInfos().front();
    other.sourceId = vertices.getViewId();
    otherCells.addActorInfo(other);
    manager->setCurrentSelection(otherCells);
    expectRows(0);
    manager->setCurrentSelection(selected);
    expectRows(1);
    vtkNew<vtkSelection> empty;
    link->SetCurrentSelection(empty);
    const auto cleared = manager->currentSelection();
    EXPECT_TRUE(cleared.isEmpty());
    EXPECT_EQ(cleared.fieldAssociation(), cvSelectionData::CELLS);
    ASSERT_EQ(cleared.actorInfos().size(), 1);
    EXPECT_EQ(cleared.actorInfos().front().sourceId, mesh.getViewId());
    expectRows(0);

    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    const QString csvPath = temp.filePath("cells.csv");
    const bool nativeDialogsDisabled =
            QApplication::testAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs, true);
    bool acceptedDialog = false;
    QTimer::singleShot(0, &view, [&]() {
        if (auto* dialog = qobject_cast<QFileDialog*>(
                    QApplication::activeModalWidget())) {
            dialog->selectFile(csvPath);
            acceptedDialog = QMetaObject::invokeMethod(dialog, "accept",
                                                       Qt::DirectConnection);
        }
    });
    QTimer::singleShot(5000, &view, []() {
        if (auto* dialog = qobject_cast<QFileDialog*>(
                    QApplication::activeModalWidget()))
            dialog->reject();
    });
    const bool invoked = QMetaObject::invokeMethod(&view, "onExportCSV",
                                                   Qt::DirectConnection);
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs,
                               nativeDialogsDisabled);
    ASSERT_TRUE(invoked);
    ASSERT_TRUE(acceptedDialog);
    QFile csv(csvPath);
    ASSERT_TRUE(csv.open(QIODevice::ReadOnly));
    EXPECT_EQ(csv.readAll(), QByteArray("Index,\"Centroid_X\"\n0,1\n1,2\n"));
    manager->setSourceObject(nullptr);
}

TEST(ChartView, QuartilesGroupFiniteXYBeforeSamplingAndKeepNativeStatistics) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float xy[][2] = {{2, 1},   {2, 2},     {2, 3},     {2, 4},
                           {1, 10},  {1, 20},    {3, 7},     {2, nan},
                           {2, inf}, {nan, 999}, {inf, 999}, {4, nan}};
    ccPointCloud source("grouped source");
    ASSERT_TRUE(source.reserve(12));
    // Point ingestion replaces NaN coordinates with (0,0,0). Scalar arrays
    // preserve invalid observations and exercise the actual finite-pair path.
    for (int i = 0; i < 12; ++i) source.addPoint(CCVector3(i, 0, 0));
    ASSERT_EQ(source.addScalarField("sampleX"), 0);
    ASSERT_EQ(source.addScalarField("sampleY"), 1);
    for (int field = 0; field < 2; ++field) {
        source.setCurrentScalarField(field);
        for (unsigned i = 0; i < 12; ++i)
            source.setPointScalarValue(i, xy[i][field]);
    }
    vtkChartView view(vtkChartView::QUARTILE_CHART);
    view.setMaxChartPoints(1);
    view.setEntity(&source);
    selectChartFields(view, {"sampleY"});
    QCheckBox* useIndex = nullptr;
    for (auto* check : view.findChildren<QCheckBox*>())
        if (check->text() == "Use Index for X Axis") useIndex = check;
    ASSERT_NE(useIndex, nullptr);
    QComboBox* xArray = nullptr;
    for (auto* combo : view.findChildren<QComboBox*>())
        if (combo->findText("sampleX") >= 0) xArray = combo;
    ASSERT_NE(xArray, nullptr);
    xArray->setCurrentIndex(xArray->findText("sampleX"));
    useIndex->setChecked(false);
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    ASSERT_EQ(chart->GetNumberOfPlots(), 3);
    EXPECT_TRUE(chart->GetPlot(0)->IsA("vtkPlotArea"));
    EXPECT_TRUE(chart->GetPlot(1)->IsA("vtkPlotArea"));
    EXPECT_TRUE(chart->GetPlot(2)->IsA("vtkPlotLine"));
    auto* table = chart->GetPlot(0)->GetInput();
    ASSERT_NE(table, nullptr);
    EXPECT_EQ(chart->GetPlot(1)->GetInput(), table);
    EXPECT_EQ(chart->GetPlot(2)->GetInput(), table);
    ASSERT_EQ(table->GetNumberOfRows(), 3);
    ASSERT_EQ(table->GetNumberOfColumns(), 8);
    const char* names[] = {"__quartile_x", "min(sampleY)", "q1(sampleY)",
                           "med(sampleY)", "q3(sampleY)",  "max(sampleY)",
                           "avg(sampleY)", "sum(sampleY)"};
    const double expected[][8] = {{1, 10, 10, 15, 20, 20, 15, 30},
                                  {2, 1, 1.5, 2.5, 3.5, 4, 2.5, 10},
                                  {3, 7, 7, 7, 7, 7, 7, 7}};
    for (int col = 0; col < 8; ++col) {
        EXPECT_STREQ(table->GetColumnName(col), names[col]);
        for (int row = 0; row < 3; ++row)
            EXPECT_DOUBLE_EQ(table->GetValue(row, col).ToDouble(),
                             expected[row][col]);
    }
    EXPECT_EQ(chart->GetAxis(vtkAxis::BOTTOM)->GetTitle(), "sampleX");
    useIndex->setChecked(true);
    ASSERT_EQ(chart->GetNumberOfPlots(), 3);
    table = chart->GetPlot(0)->GetInput();
    ASSERT_NE(table, nullptr);
    const int sourceIds[] = {0, 1, 2, 3, 4, 5, 6, 9, 10};
    ASSERT_EQ(table->GetNumberOfRows(), 9);
    for (int row = 0; row < 9; ++row) {
        const int id = sourceIds[row];
        EXPECT_DOUBLE_EQ(table->GetValue(row, 0).ToDouble(), id);
        for (int col = 1; col < 8; ++col)
            EXPECT_DOUBLE_EQ(table->GetValue(row, col).ToDouble(), xy[id][1]);
    }
    for (int role = vtkChartView::Minimum; role <= vtkChartView::QuartileBand;
         ++role) {
        view.setQuartileRoleVisible(
                static_cast<vtkChartView::QuartileRole>(role), true);
        EXPECT_TRUE(view.quartileRoleVisible(
                static_cast<vtkChartView::QuartileRole>(role)));
    }
    EXPECT_EQ(chart->GetNumberOfPlots(), 9);
    view.setQuartileRoleVisible(vtkChartView::Median, false);
    EXPECT_EQ(chart->GetNumberOfPlots(), 8);
    view.setQuartileRoleVisible(vtkChartView::MinMaxBand, false);
    EXPECT_EQ(chart->GetNumberOfPlots(), 7);
    selectChartFields(view, {});
    EXPECT_EQ(chart->GetNumberOfPlots(), 0);
    EXPECT_FALSE(chart->GetVisible());
}

// Inspect the native plot's mapped pixels, not only its input/LUT properties.
class MappedImagePlot : public vtkPlotHistogram2D {
public:
    static MappedImagePlot* New() {
        auto* plot = new MappedImagePlot;
        plot->InitializeObjectBase();
        return plot;
    }
    vtkImageData* mappedImage() { return this->Output; }
};

TEST(ChartView, ImageChartPreservesNativePixelsComponentsAndEmptySources) {
    QImage pixels(3, 2, QImage::Format_ARGB32);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x)
            pixels.setPixel(
                    x, y,
                    qRgba(10 + x, 20 + y, 30 + x + 3 * y, 100 + x + 3 * y));
    ccImage source(pixels, "pixels");
    source.setAspectRatio(3.0f);  // 3x2 image, non-square pixels: dx = 2
    vtkChartView view(vtkChartView::IMAGE_CHART);
    view.setEntity(&source);
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    auto* plot = vtkPlotHistogram2D::SafeDownCast(chart->GetPlot(0));
    ASSERT_NE(plot, nullptr);
    const QStringList components{"Red", "Green", "Blue", "Alpha"};
    vtkNew<MappedImagePlot> mapped;
    for (int component = 0; component < components.size(); ++component) {
        selectChartFields(view, {components[component]});
        auto* image = plot->GetInputImageData();
        ASSERT_NE(image, nullptr);
        EXPECT_TRUE(chart->GetVisible());
        EXPECT_EQ(image->GetDimensions()[0], 3);
        EXPECT_EQ(image->GetDimensions()[1], 2);
        EXPECT_EQ(image->GetDimensions()[2], 1);
        EXPECT_EQ(image->GetNumberOfScalarComponents(), 1);
        EXPECT_DOUBLE_EQ(image->GetSpacing()[0], 2.0);
        EXPECT_DOUBLE_EQ(image->GetSpacing()[1], 1.0);
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_DOUBLE_EQ(image->GetOrigin()[axis], 0.0);
        ASSERT_NE(plot->GetTransferFunction(), nullptr);
        EXPECT_EQ(plot->GetTransferFunction()->GetVectorComponent(), 0);
        mapped->SetInputData(image);
        mapped->SetTransferFunction(plot->GetTransferFunction());
        mapped->Update();
        ASSERT_NE(mapped->mappedImage(), nullptr);
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 3; ++x) {
                const QRgb pixel = pixels.pixel(x, y);
                const int values[] = {qRed(pixel), qGreen(pixel), qBlue(pixel),
                                      qAlpha(pixel)};
                EXPECT_DOUBLE_EQ(
                        image->GetScalarComponentAsDouble(x, 1 - y, 0, 0),
                        values[component]);
                const auto* expected = plot->GetTransferFunction()->MapValue(
                        values[component]);
                const auto* actual = static_cast<const unsigned char*>(
                        mapped->mappedImage()->GetScalarPointer(x, 1 - y, 0));
                for (int c = 0; c < 4; ++c) EXPECT_EQ(actual[c], expected[c]);
            }
        }
    }
    auto expectEmpty = [&]() {
        EXPECT_FALSE(chart->GetVisible());
        auto* image = plot->GetInputImageData();
        EXPECT_TRUE(!image || image->GetNumberOfPoints() == 0);
    };
    selectChartFields(view, {});
    expectEmpty();
    view.setEntity(&source);
    view.setEntity(nullptr);
    expectEmpty();
    ccPointCloud cloud("not an image");
    view.setEntity(&source);
    view.setEntity(&cloud);
    expectEmpty();
    ccImage empty(QImage(), "empty");
    view.setEntity(&empty);
    expectEmpty();

#if QT_VERSION >= QT_VERSION_CHECK(5, 13, 0)
    // Width 3 requires Qt scanline padding; padding is not a fourth pixel.
    QImage grayscale(3, 2, QImage::Format_Grayscale16);
    grayscale.fill(0);
    for (int y = 0; y < 2; ++y) {
        auto* row = reinterpret_cast<quint16*>(grayscale.scanLine(y));
        for (int x = 0; x < 3; ++x) row[x] = 1000 * (1 + x + 3 * y);
    }
    ccImage graySource(grayscale, "16-bit intensity");
    view.setEntity(&graySource);
    selectChartFields(view, {"Intensity"});
    auto* image = plot->GetInputImageData();
    ASSERT_NE(image, nullptr);
    EXPECT_EQ(image->GetScalarType(), VTK_UNSIGNED_SHORT);
    EXPECT_EQ(image->GetNumberOfScalarComponents(), 1);
    EXPECT_EQ(image->GetNumberOfPoints(), 6);
    EXPECT_DOUBLE_EQ(image->GetSpacing()[0], 1.0);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x)
            EXPECT_DOUBLE_EQ(image->GetScalarComponentAsDouble(x, 1 - y, 0, 0),
                             1000.0 * (1 + x + 3 * y));
#endif
}

TEST(PythonView, RunExecutesSideEffectsOnceAndInspectsSameProcess) {
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    ecvPythonView view;
    auto* editor = view.findChild<ecvPythonCodeEditor*>();
    ASSERT_NE(editor, nullptr);
    const QString counter = temp.filePath("runs.txt");
    editor->setPlainText(QString("with open(r'%1', 'a') as f: "
                                 "f.write('run\\n')\nanswer = 42\n")
                                 .arg(counter));
    ASSERT_TRUE(QMetaObject::invokeMethod(&view, "onRunScript",
                                          Qt::DirectConnection));
    auto* process = view.findChild<QProcess*>();
    ASSERT_NE(process, nullptr)
            << "Run must return with an asynchronous process";
    QEventLoop loop;
    QObject::connect(process,
                     qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
                     &loop, &QEventLoop::quit);
    QTimer::singleShot(5000, &loop, &QEventLoop::quit);
    loop.exec();
    QFile result(counter);
    ASSERT_TRUE(result.open(QIODevice::ReadOnly));
    EXPECT_EQ(result.readAll(), QByteArray("run\n"));
    auto* variables = view.findChild<QTableWidget*>();
    ASSERT_NE(variables, nullptr);
    bool hasAnswer = false;
    for (int row = 0; row < variables->rowCount(); ++row) {
        if (variables->item(row, 0)->text() == "answer") {
            hasAnswer = true;
            EXPECT_EQ(variables->item(row, 3)->text(), "42");
        }
    }
    EXPECT_TRUE(hasAnswer);
}

TEST(PythonView,
     ResultLifecycleClearsOldImagesAndPreservesTimeoutAndCancellation) {
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    const QString imagePath = temp.filePath("input.png");
    QImage image(8, 8, QImage::Format_RGB32);
    image.fill(Qt::red);
    ASSERT_TRUE(image.save(imagePath));
    ecvPythonView view;
    auto* editor = view.findChild<ecvPythonCodeEditor*>();
    ASSERT_NE(editor, nullptr);
    auto run = [&view, editor](const QString& script, bool expire = false) {
        editor->setPlainText(script);
        EXPECT_TRUE(QMetaObject::invokeMethod(&view, "onRunScript",
                                              Qt::DirectConnection));
        auto* process = view.findChild<QProcess*>();
        if (!process) {
            ADD_FAILURE() << "expected asynchronous script process";
            return;
        }
        auto* deadline = process->findChild<QTimer*>();
        EXPECT_NE(deadline, nullptr);
        if (expire && deadline) {
            QObject::connect(process, &QProcess::started, deadline,
                             [deadline]() { deadline->start(1); });
        }
        QEventLoop loop;
        QObject::connect(
                process,
                qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
                &loop, &QEventLoop::quit);
        QTimer::singleShot(5000, &loop, &QEventLoop::quit);
        loop.exec();
        EXPECT_EQ(process->state(), QProcess::NotRunning);
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    };
    run(QString("import os, shutil\nshutil.copyfile(r'%1', "
                "os.environ['PLOT_FILE'])\n")
                .arg(imagePath));
    QLabel* plot = nullptr;
    for (auto* label : view.findChildren<QLabel*>()) {
        if (!qtCompatLabelPixmap(label).isNull()) plot = label;
    }
    ASSERT_NE(plot, nullptr);
    run("answer = 7\n");
    EXPECT_TRUE(qtCompatLabelPixmap(plot).isNull());
    run("raise RuntimeError('expected failure')\n");
    EXPECT_EQ(view.findChild<QTableWidget*>()->rowCount(), 0);
    run("import time\ntime.sleep(10)\n", true);
    bool timedOut = false;
    for (auto* label : view.findChildren<QLabel*>())
        timedOut |= label->text() == "Timed out";
    EXPECT_TRUE(timedOut) << "finished must not overwrite deadline outcome";
    run("recovered = 42\n");
    EXPECT_GT(view.findChild<QTableWidget*>()->rowCount(), 0);

    ccPointCloud source("new source");
    editor->setPlainText("import time\ntime.sleep(10)\n");
    ASSERT_TRUE(QMetaObject::invokeMethod(&view, "onRunScript",
                                          Qt::DirectConnection));
    QPointer<QProcess> process = view.findChild<QProcess*>();
    ASSERT_NE(process.data(), nullptr);
    view.setEntity(&source);
    if (process && process->state() != QProcess::NotRunning)
        process->waitForFinished(3000);
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    EXPECT_EQ(view.findChild<QTableWidget*>()->rowCount(), 0);
    bool sourceStatus = false;
    for (auto* label : view.findChildren<QLabel*>())
        sourceStatus |= label->text() == "Entity: new source";
    EXPECT_TRUE(sourceStatus)
            << "cancelled execution must not overwrite source status";
}

TEST(PythonView, FailedInterpreterStartReleasesProcessForRetry) {
#ifndef Q_OS_UNIX
    GTEST_SKIP() << "executable shebang failure fixture requires Unix";
#else
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    QFile fakePython(temp.filePath("python3"));
    ASSERT_TRUE(fakePython.open(QIODevice::WriteOnly));
    fakePython.write("#!/acv-test-interpreter-that-does-not-exist\n");
    fakePython.close();
    ASSERT_TRUE(fakePython.setPermissions(QFileDevice::ReadOwner |
                                          QFileDevice::WriteOwner |
                                          QFileDevice::ExeOwner));
    ecvPythonView view;
    auto* editor = view.findChild<ecvPythonCodeEditor*>();
    ASSERT_NE(editor, nullptr);
    editor->setPlainText("answer = 42\n");
    const QByteArray originalPath = qgetenv("PATH");
    const bool hadPath = qEnvironmentVariableIsSet("PATH");
    qputenv("PATH", temp.path().toLocal8Bit());
    const bool invoked = QMetaObject::invokeMethod(&view, "onRunScript",
                                                   Qt::DirectConnection);
    if (hadPath)
        qputenv("PATH", originalPath);
    else
        qunsetenv("PATH");
    ASSERT_TRUE(invoked);
    auto* process = view.findChild<QProcess*>();
    ASSERT_NE(process, nullptr);
    QEventLoop failed;
    QObject::connect(process, &QProcess::errorOccurred, &failed,
                     &QEventLoop::quit);
    QTimer::singleShot(5000, &failed, &QEventLoop::quit);
    failed.exec();
    EXPECT_EQ(process->error(), QProcess::FailedToStart);
    bool startFailed = false;
    for (auto* label : view.findChildren<QLabel*>())
        startFailed |= label->text() == "Start failed";
    EXPECT_TRUE(startFailed);
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    EXPECT_EQ(view.findChild<QProcess*>(), nullptr);

    ASSERT_TRUE(QMetaObject::invokeMethod(&view, "onRunScript",
                                          Qt::DirectConnection));
    process = view.findChild<QProcess*>();
    ASSERT_NE(process, nullptr);
    QEventLoop recovered;
    QObject::connect(process,
                     qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
                     &recovered, &QEventLoop::quit);
    QTimer::singleShot(5000, &recovered, &QEventLoop::quit);
    recovered.exec();
    EXPECT_EQ(process->exitCode(), 0);
    auto* variables = view.findChild<QTableWidget*>();
    ASSERT_NE(variables, nullptr);
    ASSERT_EQ(variables->rowCount(), 1);
    EXPECT_EQ(variables->item(0, 0)->text(), "answer");
    EXPECT_EQ(variables->item(0, 3)->text(), "42");
#endif
}

TEST(ComparativeView,
     ChartSourcesOverlayAndCuesScanRealRowsWithTypedSelections) {
    ccPointCloud sourceA("source A"), sourceB("source B");
    sourceA.reserve(2);
    sourceB.reserve(3);
    sourceA.addPoint(CCVector3(0, 1, 2));
    sourceA.addPoint(CCVector3(1, 2, 3));
    sourceB.addPoint(CCVector3(2, 3, 4));
    sourceB.addPoint(CCVector3(3, 4, 5));
    sourceB.addPoint(CCVector3(4, 5, 6));
    // The same field occupies different scalar-array positions per source.
    ASSERT_EQ(sourceA.addScalarField("signal"), 0);
    ASSERT_EQ(sourceB.addScalarField("other"), 0);
    ASSERT_EQ(sourceB.addScalarField("signal"), 1);
    sourceA.setCurrentScalarField(0);
    sourceB.setCurrentScalarField(1);
    for (unsigned i = 0; i < sourceA.size(); ++i)
        sourceA.setPointScalarValue(i, 11 + i);
    for (unsigned i = 0; i < sourceB.size(); ++i)
        sourceB.setPointScalarValue(i, 21 + i);
    for (auto type : {vtkComparativeViewWidget::LINE_CHART,
                      vtkComparativeViewWidget::BAR_CHART}) {
        QList<ccHObject*> sources{&sourceA, &sourceB};
        vtkComparativeViewWidget view(type);
        view.setEntityListProvider([&sources]() { return sources; });
        view.setInitialEntity(&sourceA);
        view.setupGrid();
        auto* combo = view.findChild<QComboBox*>();
        ASSERT_NE(combo, nullptr);
        auto* model = qobject_cast<QStandardItemModel*>(combo->model());
        ASSERT_NE(model, nullptr);
        ASSERT_EQ(model->rowCount(), 3);
        ASSERT_EQ(view.subWidgets().size(), 4);
        auto expectSource = [&view, &sourceA](ccHObject* expected) {
            for (auto* widget : view.subWidgets()) {
                auto* chart = qobject_cast<vtkChartView*>(widget);
                ASSERT_NE(chart, nullptr);
                auto* nativeChart = renderedChart(*chart);
                ASSERT_NE(nativeChart, nullptr);
                if (!expected) {
                    EXPECT_EQ(nativeChart->GetNumberOfPlots(), 0);
                } else {
                    ASSERT_GT(nativeChart->GetNumberOfPlots(), 0);
                    auto* input = nativeChart->GetPlot(0)->GetInput();
                    ASSERT_NE(input, nullptr);
                    EXPECT_EQ(input->GetNumberOfRows(),
                              expected == &sourceA ? 2 : 3);
                }
            }
        };
        expectSource(&sourceA);
        ecvViewManager::instance().entitySelectionChanged(&sourceB);
        expectSource(&sourceA);
        auto* firstChart =
                qobject_cast<vtkChartView*>(view.subWidgets().front());
        auto* fields = firstChart->findChild<QListWidget*>();
        ASSERT_NE(fields, nullptr);
        ASSERT_GE(fields->count(), 2);
        fields->item(1)->setSelected(true);
        ASSERT_EQ(renderedChart(*firstChart)->GetNumberOfPlots(), 2);
        view.setEntityListProvider([&sources]() { return sources; });
        EXPECT_EQ(renderedChart(*firstChart)->GetNumberOfPlots(), 2)
                << "an unchanged source list must not reset per-pane fields";
        auto* manager = cvViewSelectionManager::instance();
        QObject observer;
        int publications = 0;
        QObject::connect(manager,
                         qOverload<const cvSelectionData&>(
                                 &cvViewSelectionManager::selectionChanged),
                         &observer, [&publications](const cvSelectionData&) {
                             ++publications;
                         });
        vtkNew<vtkIdTypeArray> rows;
        rows->InsertNextValue(1);
        vtkNew<vtkSelectionNode> node;
        node->SetContentType(vtkSelectionNode::INDICES);
        node->SetFieldType(vtkSelectionNode::ROW);
        node->SetSelectionList(rows);
        vtkNew<vtkSelection> gesture;
        gesture->AddNode(node);
        auto* link = renderedChart(*firstChart)->GetAnnotationLink();
        ASSERT_NE(link, nullptr);
        publications = 0;
        // The native setter itself emits the gesture event.
        link->SetCurrentSelection(gesture);
        EXPECT_EQ(publications, 1) << "one gesture has one publication owner";
        const auto selected = manager->currentSelection();
        EXPECT_EQ(selected.fieldAssociation(), cvSelectionData::POINTS);
        ASSERT_EQ(selected.actorInfos().size(), 1);
        EXPECT_EQ(selected.actorInfos().front().sourceId, sourceA.getViewId());
        EXPECT_EQ(selected.actorInfos().front().sourceIds,
                  QVector<qint64>({1}));
        for (auto* widget : view.subWidgets()) {
            auto* native = renderedChart(*qobject_cast<vtkChartView*>(widget));
            native->Update();
            auto* ids = native->GetPlot(0)->GetSelection();
            ASSERT_NE(ids, nullptr);
            ASSERT_EQ(ids->GetNumberOfValues(), 1);
            EXPECT_EQ(ids->GetValue(0), 1);
        }
        EXPECT_EQ(publications, 1) << "incoming selections must not feed back";
        vtkNew<vtkSelection> empty;
        publications = 0;
        link->SetCurrentSelection(empty);
        EXPECT_EQ(publications, 1);
        const auto cleared = manager->currentSelection();
        EXPECT_TRUE(cleared.isEmpty());
        ASSERT_EQ(cleared.actorInfos().size(), 1);
        EXPECT_EQ(cleared.actorInfos().front().sourceId, sourceA.getViewId());
        EXPECT_TRUE(cleared.actorInfos().front().sourceIds.isEmpty());
        manager->setSourceObject(nullptr);
        EXPECT_EQ(model->item(1)->checkState(), Qt::Checked);
        model->item(2)->setCheckState(Qt::Checked);
        EXPECT_EQ(model->item(1)->checkState(), Qt::Checked);
        EXPECT_EQ(model->item(0)->checkState(), Qt::Unchecked);
        expectSource(&sourceA);
        ASSERT_EQ(renderedChart(*firstChart)->GetNumberOfPlots(), 4);

        using Comparative = vtkComparativeViewWidget;
        Comparative::ParameterCue cue;
        cue.field = "Points_X";
        cue.direction = Comparative::AcrossColumns;
        cue.minimum = 0;
        cue.maximum = 4;
        cue.cellValues[3] = 3.5;
        ASSERT_TRUE(view.setCues({cue}));
        EXPECT_DOUBLE_EQ(view.cueValue(0, 0, 0), 0);
        EXPECT_DOUBLE_EQ(view.cueValue(0, 1, 0), 4);
        EXPECT_DOUBLE_EQ(view.cueValue(0, 0, 1), 0);
        EXPECT_DOUBLE_EQ(view.cueValue(0, 1, 1), 3.5);
        auto invalid = cue;
        invalid.field.clear();
        EXPECT_FALSE(view.setCues({invalid}));
        ASSERT_EQ(view.cues().size(), 1);
        for (int cell = 0; cell < view.subWidgets().size(); ++cell) {
            auto* native = renderedChart(
                    *qobject_cast<vtkChartView*>(view.subWidgets()[cell]));
            std::set<vtkTable*> tables;
            for (vtkIdType pi = 0; pi < native->GetNumberOfPlots(); ++pi)
                tables.insert(native->GetPlot(pi)->GetInput());
            EXPECT_EQ(tables.size(), cell % 2 ? 1 : 2);
            for (auto* table : tables) {
                auto* x = table->GetColumnByName("Points_X");
                auto* ids = vtkIdTypeArray::SafeDownCast(
                        table->GetFieldData()->GetArray("__acv_original_ids"));
                auto* source = vtkStringArray::SafeDownCast(
                        table->GetFieldData()->GetAbstractArray(
                                "__acv_source_id"));
                ASSERT_NE(x, nullptr);
                ASSERT_NE(ids, nullptr);
                ASSERT_NE(source, nullptr);
                EXPECT_EQ(table->GetNumberOfRows(),
                          cell % 2 ? 1
                                   : (QString::fromStdString(source->GetValue(
                                              0)) == sourceA.getViewId()
                                              ? 2
                                              : 3));
                for (vtkIdType row = 0; row < table->GetNumberOfRows(); ++row) {
                    EXPECT_GE(x->GetVariantValue(row).ToDouble(),
                              view.cueValue(0, cell % 2, cell / 2));
                    EXPECT_DOUBLE_EQ(
                            x->GetVariantValue(row).ToDouble(),
                            ids->GetValue(row) +
                                    (QString::fromStdString(source->GetValue(
                                             0)) == sourceA.getViewId()
                                             ? 0
                                             : 2));
                }
            }
        }
        // Filtering precedes sampling. Native plot selections then map each
        // input row through its own original-ID array in one publication.
        firstChart->setMaxChartPoints(2);
        auto* native = renderedChart(*firstChart);
        vtkNew<vtkSelection> overlayGesture;
        for (int pi : {0, 2}) {
            auto* plot = native->GetPlot(pi);
            vtkNew<vtkIdTypeArray> picked;
            picked->InsertNextValue(1);
            vtkNew<vtkSelectionNode> pickedNode;
            pickedNode->SetContentType(vtkSelectionNode::INDICES);
            pickedNode->SetFieldType(vtkSelectionNode::POINT);
            pickedNode->GetProperties()->Set(vtkSelectionNode::PROP(), plot);
            pickedNode->GetProperties()->Set(vtkSelectionNode::SOURCE(),
                                             plot->GetInput());
            pickedNode->SetSelectionList(picked);
            overlayGesture->AddNode(pickedNode);
        }
        publications = 0;
        native->GetAnnotationLink()->SetCurrentSelection(overlayGesture);
        EXPECT_EQ(publications, 1);
        const auto overlaySelection = manager->currentSelection();
        EXPECT_EQ(overlaySelection.fieldAssociation(), cvSelectionData::POINTS);
        ASSERT_EQ(overlaySelection.actorInfos().size(), 2);
        for (const auto& info : overlaySelection.actorInfos())
            EXPECT_EQ(info.sourceIds,
                      QVector<qint64>(
                              {info.sourceId == sourceA.getViewId() ? 1 : 2}));
        for (auto* widget : view.subWidgets()) {
            auto* chart = renderedChart(*qobject_cast<vtkChartView*>(widget));
            chart->Update();
            for (vtkIdType pi = 0; pi < chart->GetNumberOfPlots(); ++pi) {
                auto* plot = chart->GetPlot(pi);
                auto* table = plot->GetInput();
                auto* source = vtkStringArray::SafeDownCast(
                        table->GetFieldData()->GetAbstractArray(
                                "__acv_source_id"));
                auto* original = vtkIdTypeArray::SafeDownCast(
                        table->GetFieldData()->GetArray("__acv_original_ids"));
                auto* picked = plot->GetSelection();
                ASSERT_NE(picked, nullptr);
                ASSERT_EQ(picked->GetNumberOfValues(), 1);
                EXPECT_EQ(original->GetValue(picked->GetValue(0)),
                          QString::fromStdString(source->GetValue(0)) ==
                                          sourceA.getViewId()
                                  ? 1
                                  : 2);
            }
        }
        EXPECT_EQ(publications, 1);
        publications = 0;
        firstChart->clearSelection();
        EXPECT_EQ(publications, 1);
        EXPECT_TRUE(manager->currentSelection().isEmpty());
        ASSERT_TRUE(view.setCues({}));
        selectChartFields(*firstChart, {"signal"});
        auto* named = renderedChart(*firstChart);
        ASSERT_EQ(named->GetNumberOfPlots(), 2);
        EXPECT_DOUBLE_EQ(
                named->GetPlot(0)->GetInput()->GetValue(0, 1).ToDouble(), 11);
        EXPECT_DOUBLE_EQ(
                named->GetPlot(1)->GetInput()->GetValue(0, 1).ToDouble(), 21);
        Comparative::ParameterCue minimum, maximum;
        maximum.property = Comparative::ThresholdMaximum;
        minimum.field = maximum.field = "signal";
        minimum.minimum = minimum.maximum = 12;
        maximum.minimum = maximum.maximum = 22;
        maximum.cellValues.clear();
        ASSERT_TRUE(view.setCues({minimum, maximum}));
        ASSERT_EQ(named->GetPlot(0)->GetInput()->GetNumberOfRows(), 1);
        ASSERT_EQ(named->GetPlot(1)->GetInput()->GetNumberOfRows(), 2);
        EXPECT_DOUBLE_EQ(
                named->GetPlot(0)->GetInput()->GetValue(0, 1).ToDouble(), 12);
        EXPECT_DOUBLE_EQ(
                named->GetPlot(1)->GetInput()->GetValue(0, 1).ToDouble(), 21);
        ASSERT_TRUE(view.setCues({}));
        model->item(2)->setCheckState(Qt::Unchecked);
        expectSource(&sourceA);
        model->item(1)->setCheckState(Qt::Unchecked);
        EXPECT_EQ(model->item(0)->checkState(), Qt::Checked);
        expectSource(nullptr);
        model->item(1)->setCheckState(Qt::Checked);
        sources.removeOne(&sourceA);
        view.setEntityListProvider([&sources]() { return sources; });
        expectSource(nullptr);
        EXPECT_EQ(model->item(0)->checkState(), Qt::Checked);
    }
}

TEST(ComparativeView, RenderCuesCutAtEvaluatedWorldCoordinates) {
    ccPointCloud source("cue source");
    ASSERT_TRUE(source.reserve(8));
    for (float x : {-0.5f, 0.5f})
        for (float y : {-0.5f, 0.5f})
            for (float z : {-0.5f, 0.5f}) source.addPoint(CCVector3(x, y, z));
    source.setVisible(true);
    LayoutFixture fixture;
    vtkComparativeViewWidget view(vtkComparativeViewWidget::RENDER);
    source.setMetaData("time", 0.0);
    view.setEntityListProvider(
            [&source]() { return QList<ccHObject*>{&source}; });
    view.setInitialEntity(&source);
    view.setRenderViewFactory([&fixture]() { return fixture.newView(); });
    ASSERT_EQ(view.subViews().size(), 4);
    vtkNew<vtkCubeSource> cube;
    cube->Update();
    for (auto* pane : view.subViews()) {
        auto* vis = pane->getVisualizer3D();
        vtkSmartPointer<vtkPolyData> geometry = cube->GetOutput();
        ASSERT_TRUE(
                vis->addPointCloud(geometry, source.getViewId().toStdString()));
        ecvRepresentationManager::instance().ensureRepresentation(&source,
                                                                  pane);
        pane->setMultiSlicePositions(1, {0.1});
    }
    using Comparative = vtkComparativeViewWidget;
    Comparative::ParameterCue cue;
    cue.property = Comparative::SliceX;
    cue.direction = Comparative::ColumnMajor;
    cue.minimum = -0.3;
    cue.maximum = 0.3;
    ASSERT_TRUE(view.setCues({cue}));
    for (int cell = 0; cell < 4; ++cell) {
        auto* pane = view.subViews()[cell];
        const double expected = view.cueValue(0, cell % 2, cell / 2);
        EXPECT_NEAR(expected,
                    cell == 0   ? -0.3
                    : cell == 1 ? 0.1
                    : cell == 2 ? -0.1
                                : 0.3,
                    1e-12);
        ASSERT_EQ(pane->getMultiSlicePositions(0).size(), 1);
        EXPECT_DOUBLE_EQ(pane->getMultiSlicePositions(0).front(), expected);
        auto* actors =
                pane->getVisualizer3D()->getCurrentRenderer()->GetActors();
        actors->InitTraversal();
        int xCuts = 0;
        while (auto* actor = actors->GetNextActor()) {
            auto* bounds = actor->GetBounds();
            if (std::abs(bounds[1] - bounds[0]) < 1e-6 &&
                bounds[3] > bounds[2]) {
                ++xCuts;
                EXPECT_NEAR(bounds[0], expected, 1e-6);
            }
        }
        EXPECT_EQ(xCuts, 1);
    }
    ASSERT_TRUE(view.setCues({}));
    for (auto* pane : view.subViews()) {
        EXPECT_TRUE(pane->getMultiSlicePositions(0).empty());
        EXPECT_EQ(pane->getMultiSlicePositions(1), std::vector<double>({0.1}));
    }
    view.resize(900, 700);
    view.show();
    pump();
    Comparative::ParameterCue opacity, time;
    opacity.property = Comparative::Opacity;
    opacity.direction = Comparative::AcrossColumns;
    opacity.minimum = 0.2;
    opacity.maximum = 0.8;
    ASSERT_TRUE(view.setCues({opacity}));
    for (int cell = 0; cell < 4; ++cell) {
        auto* rep = ecvRepresentationManager::instance().ensureRepresentation(
                &source, view.subViews()[cell]);
        EXPECT_NEAR(rep->properties().opacity.value(), cell % 2 ? 0.8 : 0.2,
                    1e-6);
    }
    auto* renderer =
            view.subViews().front()->getVisualizer3D()->getCurrentRenderer();
    const int originalActors = renderer->GetActors()->GetNumberOfItems();
    view.setOverlayAllComparisons(true);
    auto refreshComparisons = []() {
        QEventLoop refresh;
        QTimer::singleShot(200, &refresh, &QEventLoop::quit);
        refresh.exec();
    };
    refreshComparisons();
    EXPECT_GT(renderer->GetActors()->GetNumberOfItems(), originalActors);
    EXPECT_TRUE(view.subViews().back()->asWidget()->isHidden());
    opacity.minimum = 0.3;
    opacity.maximum = 0.6;
    ASSERT_TRUE(view.setCues({opacity}));
    refreshComparisons();
    int updatedCopies = 0;
    auto* overlayActors = renderer->GetActors();
    overlayActors->InitTraversal();
    while (auto* actor = overlayActors->GetNextActor()) {
        if (!actor->GetPickable() && actor->GetMapper() &&
            actor->GetMapper()
                    ->GetInputDataObject(0, 0)
                    ->GetFieldData()
                    ->HasArray("__acv_source_id")) {
            EXPECT_NEAR(actor->GetProperty()->GetOpacity(),
                        actor->GetProperty()->GetOpacity() > 0.45 ? 0.6 : 0.3,
                        1e-6);
            if (actor->GetProperty()->GetOpacity() > 0.45) ++updatedCopies;
        }
    }
    EXPECT_GE(updatedCopies, 2);
    view.setOverlayAllComparisons(false);
    EXPECT_EQ(renderer->GetActors()->GetNumberOfItems(), originalActors);
    time.property = Comparative::Time;
    time.direction = Comparative::AcrossColumns;
    time.field = "time";
    time.minimum = 0;
    time.maximum = 1;
    ASSERT_TRUE(view.setCues({time, opacity}));
    for (int cell = 0; cell < 4; ++cell)
        EXPECT_EQ(ecvRepresentationManager::instance()
                          .ensureRepresentation(&source, view.subViews()[cell])
                          ->isVisible(),
                  cell % 2 == 0);
    QPointer<vtkGLView> retained = view.subViews().front();
    QPointer<vtkGLView> removed = view.subViews().back();
    const int registeredViews = ecvViewManager::instance().viewCount();
    ASSERT_TRUE(view.setDimensions(1, 1));
    EXPECT_EQ(view.subViews().front(), retained.data());
    EXPECT_TRUE(removed.isNull());
    EXPECT_EQ(ecvViewManager::instance().viewCount(), registeredViews - 3);
    ASSERT_TRUE(view.setCues({}));
}

TEST(FollowupAudit, XYAllowsSameFieldOnBothAxes) {
    ccPointCloud source("same-field");
    ASSERT_TRUE(source.reserve(3));
    for (int i = 0; i < 3; ++i) source.addPoint(CCVector3(i + 1, 10 + i, 0));
    vtkChartView view(vtkChartView::LINE_CHART);
    view.setEntity(&source);
    selectChartFields(view, {"Points_X"});
    QComboBox* xArray = nullptr;
    QCheckBox* useIndex = nullptr;
    for (auto* combo : view.findChildren<QComboBox*>())
        if (combo->findText("Points_X") >= 0) xArray = combo;
    for (auto* check : view.findChildren<QCheckBox*>())
        if (check->text() == "Use Index for X Axis") useIndex = check;
    ASSERT_NE(xArray, nullptr);
    ASSERT_NE(useIndex, nullptr);
    xArray->setCurrentIndex(xArray->findText("Points_X"));
    useIndex->setChecked(false);
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    ASSERT_EQ(chart->GetNumberOfPlots(), 1);
    auto* input = chart->GetPlot(0)->GetInput();
    ASSERT_NE(input, nullptr);
    // Both roles bind by name to the same valid array, without duplicate
    // columns.
    auto* sameField = input->GetColumnByName("Points_X");
    ASSERT_NE(sameField, nullptr);
    EXPECT_DOUBLE_EQ(sameField->GetVariantValue(0).ToDouble(), 1);
}

TEST(FollowupAudit, HistogramKeepsSelectedSeriesIdentity) {
    ccPointCloud source("histogram-series");
    ASSERT_TRUE(source.reserve(4));
    source.addPoint(CCVector3(0, 1000, 0));
    source.addPoint(CCVector3(0, 1000, 0));
    source.addPoint(CCVector3(9, 100, 0));
    source.addPoint(CCVector3(10, 100, 0));
    vtkChartView view(vtkChartView::HISTOGRAM);
    for (auto* spin : view.findChildren<QSpinBox*>())
        if (spin->minimum() == 2 && spin->maximum() == 1000) spin->setValue(2);
    view.setEntity(&source);
    selectChartFields(view, {"Points_X", "Points_Y"});
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    ASSERT_EQ(chart->GetNumberOfPlots(), 2);
    QVector<unsigned> selected;
    QObject::connect(&view, &vtkChartView::pointsHighlighted,
                     [&selected](ccHObject*, const QVector<unsigned>& ids) {
                         selected = ids;
                     });
    vtkNew<vtkIdTypeArray> rows;
    rows->InsertNextValue(0);
    vtkNew<vtkSelectionNode> node;
    node->SetContentType(vtkSelectionNode::INDICES);
    node->SetFieldType(vtkSelectionNode::ROW);
    node->SetSelectionList(rows);
    node->GetProperties()->Set(vtkSelectionNode::PROP(), chart->GetPlot(0));
    node->GetProperties()->Set(vtkSelectionNode::SOURCE(),
                               chart->GetPlot(0)->GetInput());
    vtkNew<vtkSelection> selection;
    selection->AddNode(node);
    chart->GetAnnotationLink()->SetCurrentSelection(selection);
    EXPECT_EQ(selected, QVector<unsigned>({0, 1}));
}

TEST(FollowupAudit, BoxUsesAveragedStepOrderStatistics) {
    ccPointCloud source("box-median");
    ASSERT_TRUE(source.reserve(4));
    for (int i = 1; i <= 4; ++i) source.addPoint(CCVector3(i, 0, 0));
    vtkChartView view(vtkChartView::BOX_CHART);
    view.setEntity(&source);
    selectChartFields(view, {"Points_X"});
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    ASSERT_GT(chart->GetNumberOfPlots(), 0);
    auto* input = chart->GetPlot(0)->GetInput();
    ASSERT_NE(input, nullptr);
    ASSERT_EQ(input->GetNumberOfRows(), 5);
    EXPECT_DOUBLE_EQ(input->GetValue(1, 0).ToDouble(), 1.5);
    EXPECT_DOUBLE_EQ(input->GetValue(2, 0).ToDouble(), 2.5);
    EXPECT_DOUBLE_EQ(input->GetValue(3, 0).ToDouble(), 3.5);
}

TEST(FollowupAudit, ComparativeRefreshRebuildsChangedSourceValues) {
    ccPointCloud source("refresh-source");
    ASSERT_TRUE(source.reserve(2));
    source.addPoint(CCVector3(0, 0, 0));
    source.addPoint(CCVector3(1, 0, 0));
    ASSERT_EQ(source.addScalarField("signal"), 0);
    source.setCurrentScalarField(0);
    source.setPointScalarValue(0, 11);
    source.setPointScalarValue(1, 12);
    vtkComparativeViewWidget view(vtkComparativeViewWidget::LINE_CHART);
    view.setEntityListProvider(
            [&source]() { return QList<ccHObject*>{&source}; });
    view.setInitialEntity(&source);
    view.setupGrid();
    ASSERT_EQ(view.subWidgets().size(), 4);
    auto* pane = qobject_cast<vtkChartView*>(view.subWidgets().front());
    ASSERT_NE(pane, nullptr);
    selectChartFields(*pane, {"signal"});
    auto* chart = renderedChart(*pane);
    ASSERT_NE(chart, nullptr);
    ASSERT_GT(chart->GetNumberOfPlots(), 0);
    ASSERT_DOUBLE_EQ(chart->GetPlot(0)->GetInput()->GetValue(0, 1).ToDouble(),
                     11);
    source.setPointScalarValue(0, 99);
    view.refreshSubViews();
    QEventLoop waitForRefresh;
    QTimer::singleShot(80, &waitForRefresh, &QEventLoop::quit);
    waitForRefresh.exec();
    EXPECT_DOUBLE_EQ(chart->GetPlot(0)->GetInput()->GetValue(0, 1).ToDouble(),
                     99);
}

TEST(FollowupAudit, ExplicitNoneLineStyleRemainsNone) {
    ccPointCloud source("none-style");
    ASSERT_TRUE(source.reserve(2));
    source.addPoint(CCVector3(0, 0, 0));
    source.addPoint(CCVector3(1, 1, 0));
    vtkChartView view(vtkChartView::LINE_CHART);
    view.setEntity(&source);
    selectChartFields(view, {"Points_X"});
    auto* table = view.findChild<QTableWidget*>();
    ASSERT_NE(table, nullptr);
    // Invoke the real table signal: the connected owner method is not a slot.
    // The UI cycles Solid(1), Dash(2), Dot(3), Dash-Dot(4), None(0).
    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(QMetaObject::invokeMethod(table, "cellClicked",
                                              Qt::DirectConnection,
                                              Q_ARG(int, 0), Q_ARG(int, 6)));
    ASSERT_EQ(table->item(0, 6)->text(), "None");
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    ASSERT_GT(chart->GetNumberOfPlots(), 0);
    EXPECT_EQ(chart->GetPlot(0)->GetPen()->GetLineType(), vtkPen::NO_PEN);
}

TEST(FollowupAudit, OutlineDoesNotHideRegisteredWireframeSource) {
    LayoutFixture fixture;
    auto* view = fixture.newView();
    fixture.adoptFirst(view);
    vtkNew<vtkCubeSource> geometry;
    geometry->Update();
    auto* vis = view->getVisualizer3D();
    ASSERT_TRUE(vis->addPointCloud(geometry->GetOutput(), "wireframe-source"));
    auto* actor = vis->getActorById("wireframe-source");
    ASSERT_NE(actor, nullptr);
    actor->GetProperty()->SetRepresentationToWireframe();
    actor->SetVisibility(1);
    view->setOutlineVisible(false);
    EXPECT_EQ(actor->GetVisibility(), 1);
}

TEST(FollowupAudit, SpreadsheetCopyAllRespectsVisibleFilter) {
    ccPointCloud source("clipboard-filter");
    ASSERT_TRUE(source.reserve(3));
    for (int x : {10, 20, 30}) source.addPoint(CCVector3(x, 0, 0));
    ecvSpreadSheetView view;
    view.setEntity(&source);
    auto* proxy = view.findChild<QSortFilterProxyModel*>();
    ASSERT_NE(proxy, nullptr);
    proxy->setFilterKeyColumn(-1);
    proxy->setFilterFixedString("20");
    ASSERT_EQ(proxy->rowCount(), 1);
    ASSERT_TRUE(QMetaObject::invokeMethod(&view, "copyToClipboard",
                                          Qt::DirectConnection));
    const QString copied = QApplication::clipboard()->text();
    const auto nonEmptyLines = copied.split('\n', QtCompat::SkipEmptyParts);
    EXPECT_EQ(nonEmptyLines.size(), proxy->rowCount() + 1)
            << copied.toStdString();
}

TEST(ViewCapabilities, ChartComparisonsTransformValuesAndKeepQualifiedRows) {
    ccPointCloud source("comparison input");
    ASSERT_TRUE(source.reserve(3));
    for (int i = 0; i < 3; ++i) source.addPoint(CCVector3(i + 1, 10 + i, 0));
    vtkChartView view(vtkChartView::LINE_CHART);
    view.setEntity(&source);
    selectChartFields(view, {"Points_X"});
    for (auto* combo : view.findChildren<QComboBox*>())
        if (combo->findText("Points_X") >= 0)
            combo->setCurrentIndex(combo->findText("Points_X"));
    for (auto* check : view.findChildren<QCheckBox*>())
        if (check->text() == "Use Index for X Axis") check->setChecked(false);
    vtkChartView::ComparisonParameters a, b;
    a.label = "a";
    a.filters.insert("Points_X", {1, 2});
    b.label = "b";
    b.filters.insert("Points_X", {2, 3});
    b.xScale = 2;
    b.yScale = 3;
    b.opacity = 0.5;
    b.lineWidth = 4;
    view.setComparisons({a, b});
    auto* chart = renderedChart(view);
    ASSERT_NE(chart, nullptr);
    ASSERT_EQ(chart->GetNumberOfPlots(), 2);
    auto* first = chart->GetPlot(0)->GetInput();
    auto* second = chart->GetPlot(1)->GetInput();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(first->GetNumberOfRows(), 2);
    ASSERT_EQ(second->GetNumberOfRows(), 2);
    EXPECT_DOUBLE_EQ(first->GetValue(0, 0).ToDouble(), 1);
    EXPECT_DOUBLE_EQ(second->GetValue(0, 0).ToDouble(), 4);
    EXPECT_DOUBLE_EQ(second->GetValue(0, 1).ToDouble(), 6);
    auto* ids = vtkIdTypeArray::SafeDownCast(
            second->GetFieldData()->GetArray("__acv_original_ids"));
    ASSERT_NE(ids, nullptr);
    EXPECT_EQ(ids->GetValue(0), 1);
    EXPECT_EQ(ids->GetValue(1), 2);
    EXPECT_NEAR(chart->GetPlot(1)->GetPen()->GetWidth(), 4, 1e-6);
    EXPECT_NEAR(chart->GetPlot(1)->GetPen()->GetOpacity() / 255.0, 0.5, 0.01);
    b.yScale = std::numeric_limits<double>::quiet_NaN();
    view.setComparisons({b});
    EXPECT_EQ(chart->GetNumberOfPlots(), 2)
            << "invalid parameters leave the valid comparison intact";
    EXPECT_DOUBLE_EQ(source.getPoint(1)->x, 2);
}

TEST(ViewCapabilities, ComparativeGridTimeScalingAndOverlayShareTheChartOwner) {
    ccPointCloud source("temporal samples");
    ASSERT_TRUE(source.reserve(4));
    for (int i = 0; i < 4; ++i) source.addPoint(CCVector3(i + 1, 10 + i, 0));
    ASSERT_EQ(source.addScalarField("time"), 0);
    source.setCurrentScalarField(0);
    for (int i = 0; i < 4; ++i) source.setPointScalarValue(i, i / 2);
    using Comparative = vtkComparativeViewWidget;
    Comparative view(Comparative::LINE_CHART);
    view.setEntityListProvider([&]() { return QList<ccHObject*>{&source}; });
    view.setInitialEntity(&source);
    view.setupGrid();
    QPointer<QWidget> first = view.subWidgets().front();
    QPointer<QWidget> removed = view.subWidgets().back();
    auto* firstChart = qobject_cast<vtkChartView*>(first.data());
    ASSERT_NE(firstChart, nullptr);
    selectChartFields(*firstChart, {"Points_Y"});
    ASSERT_TRUE(view.setDimensions(1, 2));
    EXPECT_EQ(view.subWidgets().front(), first.data());
    EXPECT_TRUE(removed.isNull());
    EXPECT_EQ(renderedChart(*firstChart)->GetNumberOfPlots(), 1);
    Comparative::ParameterCue time, scale, width;
    time.property = Comparative::Time;
    time.direction = Comparative::AcrossColumns;
    time.field = "time";
    time.minimum = 0;
    time.maximum = 1;
    scale.property = Comparative::YScale;
    scale.minimum = 1;
    scale.maximum = 2;
    width.property = Comparative::LineWidth;
    width.minimum = 2;
    width.maximum = 4;
    ASSERT_TRUE(view.setCues({time, scale, width}));
    for (int cell = 0; cell < 2; ++cell) {
        auto* pane = qobject_cast<vtkChartView*>(view.subWidgets()[cell]);
        ASSERT_NE(pane, nullptr);
        selectChartFields(*pane, {"Points_Y"});
        auto* chart = renderedChart(*pane);
        ASSERT_EQ(chart->GetNumberOfPlots(), 1);
        auto* data = chart->GetPlot(0)->GetInput();
        ASSERT_EQ(data->GetNumberOfRows(), 2);
        EXPECT_DOUBLE_EQ(data->GetValue(0, 1).ToDouble(), cell == 0 ? 10 : 24);
        EXPECT_DOUBLE_EQ(chart->GetPlot(0)->GetPen()->GetWidth(),
                         cell == 0 ? 2 : 4);
    }
    view.setOverlayAllComparisons(true);
    EXPECT_TRUE(view.overlayAllComparisons());
    EXPECT_EQ(renderedChart(*firstChart)->GetNumberOfPlots(), 2);
    EXPECT_TRUE(view.subWidgets()[1]->isHidden());
    view.setOverlayAllComparisons(false);
    EXPECT_EQ(renderedChart(*firstChart)->GetNumberOfPlots(), 1);
    EXPECT_FALSE(view.subWidgets()[1]->isHidden());
    ASSERT_TRUE(view.setDimensions(2, 3));
    EXPECT_EQ(view.subWidgets().size(), 6);
    EXPECT_EQ(view.subWidgets().front(), first.data());
    EXPECT_DOUBLE_EQ(view.cueValue(0, 2, 0), 1);
    auto invalid = time;
    invalid.cellValues.insert(6, 0);
    EXPECT_FALSE(view.setCues({invalid}));
    ASSERT_TRUE(view.setCues({}));
    EXPECT_EQ(renderedChart(*firstChart)
                      ->GetPlot(0)
                      ->GetInput()
                      ->GetNumberOfRows(),
              4);
    EXPECT_EQ(source.size(), 4);
}

TEST(ViewCapabilities, OrthoCutsGenericVolumesAndDetachesReplacedInputs) {
    vtkNew<vtkImageData> image;
    image->SetDimensions(3, 3, 3);
    image->AllocateScalars(VTK_DOUBLE, 1);
    image->GetPointData()->GetScalars()->SetName("density");
    vtkNew<vtkDataSetMapper> mapper;
    mapper->SetInputData(image);
    mapper->SetScalarModeToUsePointFieldData();
    mapper->SelectColorArray("density");
    vtkNew<vtkLookupTable> lut;
    lut->SetRange(2, 9);
    mapper->SetLookupTable(lut);
    vtkNew<vtkPlane> clip;
    clip->SetOrigin(0.25, 0, 0);
    mapper->AddClippingPlane(clip);
    vtkNew<vtkActor> appearance;
    appearance->SetMapper(mapper);
    appearance->GetProperty()->SetOpacity(0.35);
    vtkNew<vtkRenderer> top, side, front, perspective;
    vtkRenderer* renderers[3] = {top, side, front};
    vtkOrthoSliceRepresentation representation;
    representation.setInputData(image, appearance);
    representation.setSliceOrigin(1, 1, 1);
    representation.addToRenderers(renderers, perspective);
    for (int axis = 0; axis < 3; ++axis) {
        auto plane = static_cast<vtkOrthoSliceRepresentation::PlaneIndex>(axis);
        ASSERT_TRUE(representation.hasValidSlice(plane));
        auto* actor = representation.sliceActor3D(plane);
        ASSERT_NE(actor, nullptr);
        actor->GetMapper()->Update();
        auto* cut = vtkPolyData::SafeDownCast(
                actor->GetMapper()->GetInputDataObject(0, 0));
        ASSERT_NE(cut, nullptr);
        EXPECT_GT(cut->GetNumberOfCells(), 0);
        EXPECT_DOUBLE_EQ(actor->GetProperty()->GetOpacity(), 0.35);
        EXPECT_NE(actor->GetMapper()->GetLookupTable(), lut.GetPointer());
        EXPECT_NE(actor->GetMapper()->GetClippingPlanes()->GetItem(0),
                  clip.GetPointer());
        const int coordinate = axis == 0 ? 1 : axis == 1 ? 0 : 2;
        EXPECT_NEAR(cut->GetBounds()[coordinate * 2], 1, 1e-6);
        EXPECT_NEAR(cut->GetBounds()[coordinate * 2 + 1], 1, 1e-6);
    }
    auto* surface = representation.surfaceActor3D();
    ASSERT_NE(surface, nullptr);
    surface->GetMapper()->Update();
    auto* shell = vtkPolyData::SafeDownCast(
            surface->GetMapper()->GetInputDataObject(0, 0));
    ASSERT_NE(shell, nullptr);
    EXPECT_NE(shell->GetPointData()->GetArray("vtkOriginalPointIds"), nullptr);
    EXPECT_NE(shell->GetCellData()->GetArray("vtkOriginalCellIds"), nullptr);
    representation.setInputData(nullptr);
    EXPECT_EQ(top->GetActors()->GetNumberOfItems(), 0);
    EXPECT_EQ(perspective->GetActors()->GetNumberOfItems(), 0);
    EXPECT_EQ(
            representation.sliceActor3D(vtkOrthoSliceRepresentation::PLANE_TOP),
            nullptr);
    representation.setInputData(image, appearance);
    EXPECT_GT(perspective->GetActors()->GetNumberOfItems(), 0);
    EXPECT_DOUBLE_EQ(appearance->GetProperty()->GetOpacity(), 0.35);
}

TEST(ViewCapabilities,
     OrthoNativeSnapshotsPreserveTransformsStylesAndRefreshScope) {
    ccPointCloud source("native source");
    source.setVisible(true);
    LayoutFixture fixture;
    auto* sourceView = fixture.newView();
    auto* vis = sourceView->getVisualizer3D();
    vtkNew<vtkCubeSource> cube;
    cube->Update();
    vtkSmartPointer<vtkPolyData> geometry = cube->GetOutput();
    vtkNew<vtkDoubleArray> values;
    values->SetName("signal");
    values->SetNumberOfTuples(geometry->GetNumberOfPoints());
    values->FillComponent(0, 7);
    geometry->GetPointData()->SetScalars(values);
    ASSERT_TRUE(vis->addPointCloud(geometry, source.getViewId().toStdString()));
    ecvRepresentationManager::instance().ensureRepresentation(&source,
                                                              sourceView);
    auto native = vis->representationActors(source.getViewId().toStdString());
    ASSERT_EQ(native.size(), 1);
    auto* actor = native.front().actor;
    actor->SetPosition(5, 0, 0);
    actor->SetScale(2, 1, 1);
    actor->GetProperty()->SetOpacity(0.4);
    auto copies =
            sourceView->createVisibleRepresentationActors({source.getViewId()});
    ASSERT_EQ(copies.size(), 1);
    EXPECT_NEAR(copies[0]->GetBounds()[0], 4, 1e-6);
    EXPECT_NEAR(copies[0]->GetBounds()[1], 6, 1e-6);
    EXPECT_DOUBLE_EQ(copies[0]->GetPosition()[0], 0);
    auto* copiedData = vtkDataSet::SafeDownCast(
            copies[0]->GetMapper()->GetInputDataObject(0, 0));
    ASSERT_NE(copiedData, nullptr);
    ASSERT_NE(copiedData->GetPointData()->GetScalars(), nullptr);
    copiedData->GetPointData()->GetScalars()->SetTuple1(0, 99);
    EXPECT_DOUBLE_EQ(values->GetValue(0), 7);
    vtkOrthoSliceViewWidget view;
    view.populateFromView(sourceView);
    auto* inherited = view.findChild<QCheckBox*>("orthoSourceAppearance");
    ASSERT_NE(inherited, nullptr);
    ASSERT_TRUE(inherited->isChecked());
    double origin[3];
    view.getSlicePosition(origin);
    EXPECT_NEAR(origin[0], 5, 1e-6);
    auto* refresh = view.findChild<QAbstractButton*>("orthoRefresh");
    ASSERT_NE(refresh, nullptr);
    refresh->click();
    EXPECT_TRUE(inherited->isChecked())
            << "Refresh is not a uniform-style edit";
    QComboBox* showing = nullptr;
    for (auto* combo : view.findChildren<QComboBox*>())
        if (combo->findText("Visible representations") >= 0) showing = combo;
    ASSERT_NE(showing, nullptr);
    showing->setCurrentIndex(showing->findText("None"));
    refresh->click();
    auto* actors = view.getRenderer(vtkOrthoSliceViewWidget::PERSPECTIVE_VIEW)
                           ->GetActors();
    actors->InitTraversal();
    while (auto* derived = actors->GetNextActor())
        EXPECT_NE(derived->GetProperty()->GetOpacity(), 0.4);
    view.populateFromView(nullptr);
    refresh->click();  // an unavailable native scope is never an entity pointer
    EXPECT_DOUBLE_EQ(actor->GetProperty()->GetOpacity(), 0.4);
    EXPECT_DOUBLE_EQ(actor->GetPosition()[0], 5);
}

TEST(ViewCapabilities,
     PythonCallbacksReceiveCompositeTypesSelectedArraysAndNativeImages) {
    if (QProcess::execute("python3", {"-c", "import vtkmodules, matplotlib"}) !=
        0)
        GTEST_SKIP() << "Python VTK and matplotlib are optional callback "
                        "dependencies";
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    vtkNew<vtkImageData> image;
    image->SetDimensions(3, 3, 3);
    image->AllocateScalars(VTK_DOUBLE, 1);
    image->GetPointData()->GetScalars()->SetName("keep");
    image->GetPointData()->GetScalars()->FillComponent(0, 5);
    vtkNew<vtkDoubleArray> drop;
    drop->SetName("drop");
    drop->SetNumberOfTuples(image->GetNumberOfPoints());
    image->GetPointData()->AddArray(drop);
    vtkNew<vtkTable> table;
    vtkNew<vtkDoubleArray> rows;
    rows->SetName("rows");
    rows->InsertNextValue(42);
    table->AddColumn(rows);
    vtkNew<vtkMultiBlockDataSet> composite;
    composite->SetBlock(0, image);
    composite->SetBlock(1, table);
    ecvPythonView view;
    view.setDataObjectProvider([&](ccHObject*) {
        return QList<vtkSmartPointer<vtkDataObject>>{composite};
    });
    auto* source = view.findChild<QComboBox*>("pythonSource");
    ASSERT_NE(source, nullptr);
    source->setCurrentIndex(source->findText("Visible representations"));
    auto* editor = view.findChild<ecvPythonCodeEditor*>();
    ASSERT_NE(editor, nullptr);
    auto run = [&]() {
        ASSERT_TRUE(QMetaObject::invokeMethod(&view, "onRunScript",
                                              Qt::DirectConnection));
        auto* process = view.findChild<QProcess*>();
        ASSERT_NE(process, nullptr);
        QEventLoop loop;
        QObject::connect(
                process,
                qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
                &loop, &QEventLoop::quit);
        QTimer::singleShot(10000, &loop, &QEventLoop::quit);
        loop.exec();
        pump();
        for (auto* output : view.findChildren<QPlainTextEdit*>())
            if (output != editor)
                EXPECT_FALSE(output->toPlainText().contains("Traceback"))
                        << output->toPlainText().toStdString();
    };
    const QString counter = temp.filePath("callbacks.txt");
    editor->setPlainText(QString(R"PY(
with open(r'%1', 'a') as f: f.write('top\n')
def setup_data(view):
    assert view.GetNumberOfVisibleDataObjects() == 1
    data = view.GetVisibleDataObjectForSetup(0)
    assert data.IsA('vtkMultiBlockDataSet')
    assert data.GetBlock(0).IsA('vtkImageData')
    assert data.GetBlock(1).IsA('vtkTable')
    assert view.GetNumberOfAttributeArrays(0, 0) == 2
    assert view.GetAttributeArrayStatus(0, 0, 'keep') == 0
    view.SetAttributeArrayStatus(0, 0, 'keep', 1)
    view.SetAttributeArrayStatus(0, 6, 'rows', 1)
    with open(r'%1', 'a') as f: f.write('setup\n')
def render(view, width, height):
    assert width > 0 and height > 0
    data = view.GetVisibleDataObjectForRendering(0)
    image = data.GetBlock(0)
    assert image.GetPointData().HasArray('keep')
    assert not image.GetPointData().HasArray('drop')
    assert data.GetBlock(1).GetRowData().HasArray('rows')
    image.GetPointData().GetArray('keep').SetTuple1(0, 99)
    assert view.GetVisibleDataObjectForSetup(0).GetBlock(0).GetPointData().GetArray('keep').GetTuple1(0) == 5
    import numpy as np
    pixels = np.zeros((2, 3, 4), dtype=np.uint8)
    pixels[0, :, 0] = 255
    pixels[1, :, 2] = 255
    pixels[:, :, 3] = 255
    with open(r'%1', 'a') as f: f.write('render\n')
    return python_view.numpy_to_image(pixels)
)PY")
                                 .arg(counter));
    run();
    QFile result(counter);
    ASSERT_TRUE(result.open(QIODevice::ReadOnly));
    EXPECT_EQ(result.readAll(), QByteArray("top\nsetup\nrender\n"));
    EXPECT_DOUBLE_EQ(image->GetPointData()->GetScalars()->GetTuple1(0), 5);
    bool redImage = false;
    for (auto* label : view.findChildren<QLabel*>()) {
        const auto pixmap = qtCompatLabelPixmap(label);
        if (!pixmap.isNull()) {
            const auto displayed = pixmap.toImage();
            redImage = redImage ||
                       (displayed.pixelColor(0, 0).blue() == 255 &&
                        displayed.pixelColor(0, displayed.height() - 1).red() ==
                                255);
        }
    }
    EXPECT_TRUE(redImage);
    editor->setPlainText(
            "def render(view, figure):\n"
            "    figure.add_subplot(111).plot([1, 2], [3, 4])\n");
    run();
    source->setCurrentIndex(source->findText("None"));
    editor->setPlainText(
            "def setup_data(view):\n"
            "    assert view.GetNumberOfVisibleDataObjects() == 0\n");
    run();
}

int main(int argc, char** argv) {
    // Match the application's VTK surface contract before Qt creates contexts.
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}