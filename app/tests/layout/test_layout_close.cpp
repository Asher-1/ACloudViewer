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

#include <Tools/SelectionTools/cvSelectionData.h>
#include <Tools/SelectionTools/cvViewSelectionManager.h>
#include <VTKExtensions/Views/vtkChartView.h>
#include <VTKExtensions/Views/vtkComparativeViewWidget.h>
#include <VTKExtensions/Widgets/QVTKWidgetCustom.h>
#include <Visualization/VtkDisplayTools.h>
#include <Visualization/VtkVis.h>
#include <Visualization/vtkGLView.h>
#include <ecvPointCloud.h>
#include <ecvRepresentationManager.h>
#include <ecvViewLayoutProxy.h>
#include <ecvViewManager.h>
#include <gtest/gtest.h>
#include <vtkActorCollection.h>
#include <vtkAnnotationLink.h>
#include <vtkAxis.h>
#include <vtkChart.h>
#include <vtkCommand.h>
#include <vtkContextActor.h>
#include <vtkContextScene.h>
#include <vtkCubeSource.h>
#include <vtkIdTypeArray.h>
#include <vtkMapper.h>
#include <vtkNew.h>
#include <vtkPlane.h>
#include <vtkPlaneCollection.h>
#include <vtkPlot.h>
#include <vtkPropCollection.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkTable.h>

#include <QApplication>
#include <QComboBox>
#include <QCoreApplication>
#include <QEventLoop>
#include <QFile>
#include <QFrame>
#include <QHBoxLayout>
#include <QListWidget>
#include <QMainWindow>
#include <QPointer>
#include <QProcess>
#include <QSettings>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <set>

#include "MainWindow.h"
#include "ecvMultiViewWidget.h"
#include "ecvPythonCodeEditor.h"
#include "ecvPythonView.h"
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

vtkChart* renderedChart(vtkChartView& view) {
    auto* widget = view.findChild<QVTKOpenGLNativeWidget*>();
    if (!widget) return nullptr;
    auto* renderers = widget->renderWindow()->GetRenderers();
    renderers->InitTraversal();
    while (auto* renderer = renderers->GetNextItem()) {
        auto* props = renderer->GetViewProps();
        props->InitTraversal();
        while (auto* prop = props->GetNextProp()) {
            if (auto* actor = vtkContextActor::SafeDownCast(prop))
                return vtkChart::SafeDownCast(actor->GetScene()->GetItem(0));
        }
    }
    return nullptr;
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

TEST(ComparativeView, ChartSourceIsExclusiveAndInitialAndRemovedSourcesApply) {
    ccPointCloud sourceA("source A"), sourceB("source B");
    sourceA.reserve(2);
    sourceB.reserve(3);
    sourceA.addPoint(CCVector3(0, 1, 2));
    sourceA.addPoint(CCVector3(1, 2, 3));
    sourceB.addPoint(CCVector3(2, 3, 4));
    sourceB.addPoint(CCVector3(3, 4, 5));
    sourceB.addPoint(CCVector3(4, 5, 6));
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
        EXPECT_EQ(model->item(1)->checkState(), Qt::Checked);
        model->item(2)->setCheckState(Qt::Checked);
        EXPECT_EQ(model->item(1)->checkState(), Qt::Unchecked);
        EXPECT_EQ(model->item(0)->checkState(), Qt::Unchecked);
        expectSource(&sourceB);
        model->item(2)->setCheckState(Qt::Unchecked);
        EXPECT_EQ(model->item(0)->checkState(), Qt::Checked);
        expectSource(nullptr);
        model->item(1)->setCheckState(Qt::Checked);
        sources.removeOne(&sourceA);
        view.setEntityListProvider([&sources]() { return sources; });
        expectSource(nullptr);
        EXPECT_EQ(model->item(0)->checkState(), Qt::Checked);
    }
}

int main(int argc, char** argv) {
    // Match the application's VTK surface contract before Qt creates contexts.
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}