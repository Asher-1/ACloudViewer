// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Links the production VTK library. No application CLI or model assets needed.
#include <Shortcuts/ecvKeySequences.h>
#include <VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h>
#include <VTKExtensions/Widgets/QVTKWidgetCustom.h>
#include <Visualization/VtkDisplayTools.h>
#include <Visualization/VtkVis.h>
#include <Visualization/vtkGLView.h>
#include <ecvViewManager.h>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCameraOrientationWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>

#include <QAction>
#include <QApplication>
#include <QMainWindow>
#include <QSettings>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>

#include "test_utility/QtEvents.h"

using cloudviewer::tests::pressSequence;

TEST(VtkShortcuts, ProductionOperationRemappingAndViewLifetime) {
    QTemporaryDir settings;
    QCoreApplication::setOrganizationName("ShortcutVtkIntegration");
    QCoreApplication::setApplicationName("Isolated");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       settings.path());
    QMainWindow window;
    QVTKWidgetCustom widget(&window, nullptr);
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    vtkNew<vtkRenderer> renderer;
    renderWindow->AddRenderer(renderer);
    widget.setRenderWindow(renderWindow);
    auto* interactor = widget.interactor();
    ASSERT_TRUE(interactor != nullptr) << "production widget interactor exists";
    interactor->SetEnableRender(false);
    vtkNew<VTKExtensions::vtkCustomInteractorStyle> style;
    interactor->SetInteractorStyle(style);
    widget.initVtk(interactor);
    widget.setCustomInteractorStyle(style);
    auto* action = widget.findChild<QAction*>("vtk_toggle_proj");
    ASSERT_TRUE(action && action->isEnabled())
            << "production widget registers projection command";
    ASSERT_TRUE(action->shortcutContext() == Qt::WidgetShortcut &&
                widget.actions().contains(action))
            << "VTK command associated with actual render widget";
    auto* camera = renderer->GetActiveCamera();
    const int original = camera->GetParallelProjection();
    action->trigger();
    ASSERT_TRUE(camera->GetParallelProjection() != original)
            << "QAction executes actual VTK projection operation";
    auto& registry = ecvKeySequences::instance();
    auto candidate = registry.bindings();
    candidate["vtk_toggle_proj"] = {QKeySequence("Ctrl+K, Ctrl+P"),
                                    QKeySequence("Meta+P")};
    QStringList errors;
    ASSERT_TRUE(registry.applyBindings(candidate, &errors, false))
            << "VTK chord and Meta configuration accepted";
    ASSERT_TRUE(action->shortcuts() == candidate.value("vtk_toggle_proj"))
            << "actual VTK QAction rebound";
    const int beforeOldKey = camera->GetParallelProjection();
    interactor->SetKeyEventInformation(1, 0, 'o', 0, "o");
    interactor->SetAltKey(1);
    interactor->KeyPressEvent();
    interactor->CharEvent();
    ASSERT_TRUE(camera->GetParallelProjection() == beforeOldKey)
            << "raw old VTK default does not execute after remap";
    action->trigger();
    ASSERT_TRUE(camera->GetParallelProjection() != beforeOldKey)
            << "rebound command retains same VTK operation";
    widget.setCustomInteractorStyle(nullptr);
    ASSERT_TRUE(!action->isEnabled())
            << "unavailable VTK style disables executable command";
    QVTKWidgetCustom second(&window, nullptr);
    auto* secondAction = second.findChild<QAction*>("vtk_toggle_proj");
    ASSERT_TRUE(secondAction &&
                secondAction->shortcuts() == candidate.value("vtk_toggle_proj"))
            << "new render widget consumes committed binding";
    ASSERT_TRUE(registry.conflicts(registry.bindings()).isEmpty())
            << "two render widgets do not falsely conflict";
    widget.setCustomInteractorStyle(style);
    if (qEnvironmentVariableIsSet("ACV_SHORTCUT_GRAPHICS_TEST")) {
        // A shown production widget needs its registered display owner; a
        // raw standalone widget intentionally has none during offscreen checks.
        auto& views = ecvViewManager::instance();
        views.initDisplayTools(new Visualization::VtkDisplayTools(), &window);
        auto* view = vtkGLView::Create(&window);
        views.setActiveView(view);
        auto* liveWidget = view->getVtkWidget();
        auto* liveCamera = liveWidget->getVtkRender()->GetActiveCamera();
        window.setCentralWidget(liveWidget);
        window.resize(640, 480);
        window.show();
        liveWidget->show();
        window.activateWindow();
        liveWidget->setFocus();
        for (int i = 0; i < 10; ++i) {
            QApplication::processEvents();
            QThread::msleep(10);
        }
        ASSERT_TRUE(liveWidget->hasFocus())
                << "actual render widget has keyboard focus";
        int projection = liveCamera->GetParallelProjection();
        pressSequence(liveWidget, QKeySequence("Ctrl+Alt+O"));
        ASSERT_TRUE(liveCamera->GetParallelProjection() == projection)
                << "old default key is silent in Qt dispatch";
        pressSequence(liveWidget, QKeySequence("Ctrl+K, Ctrl+P"));
        ASSERT_TRUE(liveCamera->GetParallelProjection() != projection)
                << "chord key executes actual VTK operation";
        projection = liveCamera->GetParallelProjection();
        pressSequence(liveWidget, QKeySequence("Meta+P"));
        ASSERT_TRUE(liveCamera->GetParallelProjection() != projection)
                << "Meta key executes actual VTK operation";
        window.takeCentralWidget();
        liveWidget->hide();
        delete view;
        views.releaseDisplayTools();
    }
}

// Clicking the camera orientation gizmo must preserve the user's zoom:
// VTK's EndSelectAction orients the camera then calls ResetCamera(), and the
// VtkVis observer restores focal/distance/parallel scale afterwards.
TEST(VtkShortcuts, OrientationGizmoObserverPreservesZoom) {
    auto& views = ecvViewManager::instance();
    const bool ownTools = (views.displayTools() == nullptr);
    QMainWindow window;
    if (ownTools) {
        views.initDisplayTools(new Visualization::VtkDisplayTools(), &window);
    }
    auto* view = vtkGLView::Create(&window);
    ASSERT_TRUE(view != nullptr);
    auto* vis = view->getVisualizer3D();
    ASSERT_TRUE(vis != nullptr);
    vis->ToggleCameraOrientationWidget(true);
    auto* gizmo = vis->getCameraOrientationWidget();
    ASSERT_TRUE(gizmo != nullptr);
    auto* ren = vis->getCurrentRenderer();
    ASSERT_TRUE(ren != nullptr);
    auto* cam = ren->GetActiveCamera();

    // Zoomed-in state around an arbitrary focal point.
    cam->SetParallelProjection(1);
    cam->SetFocalPoint(10, 20, 30);
    cam->SetPosition(10, 20, 35);  // distance 5
    cam->SetViewUp(0, 1, 0);
    cam->SetParallelScale(2.5);

    // Geometry far from the focal point so ResetCamera would refit.
    vtkNew<vtkSphereSource> sphere;
    sphere->SetRadius(50);
    sphere->SetCenter(-100, -100, -100);
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(sphere->GetOutputPort());
    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);
    ren->AddActor(actor);

    gizmo->InvokeEvent(vtkCommand::StartInteractionEvent);
    // Reproduce vtkCameraOrientationWidget::EndSelectAction: orient the
    // camera (focal preserving) then ResetCamera (the destructive refit).
    cam->SetPosition(15, 20, 30);  // look from +X, distance 5
    cam->SetViewUp(0, 1, 0);
    ren->ResetCamera();
    gizmo->InvokeEvent(vtkCommand::EndInteractionEvent);

    EXPECT_DOUBLE_EQ(cam->GetDistance(), 5.0);
    EXPECT_DOUBLE_EQ(cam->GetParallelScale(), 2.5);
    double f[3];
    cam->GetFocalPoint(f);
    EXPECT_DOUBLE_EQ(f[0], 10);
    EXPECT_DOUBLE_EQ(f[1], 20);
    EXPECT_DOUBLE_EQ(f[2], 30);
    double vpn[3];
    cam->GetViewPlaneNormal(vpn);
    EXPECT_NEAR(vpn[0], 1.0, 1e-6) << "new orientation is kept";

    ren->RemoveActor(actor);
    window.takeCentralWidget();
    view->asWidget()->hide();
    delete view;
    if (ownTools) views.releaseDisplayTools();
}

int main(int argc, char** argv) {
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}