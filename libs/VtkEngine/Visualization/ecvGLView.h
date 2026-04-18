// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvGenericGLDisplay.h>
#include <ecvViewportParameters.h>

#include <QObject>
#include <memory>

#include "qVTK.h"

class ccHObject;
class QMainWindow;
class QVTKWidgetCustom;

namespace Visualization {
class VtkVis;
using VtkVisPtr = std::shared_ptr<VtkVis>;
}  // namespace Visualization

/// Per-window 3D view implementing ecvGenericGLDisplay backed by VTK.
///
/// Each ecvGLView owns an independent VTK rendering pipeline:
///   - QVTKWidgetCustom (Qt/VTK bridge widget)
///   - VtkVis (vtkRenderer + vtkRenderWindow + vtkInteractor)
///   - Viewport parameters (camera, perspective, zoom)
///   - Window-local DB root (m_winDBRoot)
///   - Reference to the shared global scene DB
///
/// Design references:
///   ParaView pqRenderView  — per-view vtkRenderWindow
///   CloudCompare ccGLWindow — per-window camera/viewport
///   MeshLab GLArea          — independent paintGL, shared document
class QVTK_ENGINE_LIB_API ecvGLView : public QObject,
                                      public ecvGenericGLDisplay {
    Q_OBJECT

public:
    /// Factory method — creates and initializes a new GL view.
    static ecvGLView* Create(QMainWindow* parent, bool stereoMode = false);

    ~ecvGLView() override;

    // -- ecvGenericGLDisplay implementation --

    int getUniqueID() const override { return m_uniqueID; }
    QString getTitle() const override { return m_title; }
    void redraw(bool only2D = false, bool forceRedraw = true) override;
    void refresh(bool only2D = false) override;
    void toBeRefreshed() override;
    const ecvViewportParameters& getViewportParameters() const override;
    void setViewportParameters(const ecvViewportParameters& params) override;
    void setPerspectiveState(bool state, bool objectCenteredView) override;
    bool perspectiveView() const override;
    bool objectCenteredView() const override;
    void setSceneDB(ccHObject* root) override;
    ccHObject* getSceneDB() override;
    ccHObject* getOwnDB() override;
    void addToOwnDB(ccHObject* obj, bool noDependency = true) override;
    void removeFromOwnDB(ccHObject* obj) override;
    QWidget* asWidget() override;
    const QWidget* asWidget() const override;
    bool hasOverriddenDisplayParameters() const override;
    void aboutToBeRemoved(ccDrawableObject* obj) override;

    // -- VTK-specific accessors (not exposed to CV_db layer) --

    QVTKWidgetCustom* getVtkWidget() const { return m_vtkWidget; }
    Visualization::VtkVis* getVisualizer3D() const;
    Visualization::VtkVisPtr getVisualizer3DSP() const {
        return m_visualizer3D;
    }

    /// Reset camera to fit all visible scene geometry.
    void zoomGlobal();

signals:
    void aboutToClose(ecvGLView* self);
    void viewActivated(ecvGLView* self);

protected:
    explicit ecvGLView(QMainWindow* parent);

private:
    void initVtkPipeline(QMainWindow* parent, bool stereoMode);

    int m_uniqueID;
    QString m_title;
    QVTKWidgetCustom* m_vtkWidget = nullptr;
    Visualization::VtkVisPtr m_visualizer3D;
    ecvViewportParameters m_viewportParams;
    ccHObject* m_globalDBRoot = nullptr;
    ccHObject* m_winDBRoot = nullptr;
    bool m_shouldBeRefreshed = false;

    static int s_nextWindowID;
};
