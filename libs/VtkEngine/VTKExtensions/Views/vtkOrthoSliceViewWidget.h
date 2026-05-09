// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Orthographic Slice View Widget — single widget with 4 renderers.
// Modeled after ParaView's vtkPVOrthographicSliceView pattern:
//   Top-left:     Top View     (XZ plane, camera looking -Y)
//   Top-right:    Right Side   (YZ plane, camera looking -X)
//   Bottom-left:  Front View   (XY plane, camera looking -Z)
//   Bottom-right: 3D Perspective
// All 4 renderers share a single vtkRenderWindow via SetViewport().
// ----------------------------------------------------------------------------

#pragma once

#include "qVTK.h"

#include <QWidget>

class QVTKOpenGLNativeWidget;
class vtkRenderer;
class vtkGenericOpenGLRenderWindow;
class vtkProp;

class QVTK_ENGINE_LIB_API vtkOrthoSliceViewWidget : public QWidget {
    Q_OBJECT

public:
    explicit vtkOrthoSliceViewWidget(QWidget* parent = nullptr);
    ~vtkOrthoSliceViewWidget() override;

    QString title() const { return tr("Orthographic Slice View"); }

    vtkRenderer* getRenderer(int index) const;
    vtkRenderer* mainRenderer() const;
    QVTKOpenGLNativeWidget* vtkWidget() const;

    enum ViewIndex {
        TOP_VIEW = 0,
        SIDE_VIEW = 1,
        FRONT_VIEW = 2,
        PERSPECTIVE_VIEW = 3
    };

    void setSlicePosition(double x, double y, double z);
    void getSlicePosition(double pos[3]) const;
    void resetCameras();
    void render();
    void addActorToAll(vtkProp* actor);
    void setGeometryBounds(const double bounds[6]);

    void setSliceStep(double step) { m_sliceStep = step; }
    double sliceStep() const { return m_sliceStep; }

    void setSliceIncrement(int axis, double step);
    double sliceIncrement(int axis) const;

    bool annotationsVisible() const { return m_annotationsVisible; }
    void setAnnotationsVisible(bool visible);

signals:
    void slicePositionChanged(double x, double y, double z);

protected:
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;

private:
    int hitTestViewIndex(const QPoint& pos) const;

    struct Impl;
    Impl* d;
    double m_sliceStep = 0.1;
    double m_sliceIncrements[3] = {0.1, 0.1, 0.1};
    bool m_annotationsVisible = true;
};
