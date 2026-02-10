// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QTimer>

#ifdef _MSC_VER
/* The below is MANDATORY for Windows builds or you will take an exception in
 * vtkRenderer::SetRenderWindow(vtkRenderWindow *renwin) */
#include <vtkAutoInit.h>
/* VTK was built with vtkRenderingOpenGL2 */
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

#endif

#include "qPCL.h"

// CV_CORE_LIB
#include <CVGeom.h>

// CV_DB_LIB
#include <ecvColorTypes.h>

// VTK
// #include <QVTKWidget.h>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkVersionMacros.h>
#include <vtkDataSet.h>
#include <vtkLODActor.h>
#include <vtkPlanes.h>
#include <vtkPolyData.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>

// QPCL_ENGINE_LIB
#include <Utils/PCLCloud.h>

#include "ScaleBarWidget.h"

// SYSTEM
#include <assert.h>

class QMainWindow;
class ecvDisplayTools;

class vtkCamera;
class vtkRenderer;
class vtkRenderWindowInteractor;
class vtkLogoWidget;
class vtkLookupTable;
class vtkScalarBarWidget;
class vtkRenderWindowInteractor;
class vtkOrientationMarkerWidget;
class VtkWidgetPrivate;

//! Container widget for vtk
class QPCL_ENGINE_LIB_API QVTKWidgetCustom : public QVTKOpenGLNativeWidget {
    Q_OBJECT

public:
    explicit QVTKWidgetCustom(QMainWindow* parentWindow,
                              ecvDisplayTools* tools,
                              bool stereoMode = false);
    virtual ~QVTKWidgetCustom();

    inline vtkRenderer* getVtkRender() { return this->m_render; }
    inline vtkRenderWindowInteractor* getVtkInteractor() {
        return this->m_interactor;
    }
    void initVtk(vtkSmartPointer<vtkRenderWindowInteractor> interactor,
                 bool useVBO = false);

    void setMultiViewports(bool multi = true);
    bool multiViewports() const;

    void addActor(vtkProp* actor, const QColor& clr = Qt::black);
    void addViewProp(vtkProp* prop);
    QList<vtkProp*> actors() const;

    void setActorsVisible(bool visible);
    void setActorVisible(vtkProp* actor, bool visible);
    bool actorVisible(vtkProp* actor);

    void setBackgroundColor(const QColor& clr);
    QColor backgroundColor() const;

    vtkRenderer* defaultRenderer();
    bool defaultRendererTaken() const;

    void transformCameraView(const double* viewMat);
    void transformCameraProjection(const double* projMat);

    void updateScene();
    vtkRenderWindow* GetRenderWindow() { return this->renderWindow(); }
    void SetRenderWindow(vtkRenderWindow* win) {
        return this->setRenderWindow(win);
    }
    QVTKInteractor* GetInteractor() { return this->interactor(); }

protected:
    void setBounds(double* bounds);

    double xMin() const;
    double xMax() const;
    double yMin() const;
    double yMax() const;
    double zMin() const;
    double zMax() const;

public:
    void toWorldPoint(const CCVector3d& input2D, CCVector3d& output3D);
    void toWorldPoint(const CCVector3& input2D, CCVector3d& output3D);
    void toDisplayPoint(const CCVector3d& worldPos, CCVector3d& displayPos);
    void toDisplayPoint(const CCVector3& worldPos, CCVector3d& displayPos);
    void setCameraPosition(const CCVector3d& pos);
    void setCameraFocalPoint(const CCVector3d& pos);
    void setCameraViewUp(const CCVector3d& pos);

    void setBackgroundColor(const ecvColor::Rgbf& bkg1,
                            const ecvColor::Rgbf& bkg2,
                            bool gradient);

    vtkSmartPointer<vtkLookupTable> createLookupTable(double min, double max);

    QMainWindow* getWin() { return m_win; }

    void setScaleBarVisible(bool visible) {
        if (m_scaleBar) m_scaleBar->setVisible(visible);
    }

protected:
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 4, 0)
    // FIX: VTK 9.4+ glad loading race condition with Qt6.
    // Ensure glad is loaded before VTK uses GL functions in initializeGL().
    virtual void initializeGL() override;
#endif

    // events handling
    virtual bool event(QEvent* evt) override;
    virtual void wheelEvent(QWheelEvent* event) override;
    virtual void keyPressEvent(QKeyEvent* event) override;
    virtual void mouseMoveEvent(QMouseEvent* event) override;
    virtual void mousePressEvent(QMouseEvent* event) override;
    virtual void mouseReleaseEvent(QMouseEvent* event) override;
    virtual void mouseDoubleClickEvent(QMouseEvent* event) override;
    virtual void dragEnterEvent(QDragEnterEvent* event) override;
    virtual void dropEvent(QDropEvent* event) override;

    virtual void updateActivateditems(
            int x, int y, int dx, int dy, bool updatePosition = false);

protected:
    bool m_unclosable = true;
    bool m_useVBO = false;
    vtkRenderer* m_render;
    QMainWindow* m_win;
    ecvDisplayTools* m_tools;

    vtkDataObject* m_dataObject;
    vtkActor* m_modelActor = nullptr;
    vtkLODActor* m_filterActor = nullptr;

    QColor m_color1 = Qt::blue;
    QColor m_color2 = Qt::red;

    vtkSmartPointer<vtkPlanes> m_planes;

    vtkSmartPointer<vtkCamera> m_camera;

    vtkSmartPointer<vtkRendererCollection> m_renders;

    vtkSmartPointer<vtkRenderWindowInteractor> m_interactor;

    /** \brief Internal pointer to widget which contains a logo_Widget_ */
    vtkSmartPointer<vtkLogoWidget> m_logoWidget;

    /** \brief Internal pointer to widget which contains a scalarbar_Widget_ */
    vtkSmartPointer<vtkScalarBarWidget> m_scalarbarWidget;

    /** \brief Internal pointer to widget which contains a set of axes */
    vtkSmartPointer<vtkOrientationMarkerWidget> m_axesWidget;

    ScaleBarWidget* m_scaleBar = nullptr;

    VtkWidgetPrivate* d_ptr;

    // Timer for delayed 2D label update after wheel zoom
    QTimer* m_wheelZoomUpdateTimer = nullptr;
};
