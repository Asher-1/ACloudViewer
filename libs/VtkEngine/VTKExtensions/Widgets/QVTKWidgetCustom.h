// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file QVTKWidgetCustom.h
 * @brief Custom QVTKOpenGLNativeWidget with multi-viewport, actor management,
 *        and camera control for embedding VTK rendering in Qt.
 */

#include <vtkAutoInit.h>

#include <QTimer>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);
VTK_MODULE_INIT(vtkRenderingFreeType);
VTK_MODULE_INIT(vtkInteractionStyle);

#include "qVTK.h"

// CV_CORE_LIB
#include <CVGeom.h>

// CV_DB_LIB
#include <ecvColorTypes.h>

// VTK
// #include <QVTKWidget.h>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkDataSet.h>
#include <vtkLODActor.h>
#include <vtkPlanes.h>
#include <vtkPolyData.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>

#include "ScaleBarWidget.h"

// SYSTEM
#include <assert.h>

/**
 * @file QVTKWidgetCustom.h
 * @brief Custom QVTK widget with multi-viewport, scale bar, and coordinate
 * conversion.
 */

class QMainWindow;
class ccPolyline;
class ecvGLView;
class ecvDisplayTools;
namespace Visualization {
class ImageVis;
}
#include <ecvDisplayTypes.h>
#include <ecvViewContext.h>

namespace VTKExtensions {
class vtkCustomInteractorStyle;
}

class vtkCamera;
class vtkRenderer;
class vtkRenderWindowInteractor;
class vtkLogoWidget;
class vtkLookupTable;
class vtkScalarBarWidget;
class vtkRenderWindowInteractor;
class vtkOrientationMarkerWidget;
class VtkWidgetPrivate;

/**
 * @class QVTKWidgetCustom
 * @brief Custom VTK widget with multi-viewport, scale bar, and CloudViewer
 * integration.
 */
class QVTK_ENGINE_LIB_API QVTKWidgetCustom : public QVTKOpenGLNativeWidget {
    Q_OBJECT

public:
    /// @param parentWindow Parent main window
    /// @param tools Display tools
    /// @param stereoMode Whether to enable stereo rendering
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

    void setCustomInteractorStyle(
            VTKExtensions::vtkCustomInteractorStyle* style) {
        m_customStyle = style;
    }

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
    /// @param input2D 2D display coordinates
    /// @param output3D Output 3D world coordinates
    void toWorldPoint(const CCVector3d& input2D, CCVector3d& output3D);
    void toWorldPoint(const CCVector3& input2D, CCVector3d& output3D);
    /// @param worldPos 3D world coordinates
    /// @param displayPos Output 2D display coordinates
    void toDisplayPoint(const CCVector3d& worldPos, CCVector3d& displayPos);
    void toDisplayPoint(const CCVector3& worldPos, CCVector3d& displayPos);
    /// @param pos Camera position
    void setCameraPosition(const CCVector3d& pos);
    /// @param pos Camera focal point
    void setCameraFocalPoint(const CCVector3d& pos);
    /// @param pos Camera view-up vector
    void setCameraViewUp(const CCVector3d& pos);

    /// @param bkg1 Top gradient color
    /// @param bkg2 Bottom gradient color
    /// @param gradient Whether to use gradient
    void setBackgroundColor(const ecvColor::Rgbf& bkg1,
                            const ecvColor::Rgbf& bkg2,
                            bool gradient);

    /// @param min Minimum scalar value
    /// @param max Maximum scalar value
    /// @return Lookup table
    vtkSmartPointer<vtkLookupTable> createLookupTable(double min, double max);

    /// @return Parent main window
    QMainWindow* getWin() { return m_win; }

    /// @param visible Whether to show the scale bar
    void setScaleBarVisible(bool visible) {
        if (m_scaleBar) {
            if (visible && !isVisible()) return;
            m_scaleBar->setVisible(visible);
            if (visible && m_scaleBar->isLayoutReady()) {
                m_scaleBar->update(m_render, m_interactor);
            }
        }
    }

protected:
    // events handling
    virtual bool event(QEvent* evt) override;
    void paintGL() override;
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

public:
    ecvHotZone* localHotZone() const { return m_localHotZone; }
    void setLocalHotZone(ecvHotZone* hz) { m_localHotZone = hz; }
    bool localClickableItemsVisible() const { return m_localClickableVisible; }

    /// Phase C: per-view owner.  Set to the ecvGLView that created this
    /// widget.  nullptr for the primary (singleton) widget.
    void setOwnerView(ecvGLView* view) { m_ownerView = view; }
    ecvGLView* ownerView() const { return m_ownerView; }

    /// Resolve the ecvGenericGLDisplay associated with this widget.
    ecvGenericGLDisplay* resolveDisplay() const;

    /// Return the best display for per-view method calls.
    /// Prefers resolveDisplay(); falls back to m_tools.
    ecvGenericGLDisplay* displayTarget() const;

    /// Forward this widget's input signals to the given ecvDisplayTools.
    void connectSignalsTo(ecvDisplayTools* target);

    /// Per-view 2D overlay (lazy-created for secondary views).
    std::shared_ptr<Visualization::ImageVis> localImageVis() const {
        return m_localImageVis;
    }
    void setLocalImageVis(std::shared_ptr<Visualization::ImageVis> vis) {
        m_localImageVis = vis;
    }

    /// Per-view default point size (independent of global singleton value).
    float localDefaultPointSize() const { return m_localDefaultPointSize; }
    void setLocalDefaultPointSize(float s) { m_localDefaultPointSize = s; }

    /// Per-view default line width (independent of global singleton value).
    float localDefaultLineWidth() const { return m_localDefaultLineWidth; }
    void setLocalDefaultLineWidth(float w) { m_localDefaultLineWidth = w; }

    // ================================================================
    // Per-view state accessors
    //
    // curCtx() returns the canonical context for this widget:
    //   secondary views → m_ownerView->viewContext()
    //   primary view    → ecvViewManager::resolveViewContext()
    //
    // All cur*() helpers delegate to curCtx() so the branch exists
    // in exactly one place.
    // ================================================================

    ecvViewContext& curCtx();
    const ecvViewContext& curCtx() const;
    ecvViewContext* ownerCtx();

    ecvGenericGLDisplay::INTERACTION_FLAGS& curInteractionFlags() {
        return curCtx().interactionFlags;
    }
    ecvViewportParameters& curViewportParams() {
        return curCtx().viewportParams;
    }
    const ecvViewportParameters& curViewportParams() const {
        return curCtx().viewportParams;
    }
    QPoint& curLastMousePos() { return curCtx().lastMousePos; }
    QPoint& curLastMouseMovePos() { return curCtx().lastMouseMovePos; }
    bool& curMouseMoved() { return curCtx().mouseMoved; }
    bool& curMouseButtonPressed() { return curCtx().mouseButtonPressed; }
    bool& curIgnoreMouseReleaseEvent() {
        return curCtx().ignoreMouseReleaseEvent;
    }
    bool& curWidgetClicked() { return curCtx().widgetClicked; }
    ecvGenericGLDisplay::PICKING_MODE& curPickingMode() {
        return curCtx().pickingMode;
    }
    bool& curPickingModeLocked() { return curCtx().pickingModeLocked; }
    int& curPickRadius() { return curCtx().pickRadius; }
    bool& curAllowRectangularEntityPicking() {
        return curCtx().allowRectangularEntityPicking;
    }
    int& curLastPointIndex() { return curCtx().lastPointIndex; }
    QString& curLastPickedId() { return curCtx().lastPickedId; }
    bool& curTouchInProgress() { return curCtx().touchInProgress; }
    qreal& curTouchBaseDist() { return curCtx().touchBaseDist; }
    bool& curClickableItemsVisible() { return curCtx().clickableItemsVisible; }
    bool& curBubbleViewModeEnabled() { return curCtx().bubbleViewModeEnabled; }
    float& curBubbleViewFov_deg() { return curCtx().bubbleViewFov_deg; }
    bool& curCustomLightEnabled() { return curCtx().customLightEnabled; }
    float* curCustomLightPos() { return curCtx().customLightPos; }
    bool& curRotationAxisLocked() { return curCtx().rotationAxisLocked; }
    CCVector3d& curLockedRotationAxis() { return curCtx().lockedRotationAxis; }
    ecvGenericGLDisplay::PivotVisibility& curPivotVisibility() {
        return curCtx().pivotVisibility;
    }
    bool& curPivotSymbolShown() { return curCtx().pivotSymbolShown; }
    bool& curAutoPickPivotAtCenter() { return curCtx().autoPickPivotAtCenter; }
    bool& curShowCursorCoordinates() { return curCtx().showCursorCoordinates; }
    qint64& curLastClickTime() { return curCtx().lastClickTime_ticks; }

    ccPolyline*& curRectPickingPoly();
    std::list<ccInteractor*>& curActiveItems();
    ecvHotZone*& curHotZone();

signals:
    void rightButtonClicked(int x, int y);
    void leftButtonClicked(int x, int y);
    void doubleButtonClicked(int x, int y);
    void mouseWheelChanged(QWheelEvent* event);
    void mouseWheelRotated(float wheelDelta_deg);
    void mousePosChanged(const QPoint& pos);
    void mouseMoved(int x, int y, Qt::MouseButtons buttons);
    void translation(const CCVector3d& t);
    void rotation(const ccGLMatrixd& rotMat);
    void viewMatRotated(const ccGLMatrixd& rotMat);
    void buttonReleased();
    void filesDropped(const QStringList& filenames, bool displayDialog);
    void exclusiveFullScreenToggled(bool exclusive);
    void cameraParamChanged();
    void labelmove2D(int x, int y, int dx, int dy);

protected:
    bool m_unclosable = true;
    bool m_useVBO = false;
    vtkRenderer* m_render;
    QMainWindow* m_win;
    ecvDisplayTools* m_tools;
    ecvGLView* m_ownerView = nullptr;
    ecvHotZone* m_localHotZone = nullptr;
    bool m_localClickableVisible = false;
    std::shared_ptr<Visualization::ImageVis> m_localImageVis;
    float m_localDefaultPointSize = 1.0f;
    float m_localDefaultLineWidth = 2.0f;

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

    VTKExtensions::vtkCustomInteractorStyle* m_customStyle = nullptr;

    // Timer for delayed 2D label update after wheel zoom
    QTimer* m_wheelZoomUpdateTimer = nullptr;

    // True when a cc2DLabel was directly hit in mousePressEvent
    bool m_labelClickedOnPress = false;
    // True when right-click landed on a cc2DLabel (collapse toggle)
    bool m_rightClickOnLabel = false;
};
