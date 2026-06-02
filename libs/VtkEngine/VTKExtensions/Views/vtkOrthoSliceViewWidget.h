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

#include <Tools/SelectionTools/cvSelectionHighlighter.h>

#include <QHash>
#include <QList>
#include <QRubberBand>

#include <vtkActor.h>
#include <vtkSmartPointer.h>
#include <QSet>
#include <QWidget>
#include <QEvent>
#include <functional>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QMouseEvent;
class QSlider;
class QSpinBox;
class QVTKOpenGLNativeWidget;
class vtkRenderer;
class vtkGenericOpenGLRenderWindow;
class vtkProp;
class vtkCameraOrientationWidget;
class ccHObject;
class vtkPlaneSource;
class vtkPlane;

class QVTK_ENGINE_LIB_API vtkOrthoSliceViewWidget : public QWidget {
    Q_OBJECT

public:
    explicit vtkOrthoSliceViewWidget(QWidget* parent = nullptr);
    ~vtkOrthoSliceViewWidget() override;

    QString title() const { return m_title; }

    using EntityListProvider = std::function<QList<ccHObject*>()>;
    void setEntityListProvider(EntityListProvider provider);

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

    void set3DProjection(bool perspective);
    void zoomToFit();
    void setViewPreset(int presetIndex);
    void setSelectionMode(int mode);

    void setSliceIncrement(int axis, double step);
    double sliceIncrement(int axis) const;

    bool annotationsVisible() const { return m_annotationsVisible; }
    void setAnnotationsVisible(bool visible);

    void populateFromRenderer(vtkRenderer* sourceRenderer);

    void connectExternalHighlighter(QObject* highlighter);

    void setOrientationMarkerVisible(bool visible);
    bool orientationMarkerVisible() const;
    void toggleCameraOrientationWidget(bool visible);
    bool isCameraOrientationWidgetShown() const;

signals:
    void slicePositionChanged(double x, double y, double z);
    void clicked();

protected:
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void onSourceComboChanged(int index);
    void onSourceComboAboutToShow();

private:
    int hitTestViewIndex(const QPoint& pos) const;
    void refreshSourceCombo();

    struct Impl;
    Impl* d;
    void loadEntityIntoView(ccHObject* entity);
    void loadEntitiesIntoView(const QList<ccHObject*>& entities);
    void updateSliceSpinners();

    void applyDisplayProperties();
    void performRubberBandSelection();
    void updateOrientationWidgetViewport();
    void ensureOrientationWidgetsInitialized();
    void clearEntityDisplay();
    void setupDecoratorBarContextMenu();
    void activate3DInteractor();
    bool forwardMouseToInteractor(QMouseEvent* me, QEvent::Type eventType);
    bool isInCameraOrientationWidget(const QPoint& pos) const;
    void mapWidgetToRendererDisplay(const QPoint& pos, vtkRenderer* ren,
                                    double outXY[2]) const;
    void createPlaneIndicators(const double bounds[6]);
    void updatePlaneIndicatorFromSlice(int viewIdx, vtkPlane* plane,
                                       vtkPlaneSource* ps,
                                       const double bounds[6]);
    void updateOutlineBounds();
    void createOutlineActor();
    void disconnectExternalHighlighter();

    QWidget* m_decoratorBar = nullptr;
    QWidget* m_dispBar = nullptr;
    QWidget* m_colorBar = nullptr;
    QWidget* m_lightBar = nullptr;
    QComboBox* m_sourceCombo = nullptr;
    QDoubleSpinBox* m_sliceSpin[3] = {nullptr, nullptr, nullptr};
    QDoubleSpinBox* m_stepSpin = nullptr;
    QCheckBox* m_annotCheck = nullptr;
    QLabel* m_statusLabel = nullptr;
    EntityListProvider m_entityListProvider;
    double m_sliceStep = 1.0;
    double m_sliceIncrements[3] = {1.0, 1.0, 1.0};
    bool m_annotationsVisible = true;
    bool m_axesGridVisible = false;
    bool m_updatingSpinners = false;

    bool m_draggingSlice = false;
    int m_dragViewIdx = -1;
    int m_lastActiveQuadrant = -1;
    int m_interactionOriginView = -1;
    bool m_rotating3D = false;
    bool m_panning2D = false;
    bool m_panPending = false;
    bool m_zooming2D = false;
    int m_zoomViewIdx = -1;
    bool m_interactorDisabledFor2D = false;
    bool m_cameraWidgetCapturing = false;
    bool m_rolling2D = false;
    int m_rollViewIdx = -1;
    QPoint m_rollLastPos;
    QPoint m_panPressPos;
    QPoint m_panLastPos;
    QPoint m_dragLastPos;
    QPoint m_zoomLastPos;
    double m_zoomAnchorDisplay[2] = {0.0, 0.0};
    double m_zoomScale2D = 0.0;

    enum SelectionMode {
        SEL_NONE = 0,
        SEL_POINTS,
        SEL_CELLS,
        SEL_RUBBER_POINTS,
        SEL_RUBBER_CELLS
    };
    SelectionMode m_selectionMode = SEL_NONE;

    bool m_rubberBandActive = false;
    QPoint m_rubberBandStart;
    QPoint m_rubberBandEnd;
    QSet<unsigned> m_selectedIndices;
    QRubberBand* m_rubberBandWidget = nullptr;

    QComboBox* m_reprCombo = nullptr;
    QSlider* m_opacitySlider = nullptr;
    QLabel* m_opacityLabel = nullptr;
    QSpinBox* m_pointSizeSpin = nullptr;
    QDoubleSpinBox* m_lineWidthSpin = nullptr;

    QComboBox* m_coloringCombo = nullptr;
    QCheckBox* m_mapScalarsCheck = nullptr;
    QCheckBox* m_interpScalarsCheck = nullptr;
    QCheckBox* m_renderTubesCheck = nullptr;
    QCheckBox* m_renderSpheresCheck = nullptr;
    QCheckBox* m_showOutlineCheck = nullptr;
    int m_lastAppliedReprIdx = -1;
    QCheckBox* m_disableLightingCheck = nullptr;
    QSlider* m_diffuseSlider = nullptr;
    QLabel* m_diffuseLabel = nullptr;
    QComboBox* m_interpCombo = nullptr;
    QSlider* m_specularSlider = nullptr;
    QLabel* m_specularLabel = nullptr;
    QSpinBox* m_specPowerSpin = nullptr;
    QDoubleSpinBox* m_luminositySpin = nullptr;
    QCheckBox* m_specColorCheck = nullptr;
    QCheckBox* m_useNanColorCheck = nullptr;

    QString m_viewTypeKey;
    QString m_title;

    QHash<vtkActor*, vtkSmartPointer<vtkActor>> m_externalHighlightClones;

    QMetaObject::Connection m_hlActorAddedConn;
    QMetaObject::Connection m_hlActorRemovedConn;
    QMetaObject::Connection m_hlClearedConn;
    QMetaObject::Connection m_hlOverlayConn;

    bool m_externalHighlighterLinked = false;
    int m_selectionOverlayKind =
            cvSelectionHighlighter::SelectionOverlayNone;
};
