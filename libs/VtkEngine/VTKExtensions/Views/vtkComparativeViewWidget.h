// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// ParaView-aligned Comparative View Widget.
// Implements a single widget with a 2x2 grid of sub-viewports,
// modeled after pqComparativeRenderView / pqComparativeContextView.
// ----------------------------------------------------------------------------

#pragma once

#include "qVTK.h"

#include <QHash>
#include <QList>
#include <QMetaObject>
#include <QPointer>
#include <QSet>
#include <QWidget>

#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkSmartPointer.h>
#include <functional>
#include <vector>

class QCheckBox;
class QDoubleSpinBox;
class QGridLayout;
class QSpinBox;
class QComboBox;
class QLabel;
class ccHObject;
class vtkGLView;
class vtkChartView;
class vtkCallbackCommand;
class vtkObject;

class QVTK_ENGINE_LIB_API vtkComparativeViewWidget : public QWidget {
    Q_OBJECT

public:
    enum ComparativeType {
        RENDER,
        LINE_CHART,
        BAR_CHART,
    };

    explicit vtkComparativeViewWidget(ComparativeType type,
                                      QWidget* parent = nullptr);
    ~vtkComparativeViewWidget() override;

    QString title() const;
    ComparativeType comparativeType() const { return m_type; }

    using RenderViewFactory = std::function<vtkGLView*()>;
    void setRenderViewFactory(RenderViewFactory factory);

    using SubViewInitCallback = std::function<void(vtkGLView*)>;
    void setSubViewInitCallback(SubViewInitCallback cb);

    /// Called for each sub-view immediately before synchronous teardown (e.g. remove MainWindow event filters).
    using SubViewShutdownHook = std::function<void(vtkGLView*)>;
    void setSubViewShutdownHook(SubViewShutdownHook hook);

    void refreshSubViews();
    /// Tear down sub-views, VTK observers, and signal hooks before widget destruction.
    /// Safe to call multiple times (e.g. app exit and ~dtor).
    void shutdown();
    void setSourceView(vtkGLView* src);
    void connectExternalHighlighter(QObject* highlighter);
    void disconnectExternalHighlighter();

    using EntityListProvider = std::function<QList<ccHObject*>()>;
    void setEntityListProvider(EntityListProvider provider);
    void setInitialEntity(ccHObject* entity);

    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    void setDimensions(int rows, int cols);
    void setSpacing(int spacing);
    int spacing() const { return m_spacing; }

    QList<QWidget*> subWidgets() const { return m_subWidgets; }
    QList<vtkGLView*> subViews() const { return m_subViews; }
    vtkGLView* activeSubView() const { return m_activeSubView; }

    void setupGrid();

protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;

signals:
    void subViewCreated(QWidget* subWidget);
    void clicked();

private slots:
    void onDimensionChanged();
    void onCueParameterChanged(int index);
    void onPlayCue();
    void onToggleOverlay(bool checked);
    void onExportScreenshot();

private:
    void createRenderSubViews();
    void createChartSubViews();
    void buildToolbar();
    void refreshEntityCombo();
    void applyCueToSubViews();
    void syncCamerasFromFirst();
    void syncPivotFromFirst();
    void syncPivotFromView(int srcIdx);
    void syncRepresentationsFromFirst();
    void copyActorsAcrossSubViews();
    void scheduleSubViewRefresh(bool forceSceneDirty = false);
    void performSubViewRefresh();
    void installCameraLink();
    void removeCameraLink();
    void onSubViewInteraction(int viewIdx, bool renderOthers = true);
    static void interactionCallback(
            vtkObject* caller, unsigned long eid, void* clientData, void*);
    static void renderEndCallback(
            vtkObject* caller, unsigned long eid, void* clientData, void*);
    void forceRenderAllSubViews();
    void startInteractionTimer();
    void stopInteractionTimer();
    void scheduleCameraSyncRender();
    void performCameraSyncRender();
    void setInteractiveLOD(bool enable);
    void syncInteractionModeToSubViews();
    void syncPickingModeToSubViews();
    void syncCameraFromSourceView();
    void clearHighlightClones();

    ComparativeType m_type;
    QString m_viewTypeKey;
    QString m_title;
    int m_rows = 2;
    int m_cols = 2;
    int m_spacing = 2;
    QWidget* m_toolbar = nullptr;
    QSpinBox* m_rowSpin = nullptr;
    QSpinBox* m_colSpin = nullptr;
    QComboBox* m_cueParamCombo = nullptr;
    QComboBox* m_cueModeCombo = nullptr;
    QDoubleSpinBox* m_cueMinSpin = nullptr;
    QDoubleSpinBox* m_cueMaxSpin = nullptr;
    QLabel* m_statusLabel = nullptr;
    QCheckBox* m_overlayCheck = nullptr;
    QCheckBox* m_syncCamCheck = nullptr;
    QComboBox* m_entityCombo = nullptr;
    bool m_overlayMode = false;
    QGridLayout* m_gridLayout = nullptr;
    QList<QWidget*> m_subWidgets;
    QList<vtkGLView*> m_subViews;
    RenderViewFactory m_renderFactory;
    SubViewInitCallback m_subViewInitCb;
    SubViewShutdownHook m_subViewShutdownHook;
    EntityListProvider m_entityListProvider;
    ccHObject* m_initialEntity = nullptr;
    QPointer<vtkGLView> m_sourceView;
    bool m_cameraLinkEnabled = true;
    bool m_syncingCameras = false;
    bool m_firstShowDone = false;
    bool m_closing = false;
    bool m_shutdownDone = false;
    QSet<QWidget*> m_pendingFirstResize;
    QTimer* m_subViewRefreshTimer = nullptr;
    bool m_subViewRefreshForceDirty = false;
    bool m_interacting = false;
    int m_interactionSourceIdx = -1;
    QTimer* m_cameraSyncRenderTimer = nullptr;
    bool m_needsCameraReset = false;
    int m_cameraSyncSourceIdx = 0;

    struct InteractorObserver {
        vtkObject* observed = nullptr;
        unsigned long tag = 0;
        vtkSmartPointer<vtkCallbackCommand> callback;
    };
    std::vector<InteractorObserver> m_cameraObservers;

    QHash<vtkActor*, QList<vtkSmartPointer<vtkActor>>> m_highlightClonesBySource;

    vtkGLView* m_activeSubView = nullptr;

    QMetaObject::Connection m_hlActorAddedConn;
    QMetaObject::Connection m_hlActorRemovedConn;
    QMetaObject::Connection m_hlClearedConn;
    QMetaObject::Connection m_hlOverlayConn;
    QMetaObject::Connection m_hlSelectionFinishedConn;

    struct CameraState {
        double position[3] = {0,0,1};
        double focalPoint[3] = {0,0,0};
        double viewUp[3] = {0,1,0};
        double viewAngle = 30;
        double parallelScale = 1;
        double clippingRange[2] = {0.01, 1000};
        bool valid = false;
    };
    CameraState m_baselineCamera;
    void saveBaselineCamera();
    void restoreBaselineCamera(vtkGLView* view);
};
