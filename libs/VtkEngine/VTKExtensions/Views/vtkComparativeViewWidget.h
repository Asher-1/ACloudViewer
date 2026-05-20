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

#include <QSet>
#include <QWidget>
#include <functional>

class QCheckBox;
class QDoubleSpinBox;
class QGridLayout;
class QSpinBox;
class QComboBox;
class QLabel;
class QTimer;
class ccHObject;
class vtkGLView;
class vtkChartView;

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

    void refreshSubViews();
    void setSourceView(vtkGLView* src) { m_sourceView = src; }

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

    void setupGrid();

protected:
    void showEvent(QShowEvent* event) override;
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
    void copyActorsAcrossSubViews();
    void installCameraLink();
    void onCameraLinkTick();
    void forceRenderAllSubViews();
    void syncInteractionModeToSubViews();
    void syncPickingModeToSubViews();

    ComparativeType m_type;
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
    QComboBox* m_entityCombo = nullptr;
    bool m_overlayMode = false;
    QGridLayout* m_gridLayout = nullptr;
    QList<QWidget*> m_subWidgets;
    QList<vtkGLView*> m_subViews;
    RenderViewFactory m_renderFactory;
    SubViewInitCallback m_subViewInitCb;
    EntityListProvider m_entityListProvider;
    ccHObject* m_initialEntity = nullptr;
    vtkGLView* m_sourceView = nullptr;
    QTimer* m_cameraLinkTimer = nullptr;
    bool m_cameraLinkEnabled = true;
    bool m_syncingCameras = false;
    double m_lastCameraMTime = 0;
    bool m_firstShowDone = false;
    bool m_closing = false;
    QSet<QWidget*> m_pendingFirstResize;

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
