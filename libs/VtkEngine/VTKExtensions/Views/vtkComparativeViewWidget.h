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

#include <QWidget>
#include <functional>

class QCheckBox;
class QDoubleSpinBox;
class QGridLayout;
class QSpinBox;
class QComboBox;
class QLabel;
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

    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    void setDimensions(int rows, int cols);
    void setSpacing(int spacing);
    int spacing() const { return m_spacing; }

    QList<QWidget*> subWidgets() const { return m_subWidgets; }

    void setupGrid();

signals:
    void subViewCreated(QWidget* subWidget);

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
    void applyCueToSubViews();

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
    bool m_overlayMode = false;
    QGridLayout* m_gridLayout = nullptr;
    QList<QWidget*> m_subWidgets;
    RenderViewFactory m_renderFactory;
};
