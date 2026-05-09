// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qVTK.h"

#include <QWidget>
#include <functional>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QVTKOpenGLNativeWidget;
class QLabel;
class QListWidget;
class QSpinBox;
class QPushButton;
class QToolButton;
class ccHObject;
class ccPointCloud;

class vtkChart;
class vtkContextView;

class QVTK_ENGINE_LIB_API vtkChartView : public QWidget {
    Q_OBJECT

public:
    enum ChartType {
        LINE_CHART,
        BAR_CHART,
        HISTOGRAM,
        BOX_CHART,
        POINT_CHART,
        PARALLEL_COORDINATES,
        PLOT_MATRIX,
        IMAGE_CHART,
        QUARTILE_CHART
    };

    explicit vtkChartView(ChartType type, QWidget* parent = nullptr);
    ~vtkChartView() override;

    QString title() const;
    ChartType chartType() const { return m_chartType; }

    unsigned maxChartPoints() const { return m_maxChartPoints; }
    void setMaxChartPoints(unsigned max);

public slots:
    void setEntity(ccHObject* entity);

signals:
    void pointsHighlighted(ccPointCloud* cloud,
                           const QVector<unsigned>& indices);

private slots:
    void onEntitySelectionChanged(ccHObject* entity);
    void onSelectionChanged();
    void onBinCountChanged(int bins);
    void onResetZoom();
    void onExportPNG();
    void onExportCSV();
    void onChartTitleChanged(const QString& text);
    void onToggleLegend(bool show);
    void onToggleGridLines(bool show);
    void onToggleLogScale(bool log);
    void onAxisNotationChanged(int index);
    void onAxisPrecisionChanged(int prec);
    void onToggleCustomRange(bool use);
    void onAxisRangeChanged();
    void onActiveAxisChanged(int index);
    void onAxisTitleChanged(const QString& text);
    void onToggleAxisVisible(bool visible);
    void onTooltipNotationChanged(int index);
    void onTooltipPrecisionChanged(int prec);

private:
    using FieldValueFn = std::function<float(int fieldIdx, unsigned ptIdx)>;

    void rebuildChart();
    void rebuildXYChart(const QList<int>& selectedFields, unsigned sampleCount,
                        unsigned pointCount, const FieldValueFn& getFieldValue);
    void rebuildHistogram(const QList<int>& selectedFields,
                          unsigned pointCount,
                          const FieldValueFn& getFieldValue);
    void rebuildBoxChart(const QList<int>& selectedFields,
                         unsigned sampleCount, unsigned pointCount,
                         const FieldValueFn& getFieldValue);
    void rebuildParallelCoordinates(const QList<int>& selectedFields,
                                    unsigned sampleCount,
                                    unsigned pointCount,
                                    const FieldValueFn& getFieldValue);
    void rebuildPlotMatrix(const QList<int>& selectedFields,
                           unsigned sampleCount, unsigned pointCount,
                           const FieldValueFn& getFieldValue);
    void setupChartSelectionCallback();
    void onChartAnnotationChanged();
    QColor seriesColor(int index) const;

    struct FieldDef {
        QString name;
        int sfIndex;
    };

    ChartType m_chartType;
    QLabel* m_titleLabel = nullptr;
    QListWidget* m_fieldList = nullptr;
    QSpinBox* m_binSpin = nullptr;
    QPushButton* m_resetZoomBtn = nullptr;
    QPushButton* m_exportPngBtn = nullptr;
    QPushButton* m_exportCsvBtn = nullptr;
    QCheckBox* m_linkCheck = nullptr;
    QLineEdit* m_chartTitleEdit = nullptr;
    QCheckBox* m_legendCheck = nullptr;
    QCheckBox* m_gridCheck = nullptr;
    QComboBox* m_axisSelectCombo = nullptr;
    QLineEdit* m_axisTitleEdit = nullptr;
    QCheckBox* m_axisVisibleCheck = nullptr;
    QCheckBox* m_logScaleCheck = nullptr;
    QComboBox* m_notationCombo = nullptr;
    QSpinBox* m_axisPrecSpin = nullptr;
    QCheckBox* m_customRangeCheck = nullptr;
    QDoubleSpinBox* m_rangeMinSpin = nullptr;
    QDoubleSpinBox* m_rangeMaxSpin = nullptr;
    QComboBox* m_tooltipNotationCombo = nullptr;
    QSpinBox* m_tooltipPrecSpin = nullptr;
    int m_tooltipNotation = 0;
    int m_tooltipPrecision = 6;
    void applyTooltipFormat();
    QVTKOpenGLNativeWidget* m_vtkWidget = nullptr;
    vtkContextView* m_contextView = nullptr;
    vtkChart* m_chart = nullptr;

    ccPointCloud* m_cloud = nullptr;
    QVector<FieldDef> m_fields;
    unsigned m_sampleStride = 1;
    unsigned m_maxChartPoints = 10000;
};
