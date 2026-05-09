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
class QVTKOpenGLNativeWidget;
class QLabel;
class QListWidget;
class QSpinBox;
class QPushButton;
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
    QVTKOpenGLNativeWidget* m_vtkWidget = nullptr;
    vtkContextView* m_contextView = nullptr;
    vtkChart* m_chart = nullptr;

    ccPointCloud* m_cloud = nullptr;
    QVector<FieldDef> m_fields;
    unsigned m_sampleStride = 1;
};
