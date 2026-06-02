// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qVTK.h"

#include <QList>
#include <QMap>
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
class QTableWidget;
class QPushButton;
class QToolButton;
class ccHObject;
class ccGenericMesh;
class ccMesh;
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

    using EntityListProvider = std::function<QList<ccHObject*>()>;
    void setEntityListProvider(EntityListProvider provider);

    void setCompactMode(bool compact);

    void setRectSelectionActive(bool active);
    void setPolySelectionActive(bool active);
    void setSelectionModifier(int modifier);
    void clearSelection();

    void setXAxisScale(double factor);
    void setYAxisScale(double factor);
    void setPlotOpacity(double opacity);

protected:
    bool eventFilter(QObject* obj, QEvent* event) override;
    void showEvent(QShowEvent* event) override;

public slots:
    void setEntity(ccHObject* entity);

signals:
    void pointsHighlighted(ccPointCloud* cloud,
                           const QVector<unsigned>& indices);

private slots:
    void onEntitySelectionChanged(ccHObject* entity);
    void onSourceComboChanged(int index);
    void onSourceComboAboutToShow();
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
    void rebuildQuartileChart(const QList<int>& selectedFields,
                              unsigned sampleCount, unsigned pointCount,
                              const FieldValueFn& getFieldValue);
    void rebuildImageChart(const QList<int>& selectedFields,
                           unsigned sampleCount, unsigned pointCount,
                           const FieldValueFn& getFieldValue);
    void setupChartSelectionCallback();
    void onChartAnnotationChanged();
    QColor seriesColor(int index) const;

    struct FieldDef {
        QString name;
        int sfIndex;
    };

    void refreshSourceCombo();

    ChartType m_chartType;
    QString m_viewTypeKey;
    QString m_title;
    QComboBox* m_sourceCombo = nullptr;
    QComboBox* m_attributeCombo = nullptr;
    EntityListProvider m_entityListProvider;
    QLabel* m_titleLabel = nullptr;
    QListWidget* m_fieldList = nullptr;
    QSpinBox* m_binSpin = nullptr;
    QToolButton* m_histColorBtn = nullptr;
    QColor m_histColor{0, 0, 255};
    QCheckBox* m_histOutlineCheck = nullptr;
    QPushButton* m_resetZoomBtn = nullptr;
    QPushButton* m_exportPngBtn = nullptr;
    QPushButton* m_exportCsvBtn = nullptr;
    QCheckBox* m_linkCheck = nullptr;
    QLineEdit* m_chartTitleEdit = nullptr;
    QCheckBox* m_legendCheck = nullptr;
    QCheckBox* m_gridCheck = nullptr;
    QCheckBox* m_useIndexXAxis = nullptr;
    QComboBox* m_xArrayCombo = nullptr;
    QSpinBox* m_markerSizeSpin = nullptr;
    QComboBox* m_markerStyleCombo = nullptr;
    QComboBox* m_lineStyleCombo = nullptr;
    QDoubleSpinBox* m_lineThickSpin = nullptr;
    QDoubleSpinBox* m_lineOpacitySpin = nullptr;
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
    bool m_compactMode = false;
    bool m_useCellData = false;
    QList<QWidget*> m_chromeBars;
    bool m_selectRectActive = false;
    bool m_selectPolyActive = false;
    void applyTooltipFormat();
    void showChartContextMenu(const QPoint& pos);
    void populateSeriesTable();
    void onSeriesTableChanged();
    void onSeriesTableCellClicked(int row, int col);

    QTableWidget* m_seriesTable = nullptr;
    QMap<int, QColor> m_customSeriesColors;
    struct SeriesProps {
        float lineThickness = 2.0f;
        int lineStyle = 1;
        int markerStyle = 0;
        float markerSize = 4.0f;
        QString legendName;
    };
    QMap<int, SeriesProps> m_seriesProps;
    QVTKOpenGLNativeWidget* m_vtkWidget = nullptr;
    vtkContextView* m_contextView = nullptr;
    vtkChart* m_chart = nullptr;

    ccHObject* m_currentEntity = nullptr;
    ccHObject* currentEntity() const { return m_currentEntity; }
    ccPointCloud* m_cloud = nullptr;
    ccMesh* m_mesh = nullptr;
    ccGenericMesh* m_genericMesh = nullptr;
    QVector<FieldDef> m_fields;
    QVector<float> m_computedNormals;
    unsigned m_sampleStride = 1;
    unsigned m_maxChartPoints = 10000;

    void computeVertexNormals();
    bool hasComputedNormals() const { return !m_computedNormals.isEmpty(); }
    float computedNormal(unsigned ptIdx, int axis) const {
        unsigned base = ptIdx * 3;
        if (base + 2 < static_cast<unsigned>(m_computedNormals.size()))
            return m_computedNormals[base + axis];
        return 0.0f;
    }
    float getNormalComponent(unsigned vertIdx, int axis) const;
};
