// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkChartView.h"

static constexpr unsigned kDefaultMaxChartPoints = 10000;

#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvViewManager.h>

#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QHeaderView>
#include <QListWidget>
#include <QMenu>
#include <QTableWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QTextStream>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QVTKOpenGLNativeWidget.h>

#include <vtkAnnotationLink.h>
#include <vtkAxis.h>
#include <vtkChart.h>
#include <vtkCommand.h>
#include <vtkIdTypeArray.h>
#include <vtkChartBox.h>
#include <vtkChartParallelCoordinates.h>
#include <vtkChartXY.h>
#include <vtkTooltipItem.h>
#include <vtkColorSeries.h>
#include <vtkContextMouseEvent.h>
#include <vtkContextScene.h>
#include <vtkContextView.h>
#include <vtkFloatArray.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkIntArray.h>
#include <vtkNew.h>
#include <vtkPen.h>
#include <vtkPNGWriter.h>
#include <vtkPlotBar.h>
#include <vtkPlotBox.h>
#include <vtkPlotLine.h>
#include <vtkPlotParallelCoordinates.h>
#include <vtkPlotPoints.h>
#include <vtkScatterPlotMatrix.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkRenderer.h>
#include <vtkTable.h>
#include <vtkTextProperty.h>
#include <vtkWindowToImageFilter.h>

static const QColor kPalette[] = {
        {66, 133, 244},  {234, 67, 53},  {251, 188, 4},  {52, 168, 83},
        {171, 71, 188},  {255, 112, 67}, {0, 172, 193},   {124, 179, 66},
        {233, 30, 99},   {0, 150, 136},  {255, 167, 38},  {63, 81, 181},
};

vtkChartView::vtkChartView(ChartType type, QWidget* parent)
    : QWidget(parent), m_chartType(type) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    // === Row 1: Data source decorator (ParaView-style) ===
    auto* decoratorBar = new QWidget(this);
    decoratorBar->setObjectName("ChartDecoratorBar");
    auto* decLayout = new QHBoxLayout(decoratorBar);
    decLayout->setContentsMargins(2, 1, 2, 1);
    decLayout->setSpacing(2);

    auto* showingLabel = new QLabel(QStringLiteral("<b>Showing</b>"), decoratorBar);
    decLayout->addWidget(showingLabel);

    m_sourceCombo = new QComboBox(decoratorBar);
    m_sourceCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    m_sourceCombo->addItem(tr("None"));
    decLayout->addWidget(m_sourceCombo);

    decLayout->addSpacing(8);
    auto* attrLabel = new QLabel(QStringLiteral("<b>Attribute:</b>"), decoratorBar);
    decLayout->addWidget(attrLabel);
    m_attributeCombo = new QComboBox(decoratorBar);
    m_attributeCombo->addItem(tr("Point Data"), 0);
    m_attributeCombo->addItem(tr("Cell Data"), 1);
    decLayout->addWidget(m_attributeCombo);
    connect(m_attributeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { rebuildChart(); });

    decLayout->addStretch(1);
    layout->addWidget(decoratorBar);

    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &vtkChartView::onSourceComboChanged);
    m_sourceCombo->installEventFilter(this);

    // === Row 2: Chart properties toolbar (ParaView Display-style) ===
    auto* propsBar = new QWidget(this);
    propsBar->setObjectName("ChartPropsBar");
    auto* propsLayout = new QHBoxLayout(propsBar);
    propsLayout->setContentsMargins(2, 1, 2, 1);
    propsLayout->setSpacing(4);

    m_legendCheck = new QCheckBox(tr("Legend"), propsBar);
    m_legendCheck->setChecked(true);
    propsLayout->addWidget(m_legendCheck);

    m_gridCheck = new QCheckBox(tr("Grid"), propsBar);
    m_gridCheck->setChecked(true);
    propsLayout->addWidget(m_gridCheck);

    m_logScaleCheck = new QCheckBox(tr("Log"), propsBar);
    propsLayout->addWidget(m_logScaleCheck);

    propsLayout->addSpacing(4);
    auto* precLabel = new QLabel(tr("Prec:"), propsBar);
    propsLayout->addWidget(precLabel);
    m_axisPrecSpin = new QSpinBox(propsBar);
    m_axisPrecSpin->setRange(0, 15);
    m_axisPrecSpin->setValue(2);
    m_axisPrecSpin->setFixedWidth(50);
    propsLayout->addWidget(m_axisPrecSpin);

    m_notationCombo = new QComboBox(propsBar);
    m_notationCombo->addItem(tr("Mixed"), 0);
    m_notationCombo->addItem(tr("Scientific"), 1);
    m_notationCombo->addItem(tr("Fixed"), 2);
    m_notationCombo->setFixedWidth(80);
    propsLayout->addWidget(m_notationCombo);

    if (m_chartType == HISTOGRAM) {
        propsLayout->addSpacing(4);
        auto* binLabel = new QLabel(tr("Bins:"), propsBar);
        propsLayout->addWidget(binLabel);
        m_binSpin = new QSpinBox(propsBar);
        m_binSpin->setRange(2, 1000);
        m_binSpin->setValue(256);
        m_binSpin->setFixedWidth(65);
        propsLayout->addWidget(m_binSpin);

        propsLayout->addSpacing(4);
        auto* colorLabel = new QLabel(tr("Color:"), propsBar);
        propsLayout->addWidget(colorLabel);
        m_histColorBtn = new QToolButton(propsBar);
        m_histColorBtn->setFixedSize(20, 20);
        m_histColor = QColor(0, 0, 255);
        m_histColorBtn->setStyleSheet(
                QStringLiteral("background-color: %1; border: 1px solid gray;")
                        .arg(m_histColor.name()));
        m_histColorBtn->setToolTip(tr("Histogram bar color"));
        propsLayout->addWidget(m_histColorBtn);
    }

    if (m_chartType == PARALLEL_COORDINATES) {
        propsLayout->addSpacing(4);
        auto* opLabel = new QLabel(tr("Opacity:"), propsBar);
        propsLayout->addWidget(opLabel);
        m_lineOpacitySpin = new QDoubleSpinBox(propsBar);
        m_lineOpacitySpin->setRange(0.01, 1.0);
        m_lineOpacitySpin->setValue(0.1);
        m_lineOpacitySpin->setSingleStep(0.05);
        m_lineOpacitySpin->setFixedWidth(60);
        propsLayout->addWidget(m_lineOpacitySpin);
        connect(m_lineOpacitySpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, [this]() { rebuildChart(); });
    }

    propsLayout->addStretch(1);

    auto* resetBtn = new QToolButton(propsBar);
    resetBtn->setIcon(QIcon::fromTheme("zoom-fit-best"));
    resetBtn->setToolTip(tr("Reset Zoom"));
    propsLayout->addWidget(resetBtn);
    m_resetZoomBtn = new QPushButton(this);
    m_resetZoomBtn->hide();
    connect(resetBtn, &QToolButton::clicked, this, &vtkChartView::onResetZoom);

    auto* pngBtn = new QToolButton(propsBar);
    pngBtn->setIcon(QIcon::fromTheme("camera-photo"));
    pngBtn->setToolTip(tr("Export PNG"));
    propsLayout->addWidget(pngBtn);
    m_exportPngBtn = new QPushButton(this);
    m_exportPngBtn->hide();
    connect(pngBtn, &QToolButton::clicked, this, &vtkChartView::onExportPNG);

    auto* csvBtn = new QToolButton(propsBar);
    csvBtn->setIcon(QIcon::fromTheme("document-save"));
    csvBtn->setToolTip(tr("Export CSV"));
    propsLayout->addWidget(csvBtn);
    m_exportCsvBtn = new QPushButton(this);
    m_exportCsvBtn->hide();
    connect(csvBtn, &QToolButton::clicked, this, &vtkChartView::onExportCSV);

    layout->addWidget(propsBar);

    // === Row 3: X Axis & Series Style (ParaView Series Parameters) ===
    auto* axisBar = new QWidget(this);
    axisBar->setObjectName("ChartAxisBar");
    auto* axisLayout = new QHBoxLayout(axisBar);
    axisLayout->setContentsMargins(2, 1, 2, 1);
    axisLayout->setSpacing(4);

    m_useIndexXAxis = new QCheckBox(tr("Use Index for X Axis"), axisBar);
    m_useIndexXAxis->setChecked(true);
    axisLayout->addWidget(m_useIndexXAxis);

    auto* xLabel = new QLabel(tr("X Array:"), axisBar);
    axisLayout->addWidget(xLabel);
    m_xArrayCombo = new QComboBox(axisBar);
    m_xArrayCombo->setEnabled(false);
    m_xArrayCombo->setFixedWidth(120);
    axisLayout->addWidget(m_xArrayCombo);

    connect(m_useIndexXAxis, &QCheckBox::toggled, this, [this](bool checked) {
        m_xArrayCombo->setEnabled(!checked);
        rebuildChart();
    });
    connect(m_xArrayCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this]() {
                if (!m_useIndexXAxis->isChecked()) rebuildChart();
            });

    axisLayout->addSpacing(8);

    if (m_chartType == POINT_CHART) {
        auto* mrkLabel = new QLabel(tr("Marker:"), axisBar);
        axisLayout->addWidget(mrkLabel);
        m_markerSizeSpin = new QSpinBox(axisBar);
        m_markerSizeSpin->setRange(1, 30);
        m_markerSizeSpin->setValue(4);
        m_markerSizeSpin->setFixedWidth(50);
        m_markerSizeSpin->setToolTip(tr("Marker size"));
        axisLayout->addWidget(m_markerSizeSpin);
        connect(m_markerSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
                this, [this]() { rebuildChart(); });
    }
    if (m_chartType == LINE_CHART || m_chartType == POINT_CHART
        || m_chartType == PARALLEL_COORDINATES) {
        auto* lstLabel = new QLabel(tr("Line:"), axisBar);
        axisLayout->addWidget(lstLabel);
        m_lineStyleCombo = new QComboBox(axisBar);
        m_lineStyleCombo->addItem(tr("None"), 0);
        m_lineStyleCombo->addItem(tr("Solid"), 1);
        m_lineStyleCombo->addItem(tr("Dash"), 2);
        m_lineStyleCombo->addItem(tr("Dot"), 3);
        m_lineStyleCombo->addItem(tr("Dash-Dot"), 4);
        int defLineIdx = (m_chartType == LINE_CHART
                          || m_chartType == PARALLEL_COORDINATES) ? 1 : 0;
        m_lineStyleCombo->setCurrentIndex(defLineIdx);
        m_lineStyleCombo->setFixedWidth(80);
        axisLayout->addWidget(m_lineStyleCombo);
        connect(m_lineStyleCombo,
                QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this]() { rebuildChart(); });

        auto* ltLabel = new QLabel(tr("Width:"), axisBar);
        axisLayout->addWidget(ltLabel);
        m_lineThickSpin = new QDoubleSpinBox(axisBar);
        m_lineThickSpin->setRange(0.5, 10.0);
        m_lineThickSpin->setValue(2.0);
        m_lineThickSpin->setSingleStep(0.5);
        m_lineThickSpin->setFixedWidth(55);
        axisLayout->addWidget(m_lineThickSpin);
        connect(m_lineThickSpin,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, [this]() { rebuildChart(); });
    }
    if (m_chartType == LINE_CHART) {
        axisLayout->addSpacing(4);
        auto* mkrLabel = new QLabel(tr("Marker:"), axisBar);
        axisLayout->addWidget(mkrLabel);
        m_markerStyleCombo = new QComboBox(axisBar);
        m_markerStyleCombo->addItem(tr("None"), -1);
        m_markerStyleCombo->addItem(tr("Cross"), vtkPlotPoints::CROSS);
        m_markerStyleCombo->addItem(tr("Plus"), vtkPlotPoints::PLUS);
        m_markerStyleCombo->addItem(tr("Square"), vtkPlotPoints::SQUARE);
        m_markerStyleCombo->addItem(tr("Circle"), vtkPlotPoints::CIRCLE);
        m_markerStyleCombo->addItem(tr("Diamond"), vtkPlotPoints::DIAMOND);
        m_markerStyleCombo->setFixedWidth(75);
        axisLayout->addWidget(m_markerStyleCombo);
        connect(m_markerStyleCombo,
                QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this]() { rebuildChart(); });

        auto* mszLabel = new QLabel(tr("Size:"), axisBar);
        axisLayout->addWidget(mszLabel);
        m_markerSizeSpin = new QSpinBox(axisBar);
        m_markerSizeSpin->setRange(1, 30);
        m_markerSizeSpin->setValue(1);
        m_markerSizeSpin->setFixedWidth(50);
        axisLayout->addWidget(m_markerSizeSpin);
        connect(m_markerSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
                this, [this]() { rebuildChart(); });
    }

    axisLayout->addStretch(1);
    layout->addWidget(axisBar);

    m_titleLabel = new QLabel(title() + " - No data", this);
    m_titleLabel->setAlignment(Qt::AlignCenter);
    m_titleLabel->setContentsMargins(2, 2, 2, 2);
    layout->addWidget(m_titleLabel);

    // === Series field list (right-side collapsible, ParaView Series Editor) ===
    m_fieldList = new QListWidget(this);
    m_fieldList->setSelectionMode(QAbstractItemView::MultiSelection);
    m_fieldList->setMaximumHeight(80);
    m_fieldList->setToolTip(tr("Select fields to plot (multi-select)"));
    layout->addWidget(m_fieldList);

    if (m_chartType == LINE_CHART || m_chartType == POINT_CHART) {
        m_seriesTable = new QTableWidget(0, 4, this);
        m_seriesTable->setHorizontalHeaderLabels(
                {tr("Vis"), tr("Color"), tr("Variable"), tr("Legend Name")});
        m_seriesTable->setColumnWidth(0, 30);
        m_seriesTable->setColumnWidth(1, 30);
        m_seriesTable->setColumnWidth(2, 120);
        m_seriesTable->horizontalHeader()->setStretchLastSection(true);
        m_seriesTable->verticalHeader()->setVisible(false);
        m_seriesTable->setMaximumHeight(120);
        m_seriesTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        m_seriesTable->setEditTriggers(QAbstractItemView::DoubleClicked);
        layout->addWidget(m_seriesTable);
        connect(m_seriesTable, &QTableWidget::cellChanged,
                this, [this]() { onSeriesTableChanged(); });
        connect(m_seriesTable, &QTableWidget::cellClicked,
                this, &vtkChartView::onSeriesTableCellClicked);
    }

    m_linkCheck = new QCheckBox(this);
    m_linkCheck->setChecked(true);
    m_linkCheck->hide();

    m_chartTitleEdit = new QLineEdit(this);
    m_chartTitleEdit->hide();

    m_axisSelectCombo = new QComboBox(this);
    m_axisSelectCombo->addItem(tr("Left"), vtkAxis::LEFT);
    m_axisSelectCombo->addItem(tr("Bottom"), vtkAxis::BOTTOM);
    m_axisSelectCombo->addItem(tr("Right"), vtkAxis::RIGHT);
    m_axisSelectCombo->addItem(tr("Top"), vtkAxis::TOP);
    m_axisSelectCombo->hide();

    m_axisVisibleCheck = new QCheckBox(this);
    m_axisVisibleCheck->setChecked(true);
    m_axisVisibleCheck->hide();

    m_axisTitleEdit = new QLineEdit(this);
    m_axisTitleEdit->hide();

    m_customRangeCheck = new QCheckBox(this);
    m_customRangeCheck->hide();

    m_rangeMinSpin = new QDoubleSpinBox(this);
    m_rangeMinSpin->setRange(-1e12, 1e12);
    m_rangeMinSpin->setDecimals(4);
    m_rangeMinSpin->setValue(0.0);
    m_rangeMinSpin->setEnabled(false);
    m_rangeMinSpin->hide();

    m_rangeMaxSpin = new QDoubleSpinBox(this);
    m_rangeMaxSpin->setRange(-1e12, 1e12);
    m_rangeMaxSpin->setDecimals(4);
    m_rangeMaxSpin->setValue(10.0);
    m_rangeMaxSpin->setEnabled(false);
    m_rangeMaxSpin->hide();

    m_tooltipNotationCombo = new QComboBox(this);
    m_tooltipNotationCombo->addItem(tr("Mixed"), 0);
    m_tooltipNotationCombo->addItem(tr("Sci"), 1);
    m_tooltipNotationCombo->addItem(tr("Fix"), 2);
    m_tooltipNotationCombo->hide();

    m_tooltipPrecSpin = new QSpinBox(this);
    m_tooltipPrecSpin->setRange(0, 15);
    m_tooltipPrecSpin->setValue(6);
    m_tooltipPrecSpin->hide();

    setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, &QWidget::customContextMenuRequested, this,
            &vtkChartView::showChartContextMenu);

    m_vtkWidget = new QVTKOpenGLNativeWidget(this);
    m_vtkWidget->setMinimumSize(50, 50);
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    m_vtkWidget->setRenderWindow(renderWindow);
    layout->addWidget(m_vtkWidget, 1);
    setMinimumSize(80, 80);

    m_contextView = vtkContextView::New();
    m_contextView->SetRenderWindow(renderWindow);
    m_contextView->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

    if (m_chartType == PLOT_MATRIX) {
        auto* spm = vtkScatterPlotMatrix::New();
        m_contextView->GetScene()->AddItem(spm);
        spm->Delete();
        spm->SetBackgroundColor(0, vtkColor4ub(255, 255, 255, 0));
        spm->SetBackgroundColor(1, vtkColor4ub(255, 255, 255, 0));
        spm->SetBackgroundColor(2, vtkColor4ub(128, 128, 128, 102));
    } else {
        if (m_chartType == PARALLEL_COORDINATES) {
            m_chart = vtkChartParallelCoordinates::New();
        } else if (m_chartType == BOX_CHART) {
            m_chart = vtkChartBox::New();
        } else {
            m_chart = vtkChartXY::New();
        }
        m_contextView->GetScene()->AddItem(m_chart);
        m_chart->Delete();
        m_chart->SetShowLegend(true);

        m_chart->SetActionToButton(vtkChart::SELECT,
                                   vtkContextMouseEvent::LEFT_BUTTON);
        m_chart->SetActionToButton(vtkChart::PAN,
                                   vtkContextMouseEvent::MIDDLE_BUTTON);
        m_chart->SetActionToButton(vtkChart::ZOOM_AXIS,
                                   vtkContextMouseEvent::RIGHT_BUTTON);

        for (int i = 0; i < 4; ++i) {
            auto* axis = m_chart->GetAxis(i);
            if (!axis) continue;
            axis->GetLabelProperties()->SetColor(0.0, 0.0, 0.0);
            axis->GetTitleProperties()->SetColor(0.0, 0.0, 0.0);
            axis->GetPen()->SetColor(0, 0, 0);
            axis->GetGridPen()->SetColorF(0.95, 0.95, 0.95);
            axis->SetNotation(0);
            axis->SetPrecision(2);
            axis->SetGridVisible(true);
        }
    }

    connect(m_linkCheck, &QCheckBox::toggled, this, [this](bool checked) {
        if (!m_chart) return;
        if (checked) {
            m_chart->SetActionToButton(vtkChart::SELECT,
                                       vtkContextMouseEvent::LEFT_BUTTON);
            m_chart->SetActionToButton(vtkChart::PAN,
                                       vtkContextMouseEvent::MIDDLE_BUTTON);
        } else {
            m_chart->SetActionToButton(vtkChart::PAN,
                                       vtkContextMouseEvent::LEFT_BUTTON);
            m_chart->SetActionToButton(vtkChart::ZOOM_AXIS,
                                       vtkContextMouseEvent::MIDDLE_BUTTON);
        }
    });

    connect(m_fieldList, &QListWidget::itemSelectionChanged, this,
            &vtkChartView::onSelectionChanged);
    if (m_binSpin) {
        connect(m_binSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
                &vtkChartView::onBinCountChanged);
    }
    if (m_histColorBtn) {
        connect(m_histColorBtn, &QToolButton::clicked, this, [this]() {
            QColor c = QColorDialog::getColor(m_histColor, this,
                                              tr("Select Histogram Color"));
            if (c.isValid()) {
                m_histColor = c;
                m_histColorBtn->setStyleSheet(
                        QStringLiteral(
                                "background-color: %1; border: 1px solid gray;")
                                .arg(c.name()));
                rebuildChart();
            }
        });
    }
    connect(m_resetZoomBtn, &QPushButton::clicked, this,
            &vtkChartView::onResetZoom);
    connect(m_exportPngBtn, &QPushButton::clicked, this,
            &vtkChartView::onExportPNG);
    connect(m_exportCsvBtn, &QPushButton::clicked, this,
            &vtkChartView::onExportCSV);
    connect(m_chartTitleEdit, &QLineEdit::textChanged, this,
            &vtkChartView::onChartTitleChanged);
    connect(m_legendCheck, &QCheckBox::toggled, this,
            &vtkChartView::onToggleLegend);
    connect(m_gridCheck, &QCheckBox::toggled, this,
            &vtkChartView::onToggleGridLines);
    connect(m_axisSelectCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &vtkChartView::onActiveAxisChanged);
    connect(m_axisVisibleCheck, &QCheckBox::toggled, this,
            &vtkChartView::onToggleAxisVisible);
    connect(m_axisTitleEdit, &QLineEdit::textChanged, this,
            &vtkChartView::onAxisTitleChanged);
    connect(m_logScaleCheck, &QCheckBox::toggled, this,
            &vtkChartView::onToggleLogScale);
    connect(m_notationCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &vtkChartView::onAxisNotationChanged);
    connect(m_axisPrecSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &vtkChartView::onAxisPrecisionChanged);
    connect(m_customRangeCheck, &QCheckBox::toggled, this,
            &vtkChartView::onToggleCustomRange);
    connect(m_rangeMinSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &vtkChartView::onAxisRangeChanged);
    connect(m_rangeMaxSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &vtkChartView::onAxisRangeChanged);
    connect(m_tooltipNotationCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &vtkChartView::onTooltipNotationChanged);
    connect(m_tooltipPrecSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &vtkChartView::onTooltipPrecisionChanged);
    connect(&ecvViewManager::instance(),
            &ecvViewManager::entitySelectionChanged, this,
            &vtkChartView::onEntitySelectionChanged);
}

void vtkChartView::showChartContextMenu(const QPoint& pos) {
    QMenu menu(this);

    auto* fieldsMenu = menu.addMenu(tr("Fields"));
    for (int i = 0; i < m_fieldList->count(); ++i) {
        auto* item = m_fieldList->item(i);
        auto* act = fieldsMenu->addAction(item->text());
        act->setCheckable(true);
        act->setChecked(item->isSelected());
        connect(act, &QAction::toggled, this, [this, i](bool checked) {
            m_fieldList->item(i)->setSelected(checked);
        });
    }

    if (m_chartType == HISTOGRAM && m_binSpin) {
        menu.addSeparator();
        auto* binAct = menu.addAction(
                tr("Bins: %1").arg(m_binSpin->value()));
        binAct->setEnabled(false);
    }

    menu.addSeparator();

    auto* legendAct = menu.addAction(tr("Show Legend"));
    legendAct->setCheckable(true);
    legendAct->setChecked(m_legendCheck->isChecked());
    connect(legendAct, &QAction::toggled, m_legendCheck, &QCheckBox::setChecked);

    auto* gridAct = menu.addAction(tr("Show Grid"));
    gridAct->setCheckable(true);
    gridAct->setChecked(m_gridCheck->isChecked());
    connect(gridAct, &QAction::toggled, m_gridCheck, &QCheckBox::setChecked);

    auto* linkAct = menu.addAction(tr("Link to 3D Selection"));
    linkAct->setCheckable(true);
    linkAct->setChecked(m_linkCheck->isChecked());
    connect(linkAct, &QAction::toggled, m_linkCheck, &QCheckBox::setChecked);

    menu.addSeparator();

    auto* rectSelAct = menu.addAction(tr("Rectangle Selection (s)"));
    rectSelAct->setCheckable(true);
    rectSelAct->setChecked(m_selectRectActive);
    connect(rectSelAct, &QAction::toggled, this, [this](bool checked) {
        m_selectRectActive = checked;
        if (checked) m_selectPolyActive = false;
        if (!m_chart) return;
        if (checked) {
            m_chart->SetActionToButton(vtkChart::SELECT_RECTANGLE,
                                       vtkContextMouseEvent::LEFT_BUTTON);
        } else if (!m_selectPolyActive) {
            m_chart->SetActionToButton(vtkChart::PAN,
                                       vtkContextMouseEvent::LEFT_BUTTON);
        }
    });

    auto* polySelAct = menu.addAction(tr("Polygon Selection (d)"));
    polySelAct->setCheckable(true);
    polySelAct->setChecked(m_selectPolyActive);
    connect(polySelAct, &QAction::toggled, this, [this](bool checked) {
        m_selectPolyActive = checked;
        if (checked) m_selectRectActive = false;
        if (!m_chart) return;
        if (checked) {
            m_chart->SetActionToButton(vtkChart::SELECT_POLYGON,
                                       vtkContextMouseEvent::LEFT_BUTTON);
        } else if (!m_selectRectActive) {
            m_chart->SetActionToButton(vtkChart::PAN,
                                       vtkContextMouseEvent::LEFT_BUTTON);
        }
    });

    menu.addSeparator();

    auto* resetAct = menu.addAction(tr("Reset Zoom"));
    connect(resetAct, &QAction::triggered, this, &vtkChartView::onResetZoom);

    auto* pngAct = menu.addAction(tr("Export as PNG..."));
    connect(pngAct, &QAction::triggered, this, &vtkChartView::onExportPNG);

    auto* csvAct = menu.addAction(tr("Export as CSV..."));
    connect(csvAct, &QAction::triggered, this, &vtkChartView::onExportCSV);

    menu.exec(mapToGlobal(pos));
}

vtkChartView::~vtkChartView() {
    disconnect(&ecvViewManager::instance(), nullptr, this, nullptr);
    m_chart = nullptr;
    if (m_contextView) {
        m_contextView->GetScene()->ClearItems();
        m_contextView->SetRenderWindow(nullptr);
        m_contextView->Delete();
        m_contextView = nullptr;
    }
}

QString vtkChartView::title() const {
    switch (m_chartType) {
        case LINE_CHART:
            return tr("Line Chart View");
        case BAR_CHART:
            return tr("Bar Chart View");
        case HISTOGRAM:
            return tr("Histogram View");
        case BOX_CHART:
            return tr("Box Chart View");
        case POINT_CHART:
            return tr("Point Chart View");
        case PARALLEL_COORDINATES:
            return tr("Parallel Coordinates View");
        case PLOT_MATRIX:
            return tr("Plot Matrix View");
        case IMAGE_CHART:
            return tr("Image Chart View");
        case QUARTILE_CHART:
            return tr("Quartile Chart View");
    }
    return tr("Chart View");
}

void vtkChartView::setMaxChartPoints(unsigned max) {
    m_maxChartPoints = max > 0 ? max : kDefaultMaxChartPoints;
    rebuildChart();
}

void vtkChartView::setRectSelectionActive(bool active) {
    m_selectRectActive = active;
    if (active) m_selectPolyActive = false;
    if (!m_chart) return;
    if (active) {
        m_chart->SetActionToButton(vtkChart::SELECT_RECTANGLE,
                                   vtkContextMouseEvent::LEFT_BUTTON);
    } else if (!m_selectPolyActive) {
        m_chart->SetActionToButton(vtkChart::PAN,
                                   vtkContextMouseEvent::LEFT_BUTTON);
    }
}

void vtkChartView::setPolySelectionActive(bool active) {
    m_selectPolyActive = active;
    if (active) m_selectRectActive = false;
    if (!m_chart) return;
    if (active) {
        m_chart->SetActionToButton(vtkChart::SELECT_POLYGON,
                                   vtkContextMouseEvent::LEFT_BUTTON);
    } else if (!m_selectRectActive) {
        m_chart->SetActionToButton(vtkChart::PAN,
                                   vtkContextMouseEvent::LEFT_BUTTON);
    }
}

void vtkChartView::setSelectionModifier(int modifier) {
    if (!m_chart) return;
    m_chart->SetSelectionMode(modifier);
}

void vtkChartView::clearSelection() {
    if (!m_chart) return;
    m_chart->SetActionToButton(vtkChart::PAN,
                               vtkContextMouseEvent::LEFT_BUTTON);
    m_chart->SetSelectionMode(vtkContextScene::SELECTION_DEFAULT);
    m_selectRectActive = false;
    m_selectPolyActive = false;

    auto* annotationLink = m_chart->GetAnnotationLink();
    if (annotationLink) {
        auto* sel = annotationLink->GetCurrentSelection();
        if (sel) {
            sel->RemoveAllNodes();
            annotationLink->InvokeEvent(vtkCommand::AnnotationChangedEvent);
        }
    }
    if (m_vtkWidget) m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::setYAxisScale(double factor) {
    if (!m_chart) return;
    auto* yAxis = m_chart->GetAxis(vtkAxis::LEFT);
    if (!yAxis) return;
    double range[2];
    yAxis->GetRange(range);
    double center = (range[0] + range[1]) * 0.5;
    double halfSpan = (range[1] - range[0]) * 0.5 / factor;
    yAxis->SetRange(center - halfSpan, center + halfSpan);
    yAxis->RecalculateTickSpacing();
    if (m_vtkWidget) m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::setPlotOpacity(double opacity) {
    if (!m_chart) return;
    for (int i = 0; i < m_chart->GetNumberOfPlots(); ++i) {
        auto* plot = m_chart->GetPlot(i);
        if (!plot) continue;
        auto pen = plot->GetPen();
        if (pen) {
            unsigned char c[4];
            pen->GetColor(c);
            pen->SetColorF(c[0] / 255.0, c[1] / 255.0,
                           c[2] / 255.0, opacity);
        }
    }
    if (m_vtkWidget) m_vtkWidget->renderWindow()->Render();
}

QColor vtkChartView::seriesColor(int index) const {
    if (m_customSeriesColors.contains(index))
        return m_customSeriesColors[index];
    return kPalette[index % (sizeof(kPalette) / sizeof(kPalette[0]))];
}

void vtkChartView::computeVertexNormals() {
    m_computedNormals.clear();
    if (!m_mesh || !m_cloud) return;
    if (m_cloud->hasNormals()) return;

    unsigned numVerts = m_cloud->size();
    unsigned numTris = m_mesh->size();
    if (numTris == 0) return;

    m_computedNormals.resize(numVerts * 3);
    m_computedNormals.fill(0.0f);
    QVector<int> counts(numVerts, 0);

    bool hasExplicitTriNormals = m_mesh->hasTriNormals();

    for (unsigned t = 0; t < numTris; ++t) {
        auto* tri = m_mesh->getTriangleVertIndexes(t);
        if (!tri) continue;

        CCVector3 faceNormal(0, 0, 0);
        if (hasExplicitTriNormals) {
            CCVector3 na, nb, nc;
            if (m_mesh->getTriangleNormals(t, na, nb, nc)) {
                faceNormal = (na + nb + nc);
                float len = faceNormal.norm();
                if (len > 1e-12f) faceNormal /= len;
            }
        }
        if (faceNormal.norm2() < 1e-12f) {
            const CCVector3* A = m_cloud->getPoint(tri->i1);
            const CCVector3* B = m_cloud->getPoint(tri->i2);
            const CCVector3* C = m_cloud->getPoint(tri->i3);
            if (A && B && C) {
                CCVector3 AB = *B - *A;
                CCVector3 AC = *C - *A;
                faceNormal = AB.cross(AC);
                float len = faceNormal.norm();
                if (len > 1e-12f) faceNormal /= len;
            }
        }

        auto accum = [&](unsigned vi) {
            if (vi < numVerts) {
                unsigned b = vi * 3;
                m_computedNormals[b] += faceNormal.x;
                m_computedNormals[b + 1] += faceNormal.y;
                m_computedNormals[b + 2] += faceNormal.z;
                counts[vi]++;
            }
        };
        accum(tri->i1);
        accum(tri->i2);
        accum(tri->i3);
    }

    for (unsigned i = 0; i < numVerts; ++i) {
        if (counts[i] > 0) {
            unsigned b = i * 3;
            float inv = 1.0f / counts[i];
            float nx = m_computedNormals[b] * inv;
            float ny = m_computedNormals[b + 1] * inv;
            float nz = m_computedNormals[b + 2] * inv;
            float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-12f) {
                float invL = 1.0f / len;
                nx *= invL;
                ny *= invL;
                nz *= invL;
            }
            m_computedNormals[b] = nx;
            m_computedNormals[b + 1] = ny;
            m_computedNormals[b + 2] = nz;
        }
    }
}

void vtkChartView::setEntity(ccHObject* entity) {
    m_cloud = nullptr;
    m_mesh = nullptr;
    m_computedNormals.clear();
    if (entity) {
        m_cloud = ccHObjectCaster::ToPointCloud(entity);
        m_mesh = ccHObjectCaster::ToMesh(entity);
        if (!m_cloud && m_mesh) {
            m_cloud = ccHObjectCaster::ToPointCloud(
                    m_mesh->getAssociatedCloud());
        }
        if (!m_cloud) {
            m_cloud = ccHObjectCaster::ToPointCloud(
                    ccHObjectCaster::ToGenericPointCloud(entity));
        }
        computeVertexNormals();
    }
    m_fields.clear();

    m_fieldList->blockSignals(true);
    m_fieldList->clear();

    if (m_cloud) {
        {
            FieldDef fd;
            fd.name = QStringLiteral("Points_X");
            fd.sfIndex = -20;
            m_fields.append(fd);
            auto* item = new QListWidgetItem(fd.name, m_fieldList);
            item->setForeground(seriesColor(m_fields.size() - 1));
        }
        {
            FieldDef fd;
            fd.name = QStringLiteral("Points_Y");
            fd.sfIndex = -21;
            m_fields.append(fd);
            auto* item = new QListWidgetItem(fd.name, m_fieldList);
            item->setForeground(seriesColor(m_fields.size() - 1));
        }
        {
            FieldDef fd;
            fd.name = QStringLiteral("Points_Z");
            fd.sfIndex = -22;
            m_fields.append(fd);
            auto* item = new QListWidgetItem(fd.name, m_fieldList);
            item->setForeground(seriesColor(m_fields.size() - 1));
        }
        bool hasNorms = m_cloud->hasNormals() ||
                        (m_mesh && m_mesh->hasNormals()) ||
                        hasComputedNormals();
        if (hasNorms) {
            for (int n = 0; n < 3; ++n) {
                FieldDef fd;
                fd.name = (n == 0) ? QStringLiteral("Normals_X")
                          : (n == 1) ? QStringLiteral("Normals_Y")
                                     : QStringLiteral("Normals_Z");
                fd.sfIndex = -30 - n;
                m_fields.append(fd);
                auto* item = new QListWidgetItem(fd.name, m_fieldList);
                item->setForeground(seriesColor(m_fields.size() - 1));
            }
        }
        unsigned sfCount = m_cloud->getNumberOfScalarFields();
        for (unsigned i = 0; i < sfCount; ++i) {
            FieldDef fd;
            fd.name = QString::fromUtf8(m_cloud->getScalarFieldName(i));
            fd.sfIndex = static_cast<int>(i);
            m_fields.append(fd);

            auto* item = new QListWidgetItem(fd.name, m_fieldList);
            item->setForeground(seriesColor(m_fields.size() - 1));
        }
        {
            FieldDef fd;
            fd.name = QStringLiteral("Points_Magnitude");
            fd.sfIndex = -40;
            m_fields.append(fd);
            auto* item = new QListWidgetItem(fd.name, m_fieldList);
            item->setForeground(seriesColor(m_fields.size() - 1));
        }
        if (hasNorms) {
            FieldDef fd;
            fd.name = QStringLiteral("Normals_Magnitude");
            fd.sfIndex = -41;
            m_fields.append(fd);
            auto* item = new QListWidgetItem(fd.name, m_fieldList);
            item->setForeground(seriesColor(m_fields.size() - 1));
        }
        if (m_cloud->hasColors()) {
            for (int ch = 0; ch < 3; ++ch) {
                FieldDef fd;
                fd.name = (ch == 0) ? tr("Color (R)")
                          : (ch == 1) ? tr("Color (G)")
                                      : tr("Color (B)");
                fd.sfIndex = -10 - ch;
                m_fields.append(fd);
                auto* item = new QListWidgetItem(fd.name, m_fieldList);
                item->setForeground(seriesColor(m_fields.size() - 1));
            }
        }
        if (m_mesh) {
            auto* texTable = m_mesh->getTexCoordinatesTable();
            if (texTable && texTable->size() > 0) {
                {
                    FieldDef fd;
                    fd.name = QStringLiteral("TCoords_S");
                    fd.sfIndex = -50;
                    m_fields.append(fd);
                    auto* item = new QListWidgetItem(fd.name, m_fieldList);
                    item->setForeground(seriesColor(m_fields.size() - 1));
                }
                {
                    FieldDef fd;
                    fd.name = QStringLiteral("TCoords_T");
                    fd.sfIndex = -51;
                    m_fields.append(fd);
                    auto* item = new QListWidgetItem(fd.name, m_fieldList);
                    item->setForeground(seriesColor(m_fields.size() - 1));
                }
                {
                    FieldDef fd;
                    fd.name = QStringLiteral("TCoords_Magnitude");
                    fd.sfIndex = -52;
                    m_fields.append(fd);
                    auto* item = new QListWidgetItem(fd.name, m_fieldList);
                    item->setForeground(seriesColor(m_fields.size() - 1));
                }
            }
        }
        m_titleLabel->setText(
                title() + QString(" - %1 (%2 pts)")
                                  .arg(entity->getName())
                                  .arg(m_cloud->size()));

        if (m_fieldList->count() > 0) {
            if (m_chartType == PARALLEL_COORDINATES) {
                for (int i = 0; i < m_fieldList->count(); ++i)
                    m_fieldList->item(i)->setSelected(true);
            } else if (m_chartType == PLOT_MATRIX || m_chartType == BOX_CHART) {
                int limit = qMin(m_fieldList->count(), 4);
                for (int i = 0; i < limit; ++i) {
                    m_fieldList->item(i)->setSelected(true);
                }
            } else {
                m_fieldList->item(0)->setSelected(true);
            }
        }
        populateSeriesTable();
    } else {
        m_titleLabel->setText(title() + " - No data");
    }

    if (m_sourceCombo) {
        m_sourceCombo->blockSignals(true);
        for (int i = 0; i < m_sourceCombo->count(); ++i) {
            auto* stored = reinterpret_cast<ccHObject*>(
                    m_sourceCombo->itemData(i).value<quintptr>());
            if (stored == entity) {
                m_sourceCombo->setCurrentIndex(i);
                break;
            }
        }
        if (!entity && m_sourceCombo->count() > 0) {
            m_sourceCombo->setCurrentIndex(0);
        }
        m_sourceCombo->blockSignals(false);
    }

    if (m_xArrayCombo) {
        m_xArrayCombo->blockSignals(true);
        m_xArrayCombo->clear();
        for (int i = 0; i < m_fields.size(); ++i) {
            m_xArrayCombo->addItem(m_fields[i].name, i);
        }
        m_xArrayCombo->blockSignals(false);
    }

    m_fieldList->blockSignals(false);
    rebuildChart();
}

void vtkChartView::onEntitySelectionChanged(ccHObject* entity) {
    if (!entity) return;
    m_sourceCombo->blockSignals(true);
    refreshSourceCombo();
    for (int i = 0; i < m_sourceCombo->count(); ++i) {
        auto* stored = reinterpret_cast<ccHObject*>(
                m_sourceCombo->itemData(i).value<quintptr>());
        if (stored == entity) {
            m_sourceCombo->setCurrentIndex(i);
            break;
        }
    }
    m_sourceCombo->blockSignals(false);
    setEntity(entity);
}

void vtkChartView::setEntityListProvider(EntityListProvider provider) {
    m_entityListProvider = std::move(provider);
    refreshSourceCombo();
}

void vtkChartView::refreshSourceCombo() {
    if (!m_entityListProvider || !m_sourceCombo) return;

    ccHObject* current = nullptr;
    int curIdx = m_sourceCombo->currentIndex();
    if (curIdx >= 0) {
        current = reinterpret_cast<ccHObject*>(
                m_sourceCombo->itemData(curIdx).value<quintptr>());
    }

    m_sourceCombo->blockSignals(true);
    m_sourceCombo->clear();
    m_sourceCombo->addItem(tr("None"), QVariant::fromValue<quintptr>(0));

    auto entities = m_entityListProvider();
    int newIdx = 0;
    for (int i = 0; i < entities.size(); ++i) {
        ccHObject* e = entities[i];
        if (!e) continue;
        m_sourceCombo->addItem(
                e->getName(),
                QVariant::fromValue<quintptr>(reinterpret_cast<quintptr>(e)));
        if (e == current) {
            newIdx = m_sourceCombo->count() - 1;
        }
    }
    m_sourceCombo->setCurrentIndex(newIdx);
    m_sourceCombo->blockSignals(false);
}

void vtkChartView::onSourceComboChanged(int index) {
    if (index < 0) return;
    auto ptr = m_sourceCombo->itemData(index).value<quintptr>();
    auto* entity = reinterpret_cast<ccHObject*>(ptr);
    setEntity(entity);
}

void vtkChartView::onSourceComboAboutToShow() {
    refreshSourceCombo();
}

bool vtkChartView::eventFilter(QObject* obj, QEvent* event) {
    if (obj == m_sourceCombo && event->type() == QEvent::MouseButtonPress) {
        refreshSourceCombo();
    }
    return QWidget::eventFilter(obj, event);
}

void vtkChartView::onSelectionChanged() {
    rebuildChart();
}

void vtkChartView::onBinCountChanged(int) {
    rebuildChart();
}

void vtkChartView::onExportPNG() {
    QString path = QFileDialog::getSaveFileName(
            this, tr("Export Chart as PNG"), QString(), tr("PNG (*.png)"));
    if (path.isEmpty()) return;

    auto* rw = m_vtkWidget->renderWindow();
    rw->Render();

    vtkNew<vtkWindowToImageFilter> filter;
    filter->SetInput(rw);
    filter->SetScale(2);
    filter->SetInputBufferTypeToRGBA();
    filter->Update();

    vtkNew<vtkPNGWriter> writer;
    writer->SetFileName(path.toUtf8().constData());
    writer->SetInputConnection(filter->GetOutputPort());
    writer->Write();
}

void vtkChartView::onExportCSV() {
    if (!m_cloud || m_fields.isEmpty()) return;

    QString path = QFileDialog::getSaveFileName(
            this, tr("Export Chart Data as CSV"), QString(),
            tr("CSV (*.csv)"));
    if (path.isEmpty()) return;

    QList<int> selectedFields;
    for (int i = 0; i < m_fieldList->count(); ++i) {
        if (m_fieldList->item(i)->isSelected()) {
            selectedFields.append(i);
        }
    }
    if (selectedFields.isEmpty()) return;

    unsigned pointCount = m_cloud->size();
    if (pointCount == 0) return;

    auto getFieldValue = [&](int fieldIdx, unsigned ptIdx) -> float {
        const auto& fd = m_fields[fieldIdx];
        if (fd.sfIndex >= 0) {
            auto* sf = m_cloud->getScalarField(fd.sfIndex);
            return sf ? sf->getValue(ptIdx) : 0.0f;
        }
        if (fd.sfIndex >= -22 && fd.sfIndex <= -20) {
            const CCVector3* pt = m_cloud->getPoint(ptIdx);
            if (!pt) return 0.0f;
            int axis = -(fd.sfIndex + 20);
            return static_cast<float>(pt->u[axis]);
        }
        if (fd.sfIndex >= -32 && fd.sfIndex <= -30) {
            int axis = -(fd.sfIndex + 30);
            if (m_cloud->hasNormals()) {
                const CCVector3& n = m_cloud->getPointNormal(ptIdx);
                return static_cast<float>(n.u[axis]);
            }
            if (hasComputedNormals()) {
                return computedNormal(ptIdx, axis);
            }
            return 0.0f;
        }
        if (fd.sfIndex >= -12 && fd.sfIndex <= -10 && m_cloud->hasColors()) {
            const ecvColor::Rgb& c = m_cloud->getPointColor(ptIdx);
            int ch = -(fd.sfIndex + 10);
            if (ch == 0) return c.r;
            if (ch == 1) return c.g;
            return c.b;
        }
        if (fd.sfIndex == -40) {
            const CCVector3* pt = m_cloud->getPoint(ptIdx);
            if (!pt) return 0.0f;
            return std::sqrt(pt->x * pt->x + pt->y * pt->y + pt->z * pt->z);
        }
        if (fd.sfIndex == -41) {
            if (m_cloud->hasNormals()) {
                const CCVector3& n = m_cloud->getPointNormal(ptIdx);
                return std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
            }
            if (hasComputedNormals()) {
                float nx = computedNormal(ptIdx, 0);
                float ny = computedNormal(ptIdx, 1);
                float nz = computedNormal(ptIdx, 2);
                return std::sqrt(nx * nx + ny * ny + nz * nz);
            }
            return 0.0f;
        }
        if (fd.sfIndex >= -52 && fd.sfIndex <= -50 && m_mesh) {
            auto* texTable = m_mesh->getTexCoordinatesTable();
            if (texTable && ptIdx < texTable->size()) {
                const TexCoords2D& tc = texTable->getValue(ptIdx);
                if (fd.sfIndex == -50) return tc.tx;
                if (fd.sfIndex == -51) return tc.ty;
                return std::sqrt(tc.tx * tc.tx + tc.ty * tc.ty);
            }
            return 0.0f;
        }
        return 0.0f;
    };

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream out(&file);

    out << "Index";
    for (int fi : selectedFields) {
        out << "," << m_fields[fi].name;
    }
    out << "\n";

    for (unsigned i = 0; i < pointCount; ++i) {
        out << i;
        for (int fi : selectedFields) {
            out << "," << getFieldValue(fi, i);
        }
        out << "\n";
    }
}

void vtkChartView::onResetZoom() {
    if (m_chart) {
        m_chart->RecalculateBounds();
    } else if (m_chartType == PLOT_MATRIX && m_contextView) {
        auto* spm = vtkScatterPlotMatrix::SafeDownCast(
                m_contextView->GetScene()->GetItem(0));
        if (spm) spm->Update();
    }
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onChartTitleChanged(const QString& text) {
    if (m_chart) {
        m_chart->SetTitle(text.toStdString());
        m_chart->SetShowLegend(m_chart->GetShowLegend());
    } else if (m_chartType == PLOT_MATRIX && m_contextView) {
        auto* spm = vtkScatterPlotMatrix::SafeDownCast(
                m_contextView->GetScene()->GetItem(0));
        if (spm) spm->SetTitle(text.toStdString());
    }
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onToggleLegend(bool show) {
    if (!m_chart) return;
    m_chart->SetShowLegend(show);
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onToggleGridLines(bool show) {
    if (!m_chart) return;
    for (int i = 0; i < 2; ++i) {
        auto* axis = m_chart->GetAxis(i);
        if (axis) {
            axis->SetGridVisible(show);
        }
    }
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onActiveAxisChanged(int /*index*/) {
    if (!m_chart || !m_axisSelectCombo) return;
    int axisId = m_axisSelectCombo->currentData().toInt();
    auto* axis = m_chart->GetAxis(axisId);
    if (!axis) return;

    m_axisVisibleCheck->blockSignals(true);
    m_axisVisibleCheck->setChecked(axis->GetVisible());
    m_axisVisibleCheck->blockSignals(false);

    m_axisTitleEdit->blockSignals(true);
    m_axisTitleEdit->setText(QString::fromStdString(axis->GetTitle()));
    m_axisTitleEdit->blockSignals(false);

    m_logScaleCheck->blockSignals(true);
    m_logScaleCheck->setChecked(axis->GetLogScale());
    m_logScaleCheck->blockSignals(false);

    m_notationCombo->blockSignals(true);
    int notation = axis->GetNotation();
    int notIdx = m_notationCombo->findData(notation);
    m_notationCombo->setCurrentIndex(notIdx >= 0 ? notIdx : 0);
    m_notationCombo->blockSignals(false);

    m_axisPrecSpin->blockSignals(true);
    m_axisPrecSpin->setValue(axis->GetPrecision());
    m_axisPrecSpin->blockSignals(false);

    bool isFixed = (axis->GetBehavior() == vtkAxis::FIXED);
    m_customRangeCheck->blockSignals(true);
    m_customRangeCheck->setChecked(isFixed);
    m_customRangeCheck->blockSignals(false);
    m_rangeMinSpin->setEnabled(isFixed);
    m_rangeMaxSpin->setEnabled(isFixed);
    if (isFixed) {
        m_rangeMinSpin->blockSignals(true);
        m_rangeMinSpin->setValue(axis->GetUnscaledMinimum());
        m_rangeMinSpin->blockSignals(false);
        m_rangeMaxSpin->blockSignals(true);
        m_rangeMaxSpin->setValue(axis->GetUnscaledMaximum());
        m_rangeMaxSpin->blockSignals(false);
    }
}

void vtkChartView::onToggleAxisVisible(bool visible) {
    if (!m_chart || !m_axisSelectCombo) return;
    int axisId = m_axisSelectCombo->currentData().toInt();
    auto* axis = m_chart->GetAxis(axisId);
    if (axis) axis->SetVisible(visible);
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onAxisTitleChanged(const QString& text) {
    if (!m_chart || !m_axisSelectCombo) return;
    int axisId = m_axisSelectCombo->currentData().toInt();
    auto* axis = m_chart->GetAxis(axisId);
    if (axis) axis->SetTitle(text.toStdString());
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onToggleLogScale(bool log) {
    if (!m_chart) return;

    if (m_chartType == PARALLEL_COORDINATES) {
        auto* pcChart = vtkChartParallelCoordinates::SafeDownCast(m_chart);
        if (pcChart) {
            for (int i = 0; i < pcChart->GetNumberOfAxes(); ++i) {
                auto* axis = pcChart->GetAxis(i);
                if (!axis) continue;
                axis->SetLogScale(log);
                if (log) {
                    axis->SetUnscaledMinimumLimit(1e-10);
                } else {
                    axis->SetUnscaledMinimumLimit(0.0);
                }
            }
        }
    } else {
        auto* leftAxis = m_chart->GetAxis(vtkAxis::LEFT);
        if (leftAxis) {
            leftAxis->SetLogScale(log);
            if (log) {
                leftAxis->SetUnscaledMinimumLimit(
                        m_chartType == HISTOGRAM ? 0.5 : 1e-10);
                if (m_chartType == HISTOGRAM)
                    leftAxis->SetMinimumLimit(0.5);
            } else {
                leftAxis->SetUnscaledMinimumLimit(0.0);
                leftAxis->SetMinimumLimit(0.0);
            }
        }
        auto* bottomAxis = m_chart->GetAxis(vtkAxis::BOTTOM);
        if (bottomAxis && m_chartType != HISTOGRAM) {
            bottomAxis->SetLogScale(log);
            if (log) {
                bottomAxis->SetUnscaledMinimumLimit(1e-10);
            } else {
                bottomAxis->SetUnscaledMinimumLimit(0.0);
            }
        }
    }

    m_chart->RecalculateBounds();
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onAxisNotationChanged(int index) {
    if (!m_chart || !m_axisSelectCombo) return;
    int axisId = m_axisSelectCombo->currentData().toInt();
    int notation = m_notationCombo->itemData(index).toInt();
    auto* axis = m_chart->GetAxis(axisId);
    if (axis) axis->SetNotation(notation);
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onAxisPrecisionChanged(int prec) {
    if (!m_chart || !m_axisSelectCombo) return;
    int axisId = m_axisSelectCombo->currentData().toInt();
    auto* axis = m_chart->GetAxis(axisId);
    if (axis) axis->SetPrecision(prec);
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onToggleCustomRange(bool use) {
    m_rangeMinSpin->setEnabled(use);
    m_rangeMaxSpin->setEnabled(use);
    if (!m_chart || !m_axisSelectCombo) return;
    int axisId = m_axisSelectCombo->currentData().toInt();
    auto* axis = m_chart->GetAxis(axisId);
    if (!axis) return;
    if (use) {
        axis->SetBehavior(vtkAxis::FIXED);
        axis->SetUnscaledMinimum(m_rangeMinSpin->value());
        axis->SetUnscaledMaximum(m_rangeMaxSpin->value());
    } else {
        axis->SetBehavior(vtkAxis::AUTO);
        m_chart->RecalculateBounds();
    }
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onAxisRangeChanged() {
    if (!m_chart || !m_customRangeCheck || !m_customRangeCheck->isChecked())
        return;
    if (!m_axisSelectCombo) return;
    int axisId = m_axisSelectCombo->currentData().toInt();
    auto* axis = m_chart->GetAxis(axisId);
    if (!axis) return;
    axis->SetUnscaledMinimum(m_rangeMinSpin->value());
    axis->SetUnscaledMaximum(m_rangeMaxSpin->value());
    if (m_vtkWidget && m_vtkWidget->renderWindow())
        m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::onTooltipNotationChanged(int index) {
    m_tooltipNotation = index;
    applyTooltipFormat();
}

void vtkChartView::onTooltipPrecisionChanged(int prec) {
    m_tooltipPrecision = prec;
    applyTooltipFormat();
}

void vtkChartView::applyTooltipFormat() {
    if (!m_chart) return;
    auto* chartXY = vtkChartXY::SafeDownCast(m_chart);
    if (!chartXY) return;

    auto* tooltip = chartXY->GetTooltip();
    if (!tooltip) return;

    auto* textProp = tooltip->GetTextProperties();
    if (textProp) {
        textProp->SetFontSize(12);
    }
}

void vtkChartView::setupChartSelectionCallback() {
    if (!m_chart || !m_cloud) return;

    m_chart->SetSelectionMethod(vtkChart::SELECTION_ROWS);

    auto* link = m_chart->GetAnnotationLink();
    if (!link) return;

    link->AddObserver(
            vtkCommand::AnnotationChangedEvent, this,
            &vtkChartView::onChartAnnotationChanged);
}

void vtkChartView::onChartAnnotationChanged() {
    if (!m_chart || !m_cloud) return;

    auto* link = m_chart->GetAnnotationLink();
    if (!link) return;

    auto* sel = link->GetCurrentSelection();
    if (!sel || sel->GetNumberOfNodes() == 0) {
        emit pointsHighlighted(m_cloud, {});
        return;
    }

    QVector<unsigned> indices;
    for (unsigned n = 0; n < sel->GetNumberOfNodes(); ++n) {
        auto* node = sel->GetNode(n);
        if (!node) continue;
        auto* baseArr = node->GetSelectionList();
        if (!baseArr) continue;
        auto* arr = vtkIdTypeArray::SafeDownCast(baseArr);
        if (!arr) continue;
        for (vtkIdType i = 0; i < arr->GetNumberOfTuples(); ++i) {
            vtkIdType row = arr->GetValue(i);
            unsigned ptIdx =
                    static_cast<unsigned>(row) * m_sampleStride;
            if (ptIdx < m_cloud->size()) indices.append(ptIdx);
        }
    }

    emit pointsHighlighted(m_cloud, indices);
}

void vtkChartView::rebuildChart() {
    if (m_chart) m_chart->ClearPlots();

    if (!m_cloud || m_fields.isEmpty()) {
        m_vtkWidget->renderWindow()->Render();
        return;
    }

    QList<int> selectedFields;
    for (int i = 0; i < m_fieldList->count(); ++i) {
        if (m_fieldList->item(i)->isSelected()) {
            selectedFields.append(i);
        }
    }
    if (selectedFields.isEmpty()) {
        m_vtkWidget->renderWindow()->Render();
        return;
    }

    unsigned pointCount = m_cloud->size();
    if (pointCount == 0) {
        m_vtkWidget->renderWindow()->Render();
        return;
    }

    auto getFieldValue = [&](int fieldIdx, unsigned ptIdx) -> float {
        const auto& fd = m_fields[fieldIdx];
        if (fd.sfIndex >= 0) {
            auto* sf = m_cloud->getScalarField(fd.sfIndex);
            return sf ? sf->getValue(ptIdx) : 0.0f;
        }
        if (fd.sfIndex >= -22 && fd.sfIndex <= -20) {
            const CCVector3* pt = m_cloud->getPoint(ptIdx);
            if (!pt) return 0.0f;
            int axis = -(fd.sfIndex + 20);
            return static_cast<float>(pt->u[axis]);
        }
        if (fd.sfIndex >= -32 && fd.sfIndex <= -30) {
            int axis = -(fd.sfIndex + 30);
            if (m_cloud->hasNormals()) {
                const CCVector3& n = m_cloud->getPointNormal(ptIdx);
                return static_cast<float>(n.u[axis]);
            }
            if (hasComputedNormals()) {
                return computedNormal(ptIdx, axis);
            }
            return 0.0f;
        }
        if (fd.sfIndex >= -12 && fd.sfIndex <= -10 && m_cloud->hasColors()) {
            const ecvColor::Rgb& c = m_cloud->getPointColor(ptIdx);
            int ch = -(fd.sfIndex + 10);
            if (ch == 0) return c.r;
            if (ch == 1) return c.g;
            return c.b;
        }
        if (fd.sfIndex == -40) {
            const CCVector3* pt = m_cloud->getPoint(ptIdx);
            if (!pt) return 0.0f;
            return std::sqrt(pt->x * pt->x + pt->y * pt->y + pt->z * pt->z);
        }
        if (fd.sfIndex == -41) {
            if (m_cloud->hasNormals()) {
                const CCVector3& n = m_cloud->getPointNormal(ptIdx);
                return std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
            }
            if (hasComputedNormals()) {
                float nx = computedNormal(ptIdx, 0);
                float ny = computedNormal(ptIdx, 1);
                float nz = computedNormal(ptIdx, 2);
                return std::sqrt(nx * nx + ny * ny + nz * nz);
            }
            return 0.0f;
        }
        if (fd.sfIndex >= -52 && fd.sfIndex <= -50 && m_mesh) {
            auto* texTable = m_mesh->getTexCoordinatesTable();
            if (texTable && ptIdx < texTable->size()) {
                const TexCoords2D& tc = texTable->getValue(ptIdx);
                int comp = -(fd.sfIndex + 50);
                if (comp == 0) return tc.tx;
                if (comp == 1) return tc.ty;
                return std::sqrt(tc.tx * tc.tx + tc.ty * tc.ty);
            }
            return 0.0f;
        }
        return 0.0f;
    };

    m_sampleStride = 1;
    if (pointCount > m_maxChartPoints)
        m_sampleStride = pointCount / m_maxChartPoints;
    unsigned sampleCount =
            (pointCount + m_sampleStride - 1) / m_sampleStride;

    if (m_chartType == PLOT_MATRIX) {
        rebuildPlotMatrix(selectedFields, sampleCount, pointCount,
                          getFieldValue);
    } else if (m_chartType == HISTOGRAM) {
        rebuildHistogram(selectedFields, pointCount, getFieldValue);
    } else if (m_chartType == BOX_CHART) {
        rebuildBoxChart(selectedFields, sampleCount, pointCount, getFieldValue);
    } else if (m_chartType == PARALLEL_COORDINATES) {
        rebuildParallelCoordinates(selectedFields, sampleCount, pointCount,
                                   getFieldValue);
    } else {
        rebuildXYChart(selectedFields, sampleCount, pointCount, getFieldValue);
    }

    if (m_chart) {
        m_chart->RecalculateBounds();
        if (m_chartTitleEdit && !m_chartTitleEdit->text().isEmpty())
            m_chart->SetTitle(m_chartTitleEdit->text().toStdString());
        if (m_legendCheck)
            m_chart->SetShowLegend(m_legendCheck->isChecked());
        for (int i = 0; i < 2; ++i) {
            auto* axis = m_chart->GetAxis(i);
            if (!axis) continue;
            if (m_gridCheck) axis->SetGridVisible(m_gridCheck->isChecked());
            if (m_notationCombo)
                axis->SetNotation(
                        m_notationCombo->currentData().toInt());
            if (m_axisPrecSpin) axis->SetPrecision(m_axisPrecSpin->value());
        }
        if (m_logScaleCheck && m_logScaleCheck->isChecked()) {
            auto* leftAxis = m_chart->GetAxis(vtkAxis::LEFT);
            if (leftAxis) {
                leftAxis->SetLogScale(true);
                leftAxis->SetUnscaledMinimumLimit(1e-10);
            }
        }
        if (m_customRangeCheck && m_customRangeCheck->isChecked()) {
            auto* leftAxis = m_chart->GetAxis(vtkAxis::LEFT);
            if (leftAxis) {
                leftAxis->SetBehavior(vtkAxis::FIXED);
                leftAxis->SetUnscaledMinimum(m_rangeMinSpin->value());
                leftAxis->SetUnscaledMaximum(m_rangeMaxSpin->value());
            }
        }
    }
    setupChartSelectionCallback();
    m_vtkWidget->renderWindow()->Render();
}

void vtkChartView::rebuildXYChart(const QList<int>& selectedFields,
                                  unsigned sampleCount, unsigned pointCount,
                                  const FieldValueFn& getFieldValue) {
    vtkNew<vtkTable> table;

    bool useIndex = !m_useIndexXAxis || m_useIndexXAxis->isChecked();
    int xFieldIdx = -1;
    if (!useIndex && m_xArrayCombo && m_xArrayCombo->currentIndex() >= 0) {
        xFieldIdx = m_xArrayCombo->currentData().toInt();
    }

    if (useIndex || xFieldIdx < 0 || xFieldIdx >= m_fields.size()) {
        vtkNew<vtkIntArray> indexArr;
        indexArr->SetName("Point Index");
        indexArr->SetNumberOfTuples(sampleCount);
        for (unsigned s = 0; s < sampleCount; ++s) {
            unsigned idx = s * m_sampleStride;
            if (idx >= pointCount) idx = pointCount - 1;
            indexArr->SetValue(s, idx);
        }
        table->AddColumn(indexArr);
    } else {
        QByteArray xName = m_fields[xFieldIdx].name.toUtf8();
        vtkNew<vtkFloatArray> xArr;
        xArr->SetName(xName.constData());
        xArr->SetNumberOfTuples(sampleCount);
        for (unsigned s = 0; s < sampleCount; ++s) {
            unsigned idx = s * m_sampleStride;
            if (idx >= pointCount) idx = pointCount - 1;
            xArr->SetValue(s, getFieldValue(xFieldIdx, idx));
        }
        table->AddColumn(xArr);
    }

    for (int si = 0; si < selectedFields.size(); ++si) {
        int fi = selectedFields[si];
        QByteArray nameBytes = m_fields[fi].name.toUtf8();
        vtkNew<vtkFloatArray> valueArr;
        valueArr->SetName(nameBytes.constData());
        valueArr->SetNumberOfTuples(sampleCount);

        for (unsigned s = 0; s < sampleCount; ++s) {
            unsigned idx = s * m_sampleStride;
            if (idx >= pointCount) idx = pointCount - 1;
            valueArr->SetValue(s, getFieldValue(fi, idx));
        }
        table->AddColumn(valueArr);
    }

    float markerSize = 4.0f;
    if (m_markerSizeSpin) markerSize = m_markerSizeSpin->value();
    float lineWidth = 2.0f;
    if (m_lineThickSpin) lineWidth = m_lineThickSpin->value();
    int lineStyle = 1;
    if (m_lineStyleCombo) lineStyle = m_lineStyleCombo->currentData().toInt();

    for (int si = 0; si < selectedFields.size(); ++si) {
        int fi = selectedFields[si];
        int plotType;
        switch (m_chartType) {
            case BAR_CHART:
                plotType = vtkChart::BAR;
                break;
            case POINT_CHART:
                plotType = vtkChart::POINTS;
                break;
            case QUARTILE_CHART:
                plotType = vtkChart::LINE;
                break;
            default:
                plotType = vtkChart::LINE;
                break;
        }
        auto* plot = m_chart->AddPlot(plotType);
        plot->SetInputData(table, 0, si + 1);
        QColor col = seriesColor(fi);
        plot->SetColor(col.redF(), col.greenF(), col.blueF(), 1.0);
        plot->GetPen()->SetColorF(col.redF(), col.greenF(), col.blueF());
        plot->GetPen()->SetOpacityF(1.0);

        if (m_chartType == LINE_CHART) {
            plot->SetWidth(lineWidth);
            plot->GetPen()->SetLineType(lineStyle);
            int mkrStyle = m_markerStyleCombo
                                   ? m_markerStyleCombo->currentData().toInt()
                                   : -1;
            auto* plotPts = vtkPlotPoints::SafeDownCast(plot);
            if (plotPts && mkrStyle >= 0) {
                plotPts->SetMarkerStyle(mkrStyle);
                plotPts->SetMarkerSize(markerSize);
            }
        } else if (m_chartType == POINT_CHART) {
            auto* plotPts = vtkPlotPoints::SafeDownCast(plot);
            if (plotPts) {
                plotPts->SetMarkerStyle(vtkPlotPoints::CROSS);
                plotPts->SetMarkerSize(markerSize);
            }
            if (lineStyle > 0) {
                plot->SetWidth(lineWidth);
                plot->GetPen()->SetLineType(lineStyle);
            } else {
                plot->GetPen()->SetLineType(vtkPen::NO_PEN);
            }
        } else if (m_chartType == BAR_CHART) {
            plot->SetWidth(1.0);
        }
    }

    QString xTitle = useIndex ? tr("Point Index")
                              : (xFieldIdx >= 0 ? m_fields[xFieldIdx].name
                                                : tr("Point Index"));
    m_chart->GetAxis(vtkAxis::BOTTOM)->SetTitle(xTitle.toStdString());
    m_chart->GetAxis(vtkAxis::LEFT)->SetTitle("Value");
}

void vtkChartView::rebuildHistogram(const QList<int>& selectedFields,
                                    unsigned pointCount,
                                    const FieldValueFn& getFieldValue) {
    int numBins = m_binSpin ? m_binSpin->value() : 50;

    for (int si = 0; si < selectedFields.size(); ++si) {
        int fi = selectedFields[si];
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();

        for (unsigned i = 0; i < pointCount; ++i) {
            float v = getFieldValue(fi, i);
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
        if (maxVal <= minVal) maxVal = minVal + 1.0f;
        float binWidth = (maxVal - minVal) / numBins;

        vtkNew<vtkTable> table;
        vtkNew<vtkFloatArray> binCenters;
        binCenters->SetName("Bin Center");
        binCenters->SetNumberOfTuples(numBins);

        QByteArray nameBytes = m_fields[fi].name.toUtf8();
        vtkNew<vtkFloatArray> counts;
        counts->SetName(nameBytes.constData());
        counts->SetNumberOfTuples(numBins);

        for (int b = 0; b < numBins; ++b) {
            binCenters->SetValue(b, minVal + (b + 0.5f) * binWidth);
            counts->SetValue(b, 0.0f);
        }

        for (unsigned i = 0; i < pointCount; ++i) {
            float v = getFieldValue(fi, i);
            int bin = static_cast<int>((v - minVal) / binWidth);
            if (bin >= numBins) bin = numBins - 1;
            if (bin < 0) bin = 0;
            counts->SetValue(bin, counts->GetValue(bin) + 1.0f);
        }

        bool isLog = m_logScaleCheck && m_logScaleCheck->isChecked();
        if (isLog) {
            for (int b = 0; b < numBins; ++b) {
                if (counts->GetValue(b) < 0.5f)
                    counts->SetValue(b, 0.5f);
            }
        }

        table->AddColumn(binCenters);
        table->AddColumn(counts);

        auto* bar = m_chart->AddPlot(vtkChart::BAR);
        bar->SetInputData(table, 0, 1);
        QColor col = seriesColor(fi);
        if (selectedFields.size() == 1 && m_histColor.isValid())
            col = m_histColor;
        float alpha = (selectedFields.size() <= 2) ? 0.85f : 0.65f;
        bar->SetColor(col.redF(), col.greenF(), col.blueF(), alpha);
        bar->SetWidth(1.0);
    }

    m_chart->GetAxis(vtkAxis::BOTTOM)->SetTitle(
            selectedFields.size() == 1
                    ? m_fields[selectedFields.first()].name.toStdString()
                    : std::string("Value"));
    m_chart->GetAxis(vtkAxis::LEFT)->SetTitle("Frequency");
}

void vtkChartView::rebuildBoxChart(const QList<int>& selectedFields,
                                   unsigned /*sampleCount*/, unsigned pointCount,
                                   const FieldValueFn& getFieldValue) {
    auto* boxChart = vtkChartBox::SafeDownCast(m_chart);
    if (!boxChart) return;

    vtkNew<vtkTable> table;
    table->SetNumberOfRows(5);

    QVector<QByteArray> nameStorage;
    for (int si = 0; si < selectedFields.size(); ++si) {
        int fi = selectedFields[si];
        nameStorage.append(m_fields[fi].name.toUtf8());

        std::vector<float> values;
        values.reserve(pointCount);
        for (unsigned i = 0; i < pointCount; ++i) {
            values.push_back(getFieldValue(fi, i));
        }
        std::sort(values.begin(), values.end());

        float minVal = values.front();
        float maxVal = values.back();
        float median = values[values.size() / 2];
        float q1 = values[values.size() / 4];
        float q3 = values[3 * values.size() / 4];

        vtkNew<vtkFloatArray> col;
        col->SetName(nameStorage.last().constData());
        col->SetNumberOfTuples(5);
        col->SetValue(0, minVal);
        col->SetValue(1, q1);
        col->SetValue(2, median);
        col->SetValue(3, q3);
        col->SetValue(4, maxVal);
        table->AddColumn(col);
    }

    auto* boxPlot = vtkPlotBox::SafeDownCast(boxChart->GetPlot(0));
    if (!boxPlot) return;
    boxPlot->SetInputData(table);
    for (int si = 0; si < selectedFields.size(); ++si) {
        int fi = selectedFields[si];
        boxChart->SetColumnVisibility(nameStorage[si].constData(), true);
        QColor col = seriesColor(fi);
        double rgb[3] = {col.redF(), col.greenF(), col.blueF()};
        boxPlot->SetColumnColor(nameStorage[si].constData(), rgb);
    }
}

void vtkChartView::rebuildParallelCoordinates(
        const QList<int>& selectedFields, unsigned sampleCount,
        unsigned pointCount, const FieldValueFn& getFieldValue) {
    auto* pcChart = vtkChartParallelCoordinates::SafeDownCast(m_chart);
    if (!pcChart) return;

    vtkNew<vtkTable> table;
    table->SetNumberOfRows(sampleCount);

    QVector<QByteArray> nameStorage;
    for (int si = 0; si < selectedFields.size(); ++si) {
        int fi = selectedFields[si];
        nameStorage.append(m_fields[fi].name.toUtf8());
        vtkNew<vtkFloatArray> col;
        col->SetName(nameStorage.last().constData());
        col->SetNumberOfTuples(sampleCount);

        for (unsigned s = 0; s < sampleCount; ++s) {
            unsigned idx = s * m_sampleStride;
            if (idx >= pointCount) idx = pointCount - 1;
            col->SetValue(s, getFieldValue(fi, idx));
        }
        table->AddColumn(col);
    }

    auto* pcPlot = pcChart->GetPlot(0);
    if (!pcPlot) return;
    pcPlot->SetInputData(table);
    for (int si = 0; si < selectedFields.size(); ++si) {
        pcChart->SetColumnVisibility(nameStorage[si].constData(), true);
    }

    float opacity = 0.1f;
    if (m_lineOpacitySpin) opacity = m_lineOpacitySpin->value();
    float lineWidth = 2.0f;
    if (m_lineThickSpin) lineWidth = m_lineThickSpin->value();

    auto* plot = pcPlot;
    if (plot && selectedFields.size() > 0) {
        plot->SetColor(0.0, 0.0, 0.0, opacity);
        plot->SetWidth(lineWidth);
        int lineStyle = 1;
        if (m_lineStyleCombo)
            lineStyle = m_lineStyleCombo->currentData().toInt();
        plot->GetPen()->SetLineType(lineStyle > 0 ? lineStyle
                                                   : vtkPen::SOLID_LINE);
    }

    for (int i = 0; i < pcChart->GetNumberOfAxes(); ++i) {
        auto* axis = pcChart->GetAxis(i);
        if (!axis) continue;
        if (i < nameStorage.size()) {
            axis->SetTitle(nameStorage[i].constData());
        }
        axis->GetLabelProperties()->SetColor(0.0, 0.0, 0.0);
        axis->GetLabelProperties()->SetFontSize(12);
        axis->GetLabelProperties()->SetFontFamilyToArial();
        axis->GetTitleProperties()->SetColor(0.0, 0.0, 0.0);
        axis->GetTitleProperties()->SetFontSize(12);
        axis->GetTitleProperties()->SetBold(1);
        axis->GetTitleProperties()->SetFontFamilyToArial();
        axis->GetPen()->SetColor(0, 0, 0);
        axis->GetPen()->SetWidth(1.0);
        axis->GetGridPen()->SetColorF(0.85, 0.85, 0.85);
    }
}

void vtkChartView::rebuildPlotMatrix(const QList<int>& selectedFields,
                                     unsigned sampleCount,
                                     unsigned pointCount,
                                     const FieldValueFn& getFieldValue) {
    auto* spm = vtkScatterPlotMatrix::SafeDownCast(
            m_contextView->GetScene()->GetItem(0));
    if (!spm) return;

    vtkNew<vtkTable> table;
    table->SetNumberOfRows(sampleCount);

    QVector<QByteArray> nameStorage;
    for (int si = 0; si < selectedFields.size(); ++si) {
        int fi = selectedFields[si];
        nameStorage.append(m_fields[fi].name.toUtf8());
        vtkNew<vtkFloatArray> col;
        col->SetName(nameStorage.last().constData());
        col->SetNumberOfTuples(sampleCount);

        for (unsigned s = 0; s < sampleCount; ++s) {
            unsigned idx = s * m_sampleStride;
            if (idx >= pointCount) idx = pointCount - 1;
            col->SetValue(s, getFieldValue(fi, idx));
        }
        table->AddColumn(col);
    }

    spm->SetInput(table);

    for (int si = 0; si < selectedFields.size(); ++si) {
        int fi = selectedFields[si];
        spm->SetColumnVisibility(nameStorage[si].constData(), true);
    }

    spm->SetTitle("Plot Matrix");
    spm->GetTitleProperties()->SetColor(0.0, 0.0, 0.0);
    spm->SetGutter(vtkVector2f(20.0f, 20.0f));
    spm->SetPadding(0.02f);
    spm->SetBorders(60, 50, 60, 50);

    spm->SetPlotMarkerStyle(vtkScatterPlotMatrix::SCATTERPLOT,
                             vtkPlotPoints::CIRCLE);
    spm->SetPlotMarkerSize(vtkScatterPlotMatrix::SCATTERPLOT, 3.0f);
    spm->SetPlotMarkerStyle(vtkScatterPlotMatrix::ACTIVEPLOT,
                             vtkPlotPoints::CIRCLE);
    spm->SetPlotMarkerSize(vtkScatterPlotMatrix::ACTIVEPLOT, 5.0f);

    spm->SetScatterPlotSelectedRowColumnColor(
            vtkColor4ub(255, 0, 0, 255));
    spm->SetScatterPlotSelectedActiveColor(
            vtkColor4ub(255, 0, 0, 255));

    spm->SetPlotColor(vtkScatterPlotMatrix::SCATTERPLOT,
                       vtkColor4ub(0, 0, 0, 200));
    spm->SetPlotColor(vtkScatterPlotMatrix::ACTIVEPLOT,
                       vtkColor4ub(255, 0, 0, 255));
    spm->SetPlotColor(vtkScatterPlotMatrix::HISTOGRAM,
                       vtkColor4ub(0, 0, 0, 255));

    spm->SetBackgroundColor(vtkScatterPlotMatrix::SCATTERPLOT,
                             vtkColor4ub(240, 240, 240, 255));
    spm->SetBackgroundColor(vtkScatterPlotMatrix::ACTIVEPLOT,
                             vtkColor4ub(220, 220, 220, 255));
    spm->SetBackgroundColor(vtkScatterPlotMatrix::HISTOGRAM,
                             vtkColor4ub(200, 255, 200, 200));

    spm->Update();
}

void vtkChartView::populateSeriesTable() {
    if (!m_seriesTable) return;
    m_seriesTable->blockSignals(true);
    m_seriesTable->setRowCount(0);

    for (int i = 0; i < m_fieldList->count(); ++i) {
        auto* item = m_fieldList->item(i);
        int row = m_seriesTable->rowCount();
        m_seriesTable->insertRow(row);

        auto* visItem = new QTableWidgetItem();
        visItem->setCheckState(item->isSelected() ? Qt::Checked : Qt::Unchecked);
        visItem->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
        m_seriesTable->setItem(row, 0, visItem);

        QColor col = seriesColor(i);
        auto* colorItem = new QTableWidgetItem();
        colorItem->setBackground(col);
        colorItem->setFlags(Qt::ItemIsEnabled);
        m_seriesTable->setItem(row, 1, colorItem);

        auto* varItem = new QTableWidgetItem(item->text());
        varItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        m_seriesTable->setItem(row, 2, varItem);

        auto* legendItem = new QTableWidgetItem(item->text());
        legendItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsEditable |
                             Qt::ItemIsSelectable);
        m_seriesTable->setItem(row, 3, legendItem);
    }
    m_seriesTable->blockSignals(false);
}

void vtkChartView::onSeriesTableChanged() {
    if (!m_seriesTable || !m_fieldList) return;
    m_fieldList->blockSignals(true);
    for (int r = 0; r < m_seriesTable->rowCount() &&
                    r < m_fieldList->count(); ++r) {
        auto* visItem = m_seriesTable->item(r, 0);
        if (visItem) {
            bool sel = (visItem->checkState() == Qt::Checked);
            m_fieldList->item(r)->setSelected(sel);
        }
    }
    m_fieldList->blockSignals(false);
    rebuildChart();
}

void vtkChartView::onSeriesTableCellClicked(int row, int col) {
    if (col != 1 || !m_seriesTable) return;
    QColor current = seriesColor(row);
    QColor picked = QColorDialog::getColor(current, this,
                                           tr("Pick Series Color"));
    if (!picked.isValid()) return;
    m_customSeriesColors[row] = picked;
    auto* colorItem = m_seriesTable->item(row, 1);
    if (colorItem) colorItem->setBackground(picked);
    rebuildChart();
}
