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
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvViewManager.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
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

    m_titleLabel = new QLabel(title() + " - No data", this);
    m_titleLabel->setAlignment(Qt::AlignCenter);
    m_titleLabel->setStyleSheet(
            "QLabel { background: #2d2d2d; color: #ccc; padding: 4px; "
            "font-weight: bold; }");
    layout->addWidget(m_titleLabel);

    auto* toolbar = new QWidget(this);
    auto* tbLayout = new QHBoxLayout(toolbar);
    tbLayout->setContentsMargins(4, 2, 4, 2);
    tbLayout->setSpacing(4);
    toolbar->setStyleSheet("QWidget { background: #333; }");

    auto* fieldLabel = new QLabel(tr("Fields:"), toolbar);
    fieldLabel->setStyleSheet("QLabel { color: #ccc; }");
    tbLayout->addWidget(fieldLabel);

    m_fieldList = new QListWidget(toolbar);
    m_fieldList->setSelectionMode(QAbstractItemView::MultiSelection);
    m_fieldList->setMaximumHeight(80);
    m_fieldList->setStyleSheet(
            "QListWidget { background: #1e1e1e; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; }"
            "QListWidget::item:selected { background: #264f78; }");
    tbLayout->addWidget(m_fieldList, 1);

    if (m_chartType == HISTOGRAM) {
        auto* binLabel = new QLabel(tr("Bins:"), toolbar);
        binLabel->setStyleSheet("QLabel { color: #ccc; }");
        tbLayout->addWidget(binLabel);

        m_binSpin = new QSpinBox(toolbar);
        m_binSpin->setRange(5, 500);
        m_binSpin->setValue(50);
        m_binSpin->setStyleSheet(
                "QSpinBox { background: #1e1e1e; color: #ddd; border: 1px "
                "solid #555; border-radius: 3px; padding: 2px; }");
        tbLayout->addWidget(m_binSpin);
    }

    m_linkCheck = new QCheckBox(tr("Link 3D"), toolbar);
    m_linkCheck->setStyleSheet("QCheckBox { color: #ccc; }");
    m_linkCheck->setToolTip(
            tr("Highlight selected chart points in the 3D view"));
    tbLayout->addWidget(m_linkCheck);

    m_resetZoomBtn = new QPushButton(tr("Reset Zoom"), toolbar);
    auto btnSS = QStringLiteral(
            "QPushButton { background: #3a5f8f; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 2px 8px; }"
            "QPushButton:hover { background: #4a7fbf; }");
    m_resetZoomBtn->setStyleSheet(btnSS);
    tbLayout->addWidget(m_resetZoomBtn);

    m_exportPngBtn = new QPushButton(tr("PNG"), toolbar);
    m_exportPngBtn->setStyleSheet(btnSS);
    m_exportPngBtn->setToolTip(tr("Export chart as PNG image"));
    tbLayout->addWidget(m_exportPngBtn);

    m_exportCsvBtn = new QPushButton(tr("CSV"), toolbar);
    m_exportCsvBtn->setStyleSheet(btnSS);
    m_exportCsvBtn->setToolTip(tr("Export chart data as CSV"));
    tbLayout->addWidget(m_exportCsvBtn);

    tbLayout->addWidget(
            new QLabel(QStringLiteral("|"), toolbar));

    auto* titleLabel = new QLabel(tr("Title:"), toolbar);
    titleLabel->setStyleSheet("QLabel { color: #ccc; }");
    tbLayout->addWidget(titleLabel);

    m_chartTitleEdit = new QLineEdit(toolbar);
    m_chartTitleEdit->setPlaceholderText(tr("Chart title..."));
    m_chartTitleEdit->setMaximumWidth(150);
    m_chartTitleEdit->setStyleSheet(
            "QLineEdit { background: #1e1e1e; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 2px; }");
    tbLayout->addWidget(m_chartTitleEdit);

    m_legendCheck = new QCheckBox(tr("Legend"), toolbar);
    m_legendCheck->setStyleSheet("QCheckBox { color: #ccc; }");
    m_legendCheck->setChecked(true);
    m_legendCheck->setToolTip(tr("Show/hide chart legend"));
    tbLayout->addWidget(m_legendCheck);

    m_gridCheck = new QCheckBox(tr("Grid"), toolbar);
    m_gridCheck->setStyleSheet("QCheckBox { color: #ccc; }");
    m_gridCheck->setChecked(true);
    m_gridCheck->setToolTip(tr("Show/hide grid lines"));
    tbLayout->addWidget(m_gridCheck);

    auto comboSS = QStringLiteral(
            "QComboBox { background: #1e1e1e; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 1px 4px; "
            "min-width: 50px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #2d2d2d; "
            "color: #ddd; }");
    auto spinSS = QStringLiteral(
            "QSpinBox { background: #1e1e1e; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 1px 2px; }");
    auto editSS = QStringLiteral(
            "QLineEdit { background: #1e1e1e; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 2px; }");
    auto checkSS = QStringLiteral("QCheckBox { color: #ccc; }");

    tbLayout->addWidget(new QLabel(QStringLiteral("|"), toolbar));

    auto* axisLabel = new QLabel(tr("Axis:"), toolbar);
    axisLabel->setStyleSheet("QLabel { color: #ccc; }");
    tbLayout->addWidget(axisLabel);

    m_axisSelectCombo = new QComboBox(toolbar);
    m_axisSelectCombo->addItem(tr("Left"), vtkAxis::LEFT);
    m_axisSelectCombo->addItem(tr("Bottom"), vtkAxis::BOTTOM);
    m_axisSelectCombo->addItem(tr("Right"), vtkAxis::RIGHT);
    m_axisSelectCombo->addItem(tr("Top"), vtkAxis::TOP);
    m_axisSelectCombo->setStyleSheet(comboSS);
    m_axisSelectCombo->setToolTip(
            tr("Select axis to configure (ParaView 4-axis config)"));
    tbLayout->addWidget(m_axisSelectCombo);

    m_axisVisibleCheck = new QCheckBox(tr("Vis"), toolbar);
    m_axisVisibleCheck->setStyleSheet(checkSS);
    m_axisVisibleCheck->setChecked(true);
    m_axisVisibleCheck->setToolTip(tr("Show/hide axis"));
    tbLayout->addWidget(m_axisVisibleCheck);

    m_axisTitleEdit = new QLineEdit(toolbar);
    m_axisTitleEdit->setPlaceholderText(tr("Axis title..."));
    m_axisTitleEdit->setMaximumWidth(100);
    m_axisTitleEdit->setStyleSheet(editSS);
    m_axisTitleEdit->setToolTip(tr("Axis title (ParaView AxisTitle)"));
    tbLayout->addWidget(m_axisTitleEdit);

    m_logScaleCheck = new QCheckBox(tr("Log"), toolbar);
    m_logScaleCheck->setStyleSheet(checkSS);
    m_logScaleCheck->setToolTip(tr("Logarithmic scale for selected axis"));
    tbLayout->addWidget(m_logScaleCheck);

    auto* notLabel = new QLabel(tr("Not:"), toolbar);
    notLabel->setStyleSheet("QLabel { color: #ccc; }");
    tbLayout->addWidget(notLabel);

    m_notationCombo = new QComboBox(toolbar);
    m_notationCombo->addItem(tr("Mixed"), 0);
    m_notationCombo->addItem(tr("Scientific"), 1);
    m_notationCombo->addItem(tr("Fixed"), 2);
    m_notationCombo->setStyleSheet(comboSS);
    m_notationCombo->setToolTip(tr("Axis label notation"));
    tbLayout->addWidget(m_notationCombo);

    auto* precLabel = new QLabel(tr("Prec:"), toolbar);
    precLabel->setStyleSheet("QLabel { color: #ccc; }");
    tbLayout->addWidget(precLabel);

    m_axisPrecSpin = new QSpinBox(toolbar);
    m_axisPrecSpin->setRange(0, 15);
    m_axisPrecSpin->setValue(2);
    m_axisPrecSpin->setStyleSheet(spinSS);
    m_axisPrecSpin->setToolTip(tr("Axis label precision"));
    tbLayout->addWidget(m_axisPrecSpin);

    m_customRangeCheck = new QCheckBox(tr("Custom"), toolbar);
    m_customRangeCheck->setStyleSheet(checkSS);
    m_customRangeCheck->setToolTip(tr("Use custom range for selected axis"));
    tbLayout->addWidget(m_customRangeCheck);

    m_rangeMinSpin = new QDoubleSpinBox(toolbar);
    m_rangeMinSpin->setRange(-1e12, 1e12);
    m_rangeMinSpin->setDecimals(4);
    m_rangeMinSpin->setValue(0.0);
    m_rangeMinSpin->setEnabled(false);
    m_rangeMinSpin->setMaximumWidth(80);
    m_rangeMinSpin->setStyleSheet(
            "QDoubleSpinBox { background: #1e1e1e; color: #ddd; border: "
            "1px solid #555; border-radius: 3px; padding: 1px; }");
    tbLayout->addWidget(m_rangeMinSpin);

    m_rangeMaxSpin = new QDoubleSpinBox(toolbar);
    m_rangeMaxSpin->setRange(-1e12, 1e12);
    m_rangeMaxSpin->setDecimals(4);
    m_rangeMaxSpin->setValue(10.0);
    m_rangeMaxSpin->setEnabled(false);
    m_rangeMaxSpin->setMaximumWidth(80);
    m_rangeMaxSpin->setStyleSheet(
            "QDoubleSpinBox { background: #1e1e1e; color: #ddd; border: "
            "1px solid #555; border-radius: 3px; padding: 1px; }");
    tbLayout->addWidget(m_rangeMaxSpin);

    tbLayout->addWidget(new QLabel(QStringLiteral("|"), toolbar));

    auto* tipLabel = new QLabel(tr("Tip:"), toolbar);
    tipLabel->setStyleSheet("QLabel { color: #ccc; }");
    tbLayout->addWidget(tipLabel);

    m_tooltipNotationCombo = new QComboBox(toolbar);
    m_tooltipNotationCombo->addItem(tr("Mixed"), 0);
    m_tooltipNotationCombo->addItem(tr("Sci"), 1);
    m_tooltipNotationCombo->addItem(tr("Fix"), 2);
    m_tooltipNotationCombo->setStyleSheet(comboSS);
    m_tooltipNotationCombo->setToolTip(
            tr("Tooltip notation (ParaView TooltipNotation)"));
    tbLayout->addWidget(m_tooltipNotationCombo);

    m_tooltipPrecSpin = new QSpinBox(toolbar);
    m_tooltipPrecSpin->setRange(0, 15);
    m_tooltipPrecSpin->setValue(6);
    m_tooltipPrecSpin->setStyleSheet(spinSS);
    m_tooltipPrecSpin->setToolTip(
            tr("Tooltip precision (ParaView TooltipPrecision)"));
    tbLayout->addWidget(m_tooltipPrecSpin);

    layout->addWidget(toolbar);

    m_vtkWidget = new QVTKOpenGLNativeWidget(this);
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    m_vtkWidget->setRenderWindow(renderWindow);
    layout->addWidget(m_vtkWidget, 1);

    m_contextView = vtkContextView::New();
    m_contextView->SetRenderWindow(renderWindow);
    m_contextView->GetRenderer()->SetBackground(0.12, 0.12, 0.12);

    if (m_chartType == PLOT_MATRIX) {
        auto* spm = vtkScatterPlotMatrix::New();
        m_contextView->GetScene()->AddItem(spm);
        spm->Delete();
        spm->SetBackgroundColor(0, vtkColor4ub(30, 30, 30, 255));
        spm->SetBackgroundColor(1, vtkColor4ub(40, 40, 40, 255));
        spm->SetBackgroundColor(2, vtkColor4ub(50, 50, 50, 255));
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

        m_chart->SetActionToButton(vtkChart::PAN,
                                   vtkContextMouseEvent::LEFT_BUTTON);
        m_chart->SetActionToButton(vtkChart::ZOOM,
                                   vtkContextMouseEvent::RIGHT_BUTTON);
        m_chart->SetActionToButton(vtkChart::ZOOM_AXIS,
                                   vtkContextMouseEvent::MIDDLE_BUTTON);

        for (int i = 0; i < 4; ++i) {
            auto* axis = m_chart->GetAxis(i);
            if (!axis) continue;
            axis->GetLabelProperties()->SetColor(0.8, 0.8, 0.8);
            axis->GetTitleProperties()->SetColor(0.8, 0.8, 0.8);
            axis->GetPen()->SetColor(100, 100, 100);
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

QColor vtkChartView::seriesColor(int index) const {
    return kPalette[index % (sizeof(kPalette) / sizeof(kPalette[0]))];
}

void vtkChartView::setEntity(ccHObject* entity) {
    m_cloud = entity ? ccHObjectCaster::ToPointCloud(entity) : nullptr;
    m_fields.clear();

    m_fieldList->blockSignals(true);
    m_fieldList->clear();

    if (m_cloud) {
        unsigned sfCount = m_cloud->getNumberOfScalarFields();
        for (unsigned i = 0; i < sfCount; ++i) {
            FieldDef fd;
            fd.name = QString::fromUtf8(m_cloud->getScalarFieldName(i));
            fd.sfIndex = static_cast<int>(i);
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
        m_titleLabel->setText(
                title() + QString(" - %1 (%2 pts)")
                                  .arg(entity->getName())
                                  .arg(m_cloud->size()));

        if (m_fieldList->count() > 0) {
            m_fieldList->item(0)->setSelected(true);
        }
    } else {
        m_titleLabel->setText(title() + " - No data");
    }

    m_fieldList->blockSignals(false);
    rebuildChart();
}

void vtkChartView::onEntitySelectionChanged(ccHObject* entity) {
    setEntity(entity);
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
        if (fd.sfIndex <= -10 && m_cloud->hasColors()) {
            const ecvColor::Rgb& c = m_cloud->getPointColor(ptIdx);
            int ch = -(fd.sfIndex + 10);
            if (ch == 0) return c.r;
            if (ch == 1) return c.g;
            return c.b;
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
        m_vtkWidget->renderWindow()->Render();
    }
}

void vtkChartView::onChartTitleChanged(const QString& text) {
    if (!m_chart) return;
    m_chart->SetTitle(text.toStdString());
    m_chart->SetShowLegend(m_chart->GetShowLegend());
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
    if (!m_chart || !m_axisSelectCombo) return;
    int axisId = m_axisSelectCombo->currentData().toInt();
    auto* axis = m_chart->GetAxis(axisId);
    if (axis) {
        axis->SetLogScale(log);
        if (log) axis->SetUnscaledMinimumLimit(1e-10);
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
    if (!m_chart || !m_cloud || !m_linkCheck || !m_linkCheck->isChecked())
        return;

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
        if (fd.sfIndex <= -10 && m_cloud->hasColors()) {
            const ecvColor::Rgb& c = m_cloud->getPointColor(ptIdx);
            int ch = -(fd.sfIndex + 10);
            if (ch == 0) return c.r;
            if (ch == 1) return c.g;
            return c.b;
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
    } else if (m_chartType == IMAGE_CHART || m_chartType == QUARTILE_CHART) {
        rebuildXYChart(selectedFields, sampleCount, pointCount, getFieldValue);
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
    vtkNew<vtkIntArray> indexArr;
    indexArr->SetName("Index");
    indexArr->SetNumberOfTuples(sampleCount);
    for (unsigned s = 0; s < sampleCount; ++s) {
        unsigned idx = s * m_sampleStride;
        if (idx >= pointCount) idx = pointCount - 1;
        indexArr->SetValue(s, idx);
    }
    table->AddColumn(indexArr);

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
                plotType = vtkChart::AREA;
                break;
            default:
                plotType = vtkChart::LINE;
                break;
        }
        auto* plot = m_chart->AddPlot(plotType);
        plot->SetInputData(table, 0, si + 1);
        QColor col = seriesColor(fi);
        plot->SetColor(col.redF(), col.greenF(), col.blueF(), 1.0);
        if (m_chartType == LINE_CHART) {
            plot->SetWidth(1.5);
        } else if (m_chartType == POINT_CHART) {
            plot->SetWidth(3.0);
        }
    }

    m_chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Point Index");
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
        vtkNew<vtkIntArray> counts;
        counts->SetName(nameBytes.constData());
        counts->SetNumberOfTuples(numBins);

        for (int b = 0; b < numBins; ++b) {
            binCenters->SetValue(b, minVal + (b + 0.5f) * binWidth);
            counts->SetValue(b, 0);
        }

        for (unsigned i = 0; i < pointCount; ++i) {
            float v = getFieldValue(fi, i);
            int bin = static_cast<int>((v - minVal) / binWidth);
            if (bin >= numBins) bin = numBins - 1;
            if (bin < 0) bin = 0;
            counts->SetValue(bin, counts->GetValue(bin) + 1);
        }

        table->AddColumn(binCenters);
        table->AddColumn(counts);

        auto* bar = m_chart->AddPlot(vtkChart::BAR);
        bar->SetInputData(table, 0, 1);
        QColor col = seriesColor(fi);
        bar->SetColor(col.redF(), col.greenF(), col.blueF(),
                      0.75 / selectedFields.size() + 0.25);
    }

    m_chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Value");
    m_chart->GetAxis(vtkAxis::LEFT)->SetTitle("Count");
}

void vtkChartView::rebuildBoxChart(const QList<int>& selectedFields,
                                   unsigned sampleCount, unsigned pointCount,
                                   const FieldValueFn& getFieldValue) {
    auto* boxChart = vtkChartBox::SafeDownCast(m_chart);
    if (!boxChart) return;

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

    auto* plot = pcPlot;
    if (plot && selectedFields.size() > 0) {
        QColor col = seriesColor(0);
        plot->SetColor(col.redF(), col.greenF(), col.blueF(), 0.3);
        plot->SetWidth(1.0);
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
        QColor col = seriesColor(fi);
        spm->SetScatterPlotSelectedRowColumnColor(
                vtkColor4ub(col.red(), col.green(), col.blue(), 255));
    }

    spm->SetTitle("Plot Matrix");
    spm->GetTitleProperties()->SetColor(0.8, 0.8, 0.8);
    spm->Update();
}
