// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkChartView.h"

#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvViewManager.h>

#include <QCheckBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QTextStream>
#include <QTimer>
#include <QVBoxLayout>
#include <QVTKOpenGLNativeWidget.h>

#include <vtkAxis.h>
#include <vtkChart.h>
#include <vtkChartBox.h>
#include <vtkChartParallelCoordinates.h>
#include <vtkChartXY.h>
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

void vtkChartView::setupChartSelectionCallback() {
    if (!m_chart || !m_cloud || !m_linkCheck || !m_linkCheck->isChecked())
        return;

    m_chart->SetSelectionMethod(vtkChart::SELECTION_ROWS);
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
    if (pointCount > 10000) m_sampleStride = pointCount / 10000;
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

    if (m_chart) m_chart->RecalculateBounds();
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
