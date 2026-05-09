// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkComparativeViewWidget.h"

#include <VTKExtensions/Views/vtkChartView.h>
#include <Visualization/vtkGLView.h>
#include <Visualization/VtkVis.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>

static constexpr int COMPARATIVE_SPACING = 2;

vtkComparativeViewWidget::vtkComparativeViewWidget(ComparativeType type,
                                                   QWidget* parent)
    : QWidget(parent), m_type(type) {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    buildToolbar();
    m_toolbar->setVisible(false);

    auto* toggleBar = new QWidget(this);
    toggleBar->setStyleSheet("QWidget { background: #2a2a2a; }");
    auto* toggleLay = new QHBoxLayout(toggleBar);
    toggleLay->setContentsMargins(4, 1, 4, 1);
    auto* toggleBtn = new QPushButton(
            tr("\xe2\x96\xb6 Grid Settings"), toggleBar);
    toggleBtn->setFlat(true);
    toggleBtn->setCursor(Qt::PointingHandCursor);
    toggleBtn->setStyleSheet(
            "QPushButton { color: #aaa; font-size: 11px; border: none; }"
            "QPushButton:hover { color: #fff; }");
    toggleLay->addWidget(toggleBtn);
    toggleLay->addStretch(1);
    mainLayout->addWidget(toggleBar);
    mainLayout->addWidget(m_toolbar);

    connect(toggleBtn, &QPushButton::clicked, this,
            [this, toggleBtn]() {
                bool show = !m_toolbar->isVisible();
                m_toolbar->setVisible(show);
                toggleBtn->setText(show ? tr("\xe2\x96\xbc Grid Settings")
                                       : tr("\xe2\x96\xb6 Grid Settings"));
            });

    auto* gridContainer = new QWidget(this);
    m_gridLayout = new QGridLayout(gridContainer);
    m_gridLayout->setContentsMargins(0, 0, 0, 0);
    m_gridLayout->setSpacing(COMPARATIVE_SPACING);
    mainLayout->addWidget(gridContainer, 1);
}

vtkComparativeViewWidget::~vtkComparativeViewWidget() = default;

QString vtkComparativeViewWidget::title() const {
    switch (m_type) {
        case RENDER:
            return tr("Render View (Comparative)");
        case LINE_CHART:
            return tr("Line Chart View (Comparative)");
        case BAR_CHART:
            return tr("Bar Chart View (Comparative)");
    }
    return tr("Comparative View");
}

void vtkComparativeViewWidget::setSpacing(int spacing) {
    m_spacing = spacing;
    if (m_gridLayout) {
        m_gridLayout->setSpacing(spacing);
    }
}

void vtkComparativeViewWidget::setDimensions(int rows, int cols) {
    if (rows < 1 || cols < 1 || (rows == m_rows && cols == m_cols)) return;

    // Clear existing sub-views
    for (auto* w : m_subWidgets) {
        m_gridLayout->removeWidget(w);
        w->setParent(nullptr);
        w->deleteLater();
    }
    m_subWidgets.clear();

    m_rows = rows;
    m_cols = cols;
    setupGrid();
}

void vtkComparativeViewWidget::setRenderViewFactory(
        RenderViewFactory factory) {
    m_renderFactory = factory;
    if (m_type == RENDER && m_subWidgets.isEmpty()) {
        setupGrid();
    }
}

void vtkComparativeViewWidget::setupGrid() {
    if (m_type == RENDER) {
        createRenderSubViews();
    } else {
        createChartSubViews();
    }
}

void vtkComparativeViewWidget::createRenderSubViews() {
    if (!m_renderFactory) return;

    for (int r = 0; r < m_rows; ++r) {
        for (int c = 0; c < m_cols; ++c) {
            auto* view = m_renderFactory();
            if (!view) continue;
            QWidget* viewWidget = view->asWidget();
            if (!viewWidget) continue;
            m_gridLayout->addWidget(viewWidget, r, c);
            m_subWidgets.append(viewWidget);
            emit subViewCreated(viewWidget);
        }
    }
}

void vtkComparativeViewWidget::createChartSubViews() {
    vtkChartView::ChartType chartType =
            (m_type == BAR_CHART) ? vtkChartView::BAR_CHART
                                  : vtkChartView::LINE_CHART;

    for (int r = 0; r < m_rows; ++r) {
        for (int c = 0; c < m_cols; ++c) {
            auto* chart = new vtkChartView(chartType, this);
            m_gridLayout->addWidget(chart, r, c);
            m_subWidgets.append(chart);
            emit subViewCreated(chart);
        }
    }
}

void vtkComparativeViewWidget::buildToolbar() {
    m_toolbar = new QWidget(this);
    auto* lay = new QHBoxLayout(m_toolbar);
    lay->setContentsMargins(4, 2, 4, 2);
    lay->setSpacing(4);
    m_toolbar->setStyleSheet("QWidget { background: #333; }");

    auto labelSS = QStringLiteral("QLabel { color: #ccc; }");
    auto spinSS = QStringLiteral(
            "QSpinBox { background: #1e1e1e; color: #ddd; border: 1px solid "
            "#555; border-radius: 3px; padding: 1px 2px; }");
    auto comboSS = QStringLiteral(
            "QComboBox { background: #1e1e1e; color: #ddd; border: 1px solid "
            "#555; border-radius: 3px; padding: 1px 4px; min-width: 80px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #2d2d2d; color: "
            "#ddd; }");
    auto btnSS = QStringLiteral(
            "QPushButton { background: #3a5f8f; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 2px 8px; }"
            "QPushButton:hover { background: #4a7fbf; }");

    auto* dimLabel = new QLabel(tr("<b>Grid:</b>"), m_toolbar);
    dimLabel->setStyleSheet(labelSS);
    lay->addWidget(dimLabel);

    m_rowSpin = new QSpinBox(m_toolbar);
    m_rowSpin->setRange(1, 8);
    m_rowSpin->setValue(m_rows);
    m_rowSpin->setPrefix(tr("R:"));
    m_rowSpin->setStyleSheet(spinSS);
    lay->addWidget(m_rowSpin);

    auto* xLabel = new QLabel(tr("x"), m_toolbar);
    xLabel->setStyleSheet(labelSS);
    lay->addWidget(xLabel);

    m_colSpin = new QSpinBox(m_toolbar);
    m_colSpin->setRange(1, 8);
    m_colSpin->setValue(m_cols);
    m_colSpin->setPrefix(tr("C:"));
    m_colSpin->setStyleSheet(spinSS);
    lay->addWidget(m_colSpin);

    auto* spLabel = new QLabel(tr("Sp:"), m_toolbar);
    spLabel->setStyleSheet(labelSS);
    lay->addWidget(spLabel);

    auto* spacingSpin = new QSpinBox(m_toolbar);
    spacingSpin->setRange(0, 20);
    spacingSpin->setValue(m_spacing);
    spacingSpin->setStyleSheet(spinSS);
    spacingSpin->setToolTip(tr("Grid spacing (ParaView Spacing property)"));
    lay->addWidget(spacingSpin);

    connect(spacingSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &vtkComparativeViewWidget::setSpacing);

    lay->addWidget(new QLabel(QStringLiteral("|"), m_toolbar));

    auto* cueLabel = new QLabel(tr("<b>Cue:</b>"), m_toolbar);
    cueLabel->setStyleSheet(labelSS);
    lay->addWidget(cueLabel);

    auto dspinSS = QStringLiteral(
            "QDoubleSpinBox { background: #1e1e1e; color: #ddd; border: 1px "
            "solid #555; border-radius: 3px; padding: 1px 2px; }");

    m_cueParamCombo = new QComboBox(m_toolbar);
    m_cueParamCombo->addItem(tr("None"), 0);
    m_cueParamCombo->addItem(tr("Azimuth"), 1);
    m_cueParamCombo->addItem(tr("Elevation"), 2);
    m_cueParamCombo->addItem(tr("Opacity"), 3);
    m_cueParamCombo->addItem(tr("Zoom"), 4);
    m_cueParamCombo->setStyleSheet(comboSS);
    m_cueParamCombo->setToolTip(
            tr("Parameter to sweep across sub-views "
               "(ParaView vtkPVComparativeAnimationCue)"));
    lay->addWidget(m_cueParamCombo);

    m_cueModeCombo = new QComboBox(m_toolbar);
    m_cueModeCombo->addItem(tr("X-Range"), 0);
    m_cueModeCombo->addItem(tr("Y-Range"), 1);
    m_cueModeCombo->addItem(tr("T-Range"), 2);
    m_cueModeCombo->setStyleSheet(comboSS);
    m_cueModeCombo->setToolTip(
            tr("Sweep mode: X=vary along columns, Y=vary along rows, "
               "T=vary across all (ParaView XRANGE/YRANGE/TRANGE)"));
    lay->addWidget(m_cueModeCombo);

    auto* minLabel = new QLabel(tr("Min:"), m_toolbar);
    minLabel->setStyleSheet(labelSS);
    lay->addWidget(minLabel);

    m_cueMinSpin = new QDoubleSpinBox(m_toolbar);
    m_cueMinSpin->setRange(-360, 360);
    m_cueMinSpin->setDecimals(1);
    m_cueMinSpin->setValue(0.0);
    m_cueMinSpin->setMaximumWidth(70);
    m_cueMinSpin->setStyleSheet(dspinSS);
    lay->addWidget(m_cueMinSpin);

    auto* maxLabel = new QLabel(tr("Max:"), m_toolbar);
    maxLabel->setStyleSheet(labelSS);
    lay->addWidget(maxLabel);

    m_cueMaxSpin = new QDoubleSpinBox(m_toolbar);
    m_cueMaxSpin->setRange(-360, 360);
    m_cueMaxSpin->setDecimals(1);
    m_cueMaxSpin->setValue(90.0);
    m_cueMaxSpin->setMaximumWidth(70);
    m_cueMaxSpin->setStyleSheet(dspinSS);
    lay->addWidget(m_cueMaxSpin);

    auto* playBtn = new QPushButton(tr("Apply"), m_toolbar);
    playBtn->setStyleSheet(btnSS);
    playBtn->setToolTip(tr("Apply parameter sweep to sub-views"));
    lay->addWidget(playBtn);

    lay->addWidget(new QLabel(QStringLiteral("|"), m_toolbar));

    auto checkSS = QStringLiteral("QCheckBox { color: #ccc; }");
    m_overlayCheck = new QCheckBox(tr("Overlay"), m_toolbar);
    m_overlayCheck->setStyleSheet(checkSS);
    m_overlayCheck->setToolTip(
            tr("Overlay all comparisons into first view "
               "(ParaView OverlayAllComparisons)"));
    lay->addWidget(m_overlayCheck);

    auto* screenshotBtn = new QPushButton(tr("Screenshot"), m_toolbar);
    screenshotBtn->setStyleSheet(btnSS);
    screenshotBtn->setToolTip(
            tr("Export stitched screenshot of all sub-views"));
    lay->addWidget(screenshotBtn);

    m_statusLabel = new QLabel(m_toolbar);
    m_statusLabel->setStyleSheet(
            "QLabel { color: #999; font-size: 11px; }");
    lay->addWidget(m_statusLabel);
    lay->addStretch(1);

    connect(m_rowSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &vtkComparativeViewWidget::onDimensionChanged);
    connect(m_colSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &vtkComparativeViewWidget::onDimensionChanged);
    connect(m_cueParamCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &vtkComparativeViewWidget::onCueParameterChanged);
    connect(playBtn, &QPushButton::clicked, this,
            &vtkComparativeViewWidget::onPlayCue);
    connect(m_overlayCheck, &QCheckBox::toggled, this,
            &vtkComparativeViewWidget::onToggleOverlay);
    connect(screenshotBtn, &QPushButton::clicked, this,
            &vtkComparativeViewWidget::onExportScreenshot);
}

void vtkComparativeViewWidget::onDimensionChanged() {
    int r = m_rowSpin ? m_rowSpin->value() : m_rows;
    int c = m_colSpin ? m_colSpin->value() : m_cols;
    setDimensions(r, c);
    m_statusLabel->setText(
            tr("%1x%2 = %3 views").arg(r).arg(c).arg(r * c));
}

void vtkComparativeViewWidget::onCueParameterChanged(int index) {
    if (m_statusLabel) {
        m_statusLabel->setText(
                tr("Cue: %1").arg(m_cueParamCombo->itemText(index)));
    }
}

void vtkComparativeViewWidget::onToggleOverlay(bool checked) {
    m_overlayMode = checked;
    if (m_subWidgets.size() <= 1) return;

    if (checked) {
        for (int i = 1; i < m_subWidgets.size(); ++i) {
            m_subWidgets[i]->setVisible(false);
        }
        m_gridLayout->setColumnStretch(0, 1);
        m_gridLayout->setRowStretch(0, 1);
    } else {
        for (int i = 0; i < m_subWidgets.size(); ++i) {
            m_subWidgets[i]->setVisible(true);
        }
    }

    if (m_statusLabel) {
        m_statusLabel->setText(checked ? tr("Overlay mode ON")
                                       : tr("Overlay mode OFF"));
    }
}

void vtkComparativeViewWidget::onExportScreenshot() {
    QString path = QFileDialog::getSaveFileName(
            this, tr("Save Comparative Screenshot"), QString(),
            tr("PNG (*.png);;JPEG (*.jpg)"));
    if (path.isEmpty()) return;

    QPixmap composite(size());
    render(&composite);
    composite.save(path);

    if (m_statusLabel) {
        m_statusLabel->setText(tr("Saved: %1").arg(path));
    }
}

void vtkComparativeViewWidget::onPlayCue() {
    applyCueToSubViews();
}

void vtkComparativeViewWidget::applyCueToSubViews() {
    if (!m_cueParamCombo || m_subWidgets.isEmpty()) return;

    int cueParam = m_cueParamCombo->currentData().toInt();
    if (cueParam <= 0) return;

    int cueMode = m_cueModeCombo ? m_cueModeCombo->currentData().toInt() : 2;
    double minVal = m_cueMinSpin ? m_cueMinSpin->value() : 0.0;
    double maxVal = m_cueMaxSpin ? m_cueMaxSpin->value() : 90.0;

    int dx = m_cols;
    int dy = m_rows;
    int applied = 0;

    for (int y = 0; y < dy; ++y) {
        for (int x = 0; x < dx; ++x) {
            int index = y * dx + x;
            if (index >= m_subWidgets.size()) break;

            double value = minVal;
            switch (cueMode) {
                case 0:  // XRANGE
                    value = (dx > 1)
                            ? minVal + x * (maxVal - minVal) / (dx - 1)
                            : minVal;
                    break;
                case 1:  // YRANGE
                    value = (dy > 1)
                            ? minVal + y * (maxVal - minVal) / (dy - 1)
                            : minVal;
                    break;
                case 2:  // TRANGE
                default:
                    value = (dx * dy > 1)
                            ? minVal + (y * dx + x) * (maxVal - minVal) /
                                               (dx * dy - 1)
                            : minVal;
                    break;
            }

            if (m_type == RENDER) {
                auto* view = qobject_cast<vtkGLView*>(
                        m_subWidgets[index]->findChild<vtkGLView*>());
                if (!view) {
                    auto* obj = dynamic_cast<vtkGLView*>(
                            static_cast<QObject*>(m_subWidgets[index]));
                    if (obj) view = obj;
                }
                if (!view) {
                    view = dynamic_cast<vtkGLView*>(
                            reinterpret_cast<ecvGenericGLDisplay*>(
                                    m_subWidgets[index]->property(
                                            "vtkGLView").value<void*>()));
                }
                if (view && view->getVisualizer3D()) {
                    auto renCollection =
                            view->getVisualizer3D()->getRendererCollection();
                    if (renCollection) {
                        renCollection->InitTraversal();
                        auto* ren = renCollection->GetNextItem();
                        if (ren) {
                            auto* cam = ren->GetActiveCamera();
                            if (cam) {
                                switch (cueParam) {
                                    case 1:  // Azimuth
                                        cam->Azimuth(value);
                                        break;
                                    case 2:  // Elevation
                                        cam->Elevation(value);
                                        break;
                                    case 4:  // Zoom
                                        if (value > 0)
                                            cam->Zoom(value);
                                        break;
                                    default:
                                        break;
                                }
                                ren->ResetCameraClippingRange();
                            }
                        }
                    }
                    ++applied;
                }
            }

            if (cueParam == 3 && m_type == RENDER) {
                auto* view = qobject_cast<vtkGLView*>(
                        m_subWidgets[index]->findChild<vtkGLView*>());
                if (!view) {
                    view = dynamic_cast<vtkGLView*>(
                            static_cast<QObject*>(m_subWidgets[index]));
                }
                if (view && view->getVisualizer3D()) {
                    auto renCollection =
                            view->getVisualizer3D()->getRendererCollection();
                    if (renCollection) {
                        renCollection->InitTraversal();
                        auto* ren = renCollection->GetNextItem();
                        if (ren) {
                            auto* actors = ren->GetActors();
                            if (actors) {
                                actors->InitTraversal();
                                vtkActor* actor = nullptr;
                                while ((actor = actors->GetNextActor())) {
                                    double opacity =
                                            qBound(0.0, value / 100.0, 1.0);
                                    actor->GetProperty()->SetOpacity(opacity);
                                }
                            }
                            ren->Modified();
                        }
                    }
                    ++applied;
                }
            }
        }
    }

    for (auto* widget : m_subWidgets) {
        auto* view = qobject_cast<vtkGLView*>(
                widget->findChild<vtkGLView*>());
        if (!view) {
            view = dynamic_cast<vtkGLView*>(static_cast<QObject*>(widget));
        }
        if (view && view->getVisualizer3D()) {
            auto rw = view->getVisualizer3D()->getRenderWindow();
            if (rw) rw->Render();
        }
    }

    if (m_statusLabel) {
        m_statusLabel->setText(
                tr("Applied %1 [%2..%3] to %4/%5 views")
                        .arg(m_cueParamCombo->currentText())
                        .arg(minVal, 0, 'f', 1)
                        .arg(maxVal, 0, 'f', 1)
                        .arg(applied)
                        .arg(m_subWidgets.size()));
    }
}
