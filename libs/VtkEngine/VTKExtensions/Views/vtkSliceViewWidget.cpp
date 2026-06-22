// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkSliceViewWidget.h"

#include <VTKExtensions/Widgets/QVTKWidgetCustom.h>
#include <Visualization/vtkGLView.h>

#include <QCheckBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>

#include "vtkMultiSliceAxisWidget.h"

vtkSliceViewWidget::vtkSliceViewWidget(vtkGLView* view, QWidget* parent)
    : QWidget(parent), m_view(view) {
    setAutoFillBackground(true);

    auto* outerLayout = new QVBoxLayout(this);
    outerLayout->setContentsMargins(0, 0, 0, 0);
    outerLayout->setSpacing(0);

    auto* toolbar = new QWidget(this);
    auto* tbLayout = new QHBoxLayout(toolbar);
    tbLayout->setContentsMargins(2, 1, 2, 1);
    tbLayout->setSpacing(4);

    auto* outlineCheck = new QCheckBox(tr("Outline"), toolbar);
    outlineCheck->setChecked(m_outlineVisible);
    tbLayout->addWidget(outlineCheck);
    connect(outlineCheck, &QCheckBox::toggled, this,
            &vtkSliceViewWidget::setOutlineVisible);

    auto* helpLabel = new QLabel(tr("Double-click axis to add slice, drag to "
                                    "move, right-click to remove"),
                                 toolbar);
    helpLabel->setStyleSheet("color: gray; font-size: 9px;");
    tbLayout->addWidget(helpLabel);
    tbLayout->addStretch(1);

    m_statusLabel = new QLabel(toolbar);
    m_statusLabel->setContentsMargins(4, 0, 4, 0);
    tbLayout->addWidget(m_statusLabel);

    outerLayout->addWidget(toolbar);

    auto* gridWidget = new QWidget(this);
    auto* grid = new QGridLayout(gridWidget);
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setSpacing(0);

    m_yAxis = new vtkMultiSliceAxisWidget(vtkMultiSliceAxisWidget::Y_AXIS,
                                          gridWidget);
    m_xAxis = new vtkMultiSliceAxisWidget(vtkMultiSliceAxisWidget::X_AXIS,
                                          gridWidget);
    m_zAxis = new vtkMultiSliceAxisWidget(vtkMultiSliceAxisWidget::Z_AXIS,
                                          gridWidget);

    m_xAxis->setAxisTitle(QStringLiteral("X"));
    m_yAxis->setAxisTitle(QStringLiteral("Y"));
    m_zAxis->setAxisTitle(QStringLiteral("Z"));

    grid->addWidget(m_yAxis, 0, 1);
    grid->addWidget(m_xAxis, 1, 0);
    grid->addWidget(view->getVtkWidget(), 1, 1);
    grid->addWidget(m_zAxis, 1, 2);

    grid->setColumnStretch(1, 1);
    grid->setRowStretch(1, 1);

    outerLayout->addWidget(gridWidget, 1);

    connect(m_xAxis, &vtkMultiSliceAxisWidget::slicePositionsChanged, this,
            &vtkSliceViewWidget::onSlicePositionsChanged);
    connect(m_yAxis, &vtkMultiSliceAxisWidget::slicePositionsChanged, this,
            &vtkSliceViewWidget::onSlicePositionsChanged);
    connect(m_zAxis, &vtkMultiSliceAxisWidget::slicePositionsChanged, this,
            &vtkSliceViewWidget::onSlicePositionsChanged);
}

vtkSliceViewWidget::~vtkSliceViewWidget() = default;

void vtkSliceViewWidget::setDataBounds(const double bounds[6]) {
    m_xAxis->setDataRange(bounds[0], bounds[1]);
    m_yAxis->setDataRange(bounds[2], bounds[3]);
    m_zAxis->setDataRange(bounds[4], bounds[5]);

    if (m_xAxis->slicePositions().isEmpty() &&
        m_yAxis->slicePositions().isEmpty() &&
        m_zAxis->slicePositions().isEmpty()) {
        double cx = (bounds[0] + bounds[1]) * 0.5;
        double cy = (bounds[2] + bounds[3]) * 0.5;
        double cz = (bounds[4] + bounds[5]) * 0.5;
        m_xAxis->setSlicePositions({cx});
        m_yAxis->setSlicePositions({cy});
        m_zAxis->setSlicePositions({cz});
        onSlicePositionsChanged(0, m_xAxis->slicePositions());
        onSlicePositionsChanged(1, m_yAxis->slicePositions());
        onSlicePositionsChanged(2, m_zAxis->slicePositions());
    }
}

void vtkSliceViewWidget::setOutlineVisible(bool visible) {
    m_outlineVisible = visible;
    if (m_view) {
        m_view->setOutlineVisible(visible);
    }
}

void vtkSliceViewWidget::onSlicePositionsChanged(
        int axis, const QVector<double>& positions) {
    if (!m_view) return;
    std::vector<double> vec(positions.begin(), positions.end());
    m_view->setMultiSlicePositions(axis, vec);

    int total = m_xAxis->slicePositions().size() +
                m_yAxis->slicePositions().size() +
                m_zAxis->slicePositions().size();
    if (m_statusLabel) {
        const char* names[] = {"X", "Y", "Z"};
        m_statusLabel->setText(tr("%1 slice(s) - %2: %3")
                                       .arg(total)
                                       .arg(names[axis])
                                       .arg(positions.size()));
    }
}
