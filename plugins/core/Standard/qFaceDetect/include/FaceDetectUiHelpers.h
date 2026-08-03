// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

namespace FaceDetectUi {

constexpr int kFormLabelColumnWidth = 92;
constexpr int kCompactNumericWidth = 76;
constexpr int kBrowseButtonWidth = 72;
constexpr int kCompactPreviewSize = 72;

/** Left-aligned label for two-column form grids. */
inline QLabel* makeFormLabel(const QString& text) {
    auto* label = new QLabel(text);
    label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    label->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    return label;
}

/** Muted helper text beside action buttons (e.g. Use test data). */
inline QLabel* makeHelperCaption(const QString& text) {
    auto* label = new QLabel(text);
    label->setWordWrap(true);
    label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    label->setStyleSheet(QStringLiteral("color: palette(mid); font-size: 11px;"));
    return label;
}

inline QPushButton* makeBrowseButton(const QString& text, QWidget* parent = nullptr) {
    auto* btn = new QPushButton(text, parent);
    btn->setFixedWidth(kBrowseButtonWidth);
    btn->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    return btn;
}

/** Uniform two-column form: label | field | label | field. */
inline void setupTwoColumnFormGrid(QGridLayout* grid,
                                   int labelColumnWidth = kFormLabelColumnWidth) {
    if (!grid) return;
    grid->setHorizontalSpacing(6);
    grid->setVerticalSpacing(2);
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setColumnMinimumWidth(0, labelColumnWidth);
    grid->setColumnMinimumWidth(2, labelColumnWidth);
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(3, 1);
}

inline void setupCompactMainLayout(QVBoxLayout* layout) {
    if (!layout) return;
    layout->setContentsMargins(4, 8, 4, 2);
    layout->setSpacing(4);
}

/** Tab pane spacing so top controls (e.g. Use test data) do not overlap tabs. */
inline void applyTabWidgetPaneStyle(QTabWidget* tabs) {
    if (!tabs) return;
    tabs->setDocumentMode(false);
    tabs->setStyleSheet(QStringLiteral(
            "QTabWidget::pane { border: 0; padding: 8px 4px 4px 4px; top: 0px; }"
            "QTabBar::tab { padding: 4px 10px; min-height: 18px; }"));
}

inline void tightenGroupBox(QGroupBox* box) {
    if (!box) return;
    box->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    if (auto* grid = qobject_cast<QGridLayout*>(box->layout())) {
        grid->setContentsMargins(6, 4, 6, 4);
        grid->setVerticalSpacing(2);
    } else if (auto* vbox = qobject_cast<QVBoxLayout*>(box->layout())) {
        vbox->setContentsMargins(6, 4, 6, 4);
        vbox->setSpacing(3);
    }
}

inline void makeCompactDoubleSpin(QDoubleSpinBox* spin) {
    if (!spin) return;
    spin->setMaximumWidth(kCompactNumericWidth);
    spin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

inline void makeCompactSpin(QSpinBox* spin) {
    if (!spin) return;
    spin->setMaximumWidth(kCompactNumericWidth);
    spin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

inline void tightenFormGrid(QGridLayout* grid, int valueColumnStretch = 1) {
    setupTwoColumnFormGrid(grid);
    if (!grid) return;
    grid->setColumnStretch(1, valueColumnStretch);
    grid->setColumnStretch(3, valueColumnStretch);
}

inline QDoubleSpinBox* makeMinDetectionScoreSpin(QWidget* parent = nullptr,
                                               const QString& tooltip = {}) {
    auto* spin = new QDoubleSpinBox(parent);
    spin->setRange(0.0, 1.0);
    spin->setSingleStep(0.05);
    spin->setValue(0.5);
    makeCompactDoubleSpin(spin);
    if (!tooltip.isEmpty()) spin->setToolTip(tooltip);
    return spin;
}

}  // namespace FaceDetectUi
