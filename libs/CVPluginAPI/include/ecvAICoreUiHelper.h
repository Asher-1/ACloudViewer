// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Shared UI helpers for AICore-based plugin dialogs (qDA3, qDeepLSD,
// qFaceDetect, qLightGlue, qFreeSplatter, qRFDetr, qRMBG, qYOLO).
//
// Provides DPI-aware sizing, consistent spacing/margins, unified button
// styling, and compact grid-layout helpers so every plugin dialog behaves
// consistently across platforms (Linux, macOS, Windows), screen resolutions,
// and Qt 5/6.
// ----------------------------------------------------------------------------

#pragma once

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QScreen>
#include <QSizePolicy>
#include <QSpinBox>
#include <QStyle>
#include <QTabWidget>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

// ---------------------------------------------------------------------------
//  DPI-aware logical-pixel helpers
// ---------------------------------------------------------------------------

namespace ecvAICoreUi {

/// Multiply a nominal (96-dpi) pixel size by the screen's logical DPI factor.
/// Keeps button heights, preview thumbnails, and minimum sizes consistent
/// across monitors with different scaling.
inline int dpiScaled(int px) {
    const QScreen* screen = QGuiApplication::primaryScreen();
    const qreal dpi = screen ? screen->logicalDotsPerInch() : 96.0;
    return qMax(px, qRound(px * dpi / 96.0));
}

/// DPI-aware spinbox / combo / line-edit height so inline widgets match
/// push-button height regardless of platform style.
inline int controlHeight() { return dpiScaled(24); }

// ---------------------------------------------------------------------------
//  Spacing & margin constants
// ---------------------------------------------------------------------------

/// Tight margins for tab-page layouts — keeps content compact while
/// leaving enough breathing room.
inline QMargins tabMargins() { return QMargins(4, 4, 4, 4); }

/// Even tighter margins for in-group rows (custom model path, threshold
/// rows, etc.).
inline QMargins rowMargins() { return QMargins(0, 0, 0, 0); }

/// Standard vertical spacing between rows inside a group box / tab.
inline int vSpacing() { return 4; }

/// Tighter vertical spacing for dense parameter rows.
inline int tightVSpacing() { return 2; }

/// Horizontal spacing between adjacent controls.
inline int hSpacing() { return 6; }

/// Horizontal spacing inside a row of tightly packed controls.
inline int tightHSpacing() { return 4; }

/// Default minimum width for numeric spinboxes (threshold, threads, etc.).
inline int compactSpinWidth() { return dpiScaled(72); }

/// Default width for "Browse…" buttons.
inline int browseBtnWidth() { return dpiScaled(72); }

/// Preview thumbnail size — DPI-aware so it looks consistent on HiDPI.
inline int previewSize() { return dpiScaled(96); }

/// Small preview thumbnail size for slot-based displays (e.g. LightGlue).
inline int slotPreviewSize() { return dpiScaled(88); }

/// Maximum height of the DB-image list widget.
inline int dbListMaxHeight() { return dpiScaled(140); }

/// Maximum height of the file-pool list widget (LightGlue).
inline int filePoolMaxHeight() { return dpiScaled(120); }

// ---------------------------------------------------------------------------
//  Label helpers
// ---------------------------------------------------------------------------

/// Left-aligned form label with fixed width for two-column grids.
inline QLabel* makeLabel(const QString& text, int fixedWidth = -1) {
    auto* label = new QLabel(text);
    label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    if (fixedWidth > 0) {
        label->setFixedWidth(dpiScaled(fixedWidth));
        label->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    } else {
        label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    }
    return label;
}

/// Muted helper / hint label for descriptions.
inline QLabel* makeHintLabel(const QString& text, QWidget* parent = nullptr) {
    auto* label = new QLabel(text, parent);
    label->setWordWrap(true);
    label->setStyleSheet(
            QStringLiteral("color: #666; font-size: 11px; padding: 1px 0;"));
    label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    return label;
}

// ---------------------------------------------------------------------------
//  Button helpers
// ---------------------------------------------------------------------------

/// Teal accent "Try sample data" button — consistent across all AICore
/// plugins.
inline QPushButton* makeSampleDataBtn(QWidget* parent = nullptr) {
    auto* btn = new QPushButton(QStringLiteral("\U0001f9ea  Try sample data"),
                                parent);
    btn->setStyleSheet(
            "QPushButton { background: #00897b; color: white; font-weight: "
            "bold; border: none; border-radius: 4px; padding: 5px 12px; }"
            "QPushButton:hover { background: #00796b; }"
            "QPushButton:pressed { background: #00695c; }"
            "QPushButton:disabled { background: #b2dfdb; color: #e0f2f1; }");
    btn->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    return btn;
}

/// Standard "Browse…" button.
inline QPushButton* makeBrowseBtn(const QString& text,
                                  QWidget* parent = nullptr) {
    auto* btn = new QPushButton(text, parent);
    btn->setFixedWidth(browseBtnWidth());
    btn->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    return btn;
}

// ---------------------------------------------------------------------------
//  SpinBox helpers
// ---------------------------------------------------------------------------

inline void setCompactDoubleSpin(QDoubleSpinBox* spin) {
    if (!spin) return;
    spin->setFixedWidth(compactSpinWidth());
    spin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

inline void setCompactSpin(QSpinBox* spin) {
    if (!spin) return;
    spin->setFixedWidth(compactSpinWidth());
    spin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

// ---------------------------------------------------------------------------
//  Layout helpers
// ---------------------------------------------------------------------------

/// Apply standard tab-page layout margins and spacing.
inline void setupTabLayout(QVBoxLayout* layout) {
    if (!layout) return;
    layout->setContentsMargins(tabMargins());
    layout->setSpacing(vSpacing());
}

/// Configure a QGridLayout as a compact two-column form (4 columns: label,
/// field, label, field) with consistent spacing.
inline void setupFormGrid(QGridLayout* grid, int labelColWidth = 0) {
    if (!grid) return;
    grid->setHorizontalSpacing(hSpacing());
    grid->setVerticalSpacing(tightVSpacing());
    grid->setContentsMargins(rowMargins());
    if (labelColWidth > 0) {
        grid->setColumnMinimumWidth(0, dpiScaled(labelColWidth));
        grid->setColumnMinimumWidth(2, dpiScaled(labelColWidth));
    }
    grid->setColumnStretch(0, 0);
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(2, 0);
    grid->setColumnStretch(3, 1);
}

/// Tighten a group box: Maximum vertical size policy + reduced margins.
inline void tightenGroupBox(QGroupBox* box) {
    if (!box) return;
    box->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    // Remove the extra vertical margin that QGroupBox leaves above the
    // title so it sits tight against the preceding group.
    if (auto* grid = qobject_cast<QGridLayout*>(box->layout())) {
        grid->setContentsMargins(6, 4, 6, 4);
        grid->setVerticalSpacing(tightVSpacing());
    } else if (auto* vbox = qobject_cast<QVBoxLayout*>(box->layout())) {
        vbox->setContentsMargins(6, 4, 6, 4);
        vbox->setSpacing(vSpacing());
    }
}

/// Clean tab-widget pane style — no border so content looks seamless.
inline void styleTabWidget(QTabWidget* tabs) {
    if (!tabs) return;
    tabs->setDocumentMode(false);
    tabs->setStyleSheet(QStringLiteral(
            "QTabWidget::pane { border: 0; padding: 2px; top: 0px; }"
            "QTabBar::tab { padding: 4px 10px; min-height: 18px; }"));
}

// ---------------------------------------------------------------------------
//  Preview-image helpers
// ---------------------------------------------------------------------------

/// Wraps an ecvClickableImageLabel with a tap-to-preview hint below it.
/// The returned widget owns both the label and the hint.
inline QWidget* wrapPreviewWithHint(QWidget* previewLabel,
                                    QWidget* parent = nullptr) {
    auto* container = new QWidget(parent);
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);
    layout->addWidget(previewLabel, 0, Qt::AlignHCenter);
    auto* hint = new QLabel(QStringLiteral("Tap to preview"));
    hint->setAlignment(Qt::AlignCenter);
    hint->setStyleSheet(
            QStringLiteral("color: palette(mid); font-size: 10px;"));
    layout->addWidget(hint);
    return container;
}

// ---------------------------------------------------------------------------
//  Common action-row builder
// ---------------------------------------------------------------------------

/// Build a standard action row: left side = optional checkboxes / output
/// controls; right side = sample-data button + Run + Cancel (+ optional
/// Close).
/// Returns the layout so callers can insert extra buttons at the end.
inline QHBoxLayout* makeActionRow(QPushButton* runBtn,
                                  QPushButton* cancelBtn,
                                  QWidget* leftWidget = nullptr) {
    auto* row = new QHBoxLayout;
    row->setSpacing(hSpacing());
    if (leftWidget) {
        row->addWidget(leftWidget);
    }
    row->addStretch();
    if (runBtn) row->addWidget(runBtn);
    if (cancelBtn) row->addWidget(cancelBtn);
    return row;
}

// ---------------------------------------------------------------------------
//  Runtime parameter row (Device / Threads)
// ---------------------------------------------------------------------------

/// Create a compact device + threads row used in every AICore dialog.
/// Returns the widget so callers can add spinboxes for other params.
inline QWidget* makeRuntimeRow(QComboBox* deviceCombo,
                               QSpinBox* threadsSpin,
                               QWidget* parent = nullptr) {
    auto* widget = new QWidget(parent);
    auto* layout = new QHBoxLayout(widget);
    layout->setContentsMargins(rowMargins());
    layout->setSpacing(hSpacing());

    layout->addWidget(new QLabel(QWidget::tr("Device:")));
    layout->addWidget(deviceCombo, 1);
    layout->addWidget(new QLabel(QWidget::tr("Threads:")));
    layout->addWidget(threadsSpin);
    layout->addStretch();

    return widget;
}

// ---------------------------------------------------------------------------
//  DB-image toggle / list helpers
// ---------------------------------------------------------------------------

/// Create a collapsible DB-image section with a toggle button and a
/// content area.  Returns the toggle button so callers can connect its
/// signals.
inline QToolButton* makeDbSection(QWidget* dbContentWidget) {
    auto* btn = new QToolButton;
    btn->setArrowType(Qt::RightArrow);
    btn->setCheckable(true);
    btn->setChecked(false);
    btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    btn->setText(QWidget::tr("DB Source Images (optional)"));
    btn->setCursor(Qt::PointingHandCursor);
    btn->setStyleSheet(
            "QToolButton { border: none; font-weight: bold; padding: 4px 6px; "
            "  border-radius: 3px; color: palette(text); }"
            "QToolButton:hover { background: palette(midlight); }");
    return btn;
}

inline void connectDbToggle(QToolButton* btn, QWidget* contentWidget) {
    if (!btn || !contentWidget) return;
    QObject::connect(
            btn, &QToolButton::toggled, btn,
            [btn, contentWidget](bool checked) {
                btn->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
                contentWidget->setVisible(checked);
            });
}

// ---------------------------------------------------------------------------
//  Progress section
// ---------------------------------------------------------------------------

/// Shared download progress bar + status label (both hidden by default).
inline void setupProgressSection(QVBoxLayout* parent,
                                 QLabel*& downloadLabel,
                                 QProgressBar*& progressBar) {
    if (!parent) return;
    downloadLabel = new QLabel;
    downloadLabel->setWordWrap(true);
    downloadLabel->setVisible(false);
    parent->addWidget(downloadLabel);

    progressBar = new QProgressBar;
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setTextVisible(true);
    progressBar->setVisible(false);
    parent->addWidget(progressBar);
}

}  // namespace ecvAICoreUi