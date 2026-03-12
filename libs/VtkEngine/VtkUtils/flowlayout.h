// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file flowlayout.h
/// @brief Flow layout that arranges widgets left-to-right, wrapping to next
/// line.

#include <QLayout>
#include <QRect>
#include <QStyle>

#include "qVTK.h"

/// @class FlowLayout
/// @brief Layout that flows child widgets left-to-right and wraps when needed.
class QVTK_ENGINE_LIB_API FlowLayout : public QLayout {
public:
    /// @param parent Parent widget
    /// @param margin Layout margin (-1 for style default)
    /// @param hSpacing Horizontal spacing (-1 for style default)
    /// @param vSpacing Vertical spacing (-1 for style default)
    explicit FlowLayout(QWidget *parent,
                        int margin = -1,
                        int hSpacing = -1,
                        int vSpacing = -1);
    explicit FlowLayout(int margin = -1, int hSpacing = -1, int vSpacing = -1);
    ~FlowLayout();

    void clear();
    void addItem(QLayoutItem *item) Q_DECL_OVERRIDE;
    /// @return Horizontal spacing between items
    int horizontalSpacing() const;
    /// @return Vertical spacing between rows
    int verticalSpacing() const;
    Qt::Orientations expandingDirections() const Q_DECL_OVERRIDE;
    bool hasHeightForWidth() const Q_DECL_OVERRIDE;
    int heightForWidth(int) const Q_DECL_OVERRIDE;
    int count() const Q_DECL_OVERRIDE;
    QLayoutItem *itemAt(int index) const Q_DECL_OVERRIDE;
    QSize minimumSize() const Q_DECL_OVERRIDE;
    void setGeometry(const QRect &rect) Q_DECL_OVERRIDE;
    QSize sizeHint() const Q_DECL_OVERRIDE;
    QLayoutItem *takeAt(int index) Q_DECL_OVERRIDE;

private:
    int doLayout(const QRect &rect, bool testOnly) const;
    int smartSpacing(QStyle::PixelMetric pm) const;

    QList<QLayoutItem *> itemList;
    int m_hSpace;
    int m_vSpace;
};
