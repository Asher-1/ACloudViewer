// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_MULTI_COLUMN_HEADER_VIEW_H
#define CV_MULTI_COLUMN_HEADER_VIEW_H

#include <QHeaderView>
#include <QPair>

/**
 * @class cvMultiColumnHeaderView
 * @brief QHeaderView that supports showing multiple sections as one.
 *
 * cvMultiColumnHeaderView extends QHeaderView to support showing multiple
 * adjacent sections as a single section. This is useful for showing vector
 * quantities, for example. Instead of each component taking up header space
 * and making it confusing to understand that the various sections are part of
 * the same vector, cvMultiColumnHeaderView can show all those sections under a
 * single banner. It still supports resizing individual sections thus does not
 * inhibit usability.
 *
 * cvMultiColumnHeaderView simply combines adjacent sections with same
 * (non-empty) `QString` value for Qt::`DisplayRole`. This is done by
 * overriding `QHeaderView::paintSection` and custom painting such
 * sections spanning multiple sections.
 *
 * Reference: ParaView/Qt/Widgets/pqMultiColumnHeaderView.h
 */
class cvMultiColumnHeaderView : public QHeaderView {
    Q_OBJECT
    typedef QHeaderView Superclass;

public:
    cvMultiColumnHeaderView(Qt::Orientation orientation,
                            QWidget* parent = nullptr);
    ~cvMultiColumnHeaderView() override;

protected:
    void paintSection(QPainter* painter,
                      const QRect& rect,
                      int logicalIndex) const override;

private:
    QPair<int, int> sectionSpan(int visualIndex) const;
    QString sectionDisplayText(int logicalIndex) const;

    Q_DISABLE_COPY(cvMultiColumnHeaderView)
};

#endif  // CV_MULTI_COLUMN_HEADER_VIEW_H
