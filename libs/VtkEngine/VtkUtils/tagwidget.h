// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file tagwidget.h
 * @brief Tag/label management widget for displaying removable tag chips.
 */

#include <QWidget>

#include "qVTK.h"

namespace Widgets {

class TagWidgetPrivate;

/**
 * @class TagWidget
 * @brief Displays a set of tags as removable chips, with add/remove signals.
 */
class QVTK_ENGINE_LIB_API TagWidget : public QWidget {
    Q_OBJECT
public:
    explicit TagWidget(QWidget* parent = 0);
    ~TagWidget();

    void addTag(const QString& tagName);
    void addTags(const QStringList& tags);
    void clear();
    QStringList tags() const;

private:
    Q_DISABLE_COPY(TagWidget)
    TagWidgetPrivate* d_ptr;
};

}  // namespace Widgets
