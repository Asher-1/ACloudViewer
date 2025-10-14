// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QWidget>

#include "../qPCL.h"

namespace Widgets {
class TagWidgetPrivate;
class QPCL_ENGINE_LIB_API TagWidget : public QWidget {
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
