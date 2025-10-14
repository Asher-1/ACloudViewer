// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QComboBox>

#include "../qPCL.h"

class QListWidgetItem;
namespace Widgets {

class GradientComboBoxPrivate;
class QPCL_ENGINE_LIB_API GradientComboBox : public QComboBox {
    Q_OBJECT
public:
    explicit GradientComboBox(QWidget* parent = nullptr);
    ~GradientComboBox();

    void setCurrentIndex(int index);
    int currentIndex() const;

    QString currentName() const;
    QColor currentColor1() const;
    QColor currentColor2() const;

    static QPair<QColor, QColor> colorPair(int index);

    void showPopup();
    void hidePopup();

signals:
    void colorsChanged(const QString& name,
                       const QColor& clr1,
                       const QColor& clr2);

protected:
    void paintEvent(QPaintEvent* e);
    void resizeEvent(QResizeEvent* e);
    QSize minimumSizeHint() const;

private slots:
    void onActivated(int index);

private:
    GradientComboBoxPrivate* d_ptr;
    Q_DISABLE_COPY(GradientComboBox)
};

}  // namespace Widgets
