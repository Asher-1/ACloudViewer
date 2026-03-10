// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file doublespinboxdelegate.h
/// @brief QStyledItemDelegate for editing doubles with QDoubleSpinBox.

#include <QStyledItemDelegate>

#include "qVTK.h"

namespace VtkUtils {

/// @class DoubleSpinBoxDelegate
/// @brief Item delegate that uses QDoubleSpinBox for double editing in table
/// views.
class QVTK_ENGINE_LIB_API DoubleSpinBoxDelegate : public QStyledItemDelegate {
    Q_OBJECT

public:
    DoubleSpinBoxDelegate(QObject *parent = 0);

    QWidget *createEditor(QWidget *parent,
                          const QStyleOptionViewItem &option,
                          const QModelIndex &index) const Q_DECL_OVERRIDE;

    void setEditorData(QWidget *editor,
                       const QModelIndex &index) const Q_DECL_OVERRIDE;
    void setModelData(QWidget *editor,
                      QAbstractItemModel *model,
                      const QModelIndex &index) const Q_DECL_OVERRIDE;

    void updateEditorGeometry(QWidget *editor,
                              const QStyleOptionViewItem &option,
                              const QModelIndex &index) const Q_DECL_OVERRIDE;
};

}  // namespace VtkUtils
