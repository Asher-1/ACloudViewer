// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_CUSTOM_LABELS_WIDGET_H
#define ECV_CUSTOM_LABELS_WIDGET_H

#include <QList>
#include <QPair>
#include <QWidget>

class QTableWidget;
class QPushButton;

/**
 * @brief ParaView-style Custom Labels Editor Widget
 *
 * Table with "Value" and "Label" columns, plus Add/Remove buttons.
 */
class ecvCustomLabelsWidget : public QWidget {
    Q_OBJECT

public:
    explicit ecvCustomLabelsWidget(QWidget* parent = nullptr);
    ~ecvCustomLabelsWidget() override;

    QList<QPair<double, QString>> getLabels() const;
    void setLabels(const QList<QPair<double, QString>>& labels);
    void clearLabels();

private slots:
    void onAddButtonClicked();
    void onRemoveButtonClicked();

private:
    void setupUI();

    QTableWidget* m_table;
    QPushButton* m_addButton;
    QPushButton* m_removeButton;
};

#endif  // ECV_CUSTOM_LABELS_WIDGET_H
