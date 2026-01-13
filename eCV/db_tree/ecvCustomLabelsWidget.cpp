// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvCustomLabelsWidget.h"

#include <QHBoxLayout>
#include <QHeaderView>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>

ecvCustomLabelsWidget::ecvCustomLabelsWidget(QWidget* parent)
    : QWidget(parent) {
    setupUI();
}

ecvCustomLabelsWidget::~ecvCustomLabelsWidget() = default;

void ecvCustomLabelsWidget::setupUI() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(4);

    // Table for Value/Label pairs
    m_table = new QTableWidget(0, 2, this);
    m_table->setHorizontalHeaderLabels(QStringList()
                                       << tr("Value") << tr("Label"));
    m_table->horizontalHeader()->setStretchLastSection(true);
    m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_table->setMinimumHeight(120);
    m_table->setMaximumHeight(200);
    mainLayout->addWidget(m_table);

    // Add/Remove buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->setContentsMargins(0, 0, 0, 0);

    m_addButton = new QPushButton(tr("Add"), this);
    m_removeButton = new QPushButton(tr("Remove"), this);

    connect(m_addButton, &QPushButton::clicked, this,
            &ecvCustomLabelsWidget::onAddButtonClicked);
    connect(m_removeButton, &QPushButton::clicked, this,
            &ecvCustomLabelsWidget::onRemoveButtonClicked);

    buttonLayout->addWidget(m_addButton);
    buttonLayout->addWidget(m_removeButton);
    buttonLayout->addStretch();

    mainLayout->addLayout(buttonLayout);
}

void ecvCustomLabelsWidget::onAddButtonClicked() {
    int row = m_table->rowCount();
    m_table->insertRow(row);

    // Default value: 0.0
    QTableWidgetItem* valueItem = new QTableWidgetItem("0.0");
    m_table->setItem(row, 0, valueItem);

    // Default label: empty
    QTableWidgetItem* labelItem = new QTableWidgetItem("");
    m_table->setItem(row, 1, labelItem);

    // Select the new row
    m_table->selectRow(row);
    // Edit the value cell
    m_table->editItem(valueItem);
}

void ecvCustomLabelsWidget::onRemoveButtonClicked() {
    QList<QTableWidgetSelectionRange> ranges = m_table->selectedRanges();
    if (ranges.isEmpty()) {
        // No selection, remove last row
        if (m_table->rowCount() > 0) {
            m_table->removeRow(m_table->rowCount() - 1);
        }
    } else {
        // Remove selected rows (from bottom to top to avoid index shifting)
        QList<int> rowsToRemove;
        for (const QTableWidgetSelectionRange& range : ranges) {
            for (int row = range.topRow(); row <= range.bottomRow(); ++row) {
                if (!rowsToRemove.contains(row)) {
                    rowsToRemove.append(row);
                }
            }
        }
        std::sort(rowsToRemove.begin(), rowsToRemove.end(),
                  std::greater<int>());
        for (int row : rowsToRemove) {
            m_table->removeRow(row);
        }
    }
}

QList<QPair<double, QString>> ecvCustomLabelsWidget::getLabels() const {
    QList<QPair<double, QString>> labels;
    for (int row = 0; row < m_table->rowCount(); ++row) {
        QTableWidgetItem* valueItem = m_table->item(row, 0);
        QTableWidgetItem* labelItem = m_table->item(row, 1);

        if (valueItem && labelItem) {
            bool ok = false;
            double value = valueItem->text().toDouble(&ok);
            if (ok) {
                labels.append(qMakePair(value, labelItem->text()));
            }
        }
    }
    return labels;
}

void ecvCustomLabelsWidget::setLabels(
        const QList<QPair<double, QString>>& labels) {
    m_table->setRowCount(0);
    for (const QPair<double, QString>& label : labels) {
        int row = m_table->rowCount();
        m_table->insertRow(row);

        QTableWidgetItem* valueItem =
                new QTableWidgetItem(QString::number(label.first));
        m_table->setItem(row, 0, valueItem);

        QTableWidgetItem* labelItem = new QTableWidgetItem(label.second);
        m_table->setItem(row, 1, labelItem);
    }
}

void ecvCustomLabelsWidget::clearLabels() { m_table->setRowCount(0); }
