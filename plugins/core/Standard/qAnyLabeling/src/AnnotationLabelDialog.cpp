// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationLabelDialog.h"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QVBoxLayout>

AnnotationLabelDialog::AnnotationLabelDialog(const QStringList& labels,
                                             bool sortLabels,
                                             bool showTextField,
                                             const QString& completion,
                                             bool fitToColumn,
                                             bool fitToRow,
                                             const QString& selected,
                                             QWidget* parent)
    : QDialog(parent),
      m_labels(labels),
      m_sortLabels(sortLabels),
      m_completion(completion) {
    setWindowTitle(tr("Label object"));
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(10, 10, 10, 10);
    if (showTextField) {
        m_edit = new QLineEdit(selected, this);
        m_edit->setPlaceholderText(
                tr("Enter label; the list filters as you type"));
        layout->addWidget(m_edit);
    }
    m_list = new QListWidget(this);
    m_list->setObjectName(QStringLiteral("AnnotationLabelDialogList"));
    layout->addWidget(m_list);
    // Label Flags template fields render under the list when the template
    // defines flags for the shape's label (setFlagFields).
    m_flagLayout = new QVBoxLayout;
    m_flagLayout->setContentsMargins(0, 0, 0, 0);
    layout->addLayout(m_flagLayout);
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(m_list, &QListWidget::itemDoubleClicked, this,
            [this](QListWidgetItem*) { accept(); });
    if (m_edit)
        connect(m_edit, &QLineEdit::textChanged, this,
                [this]() { rebuildList(); });
    rebuildList();
    if (m_edit) m_edit->setFocus();
    // fit_to_content: grow with the list contents instead of a fixed size.
    adjustSize();
    if (fitToColumn || fitToRow) {
        const int rows = qMin(m_list->count(), 12);
        if (rows > 0) {
            const int rowHeight = m_list->sizeHintForRow(0);
            if (rowHeight > 0)
                resize(width(), height() + rowHeight * rows - m_list->height());
        }
    }
}

QString AnnotationLabelDialog::selectedLabel() const {
    if (m_edit) return m_edit->text().trimmed();
    if (m_list->currentItem()) return m_list->currentItem()->text().trimmed();
    return QString();
}

void AnnotationLabelDialog::setFlagFields(const QStringList& flags,
                                          const QJsonObject& currentFlags) {
    if (!m_flagLayout) return;
    m_flagNames = flags;
    m_currentFlags = currentFlags;
    // Reuse on repeated opens inside one dialog lifetime.
    while (m_flagLayout->count() > 0) {
        QLayoutItem* item = m_flagLayout->takeAt(0);
        if (item->widget()) item->widget()->deleteLater();
        delete item;
    }
    if (flags.isEmpty()) return;
    auto* title = new QLabel(tr("Flags"), this);
    m_flagLayout->addWidget(title);
    for (const QString& flag : flags) {
        auto* box = new QCheckBox(flag, this);
        box->setObjectName(QStringLiteral("AnnotationLabelFlag_%1").arg(flag));
        box->setChecked(currentFlags.value(flag).toBool(false));
        m_flagLayout->addWidget(box);
    }
    adjustSize();
}

QJsonObject AnnotationLabelDialog::selectedFlags() const {
    QJsonObject flags;
    if (!m_flagLayout) return flags;
    for (int i = 0; i < m_flagLayout->count(); ++i) {
        auto* box = qobject_cast<QCheckBox*>(m_flagLayout->itemAt(i)->widget());
        if (box) flags.insert(box->text(), box->isChecked());
    }
    return flags;
}

void AnnotationLabelDialog::rebuildList() {
    if (!m_list) return;
    m_list->clear();
    QStringList labels = m_labels;
    if (m_sortLabels) labels.sort();
    const QString filter = m_edit ? m_edit->text().trimmed() : QString();
    for (const QString& label : labels) {
        if (!filter.isEmpty()) {
            if (m_completion == QStringLiteral("startswith")) {
                if (!label.startsWith(filter, Qt::CaseInsensitive)) continue;
            } else if (m_completion == QStringLiteral("contains")) {
                if (!label.contains(filter, Qt::CaseInsensitive)) continue;
            }
        }
        m_list->addItem(label);
    }
    if (!filter.isEmpty()) {
        for (int i = 0; i < m_list->count(); ++i)
            if (m_list->item(i)->text() == filter) {
                m_list->setCurrentRow(i);
                break;
            }
    } else if (m_list->count() > 0) {
        m_list->setCurrentRow(0);
    }
}

void AnnotationLabelDialog::keyPressEvent(QKeyEvent* event) {
    // Enter accepts the highlighted list entry when the text field already
    // matches it exactly, mirroring the source dialog keyboard flow.
    if (event->key() == Qt::Key_Down || event->key() == Qt::Key_Up) {
        if (m_list) {
            const int delta = event->key() == Qt::Key_Down ? 1 : -1;
            const int row = m_list->currentRow() + delta;
            if (row >= 0 && row < m_list->count()) m_list->setCurrentRow(row);
        }
        return;
    }
    QDialog::keyPressEvent(event);
}
