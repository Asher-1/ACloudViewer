// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvShortcutDialog.h"

// Local
#include "ecvPersistentSettings.h"

// Qt
#include <QAction>
#include <QHeaderView>
#include <QKeySequenceEdit>
#include <QMessageBox>
#include <QSettings>
#include <QTableWidget>
#include <QTableWidgetItem>

constexpr int ACTION_NAME_COLUMN = 0;
constexpr int KEY_SEQUENCE_COLUMN = 1;

ecvShortcutEditDialog::ecvShortcutEditDialog(QWidget* parent)
    : QDialog(parent), m_ui(new Ui_ShortcutEditDialog) {
    m_ui->setupUi(this);
    connect(m_ui->clearButton, &QPushButton::clicked, m_ui->keySequenceEdit,
            &QKeySequenceEdit::clear);
}

ecvShortcutEditDialog::~ecvShortcutEditDialog() { delete m_ui; }

QKeySequence ecvShortcutEditDialog::keySequence() const {
    return m_ui->keySequenceEdit->keySequence();
}

void ecvShortcutEditDialog::setKeySequence(const QKeySequence& sequence) const {
    m_ui->keySequenceEdit->setKeySequence(sequence);
}

int ecvShortcutEditDialog::exec() {
    m_ui->keySequenceEdit->setFocus();
    return QDialog::exec();
}

ecvShortcutDialog::ecvShortcutDialog(const QList<QAction*>& actions,
                                     QWidget* parent)
    : QDialog(parent),
      m_ui(new Ui_ShortcutDialog),
      m_editDialog(new ecvShortcutEditDialog(this)) {
    m_ui->setupUi(this);
    m_ui->tableWidget->setRowCount(actions.count());
    m_ui->tableWidget->horizontalHeader()->setSectionResizeMode(
            QHeaderView::Stretch);
    m_ui->tableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);

    connect(m_ui->tableWidget, &QTableWidget::itemDoubleClicked, this,
            &ecvShortcutDialog::handleDoubleClick);

    int row = 0;
    for (QAction* action : actions) {
        auto* actionWidget =
                new QTableWidgetItem(action->icon(), action->text());
        actionWidget->setFlags(actionWidget->flags() & ~Qt::ItemIsEditable);
        m_ui->tableWidget->setItem(row, ACTION_NAME_COLUMN, actionWidget);

        auto* shortcutWidget =
                new QTableWidgetItem(action->shortcut().toString());
        shortcutWidget->setFlags(actionWidget->flags() & ~Qt::ItemIsEditable);
        shortcutWidget->setData(Qt::UserRole, QVariant::fromValue(action));
        m_ui->tableWidget->setItem(row, KEY_SEQUENCE_COLUMN, shortcutWidget);
        row += 1;
    }
}

ecvShortcutDialog::~ecvShortcutDialog() {
    delete m_ui;
    delete m_editDialog;
}

void ecvShortcutDialog::restoreShortcutsFromQSettings() const {
    QSettings settings;
    settings.beginGroup(ecvPS::Shortcuts());

    for (int i = 0; i < m_ui->tableWidget->rowCount(); i++) {
        QTableWidgetItem* item =
                m_ui->tableWidget->item(i, KEY_SEQUENCE_COLUMN);
        auto* action = item->data(Qt::UserRole).value<QAction*>();

        if (settings.contains(action->text())) {
            const QKeySequence defaultValue;
            const auto sequence = settings.value(action->text(), defaultValue)
                                          .value<QKeySequence>();

            item->setText(sequence.toString());
            action->setShortcut(sequence);
        }
    }
    settings.endGroup();
}

const QAction* ecvShortcutDialog::checkConflict(
        const QKeySequence& sequence) const {
    for (int i = 0; i < m_ui->tableWidget->rowCount(); i++) {
        const QTableWidgetItem* item = m_ui->tableWidget->item(i, 1);
        const auto* action = item->data(Qt::UserRole).value<QAction*>();
        if (action->shortcut() == sequence) {
            return action;
        }
    }

    return nullptr;
}

void ecvShortcutDialog::handleDoubleClick(QTableWidgetItem* item) {
    if (!item) {
        return;
    }

    if (item->column() != KEY_SEQUENCE_COLUMN) {
        item = m_ui->tableWidget->item(item->row(), KEY_SEQUENCE_COLUMN);
    }

    auto* action = item->data(Qt::UserRole).value<QAction*>();
    m_editDialog->setKeySequence(action->shortcut());

    if (m_editDialog->exec() == QDialog::Rejected) {
        return;
    }

    const QKeySequence keySequence = m_editDialog->keySequence();
    if (keySequence == action->shortcut()) {
        // User did not change it
        return;
    }

    if (!keySequence.isEmpty()) {
        const QAction* conflict = checkConflict(keySequence);
        if (conflict) {
            QMessageBox::critical(
                    this, tr("Shortcut conflict"),
                    QString(tr("The shortcut entered would conflict with the "
                               "one for `%1`"))
                            .arg(conflict->text()));
            return;
        }
    }

    item->setText(keySequence.toString());
    action->setShortcut(keySequence);

    QSettings settings;
    settings.beginGroup(ecvPS::Shortcuts());
    settings.setValue(action->text(), keySequence);
}
