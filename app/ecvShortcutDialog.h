// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ui_shorcutSettings.h"
#include "ui_shortcutEditDialog.h"

//! Widget that captures key sequences to be able to edit a shortcut assigned to
//! an action
class ecvShortcutEditDialog final : public QDialog {
    Q_OBJECT

public:
    explicit ecvShortcutEditDialog(QWidget* parent = nullptr);

    ~ecvShortcutEditDialog() override;

    QKeySequence keySequence() const;

    void setKeySequence(const QKeySequence& sequence) const;

    int exec() override;

private:
    Ui_ShortcutEditDialog* m_ui;
};

//! Shortcut edit dialog
//!
//! List shortcuts for known actions, and allows to edit them
//! Saves to QSettings on each edit
class ecvShortcutDialog final : public QDialog {
    Q_OBJECT
public:
    explicit ecvShortcutDialog(const QList<QAction*>& actions,
                               QWidget* parent = nullptr);

    ~ecvShortcutDialog() override;

    void restoreShortcutsFromQSettings() const;

private slots:
    void filterActions(const QString& searchText);

private:
    const QAction* checkConflict(const QKeySequence& sequence) const;
    void handleDoubleClick(QTableWidgetItem* item);
    void showAllRows();

    Ui_ShortcutDialog* m_ui;
    ecvShortcutEditDialog* m_editDialog;
    QList<QAction*> m_allActions;  // Store all actions for filtering
};
