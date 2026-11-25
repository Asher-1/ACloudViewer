// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Shortcuts/ecvKeySequences.h>

#include "ui_shorcutSettings.h"
#include "ui_shortcutEditDialog.h"

class QComboBox;
class QLabel;
class QListWidget;

class ecvShortcutEditDialog final : public QDialog {
    Q_OBJECT
public:
    explicit ecvShortcutEditDialog(QWidget* parent = nullptr);
    ~ecvShortcutEditDialog() override;
    QList<QKeySequence> keySequences() const;
    void setKeySequences(const QList<QKeySequence>& sequences);
    int exec() override;
    void accept() override;

signals:
    void applyRequested();

private:
    void storeCurrentBinding();
    Ui_ShortcutEditDialog* m_ui;
    QListWidget* m_bindingsList;
    int m_editingRow = -1;
};

//! Live view of the shared command registry. Every mutation is validated and
//! immediately applied by the registry; closing this dialog is not a rollback.
class ecvShortcutDialog final : public QDialog {
    Q_OBJECT
public:
    explicit ecvShortcutDialog(const QList<QAction*>& actions,
                               QWidget* parent = nullptr);
    ~ecvShortcutDialog() override;
    void registerActions(const QList<QAction*>& actions);
    void restoreShortcutsFromQSettings() const;
    QStringList detectAllConflicts() const;
    void refreshCommands();
    void selectPlugin(const QString& pluginId);

protected:
    void showEvent(QShowEvent* event) override;

private slots:
    void onResetSelected();
    void onResetAll();
    void onExportShortcuts();
    void onImportShortcuts();

private:
    bool apply(const ecvKeySequences::Bindings& bindings);
    void handleDoubleClick(QTableWidgetItem* item);
    void applyFilters();

    Ui_ShortcutDialog* m_ui;
    ecvShortcutEditDialog* m_editDialog;
    QComboBox* m_categoryCombo;
    QComboBox* m_pluginCombo;
    QLabel* m_conflictLabel;
};
