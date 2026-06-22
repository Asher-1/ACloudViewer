// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QMap>

#include "ui_shorcutSettings.h"
#include "ui_shortcutEditDialog.h"

class QComboBox;
class QLabel;

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

//! Shortcut management dialog with conflict detection.
//! Supports editing, resetting to defaults, and persistence via QSettings.
//! Also tracks standalone QShortcuts and modal shortcuts for conflict
//! awareness.
class ecvShortcutDialog final : public QDialog {
    Q_OBJECT
public:
    explicit ecvShortcutDialog(const QList<QAction*>& actions,
                               QWidget* parent = nullptr);
    ~ecvShortcutDialog() override;

    void restoreShortcutsFromQSettings() const;

    //! Register standalone QShortcut keys for conflict detection.
    void registerStandaloneShortcut(const QString& name,
                                    const QKeySequence& seq);

    //! Check all registered shortcuts for conflicts and return a list.
    QStringList detectAllConflicts() const;

    //! Refresh modal shortcuts from ecvKeySequences for conflict awareness.
    void syncModalShortcuts();

    //! Number of QAction rows (before VTK rows).
    int actionRowCount() const { return m_allActions.count(); }

private slots:
    void filterActions(const QString& searchText);
    void onCategoryChanged(int index);
    void onResetSelected();
    void onResetAll();
    void onExportShortcuts();
    void onImportShortcuts();

private:
    struct ConflictInfo {
        const QAction* action = nullptr;
        QString name;
        int row = -1;
    };

    static QString actionPersistKey(const QAction* action);
    QList<ConflictInfo> checkConflicts(const QKeySequence& sequence,
                                       const QAction* exclude = nullptr) const;
    void handleDoubleClick(QTableWidgetItem* item);
    void showAllRows();
    void refreshConflictHighlighting();

    void applyFilters();

    Ui_ShortcutDialog* m_ui;
    ecvShortcutEditDialog* m_editDialog;
    QComboBox* m_categoryCombo;
    QLabel* m_conflictLabel;
    QList<QAction*> m_allActions;
    QMap<QAction*, QKeySequence> m_defaultShortcuts;
    QMap<QString, QKeySequence> m_vtkDefaults;
    QMap<QString, QKeySequence> m_standaloneShortcuts;
    QMap<QString, QKeySequence> m_modalShortcuts;
};
