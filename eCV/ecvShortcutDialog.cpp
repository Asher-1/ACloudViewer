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
#include <QLineEdit>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSettings>
#include <QTableWidget>
#include <QTableWidgetItem>

constexpr int ACTION_NAME_COLUMN = 0;
constexpr int KEY_SEQUENCE_COLUMN = 1;

// Helper function to recursively find menu path for an action
static QString findMenuPathRecursive(QMenu* menu,
                                     QAction* targetAction,
                                     QStringList path) {
    if (!menu) {
        return QString();
    }

    // Add current menu title to path (only if not already in path to avoid
    // duplicates)
    QString menuTitle = menu->title();
    menuTitle.remove('&');
    if (!menuTitle.isEmpty() && (path.isEmpty() || path.last() != menuTitle)) {
        path.append(menuTitle);
    }

    // Check if this menu contains the target action
    for (QAction* action : menu->actions()) {
        if (action == targetAction) {
            // Found the action in this menu
            return path.join(" > ");
        }

        // Check if this action has a submenu
        QMenu* submenu = action->menu();
        if (submenu) {
            // For submenus, use the submenu's title directly (don't add action
            // text)
            QStringList subPath = path;
            QString submenuTitle = submenu->title();
            submenuTitle.remove('&');
            // Only add if different from last item to avoid duplicates
            if (!submenuTitle.isEmpty() &&
                (subPath.isEmpty() || subPath.last() != submenuTitle)) {
                subPath.append(submenuTitle);
            }
            QString result =
                    findMenuPathRecursive(submenu, targetAction, subPath);
            if (!result.isEmpty()) {
                return result;
            }
        }
    }

    return QString();
}

// Helper function to get menu path for an action
static QString getMenuPath(QAction* action, QWidget* parentWidget) {
    if (!action) {
        return QString();
    }

    // First try: traverse up the parent chain (fast path)
    QStringList path;
    QObject* parent = action->parent();

    while (parent) {
        QMenu* menu = qobject_cast<QMenu*>(parent);
        if (menu) {
            QString menuTitle = menu->title();
            menuTitle.remove('&');
            // Only add if not empty and not duplicate
            if (!menuTitle.isEmpty() &&
                (path.isEmpty() || path.first() != menuTitle)) {
                path.prepend(menuTitle);
            }
            parent = menu->parent();
        } else {
            // Check if parent is another action (for submenu actions)
            QAction* parentAction = qobject_cast<QAction*>(parent);
            if (parentAction) {
                // For submenu actions, get the submenu title instead of action
                // text
                QMenu* submenu = parentAction->menu();
                if (submenu) {
                    QString submenuTitle = submenu->title();
                    submenuTitle.remove('&');
                    if (!submenuTitle.isEmpty() &&
                        (path.isEmpty() || path.first() != submenuTitle)) {
                        path.prepend(submenuTitle);
                    }
                    parent = submenu->parent();
                } else {
                    QString actionTitle = parentAction->text();
                    actionTitle.remove('&');
                    if (!actionTitle.isEmpty() &&
                        (path.isEmpty() || path.first() != actionTitle)) {
                        path.prepend(actionTitle);
                    }
                    parent = parentAction->parent();
                }
            } else {
                break;
            }
        }
    }

    if (!path.isEmpty()) {
        return path.join(" > ");
    }

    // Second try: search through menu bar if parent widget is available
    if (parentWidget) {
        QMenuBar* menuBar = parentWidget->findChild<QMenuBar*>();
        if (!menuBar) {
            // Try to get menuBar from QMainWindow
            QMainWindow* mainWindow = qobject_cast<QMainWindow*>(parentWidget);
            if (mainWindow) {
                menuBar = mainWindow->menuBar();
            }
        }

        if (menuBar) {
            QStringList emptyPath;
            for (QAction* menuAction : menuBar->actions()) {
                QMenu* menu = menuAction->menu();
                if (menu) {
                    QString result =
                            findMenuPathRecursive(menu, action, emptyPath);
                    if (!result.isEmpty()) {
                        return result;
                    }
                }
            }
        }
    }

    return QString();
}

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
      m_editDialog(new ecvShortcutEditDialog(this)),
      m_allActions(actions) {
    m_ui->setupUi(this);
    m_ui->tableWidget->setRowCount(actions.count());
    m_ui->tableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);

    connect(m_ui->tableWidget, &QTableWidget::itemDoubleClicked, this,
            &ecvShortcutDialog::handleDoubleClick);
    connect(m_ui->searchLineEdit, &QLineEdit::textChanged, this,
            &ecvShortcutDialog::filterActions);

    int row = 0;
    for (QAction* action : actions) {
        // Build display text with full information
        QString displayText = action->text();
        QString menuPath = getMenuPath(action, parent);
        QString toolTip = action->toolTip();

        // Remove accelerator markers from display text
        QString cleanText = displayText;
        cleanText.remove('&');

        // Build full description
        QString fullDescription = cleanText;

        // Add menu path if available
        if (!menuPath.isEmpty()) {
            fullDescription += " (" + menuPath + ")";
        }

        // Add tooltip if available and different from text
        if (!toolTip.isEmpty() && toolTip != displayText &&
            toolTip != cleanText) {
            // Remove accelerator markers from tooltip
            QString cleanToolTip = toolTip;
            cleanToolTip.remove('&');
            if (cleanToolTip != cleanText && !cleanToolTip.isEmpty()) {
                fullDescription += " - " + cleanToolTip;
            }
        }

        auto* actionWidget =
                new QTableWidgetItem(action->icon(), fullDescription);
        actionWidget->setFlags(actionWidget->flags() & ~Qt::ItemIsEditable);

        // Store original text, menu path, and tooltip for searching
        actionWidget->setData(Qt::UserRole + 1, cleanText);  // Original text
        actionWidget->setData(Qt::UserRole + 2, menuPath);   // Menu path
        actionWidget->setData(Qt::UserRole + 3, toolTip);    // Tooltip

        m_ui->tableWidget->setItem(row, ACTION_NAME_COLUMN, actionWidget);

        auto* shortcutWidget =
                new QTableWidgetItem(action->shortcut().toString());
        shortcutWidget->setFlags(actionWidget->flags() & ~Qt::ItemIsEditable);
        shortcutWidget->setData(Qt::UserRole, QVariant::fromValue(action));
        m_ui->tableWidget->setItem(row, KEY_SEQUENCE_COLUMN, shortcutWidget);
        row += 1;
    }

    // Set column resize modes: first column stretches, second column has fixed
    // width This ensures the first column (Action) is much wider than the
    // second (Shortcut)

    // First, measure content to determine optimal width for shortcut column
    m_ui->tableWidget->horizontalHeader()->setSectionResizeMode(
            KEY_SEQUENCE_COLUMN, QHeaderView::ResizeToContents);
    m_ui->tableWidget->resizeColumnToContents(KEY_SEQUENCE_COLUMN);

    // Get measured width for shortcut column
    int shortcutContentWidth =
            m_ui->tableWidget->columnWidth(KEY_SEQUENCE_COLUMN);

    // Set a fixed width for the shortcut column (with some margin)
    const int minShortcutWidth = 200;  // Minimum width for shortcut column
    int shortcutWidth = qMax(shortcutContentWidth + 30, minShortcutWidth);

    // Set shortcut column to fixed width
    m_ui->tableWidget->setColumnWidth(KEY_SEQUENCE_COLUMN, shortcutWidth);
    m_ui->tableWidget->horizontalHeader()->setSectionResizeMode(
            KEY_SEQUENCE_COLUMN, QHeaderView::Fixed);

    // Set action column to stretch mode - it will take all remaining space
    m_ui->tableWidget->horizontalHeader()->setStretchLastSection(false);
    m_ui->tableWidget->horizontalHeader()->setSectionResizeMode(
            ACTION_NAME_COLUMN, QHeaderView::Stretch);
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

void ecvShortcutDialog::filterActions(const QString& searchText) {
    if (searchText.isEmpty()) {
        showAllRows();
        return;
    }

    QString searchLower = searchText.toLower();
    for (int row = 0; row < m_ui->tableWidget->rowCount(); ++row) {
        QTableWidgetItem* item =
                m_ui->tableWidget->item(row, ACTION_NAME_COLUMN);
        if (item) {
            // Search in display text
            bool matches = item->text().toLower().contains(searchLower);

            // Also search in stored data (original text, menu path, tooltip)
            if (!matches) {
                QString originalText =
                        item->data(Qt::UserRole + 1).toString().toLower();
                QString menuPath =
                        item->data(Qt::UserRole + 2).toString().toLower();
                QString toolTip =
                        item->data(Qt::UserRole + 3).toString().toLower();

                matches = originalText.contains(searchLower) ||
                          menuPath.contains(searchLower) ||
                          toolTip.contains(searchLower);
            }

            m_ui->tableWidget->setRowHidden(row, !matches);
        }
    }
}

void ecvShortcutDialog::showAllRows() {
    for (int row = 0; row < m_ui->tableWidget->rowCount(); ++row) {
        m_ui->tableWidget->setRowHidden(row, false);
    }
}
