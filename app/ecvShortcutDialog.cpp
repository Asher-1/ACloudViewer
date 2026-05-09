// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvShortcutDialog.h"

#include "ecvPersistentSettings.h"

#include <Shortcuts/ecvKeySequences.h>
#include <VTKExtensions/Widgets/VtkShortcutRegistry.h>

#include <QAction>
#include <QComboBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeySequenceEdit>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

constexpr int ACTION_NAME_COLUMN = 0;
constexpr int KEY_SEQUENCE_COLUMN = 1;

static QString findMenuPathRecursive(QMenu* menu,
                                     QAction* targetAction,
                                     QStringList path) {
    if (!menu) return {};

    QString menuTitle = menu->title();
    menuTitle.remove('&');
    if (!menuTitle.isEmpty() && (path.isEmpty() || path.last() != menuTitle)) {
        path.append(menuTitle);
    }

    for (QAction* action : menu->actions()) {
        if (action == targetAction) return path.join(" > ");

        QMenu* submenu = action->menu();
        if (submenu) {
            QStringList subPath = path;
            QString submenuTitle = submenu->title();
            submenuTitle.remove('&');
            if (!submenuTitle.isEmpty() &&
                (subPath.isEmpty() || subPath.last() != submenuTitle)) {
                subPath.append(submenuTitle);
            }
            QString result =
                    findMenuPathRecursive(submenu, targetAction, subPath);
            if (!result.isEmpty()) return result;
        }
    }
    return {};
}

static QString getMenuPath(QAction* action, QWidget* parentWidget) {
    if (!action) return {};

    QStringList path;
    QObject* parent = action->parent();

    while (parent) {
        QMenu* menu = qobject_cast<QMenu*>(parent);
        if (menu) {
            QString menuTitle = menu->title();
            menuTitle.remove('&');
            if (!menuTitle.isEmpty() &&
                (path.isEmpty() || path.first() != menuTitle)) {
                path.prepend(menuTitle);
            }
            parent = menu->parent();
        } else {
            QAction* parentAction = qobject_cast<QAction*>(parent);
            if (parentAction) {
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

    if (!path.isEmpty()) return path.join(" > ");

    if (parentWidget) {
        QMenuBar* menuBar = parentWidget->findChild<QMenuBar*>();
        if (!menuBar) {
            auto* mainWindow = qobject_cast<QMainWindow*>(parentWidget);
            if (mainWindow) menuBar = mainWindow->menuBar();
        }
        if (menuBar) {
            QStringList emptyPath;
            for (QAction* menuAction : menuBar->actions()) {
                QMenu* menu = menuAction->menu();
                if (menu) {
                    QString result =
                            findMenuPathRecursive(menu, action, emptyPath);
                    if (!result.isEmpty()) return result;
                }
            }
        }
    }
    return {};
}

// ============================================================================
// ecvShortcutEditDialog
// ============================================================================

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

void ecvShortcutEditDialog::setKeySequence(
        const QKeySequence& sequence) const {
    m_ui->keySequenceEdit->setKeySequence(sequence);
}

int ecvShortcutEditDialog::exec() {
    m_ui->keySequenceEdit->setFocus();
    return QDialog::exec();
}

// ============================================================================
// ecvShortcutDialog
// ============================================================================

QString ecvShortcutDialog::actionPersistKey(const QAction* action) {
    if (!action) return {};
    QString key = action->objectName();
    if (key.isEmpty()) key = action->text();
    key.remove('&');
    return key;
}

ecvShortcutDialog::ecvShortcutDialog(const QList<QAction*>& actions,
                                     QWidget* parent)
    : QDialog(parent),
      m_ui(new Ui_ShortcutDialog),
      m_editDialog(new ecvShortcutEditDialog(this)),
      m_allActions(actions) {
    m_ui->setupUi(this);

    const auto vtkDefs = vtkDefaultShortcuts();
    m_ui->tableWidget->setRowCount(actions.count() + vtkDefs.size());
    m_ui->tableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);

    m_categoryCombo = new QComboBox(this);
    m_categoryCombo->addItem(tr("All Categories"));
    if (auto* lay = qobject_cast<QVBoxLayout*>(layout())) {
        auto* filterRow = new QHBoxLayout;
        filterRow->addWidget(m_ui->searchLineEdit);
        filterRow->addWidget(m_categoryCombo);
        lay->removeWidget(m_ui->searchLineEdit);
        lay->insertLayout(0, filterRow);
    }

    connect(m_ui->tableWidget, &QTableWidget::itemDoubleClicked, this,
            &ecvShortcutDialog::handleDoubleClick);
    connect(m_ui->searchLineEdit, &QLineEdit::textChanged, this,
            &ecvShortcutDialog::filterActions);
    connect(m_categoryCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ecvShortcutDialog::onCategoryChanged);

    QSet<QString> categories;
    int row = 0;
    for (QAction* action : actions) {
        m_defaultShortcuts[action] = action->shortcut();

        QString displayText = action->text();
        QString menuPath = getMenuPath(action, parent);
        QString toolTip = action->toolTip();

        QString cleanText = displayText;
        cleanText.remove('&');

        QString category = menuPath.split(" > ").value(0);
        if (category.isEmpty()) category = tr("Other");
        categories.insert(category);

        QString fullDescription = cleanText;
        if (!menuPath.isEmpty()) {
            fullDescription += " (" + menuPath + ")";
        }
        if (!toolTip.isEmpty() && toolTip != displayText &&
            toolTip != cleanText) {
            QString cleanToolTip = toolTip;
            cleanToolTip.remove('&');
            if (cleanToolTip != cleanText && !cleanToolTip.isEmpty()) {
                fullDescription += " - " + cleanToolTip;
            }
        }

        auto* actionWidget =
                new QTableWidgetItem(action->icon(), fullDescription);
        actionWidget->setFlags(actionWidget->flags() & ~Qt::ItemIsEditable);
        actionWidget->setData(Qt::UserRole + 1, cleanText);
        actionWidget->setData(Qt::UserRole + 2, menuPath);
        actionWidget->setData(Qt::UserRole + 3, toolTip);
        actionWidget->setData(Qt::UserRole + 4, category);
        m_ui->tableWidget->setItem(row, ACTION_NAME_COLUMN, actionWidget);

        auto* shortcutWidget =
                new QTableWidgetItem(action->shortcut().toString());
        shortcutWidget->setFlags(actionWidget->flags() & ~Qt::ItemIsEditable);
        shortcutWidget->setData(Qt::UserRole, QVariant::fromValue(action));
        m_ui->tableWidget->setItem(row, KEY_SEQUENCE_COLUMN, shortcutWidget);
        row += 1;
    }

    // --- VTK interactor shortcut rows (editable) ---
    QSettings vtkSettings;
    vtkSettings.beginGroup(QStringLiteral("VtkShortcuts"));

    QString vtkCategory = QStringLiteral("VTK");
    categories.insert(vtkCategory);

    for (const auto& def : vtkDefs) {
        QKeySequence currentSeq = def.defaultKey;
        if (vtkSettings.contains(def.id)) {
            currentSeq = vtkSettings.value(def.id).value<QKeySequence>();
        }
        m_vtkDefaults[def.id] = def.defaultKey;

        auto* nameItem = new QTableWidgetItem(def.label);
        nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
        nameItem->setData(Qt::UserRole + 1, def.label);
        nameItem->setData(Qt::UserRole + 2, QStringLiteral("VTK Interactor"));
        nameItem->setData(Qt::UserRole + 3, QString());
        nameItem->setData(Qt::UserRole + 4, vtkCategory);
        nameItem->setData(Qt::UserRole + 5, QStringLiteral("vtk"));
        nameItem->setData(Qt::UserRole + 6, def.id);
        m_ui->tableWidget->setItem(row, ACTION_NAME_COLUMN, nameItem);

        auto* seqItem = new QTableWidgetItem(currentSeq.toString());
        seqItem->setFlags(seqItem->flags() & ~Qt::ItemIsEditable);
        seqItem->setData(Qt::UserRole, QVariant());
        seqItem->setData(Qt::UserRole + 5, QStringLiteral("vtk"));
        seqItem->setData(Qt::UserRole + 6, def.id);
        m_ui->tableWidget->setItem(row, KEY_SEQUENCE_COLUMN, seqItem);
        row += 1;
    }
    vtkSettings.endGroup();

    QStringList sortedCats = categories.values();
    sortedCats.sort();
    for (const auto& cat : sortedCats) {
        m_categoryCombo->addItem(cat);
    }

    m_ui->tableWidget->horizontalHeader()->setSectionResizeMode(
            KEY_SEQUENCE_COLUMN, QHeaderView::ResizeToContents);
    m_ui->tableWidget->resizeColumnToContents(KEY_SEQUENCE_COLUMN);

    int shortcutContentWidth =
            m_ui->tableWidget->columnWidth(KEY_SEQUENCE_COLUMN);
    constexpr int minShortcutWidth = 200;
    int shortcutWidth = qMax(shortcutContentWidth + 30, minShortcutWidth);

    m_ui->tableWidget->setColumnWidth(KEY_SEQUENCE_COLUMN, shortcutWidth);
    m_ui->tableWidget->horizontalHeader()->setSectionResizeMode(
            KEY_SEQUENCE_COLUMN, QHeaderView::Fixed);
    m_ui->tableWidget->horizontalHeader()->setStretchLastSection(false);
    m_ui->tableWidget->horizontalHeader()->setSectionResizeMode(
            ACTION_NAME_COLUMN, QHeaderView::Stretch);

    m_conflictLabel = new QLabel(this);
    m_conflictLabel->setWordWrap(true);

    auto* resetBtn = new QPushButton(tr("Reset Selected"), this);
    auto* resetAllBtn = new QPushButton(tr("Reset All Defaults"), this);
    auto* exportBtn = new QPushButton(tr("Export..."), this);
    auto* importBtn = new QPushButton(tr("Import..."), this);
    connect(resetBtn, &QPushButton::clicked, this,
            &ecvShortcutDialog::onResetSelected);
    connect(resetAllBtn, &QPushButton::clicked, this,
            &ecvShortcutDialog::onResetAll);
    connect(exportBtn, &QPushButton::clicked, this,
            &ecvShortcutDialog::onExportShortcuts);
    connect(importBtn, &QPushButton::clicked, this,
            &ecvShortcutDialog::onImportShortcuts);

    if (auto* lay = qobject_cast<QVBoxLayout*>(layout())) {
        auto* btnRow = new QHBoxLayout;
        btnRow->addWidget(resetBtn);
        btnRow->addWidget(resetAllBtn);
        btnRow->addStretch(1);
        btnRow->addWidget(importBtn);
        btnRow->addWidget(exportBtn);
        lay->insertLayout(lay->count() - 1, btnRow);
        lay->insertWidget(lay->count() - 1, m_conflictLabel);
    }

    syncModalShortcuts();
    refreshConflictHighlighting();
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
        if (item->data(Qt::UserRole + 5).toString() == "vtk") continue;

        auto* action = item->data(Qt::UserRole).value<QAction*>();
        if (!action) continue;

        QString key = actionPersistKey(action);
        if (key.isEmpty()) continue;

        if (settings.contains(key)) {
            const auto sequence =
                    settings.value(key).value<QKeySequence>();
            item->setText(sequence.toString());
            action->setShortcut(sequence);
        } else if (settings.contains(action->text())) {
            const auto sequence =
                    settings.value(action->text()).value<QKeySequence>();
            item->setText(sequence.toString());
            action->setShortcut(sequence);
        }
    }
    settings.endGroup();
}

void ecvShortcutDialog::registerStandaloneShortcut(const QString& name,
                                                    const QKeySequence& seq) {
    m_standaloneShortcuts[name] = seq;
}

void ecvShortcutDialog::syncModalShortcuts() {
    m_modalShortcuts.clear();
    auto modalSeqs = ecvKeySequences::instance().allRegisteredSequences();
    for (auto it = modalSeqs.constBegin(); it != modalSeqs.constEnd(); ++it) {
        QKeySequence seq(it.key());
        if (seq.isEmpty()) continue;
        QString label = it.value().join(", ");
        m_modalShortcuts[label] = seq;
    }
}

QList<ecvShortcutDialog::ConflictInfo> ecvShortcutDialog::checkConflicts(
        const QKeySequence& sequence, const QAction* exclude) const {
    QList<ConflictInfo> conflicts;

    for (int i = 0; i < m_ui->tableWidget->rowCount(); i++) {
        const QTableWidgetItem* item =
                m_ui->tableWidget->item(i, KEY_SEQUENCE_COLUMN);

        bool isVtk = item->data(Qt::UserRole + 5).toString() == "vtk";
        if (isVtk) {
            QKeySequence rowSeq(item->text());
            if (rowSeq == sequence) {
                ConflictInfo ci;
                ci.name = QStringLiteral("[VTK] ") +
                          item->data(Qt::UserRole + 6).toString();
                ci.row = i;
                conflicts.append(ci);
            }
            continue;
        }

        const auto* action = item->data(Qt::UserRole).value<QAction*>();
        if (!action) continue;
        if (action == exclude) continue;
        if (action->shortcut() == sequence) {
            ConflictInfo ci;
            ci.action = action;
            ci.name = action->text();
            ci.name.remove('&');
            ci.row = i;
            conflicts.append(ci);
        }
    }

    for (auto it = m_standaloneShortcuts.constBegin();
         it != m_standaloneShortcuts.constEnd(); ++it) {
        if (it.value() == sequence) {
            ConflictInfo ci;
            ci.name = QStringLiteral("[Standalone] ") + it.key();
            conflicts.append(ci);
        }
    }

    for (auto it = m_modalShortcuts.constBegin();
         it != m_modalShortcuts.constEnd(); ++it) {
        if (it.value() == sequence) {
            ConflictInfo ci;
            ci.name = QStringLiteral("[Modal] ") + it.key();
            conflicts.append(ci);
        }
    }

    return conflicts;
}

QStringList ecvShortcutDialog::detectAllConflicts() const {
    QStringList result;
    QMap<QString, QStringList> seqToNames;

    for (int i = 0; i < m_ui->tableWidget->rowCount(); i++) {
        const QTableWidgetItem* item =
                m_ui->tableWidget->item(i, KEY_SEQUENCE_COLUMN);
        bool isVtk = item->data(Qt::UserRole + 5).toString() == "vtk";

        if (isVtk) {
            QKeySequence seq(item->text());
            if (seq.isEmpty()) continue;
            seqToNames[seq.toString()].append(
                    QStringLiteral("[VTK] ") +
                    item->data(Qt::UserRole + 6).toString());
        } else {
            const auto* action =
                    item->data(Qt::UserRole).value<QAction*>();
            if (!action || action->shortcut().isEmpty()) continue;
            QString name = action->text();
            name.remove('&');
            seqToNames[action->shortcut().toString()].append(name);
        }
    }

    for (auto it = m_standaloneShortcuts.constBegin();
         it != m_standaloneShortcuts.constEnd(); ++it) {
        if (it.value().isEmpty()) continue;
        seqToNames[it.value().toString()].append(
                QStringLiteral("[Standalone] ") + it.key());
    }

    for (auto it = m_modalShortcuts.constBegin();
         it != m_modalShortcuts.constEnd(); ++it) {
        if (it.value().isEmpty()) continue;
        seqToNames[it.value().toString()].append(
                QStringLiteral("[Modal] ") + it.key());
    }

    for (auto it = seqToNames.constBegin(); it != seqToNames.constEnd();
         ++it) {
        if (it.value().size() > 1) {
            result.append(QStringLiteral("%1: %2")
                                  .arg(it.key())
                                  .arg(it.value().join(", ")));
        }
    }

    return result;
}

void ecvShortcutDialog::refreshConflictHighlighting() {
    QMap<QString, QList<int>> seqToRows;

    for (int i = 0; i < m_ui->tableWidget->rowCount(); i++) {
        const QTableWidgetItem* item =
                m_ui->tableWidget->item(i, KEY_SEQUENCE_COLUMN);
        bool isVtk = item->data(Qt::UserRole + 5).toString() == "vtk";

        if (isVtk) {
            QKeySequence seq(item->text());
            if (seq.isEmpty()) continue;
            seqToRows[seq.toString()].append(i);
        } else {
            const auto* action =
                    item->data(Qt::UserRole).value<QAction*>();
            if (!action || action->shortcut().isEmpty()) continue;
            seqToRows[action->shortcut().toString()].append(i);
        }
    }

    QSet<QString> externalSeqs;
    for (auto it = m_standaloneShortcuts.constBegin();
         it != m_standaloneShortcuts.constEnd(); ++it) {
        if (!it.value().isEmpty()) externalSeqs.insert(it.value().toString());
    }
    for (auto it = m_modalShortcuts.constBegin();
         it != m_modalShortcuts.constEnd(); ++it) {
        if (!it.value().isEmpty()) externalSeqs.insert(it.value().toString());
    }

    QColor conflictBg(120, 40, 40);
    QColor normalBg;
    int conflictCount = 0;

    for (int i = 0; i < m_ui->tableWidget->rowCount(); i++) {
        m_ui->tableWidget->item(i, ACTION_NAME_COLUMN)->setBackground(normalBg);
        m_ui->tableWidget->item(i, KEY_SEQUENCE_COLUMN)
                ->setBackground(normalBg);
    }

    for (auto it = seqToRows.constBegin(); it != seqToRows.constEnd(); ++it) {
        bool hasConflict =
                it.value().size() > 1 || externalSeqs.contains(it.key());
        if (hasConflict) {
            conflictCount += it.value().size();
            for (int row : it.value()) {
                m_ui->tableWidget->item(row, ACTION_NAME_COLUMN)
                        ->setBackground(conflictBg);
                m_ui->tableWidget->item(row, KEY_SEQUENCE_COLUMN)
                        ->setBackground(conflictBg);
            }
        }
    }

    if (m_conflictLabel) {
        if (conflictCount > 0) {
            m_conflictLabel->setText(
                    tr("<span style='color:#ff6666;'>%1 shortcut "
                       "conflict(s) detected. Conflicting rows are "
                       "highlighted in red.</span>")
                            .arg(conflictCount));
        } else {
            m_conflictLabel->setText(
                    tr("<span style='color:#66ff66;'>No shortcut "
                       "conflicts detected.</span>"));
        }
    }
}

void ecvShortcutDialog::handleDoubleClick(QTableWidgetItem* item) {
    if (!item) return;

    if (item->column() != KEY_SEQUENCE_COLUMN) {
        item = m_ui->tableWidget->item(item->row(), KEY_SEQUENCE_COLUMN);
    }

    bool isVtk = item->data(Qt::UserRole + 5).toString() == "vtk";

    if (isVtk) {
        QString vtkId = item->data(Qt::UserRole + 6).toString();
        QKeySequence oldSeq(item->text());
        m_editDialog->setKeySequence(oldSeq);

        if (m_editDialog->exec() == QDialog::Rejected) return;

        const QKeySequence keySequence = m_editDialog->keySequence();
        if (keySequence == oldSeq) return;

        if (!keySequence.isEmpty()) {
            auto conflicts = checkConflicts(keySequence, nullptr);
            QList<ConflictInfo> filtered;
            for (const auto& c : conflicts) {
                if (c.name.contains(vtkId)) continue;
                filtered.append(c);
            }
            if (!filtered.isEmpty()) {
                QStringList names;
                for (const auto& c : filtered) names.append(c.name);
                QMessageBox::critical(
                        this, tr("Shortcut Conflict"),
                        tr("The shortcut \"%1\" conflicts with:\n%2\n\n"
                           "Please choose a different shortcut.")
                                .arg(keySequence.toString())
                                .arg(names.join("\n")));
                return;
            }
        }

        item->setText(keySequence.toString());

        QSettings settings;
        settings.beginGroup(QStringLiteral("VtkShortcuts"));
        settings.setValue(vtkId, keySequence);
        settings.endGroup();

        refreshConflictHighlighting();
        return;
    }

    auto* action = item->data(Qt::UserRole).value<QAction*>();
    if (!action) return;
    m_editDialog->setKeySequence(action->shortcut());

    if (m_editDialog->exec() == QDialog::Rejected) return;

    const QKeySequence keySequence = m_editDialog->keySequence();
    if (keySequence == action->shortcut()) return;

    if (!keySequence.isEmpty()) {
        auto conflicts = checkConflicts(keySequence, action);
        if (!conflicts.isEmpty()) {
            QStringList names;
            for (const auto& c : conflicts) names.append(c.name);
            QMessageBox::critical(
                    this, tr("Shortcut Conflict"),
                    tr("The shortcut \"%1\" conflicts with:\n%2\n\n"
                       "Please choose a different shortcut.")
                            .arg(keySequence.toString())
                            .arg(names.join("\n")));
            return;
        }
    }

    item->setText(keySequence.toString());
    action->setShortcut(keySequence);

    QSettings settings;
    settings.beginGroup(ecvPS::Shortcuts());
    settings.setValue(actionPersistKey(action), keySequence);
    settings.endGroup();

    refreshConflictHighlighting();
}

void ecvShortcutDialog::onResetSelected() {
    auto selected = m_ui->tableWidget->selectedItems();
    QSet<int> rows;
    for (auto* item : selected) rows.insert(item->row());

    if (rows.isEmpty()) {
        QMessageBox::information(this, tr("Reset Shortcut"),
                                 tr("Please select one or more rows first."));
        return;
    }

    QSettings actionSettings;
    actionSettings.beginGroup(ecvPS::Shortcuts());

    QSettings vtkSettings;
    vtkSettings.beginGroup(QStringLiteral("VtkShortcuts"));

    for (int row : rows) {
        auto* item = m_ui->tableWidget->item(row, KEY_SEQUENCE_COLUMN);
        bool isVtk = item->data(Qt::UserRole + 5).toString() == "vtk";

        if (isVtk) {
            QString vtkId = item->data(Qt::UserRole + 6).toString();
            if (m_vtkDefaults.contains(vtkId)) {
                QKeySequence def = m_vtkDefaults[vtkId];
                item->setText(def.toString());
                vtkSettings.remove(vtkId);
            }
        } else {
            auto* action = item->data(Qt::UserRole).value<QAction*>();
            if (action && m_defaultShortcuts.contains(action)) {
                QKeySequence def = m_defaultShortcuts[action];
                action->setShortcut(def);
                item->setText(def.toString());
                actionSettings.remove(actionPersistKey(action));
            }
        }
    }

    actionSettings.endGroup();
    vtkSettings.endGroup();
    refreshConflictHighlighting();
}

void ecvShortcutDialog::onResetAll() {
    if (QMessageBox::question(this, tr("Reset All Shortcuts"),
                              tr("Reset all shortcuts to their default values?"),
                              QMessageBox::Yes | QMessageBox::No) !=
        QMessageBox::Yes) {
        return;
    }

    QSettings actionSettings;
    actionSettings.beginGroup(ecvPS::Shortcuts());

    QSettings vtkSettings;
    vtkSettings.beginGroup(QStringLiteral("VtkShortcuts"));

    for (int i = 0; i < m_ui->tableWidget->rowCount(); i++) {
        auto* item = m_ui->tableWidget->item(i, KEY_SEQUENCE_COLUMN);
        bool isVtk = item->data(Qt::UserRole + 5).toString() == "vtk";

        if (isVtk) {
            QString vtkId = item->data(Qt::UserRole + 6).toString();
            if (m_vtkDefaults.contains(vtkId)) {
                item->setText(m_vtkDefaults[vtkId].toString());
            }
        } else {
            auto* action = item->data(Qt::UserRole).value<QAction*>();
            if (action && m_defaultShortcuts.contains(action)) {
                QKeySequence def = m_defaultShortcuts[action];
                action->setShortcut(def);
                item->setText(def.toString());
            }
        }
    }

    actionSettings.remove("");
    actionSettings.endGroup();

    vtkSettings.remove("");
    vtkSettings.endGroup();

    refreshConflictHighlighting();
}

void ecvShortcutDialog::onExportShortcuts() {
    QString path = QFileDialog::getSaveFileName(
            this, tr("Export Shortcuts"), QString(), tr("JSON (*.json)"));
    if (path.isEmpty()) return;

    QJsonObject root;
    QJsonObject vtkObj;
    for (int i = 0; i < m_ui->tableWidget->rowCount(); i++) {
        auto* item = m_ui->tableWidget->item(i, KEY_SEQUENCE_COLUMN);
        bool isVtk = item->data(Qt::UserRole + 5).toString() == "vtk";

        if (isVtk) {
            QString vtkId = item->data(Qt::UserRole + 6).toString();
            vtkObj[vtkId] = item->text();
        } else {
            auto* action = item->data(Qt::UserRole).value<QAction*>();
            if (!action) continue;
            QString key = actionPersistKey(action);
            if (key.isEmpty()) continue;
            root[key] = action->shortcut().toString();
        }
    }
    if (!vtkObj.isEmpty()) root["__vtk__"] = vtkObj;

    QFile file(path);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        file.close();
        QMessageBox::information(
                this, tr("Export Shortcuts"),
                tr("Exported %1 shortcuts to %2").arg(root.size()).arg(path));
    }
}

void ecvShortcutDialog::onImportShortcuts() {
    QString path = QFileDialog::getOpenFileName(
            this, tr("Import Shortcuts"), QString(), tr("JSON (*.json)"));
    if (path.isEmpty()) return;

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Import Shortcuts"),
                             tr("Cannot open file: %1").arg(path));
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();
    if (!doc.isObject()) {
        QMessageBox::warning(this, tr("Import Shortcuts"),
                             tr("Invalid JSON format."));
        return;
    }

    QJsonObject root = doc.object();
    int applied = 0;

    QJsonObject vtkObj = root["__vtk__"].toObject();

    QSettings actionSettings;
    actionSettings.beginGroup(ecvPS::Shortcuts());

    QSettings vtkSettings;
    vtkSettings.beginGroup(QStringLiteral("VtkShortcuts"));

    for (int i = 0; i < m_ui->tableWidget->rowCount(); i++) {
        auto* item = m_ui->tableWidget->item(i, KEY_SEQUENCE_COLUMN);
        bool isVtk = item->data(Qt::UserRole + 5).toString() == "vtk";

        if (isVtk) {
            QString vtkId = item->data(Qt::UserRole + 6).toString();
            if (vtkObj.contains(vtkId)) {
                QKeySequence seq(vtkObj[vtkId].toString());
                item->setText(seq.toString());
                vtkSettings.setValue(vtkId, seq);
                applied++;
            }
        } else {
            auto* action = item->data(Qt::UserRole).value<QAction*>();
            if (!action) continue;
            QString key = actionPersistKey(action);
            if (key.isEmpty()) continue;

            if (root.contains(key)) {
                QKeySequence seq(root[key].toString());
                action->setShortcut(seq);
                item->setText(seq.toString());
                actionSettings.setValue(key, seq);
                applied++;
            }
        }
    }

    actionSettings.endGroup();
    vtkSettings.endGroup();
    refreshConflictHighlighting();

    QMessageBox::information(
            this, tr("Import Shortcuts"),
            tr("Applied %1 shortcut(s) from %2").arg(applied).arg(path));
}

void ecvShortcutDialog::filterActions(const QString& /*searchText*/) {
    applyFilters();
}

void ecvShortcutDialog::onCategoryChanged(int /*index*/) { applyFilters(); }

void ecvShortcutDialog::applyFilters() {
    QString searchLower = m_ui->searchLineEdit->text().toLower();
    QString selectedCategory;
    if (m_categoryCombo->currentIndex() > 0)
        selectedCategory = m_categoryCombo->currentText();

    for (int row = 0; row < m_ui->tableWidget->rowCount(); ++row) {
        QTableWidgetItem* item =
                m_ui->tableWidget->item(row, ACTION_NAME_COLUMN);
        if (!item) continue;

        bool catMatch = selectedCategory.isEmpty() ||
                        item->data(Qt::UserRole + 4).toString() ==
                                selectedCategory;

        bool textMatch = true;
        if (!searchLower.isEmpty()) {
            textMatch = item->text().toLower().contains(searchLower);
            if (!textMatch) {
                textMatch =
                        item->data(Qt::UserRole + 1)
                                .toString()
                                .toLower()
                                .contains(searchLower) ||
                        item->data(Qt::UserRole + 2)
                                .toString()
                                .toLower()
                                .contains(searchLower) ||
                        item->data(Qt::UserRole + 3)
                                .toString()
                                .toLower()
                                .contains(searchLower);
            }
            if (!textMatch) {
                auto* shortcutItem =
                        m_ui->tableWidget->item(row, KEY_SEQUENCE_COLUMN);
                if (shortcutItem) {
                    textMatch = shortcutItem->text()
                                        .toLower()
                                        .contains(searchLower);
                }
            }
        }

        m_ui->tableWidget->setRowHidden(row, !(catMatch && textMatch));
    }
}

void ecvShortcutDialog::showAllRows() {
    for (int row = 0; row < m_ui->tableWidget->rowCount(); ++row) {
        m_ui->tableWidget->setRowHidden(row, false);
    }
}
