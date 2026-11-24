// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvShortcutDialog.h"

#include <CVLog.h>

#include <QAction>
#include <QComboBox>
#include <QFileDialog>
#include <QHeaderView>
#include <QKeySequenceEdit>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QSaveFile>
#include <QSignalBlocker>
#include <QTableWidget>
#include <QVBoxLayout>

namespace {
constexpr int IdRole = Qt::UserRole;
constexpr int CategoryRole = Qt::UserRole + 1;

QString menuPath(QMenu* menu, QAction* target) {
    for (QAction* action : menu->actions()) {
        QString path;
        if (action == target)
            path = menu->title();
        else if (action->menu()) {
            path = menuPath(action->menu(), target);
            if (!path.isEmpty()) path.prepend(menu->title() + " > ");
        }
        if (!path.isEmpty()) {
            path.remove('&');
            return path;
        }
    }
    return {};
}

QStringList bindingText(const QList<QKeySequence>& keys) {
    QStringList result;
    for (const auto& key : keys) result.append(key.toString());
    return result;
}
}  // namespace

ecvShortcutEditDialog::ecvShortcutEditDialog(QWidget* parent)
    : QDialog(parent),
      m_ui(new Ui_ShortcutEditDialog),
      m_bindingsList(new QListWidget(this)) {
    m_ui->setupUi(this);
    m_ui->verticalLayout->insertWidget(
            0, new QLabel(tr("Alternative bindings (each binding may contain "
                             "multiple strokes):"),
                          this));
    m_ui->verticalLayout->insertWidget(1, m_bindingsList);
    auto* add = new QPushButton(tr("Add binding"), this);
    m_ui->horizontalLayout->addWidget(add);
    connect(m_bindingsList, &QListWidget::currentRowChanged, this,
            [this](int row) {
                storeCurrentBinding();
                m_editingRow = row;
                const auto* item = m_bindingsList->item(row);
                m_ui->keySequenceEdit->setKeySequence(
                        item ? item->data(Qt::UserRole).value<QKeySequence>()
                             : QKeySequence());
            });
    connect(add, &QPushButton::clicked, this, [this]() {
        storeCurrentBinding();
        auto* item = new QListWidgetItem(tr("New binding"), m_bindingsList);
        m_bindingsList->setCurrentItem(item);
        m_ui->keySequenceEdit->setFocus();
    });
    connect(m_ui->clearButton, &QPushButton::clicked, m_ui->keySequenceEdit,
            &QKeySequenceEdit::clear);
    connect(m_ui->keySequenceEdit, &QKeySequenceEdit::keySequenceChanged, this,
            [this]() { storeCurrentBinding(); });
    resize(500, 250);
}

ecvShortcutEditDialog::~ecvShortcutEditDialog() { delete m_ui; }

void ecvShortcutEditDialog::storeCurrentBinding() {
    if (auto* item = m_bindingsList->item(m_editingRow)) {
        const auto sequence = m_ui->keySequenceEdit->keySequence();
        item->setData(Qt::UserRole, QVariant::fromValue(sequence));
        item->setText(sequence.isEmpty() ? tr("Unbound") : sequence.toString());
    }
}

QList<QKeySequence> ecvShortcutEditDialog::keySequences() const {
    QList<QKeySequence> result;
    for (int i = 0; i < m_bindingsList->count(); ++i) {
        const auto sequence = i == m_editingRow
                                      ? m_ui->keySequenceEdit->keySequence()
                                      : m_bindingsList->item(i)
                                                ->data(Qt::UserRole)
                                                .value<QKeySequence>();
        if (!sequence.isEmpty() && !result.contains(sequence))
            result.append(sequence);
    }
    return result;
}

void ecvShortcutEditDialog::setKeySequences(
        const QList<QKeySequence>& sequences) {
    {
        QSignalBlocker blocker(m_bindingsList);
        m_bindingsList->clear();
        m_editingRow = -1;
        m_ui->keySequenceEdit->clear();
        for (const auto& sequence : sequences) {
            auto* item =
                    new QListWidgetItem(sequence.toString(), m_bindingsList);
            item->setData(Qt::UserRole, QVariant::fromValue(sequence));
        }
        if (m_bindingsList->count() == 0)
            new QListWidgetItem(tr("New binding"), m_bindingsList);
    }
    m_bindingsList->setCurrentRow(0);
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
      m_categoryCombo(new QComboBox(this)),
      m_conflictLabel(new QLabel(this)) {
    m_ui->setupUi(this);
    m_ui->tableWidget->setAlternatingRowColors(true);
    m_ui->tableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);
    auto* layout = m_ui->verticalLayout;
    auto* filters = new QHBoxLayout;
    layout->removeWidget(m_ui->searchLineEdit);
    filters->addWidget(m_ui->searchLineEdit);
    filters->addWidget(m_categoryCombo);
    layout->insertLayout(0, filters);
    m_ui->tableWidget->horizontalHeader()->setSectionResizeMode(
            0, QHeaderView::Stretch);
    m_ui->tableWidget->horizontalHeader()->setSectionResizeMode(
            1, QHeaderView::ResizeToContents);
    m_conflictLabel->setWordWrap(true);
    layout->insertWidget(layout->count() - 1, m_conflictLabel);
    auto* buttons = new QHBoxLayout;
    auto addButton = [this, buttons](const QString& label, auto slot) {
        auto* button = new QPushButton(label, this);
        buttons->addWidget(button);
        connect(button, &QPushButton::clicked, this, slot);
    };
    addButton(tr("Reset Selected"), &ecvShortcutDialog::onResetSelected);
    addButton(tr("Reset All Defaults"), &ecvShortcutDialog::onResetAll);
    buttons->addStretch();
    addButton(tr("Import..."), &ecvShortcutDialog::onImportShortcuts);
    addButton(tr("Export..."), &ecvShortcutDialog::onExportShortcuts);
    layout->insertLayout(layout->count() - 1, buttons);
    auto* note = new QLabel(
            tr("Changes are applied immediately. Text editing, mouse modifiers "
               "and contextual tool cancellation are not global command "
               "bindings."),
            this);
    note->setWordWrap(true);
    layout->insertWidget(layout->count() - 1, note);
    connect(m_ui->tableWidget, &QTableWidget::itemDoubleClicked, this,
            &ecvShortcutDialog::handleDoubleClick);
    connect(m_ui->searchLineEdit, &QLineEdit::textChanged, this,
            &ecvShortcutDialog::applyFilters);
    connect(m_categoryCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ecvShortcutDialog::applyFilters);
    registerActions(actions);
    connect(&ecvKeySequences::instance(), &ecvKeySequences::commandsChanged,
            this, [this]() {
                if (isVisible()) refreshCommands();
            });
    refreshCommands();
}

ecvShortcutDialog::~ecvShortcutDialog() { delete m_ui; }

void ecvShortcutDialog::showEvent(QShowEvent* event) {
    refreshCommands();
    QDialog::showEvent(event);
}

void ecvShortcutDialog::registerActions(const QList<QAction*>& actions) {
    auto& registry = ecvKeySequences::instance();
    for (auto* action : actions) {
        if (!action || action->property("shortcutTransient").toBool() ||
            action->property("shortcutProxy").toBool() ||
            action->property("shortcutGesture").toBool() ||
            !action->property("shortcutCommandId").toString().isEmpty())
            continue;
        if (action->objectName().isEmpty()) continue;
        QString category = action->property("shortcutCategory").toString();
        if (category.isEmpty() && parentWidget()) {
            if (auto* bar = parentWidget()->findChild<QMenuBar*>()) {
                for (auto* entry : bar->actions()) {
                    if (!entry->menu()) continue;
                    category = menuPath(entry->menu(), action)
                                       .section(" > ", 0, 0);
                    if (!category.isEmpty()) break;
                }
            }
        }
        registry.registerCommand(action->objectName(), action,
                                 category.isEmpty() ? tr("Other") : category);
    }
}

void ecvShortcutDialog::restoreShortcutsFromQSettings() const {
    QStringList errors;
    if (!ecvKeySequences::instance().restoreBindings(&errors))
        CVLog::Warning(tr("Shortcut settings were not applied:\n%1")
                               .arg(errors.join("\n")));
}

QStringList ecvShortcutDialog::detectAllConflicts() const {
    const auto& registry = ecvKeySequences::instance();
    QStringList result;
    for (const auto& conflict : registry.conflicts(registry.bindings()))
        result.append(QStringLiteral("%1 / %2: %3")
                              .arg(conflict.first, conflict.second,
                                   conflict.sequence.toString()));
    return result;
}

void ecvShortcutDialog::refreshCommands() {
    const QString selectedCategory = m_categoryCombo->currentData().toString();
    QString selectedId;
    if (auto* item = m_ui->tableWidget->currentItem())
        selectedId = m_ui->tableWidget->item(item->row(), 0)
                             ->data(IdRole)
                             .toString();
    const auto& registry = ecvKeySequences::instance();
    const auto bindings = registry.bindings();
    const auto commands = registry.commands();
    QSet<QString> categories, conflicting;
    const auto conflicts = registry.conflicts(bindings);
    for (const auto& conflict : conflicts) {
        conflicting.insert(conflict.first);
        conflicting.insert(conflict.second);
    }
    m_ui->tableWidget->setRowCount(commands.size());
    int row = 0;
    for (const auto& command : commands) {
        QAction* action = nullptr;
        for (const auto& instance : command.actions)
            if (instance) {
                action = instance;
                break;
            }
        QString name = action ? action->text() : command.label;
        name.remove('&');
        if (!action) name += tr(" [not currently loaded]");
        auto* nameItem =
                new QTableWidgetItem(action ? action->icon() : QIcon(), name);
        nameItem->setData(IdRole, command.id);
        nameItem->setData(CategoryRole, command.category);
        nameItem->setToolTip(command.id + "\n" +
                             (action ? action->toolTip()
                                     : tr("Applied when this command's window "
                                          "is created.")));
        auto* keyItem = new QTableWidgetItem(
                bindingText(bindings.value(command.id)).join("; "));
        for (auto* item : {nameItem, keyItem}) {
            item->setFlags(item->flags() & ~Qt::ItemIsEditable);
            if (conflicting.contains(command.id)) {
                const bool dark = palette().base().color().lightness() < 128;
                item->setBackground(dark ? QColor(130, 50, 50)
                                         : QColor(255, 210, 210));
            }
        }
        m_ui->tableWidget->setItem(row, 0, nameItem);
        m_ui->tableWidget->setItem(row, 1, keyItem);
        if (command.id == selectedId) m_ui->tableWidget->selectRow(row);
        categories.insert(command.category);
        ++row;
    }
    {
        QSignalBlocker blocker(m_categoryCombo);
        m_categoryCombo->clear();
        m_categoryCombo->addItem(tr("All Categories"), QString());
        auto sorted = categories.values();
        sorted.sort();
        for (const auto& category : sorted)
            m_categoryCombo->addItem(category, category);
        const int index = m_categoryCombo->findData(selectedCategory);
        if (index >= 0) m_categoryCombo->setCurrentIndex(index);
    }
    m_conflictLabel->setText(
            conflicts.isEmpty()
                    ? tr("No conflicts among registered command bindings.")
                    : tr("%1 conflicting command binding pair(s).")
                              .arg(conflicts.size()));
    applyFilters();
}

bool ecvShortcutDialog::apply(const ecvKeySequences::Bindings& bindings) {
    QStringList errors;
    if (ecvKeySequences::instance().applyBindings(bindings, &errors))
        return true;
    QMessageBox::warning(this, tr("Shortcut configuration rejected"),
                         errors.join("\n"));
    return false;
}

void ecvShortcutDialog::handleDoubleClick(QTableWidgetItem* item) {
    if (!item) return;
    const QString id =
            m_ui->tableWidget->item(item->row(), 0)->data(IdRole).toString();
    auto bindings = ecvKeySequences::instance().bindings();
    m_editDialog->setKeySequences(bindings.value(id));
    if (m_editDialog->exec() != QDialog::Accepted) return;
    bindings[id] = m_editDialog->keySequences();
    apply(bindings);
}

void ecvShortcutDialog::onResetSelected() {
    auto& registry = ecvKeySequences::instance();
    auto bindings = registry.bindings();
    const auto defaults = registry.defaultBindings();
    QSet<int> rows;
    for (auto* item : m_ui->tableWidget->selectedItems())
        rows.insert(item->row());
    if (rows.isEmpty()) return;
    for (int row : rows) {
        const auto id =
                m_ui->tableWidget->item(row, 0)->data(IdRole).toString();
        bindings[id] = defaults.value(id);
    }
    apply(bindings);
}

void ecvShortcutDialog::onResetAll() {
    if (QMessageBox::question(this, tr("Reset All Shortcuts"),
                              tr("Reset all registered commands to their "
                                 "factory defaults?")) == QMessageBox::Yes)
        apply(ecvKeySequences::instance().defaultBindings());
}

void ecvShortcutDialog::onExportShortcuts() {
    const QString path = QFileDialog::getSaveFileName(
            this, tr("Export Shortcuts"), {}, tr("JSON (*.json)"));
    if (path.isEmpty()) return;
    QSaveFile file(path);
    const auto bytes = ecvKeySequences::instance().exportBindings();
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() ||
        !file.commit())
        QMessageBox::warning(this, tr("Export Shortcuts"), file.errorString());
}

void ecvShortcutDialog::onImportShortcuts() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Import Shortcuts"), {}, tr("JSON (*.json)"));
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Import Shortcuts"), file.errorString());
        return;
    }
    const auto data = file.readAll();
    if (file.error() != QFile::NoError) {
        QMessageBox::warning(this, tr("Import Shortcuts"), file.errorString());
        return;
    }
    QStringList errors, unavailable;
    if (!ecvKeySequences::instance().importBindings(data, &errors,
                                                    &unavailable)) {
        QMessageBox::warning(this, tr("Import Shortcuts"), errors.join("\n"));
        return;
    }
    if (!unavailable.isEmpty())
        QMessageBox::information(
                this, tr("Import Shortcuts"),
                tr("Bindings retained for unavailable commands (not activated "
                   "until their owners load):\n%1")
                        .arg(unavailable.join("\n")));
}

void ecvShortcutDialog::applyFilters() {
    const auto category = m_categoryCombo->currentData().toString();
    const auto search = m_ui->searchLineEdit->text();
    for (int row = 0; row < m_ui->tableWidget->rowCount(); ++row) {
        const auto* item = m_ui->tableWidget->item(row, 0);
        const QString text = item->text() + " " + item->toolTip() + " " +
                             m_ui->tableWidget->item(row, 1)->text();
        m_ui->tableWidget->setRowHidden(
                row, (!category.isEmpty() &&
                      item->data(CategoryRole).toString() != category) ||
                             !text.contains(search, Qt::CaseInsensitive));
    }
}
