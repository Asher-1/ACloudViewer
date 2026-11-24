// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Shortcuts/ecvKeySequences.h"

#include "Shortcuts/ecvModalShortcut.h"

// app
#include <CVLog.h>

#include <QAction>
#include <QApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeySequence>
#include <QMap>
#include <QPointer>
#include <QScopedValueRollback>
#include <QSet>
#include <QSettings>
#include <QShortcut>

namespace {

/**
 * @brief Internal structure to hold all shortcuts registered for a key
 * sequence.
 */
struct Shortcuts {
    /// All shortcuts (siblings) that share the same key sequence
    QSet<ecvModalShortcut*> Siblings;

    /// Most recently used shortcut (for reordering)
    ecvModalShortcut* mruShortcut = nullptr;

    // TODO: Allow a key-sequence to be "default to last" when a shortcut is
    //       removed/disabled as an option?
    // TODO: Hold tooltip or preference name?
};

/**
 * @brief Global dictionary mapping key sequences to their shortcuts.
 */
struct Dictionary {
    QMap<QKeySequence, Shortcuts> Data;
};

/// Global instance
Dictionary g_keys;

// Qt owns key-event matching (including chords and platform modifiers). The
// registry only compares potential contexts at configuration time.
QList<QWidget*> shortcutWidgets(const QAction* action) {
    QList<QWidget*> result;
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    for (QObject* object : action->associatedObjects()) {
        if (auto* widget = qobject_cast<QWidget*>(object))
            result.append(widget);
    }
#else
    result = action->associatedWidgets();
#endif
    return result;
}

QWidget* shortcutWindow(QWidget* widget) {
    // Menus are popup windows, but their actions belong to the owning window.
    while (widget && widget->windowType() == Qt::Popup &&
           widget->parentWidget())
        widget = widget->parentWidget();
    return widget ? widget->window() : nullptr;
}

bool contextsOverlap(const QList<QWidget*>& a,
                     Qt::ShortcutContext ac,
                     const QList<QWidget*>& b,
                     Qt::ShortcutContext bc) {
    // An unassociated action has no executable Qt shortcut.
    if (a.isEmpty() || b.isEmpty()) return false;
    if (ac == Qt::ApplicationShortcut || bc == Qt::ApplicationShortcut)
        return true;
    for (QWidget* aw : a) {
        for (QWidget* bw : b) {
            if (shortcutWindow(aw) != shortcutWindow(bw)) continue;
            if (ac == Qt::WindowShortcut || bc == Qt::WindowShortcut)
                return true;
            if (aw == bw ||
                (ac == Qt::WidgetWithChildrenShortcut &&
                 aw->isAncestorOf(bw)) ||
                (bc == Qt::WidgetWithChildrenShortcut && bw->isAncestorOf(aw)))
                return true;
        }
    }
    return false;
}

bool contextsOverlap(const QAction* first, const QAction* second) {
    return contextsOverlap(shortcutWidgets(first), first->shortcutContext(),
                           shortcutWidgets(second), second->shortcutContext());
}

bool sequencesOverlap(const QKeySequence& a, const QKeySequence& b) {
    return a.matches(b) != QKeySequence::NoMatch ||
           b.matches(a) != QKeySequence::NoMatch;
}

bool validSequence(const QKeySequence& sequence) {
    if (sequence.isEmpty()) return false;
    for (int i = 0; i < sequence.count(); ++i) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        const int key = sequence[i].key();
#else
        const int key = sequence[i] & ~int(Qt::KeyboardModifierMask);
#endif
        if (key == 0 || key == Qt::Key_unknown ||
            (key >= Qt::Key_Shift && key <= Qt::Key_Meta) ||
            key == Qt::Key_AltGr)
            return false;
    }
    return true;
}

QByteArray encodeBindingFile(const ecvKeySequences::Bindings& bindings,
                             QJsonDocument::JsonFormat format) {
    QJsonObject object;
    for (auto it = bindings.constBegin(); it != bindings.constEnd(); ++it) {
        QJsonArray keys;
        for (const auto& sequence : it.value())
            keys.append(sequence.toString(QKeySequence::PortableText));
        object.insert(it.key(), keys);
    }
    QJsonObject root;
    root.insert("version", 2);
    root.insert("bindings", object);
    return QJsonDocument(root).toJson(format);
}

bool decodeBindingFile(const QByteArray& data,
                       ecvKeySequences::Bindings& bindings,
                       bool allowLegacy,
                       bool& legacy,
                       QStringList* errors) {
    QJsonParseError parseError;
    const auto document = QJsonDocument::fromJson(data, &parseError);
    auto object = document.object();
    legacy = !object.contains("version");
    QStringList problems;
    if (parseError.error != QJsonParseError::NoError || !document.isObject() ||
        (legacy && !allowLegacy) ||
        (!legacy && (object.value("version").toDouble(-1) != 2.0 ||
                     !object.value("bindings").isObject()))) {
        if (errors)
            *errors = QStringList{
                    QObject::tr("Invalid shortcut configuration format.")};
        return false;
    }
    if (legacy) {
        if (object.contains("__vtk__")) {
            const auto vtkValue = object.take("__vtk__");
            if (!vtkValue.isObject()) {
                if (errors)
                    *errors = QStringList{
                            QObject::tr("Invalid legacy VTK bindings.")};
                return false;
            }
            const auto vtk = vtkValue.toObject();
            for (auto it = vtk.constBegin(); it != vtk.constEnd(); ++it) {
                if (object.contains(it.key()))
                    problems.append(
                            QObject::tr("Duplicate command: %1").arg(it.key()));
                object.insert(it.key(), it.value());
            }
        }
    } else
        object = object.value("bindings").toObject();
    for (auto it = object.constBegin(); it != object.constEnd(); ++it) {
        QJsonArray values;
        if (!legacy && it.value().isArray())
            values = it.value().toArray();
        else if (legacy && it.value().isString()) {
            if (!it.value().toString().isEmpty()) values.append(it.value());
        } else {
            problems.append(QObject::tr("Invalid binding: %1").arg(it.key()));
            continue;
        }
        if (it.key().isEmpty())
            problems.append(QObject::tr("Empty shortcut command ID."));
        QList<QKeySequence> keys;
        for (const auto& value : values) {
            const QKeySequence sequence(value.toString(),
                                        QKeySequence::PortableText);
            if (!value.isString() || !validSequence(sequence))
                problems.append(
                        QObject::tr("Invalid binding: %1").arg(it.key()));
            else
                keys.append(sequence);
        }
        bindings[it.key()] = keys;
    }
    if (errors) *errors = problems;
    return problems.isEmpty();
}

bool commandsOverlap(const ecvKeySequences::Command& a,
                     const ecvKeySequences::Command& b) {
    for (const auto& first : a.actions)
        for (const auto& second : b.actions)
            if (first && second && contextsOverlap(first, second)) return true;
    return false;
}

}  // anonymous namespace

//-----------------------------------------------------------------------------
ecvKeySequences& ecvKeySequences::instance() {
    static ecvKeySequences inst(nullptr);
    return inst;
}

//-----------------------------------------------------------------------------
ecvKeySequences::ecvKeySequences(QObject* parent)
    : QObject(parent), m_silence(false) {
    CVLog::PrintVerbose("[ecvKeySequences] Modal shortcut manager initialized");
    if (auto* app = qobject_cast<QApplication*>(QCoreApplication::instance())) {
        connect(app, &QApplication::focusChanged, this,
                [this](QWidget*, QWidget*) {
                    if (!g_keys.Data.isEmpty()) refreshCommandShortcuts();
                });
    }
}

bool ecvKeySequences::registerCommand(const QString& id,
                                      QAction* action,
                                      const QString& category) {
    if (id.isEmpty() || !action) return false;
    const QString previousId = action->property("shortcutCommandId").toString();
    if (!previousId.isEmpty()) return previousId == id;
    auto it = m_commands.find(id);
    const bool newCommand = it == m_commands.end();
    if (newCommand) {
        Command command;
        command.id = id;
        command.category = category;
        command.label = action->text().isEmpty() ? id : action->text();
        command.defaults = action->shortcuts();
        it = m_commands.insert(id, command);
        m_bindings[id] = command.defaults;
    }
    for (const auto& existing : it->actions) {
        if (existing && contextsOverlap(existing, action)) {
            action->setShortcuts({});
            CVLog::Warning(tr("Overlapping instances of shortcut command: %1")
                                   .arg(id));
            return false;
        }
    }
    it->actions.append(action);
    {
        // Factory defaults are executable bindings too: validate even before
        // the first settings restore, not only for late-created commands.
        Bindings candidate = m_bindings;
        // Existing instances always consume the live committed binding. Only
        // a newly loaded command consults a previously unloaded plugin's data.
        if (m_bindingsRestored && newCommand && m_savedBindings.contains(id))
            candidate[id] = m_savedBindings.value(id);
        QStringList errors;
        if (!applyBindings(candidate, &errors, false)) {
            if (newCommand) {
                // Keep the new command editable, but do not introduce an
                // ambiguous dispatcher. Its declared factory key is retained.
                candidate[id] = {};
                if (applyBindings(candidate, nullptr, false)) {
                    CVLog::Warning(
                            tr("Command %1 registered without a shortcut:\n%2")
                                    .arg(id, errors.join("\n")));
                } else {
                    m_commands[id].actions.removeAll(action);
                    m_commands.remove(id);
                    m_bindings.remove(id);
                    action->setShortcuts({});
                    CVLog::Warning(errors.join("\n"));
                    return false;
                }
            } else {
                m_commands[id].actions.removeAll(action);
                action->setShortcuts({});
                CVLog::Warning(errors.join("\n"));
                return false;
            }
        }
    }
    action->setProperty("shortcutCommandId", id);
    connect(action, &QAction::changed, this, [this, id, action]() {
        if (m_rebindingCommands || m_silence) return;
        // Enabled/text changes remain cheap. Registered command keys can only
        // change through the validator, never a second QAction setter path.
        if (action->shortcuts() != effectiveBindings(id, action)) {
            CVLog::Warning(tr("Use the shortcut registry to change command %1")
                                   .arg(id));
            refreshCommandShortcuts();
        }
    });
    connect(action, &QObject::destroyed, this, [this, id]() {
        auto& actions = m_commands[id].actions;
        for (int i = actions.size() - 1; i >= 0; --i)
            if (!actions[i]) actions.removeAt(i);
        emit commandsChanged();
    });
    refreshCommandShortcuts();
    emit commandsChanged();
    return true;
}

QList<QKeySequence> ecvKeySequences::effectiveBindings(
        const QString& id, const QAction* action) const {
    auto effective = m_bindings.value(id);
    for (auto it = g_keys.Data.constBegin(); it != g_keys.Data.constEnd();
         ++it) {
        for (const auto* modal : it->Siblings) {
            if (!modal || !modal->isEnabled() || !modal->m_shortcut) continue;
            auto* widget = qobject_cast<QWidget*>(modal->m_shortcut->parent());
            if (!widget || !widget->isVisible() || !widget->isEnabled())
                continue;
            auto* focus = QApplication::focusWidget();
            const auto context = modal->m_shortcut->context();
            const bool ownsFocus =
                    context == Qt::ApplicationShortcut ||
                    (context == Qt::WindowShortcut && focus &&
                     shortcutWindow(widget) == shortcutWindow(focus)) ||
                    (context == Qt::WidgetShortcut && widget->hasFocus()) ||
                    (context == Qt::WidgetWithChildrenShortcut && focus &&
                     (widget == focus || widget->isAncestorOf(focus)));
            if (!ownsFocus ||
                !contextsOverlap(shortcutWidgets(action),
                                 action->shortcutContext(), {widget}, context))
                continue;
            for (int i = effective.size() - 1; i >= 0; --i)
                if (sequencesOverlap(effective[i], it.key()))
                    effective.removeAt(i);
        }
    }
    return effective;
}

void ecvKeySequences::refreshCommandShortcuts() {
    if (m_silence || m_rebindingCommands) return;
    QScopedValueRollback<bool> rebinding(m_rebindingCommands, true);
    // Snapshot instance lists: changed() observers can destroy another view.
    // Only effective Qt keys are suspended; configured keys and enabled state
    // survive transient-tool ownership and are restored when it ends.
    const auto commands = m_commands.values();
    for (const auto& command : commands) {
        for (const auto& action : command.actions) {
            if (!action) continue;
            const auto effective = effectiveBindings(command.id, action);
            if (action->shortcuts() != effective)
                action->setShortcuts(effective);
        }
    }
}

QList<ecvKeySequences::Command> ecvKeySequences::commands() const {
    return m_commands.values();
}

ecvKeySequences::Bindings ecvKeySequences::defaultBindings() const {
    Bindings result;
    for (const auto& command : m_commands)
        result[command.id] = command.defaults;
    return result;
}

QList<ecvKeySequences::Conflict> ecvKeySequences::conflicts(
        const Bindings& candidate) const {
    struct Entry {
        QString id;
        QKeySequence sequence;
    };
    // Only sequences with the same first stroke can conflict. This runs on
    // configuration changes, not on the key-event/rendering hot path.
    QMap<QString, QList<Entry>> buckets;
    QList<Conflict> result;
    for (auto it = candidate.constBegin(); it != candidate.constEnd(); ++it) {
        if (!m_commands.contains(it.key())) continue;
        for (const auto& sequence : it.value()) {
            if (sequence.isEmpty()) continue;
            auto& bucket =
                    buckets[QKeySequence(sequence[0])
                                    .toString(QKeySequence::PortableText)];
            for (const auto& previous : bucket) {
                if (!sequencesOverlap(sequence, previous.sequence)) continue;
                if (commandsOverlap(m_commands.value(previous.id),
                                    m_commands.value(it.key())))
                    result.append({previous.id, it.key(), sequence});
            }
            bucket.append({it.key(), sequence});
        }
    }
    return result;
}

bool ecvKeySequences::applyBindings(const Bindings& candidate,
                                    QStringList* errors,
                                    bool persist) {
    if (m_rebindingCommands) {
        if (errors)
            *errors = QStringList{tr("Shortcut bindings are being updated.")};
        return false;
    }
    QStringList problems;
    for (const auto& id : m_commands.keys()) {
        if (!candidate.contains(id))
            problems.append(tr("Missing shortcut command: %1").arg(id));
    }
    for (auto it = candidate.constBegin(); it != candidate.constEnd(); ++it) {
        if (!m_commands.contains(it.key()))
            problems.append(tr("Unknown shortcut command: %1").arg(it.key()));
        for (const auto& sequence : it.value()) {
            if (!validSequence(sequence))
                problems.append(
                        tr("Invalid binding in command: %1").arg(it.key()));
        }
    }
    for (const auto& conflict : conflicts(candidate))
        problems.append(tr("Shortcut conflict: %1 / %2 (%3)")
                                .arg(conflict.first, conflict.second,
                                     conflict.sequence.toString()));
    if (!problems.isEmpty()) {
        if (errors) *errors = problems;
        return false;
    }

    if (persist) {
        // One versioned value is the commit record; unknown plugin commands
        // retained from an earlier session survive while the plugin is
        // unloaded.
        Bindings saved = m_savedBindings;
        for (auto it = candidate.constBegin(); it != candidate.constEnd(); ++it)
            saved[it.key()] = it.value();
        QSettings settings;
        const QString key = QStringLiteral("Shortcuts/commands-v2");
        const bool hadPrevious = settings.contains(key);
        const QVariant previous = settings.value(key);
        settings.setValue(key,
                          encodeBindingFile(saved, QJsonDocument::Compact));
        settings.sync();
        if (settings.status() != QSettings::NoError) {
            // QSettings caches pending writes across instances. Revert that
            // cache as well, so a failed save cannot leak into a later restore.
            if (hadPrevious)
                settings.setValue(key, previous);
            else
                settings.remove(key);
            settings.sync();
            if (errors)
                *errors =
                        QStringList{tr("Cannot save shortcut settings; live "
                                       "bindings were not changed.")};
            return false;
        }
        m_savedBindings = saved;
    }
    m_bindings = candidate;
    refreshCommandShortcuts();
    emit commandsChanged();
    if (errors) errors->clear();
    return true;
}

QByteArray ecvKeySequences::exportBindings() const {
    Bindings profile = m_savedBindings;
    for (auto it = m_bindings.constBegin(); it != m_bindings.constEnd(); ++it)
        profile[it.key()] = it.value();
    return encodeBindingFile(profile, QJsonDocument::Indented);
}

bool ecvKeySequences::importBindings(const QByteArray& data,
                                     QStringList* errors,
                                     QStringList* unavailable) {
    if (unavailable) unavailable->clear();
    Bindings imported;
    bool legacy = false;
    if (!decodeBindingFile(data, imported, true, legacy, errors)) return false;
    Bindings candidate = m_bindings;
    Bindings saved = m_savedBindings;
    QStringList problems, missing;
    QSet<QString> resolved;
    for (auto it = imported.constBegin(); it != imported.constEnd(); ++it) {
        QString id = it.key();
        if (legacy && !m_commands.contains(id)) {
            QSet<QString> matches;
            for (const auto& command : m_commands) {
                QStringList aliases{command.label};
                for (const auto& action : command.actions)
                    if (action)
                        aliases << action->objectName() << action->text();
                for (auto alias : aliases) {
                    if (alias == id || alias.remove('&') == id)
                        matches.insert(command.id);
                }
            }
            if (matches.size() != 1) {
                problems.append(
                        tr("Unknown or ambiguous legacy command: %1").arg(id));
                continue;
            }
            id = *matches.constBegin();
        }
        if (resolved.contains(id)) {
            problems.append(tr("Duplicate shortcut command: %1").arg(id));
            continue;
        }
        resolved.insert(id);
        if (m_commands.contains(id))
            candidate[id] = it.value();
        else {
            // A versioned plugin/window profile may arrive before its owner.
            // Retain it explicitly, but never execute or claim it is validated
            // against a context until registerCommand() admits that owner.
            saved[id] = it.value();
            missing.append(id);
        }
    }
    if (!problems.isEmpty()) {
        if (errors) *errors = problems;
        return false;
    }
    const Bindings previousSaved = m_savedBindings;
    m_savedBindings = saved;
    if (!applyBindings(candidate, errors)) {
        m_savedBindings = previousSaved;
        return false;
    }
    m_bindingsRestored = true;
    if (unavailable) *unavailable = missing;
    return true;
}

bool ecvKeySequences::restoreBindings(QStringList* errors) {
    QSettings settings;
    Bindings candidate = m_bindings;
    Bindings saved;
    if (settings.contains("Shortcuts/commands-v2")) {
        bool legacy = false;
        if (!decodeBindingFile(
                    settings.value("Shortcuts/commands-v2").toByteArray(),
                    saved, false, legacy, errors))
            return false;
    } else {
        // Read legacy keys once, but never use translated labels as new IDs.
        for (const auto& command : m_commands) {
            const QString group = command.id.startsWith("vtk_")
                                          ? "VtkShortcuts/"
                                          : "Shortcuts/";
            QStringList keys{group + command.id};
            for (const auto& action : command.actions) {
                if (!action) continue;
                if (!action->objectName().isEmpty())
                    keys.append(group + action->objectName());
                QString label = action->text();
                label.remove('&');
                keys.append(group + label);
                keys.append(group + action->text());
                break;
            }
            for (const auto& key : keys) {
                if (!settings.contains(key)) continue;
                const auto sequence = settings.value(key).value<QKeySequence>();
                saved[command.id] = sequence.isEmpty()
                                            ? QList<QKeySequence>()
                                            : QList<QKeySequence>{sequence};
                break;
            }
        }
    }
    for (auto it = saved.constBegin(); it != saved.constEnd(); ++it)
        if (candidate.contains(it.key())) candidate[it.key()] = it.value();
    if (!applyBindings(candidate, errors, false)) return false;
    m_savedBindings = saved;
    m_bindingsRestored = true;
    return true;
}

//-----------------------------------------------------------------------------
ecvModalShortcut* ecvKeySequences::active(
        const QKeySequence& keySequence) const {
    ecvModalShortcut* active = nullptr;
    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(keySequence);
    if (iter == g_keys.Data.end()) {
        return active;
    }

    for (auto& sibling : iter->Siblings) {
        if (sibling && sibling->isEnabled()) {
            if (!active) active = sibling;
        }
    }
    return active;
}

//-----------------------------------------------------------------------------
ecvModalShortcut* ecvKeySequences::addModalShortcut(
        const QKeySequence& keySequence,
        QAction* action,
        QWidget* parent,
        Qt::ShortcutContext context) {
    if (keySequence.isEmpty()) {
        CVLog::Warning(
                "[ecvKeySequences] Attempted to register empty key sequence");
        return nullptr;
    }

    // Create the modal shortcut
    auto shortcut = new ecvModalShortcut(keySequence, action, parent);

    // Add to siblings list
    auto& keyData = g_keys.Data[keySequence];
    keyData.Siblings.insert(shortcut);

    // Initially disable (will be enabled below)
    shortcut->setEnabled(false);
    shortcut->setContextWidget(parent, context);

    // Connect signals
    QObject::connect(shortcut, &ecvModalShortcut::enabled, this,
                     &ecvKeySequences::disableSiblings);
    QObject::connect(shortcut, &ecvModalShortcut::enabled, this,
                     &ecvKeySequences::refreshCommandShortcuts);
    QObject::connect(shortcut, &ecvModalShortcut::disabled, this,
                     &ecvKeySequences::refreshCommandShortcuts);
    QObject::connect(shortcut, &ecvModalShortcut::unregister, this,
                     &ecvKeySequences::removeModalShortcut);

    // Enable (this will trigger disableSiblings() to ensure only one is active)
    shortcut->setEnabled(true);

    CVLog::PrintVerbose(
            QString("[ecvKeySequences] Registered modal shortcut: %1 (ID: %2)")
                    .arg(keySequence.toString())
                    .arg(shortcut->objectName().isEmpty()
                                 ? "unnamed"
                                 : shortcut->objectName()));

    return shortcut;
}

//-----------------------------------------------------------------------------
void ecvKeySequences::reorder(ecvModalShortcut* target) {
    if (!target) {
        return;
    }

    // Find the key sequence for this shortcut
    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(target->keySequence());
    if (iter == g_keys.Data.end()) {
        return;
    }

    // Check if any sibling of target is currently active
    bool hasActiveSibling = false;
    for (auto& sibling : iter->Siblings) {
        if (sibling != target && sibling && sibling->isEnabled()) {
            hasActiveSibling = true;
            break;
        }
    }

    // If no sibling is active, reorder has no effect (as per ParaView
    // documentation)
    if (!hasActiveSibling) {
        return;
    }

    // Mark target as the most recently used shortcut
    // This allows enableNextSibling() to activate it when the current
    // shortcut is disabled
    iter->mruShortcut = target;

    CVLog::PrintVerbose(
            QString("[ecvKeySequences] Reordered shortcut: %1 (MRU set)")
                    .arg(target->keySequence().toString()));
}

//-----------------------------------------------------------------------------
void ecvKeySequences::dumpShortcuts(const QKeySequence& keySequence) const {
    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(keySequence);
    if (iter == g_keys.Data.end()) {
        CVLog::Print(
                QString("[ecvKeySequences] No shortcuts registered for: %1")
                        .arg(keySequence.toString()));
        return;
    }

    CVLog::PrintVerbose(QString("[ecvKeySequences] Shortcuts for %1:")
                                .arg(keySequence.toString()));
    for (auto& sibling : iter->Siblings) {
        if (sibling) {
            CVLog::PrintVerbose(QString("  - %1: %2")
                                        .arg(sibling->objectName().isEmpty()
                                                     ? "unnamed"
                                                     : sibling->objectName())
                                        .arg(sibling->isEnabled()
                                                     ? "enabled"
                                                     : "disabled"));
        }
    }
}

//-----------------------------------------------------------------------------
QMap<QString, QStringList> ecvKeySequences::allRegisteredSequences() const {
    QMap<QString, QStringList> result;
    for (auto it = g_keys.Data.constBegin(); it != g_keys.Data.constEnd();
         ++it) {
        QStringList names;
        for (auto* sibling : it.value().Siblings) {
            if (!sibling) continue;
            QString name;
            if (sibling->parent()) {
                auto* widget = qobject_cast<QWidget*>(sibling->parent());
                if (widget && !widget->objectName().isEmpty())
                    name = widget->objectName();
            }
            if (name.isEmpty()) name = sibling->objectName();
            if (name.isEmpty()) name = QStringLiteral("modal");
            names.append(name);
        }
        result[it.key().toString()] = names;
    }
    return result;
}

//-----------------------------------------------------------------------------
void ecvKeySequences::disableSiblings() {
    if (m_silence) {
        return;
    }

    auto* shortcut = qobject_cast<ecvModalShortcut*>(this->sender());
    if (!shortcut) {
        return;
    }

    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(shortcut->keySequence());
    if (iter == g_keys.Data.end()) {
        return;
    }

    iter->mruShortcut = shortcut;
    if (!shortcut->m_shortcut) return;
    auto* owner = qobject_cast<QWidget*>(shortcut->m_shortcut->parent());
    // Equal keys are only ambiguous in overlapping contexts. Prefix-related
    // chords need the same arbitration as exact matches, while disjoint views
    // may keep their own listeners enabled.
    m_silence = true;
    int disabledCount = 0;
    for (auto entry = g_keys.Data.begin(); entry != g_keys.Data.end();
         ++entry) {
        if (!sequencesOverlap(entry.key(), shortcut->keySequence())) continue;
        for (auto* sibling : entry->Siblings) {
            if (!sibling || sibling == shortcut || !sibling->isEnabled() ||
                !sibling->m_shortcut)
                continue;
            auto* other = qobject_cast<QWidget*>(sibling->m_shortcut->parent());
            if (owner && other &&
                contextsOverlap({owner}, shortcut->m_shortcut->context(),
                                {other}, sibling->m_shortcut->context())) {
                sibling->setEnabled(false);
                ++disabledCount;
            }
        }
    }
    m_silence = false;

    if (disabledCount > 0) {
        CVLog::PrintVerbose(
                QString("[ecvKeySequences] Disabled %1 sibling(s) for %2")
                        .arg(disabledCount)
                        .arg(shortcut->keySequence().toString()));
    }
}

//-----------------------------------------------------------------------------
void ecvKeySequences::enableNextSibling() {
    if (m_silence) {
        return;
    }

    auto* shortcut = qobject_cast<ecvModalShortcut*>(this->sender());
    if (!shortcut) {
        return;
    }

    // Find the key sequence for this shortcut
    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(shortcut->keySequence());
    if (iter == g_keys.Data.end()) {
        return;
    }

    // Activate the most-recently-used sibling if available
    // This implements the reordering behavior: when a shortcut is disabled,
    // the MRU shortcut (set via reorder()) becomes the next one to activate
    if (iter->mruShortcut && iter->mruShortcut != shortcut) {
        // Check if the MRU shortcut's parent widget is still valid and enabled
        QWidget* parentWidget =
                qobject_cast<QWidget*>(iter->mruShortcut->parent());
        if (parentWidget && parentWidget->isEnabled() &&
            parentWidget->isVisible()) {
            iter->mruShortcut->setEnabled(true);
            CVLog::PrintVerbose(
                    QString("[ecvKeySequences] Enabled next sibling (MRU): %1")
                            .arg(shortcut->keySequence().toString()));
        } else {
            // MRU shortcut's widget is disabled/hidden, try to find another
            // enabled sibling
            for (auto& sibling : iter->Siblings) {
                if (sibling != shortcut && sibling) {
                    QWidget* siblingParent =
                            qobject_cast<QWidget*>(sibling->parent());
                    if (siblingParent && siblingParent->isEnabled() &&
                        siblingParent->isVisible()) {
                        sibling->setEnabled(true);
                        CVLog::PrintVerbose(QString("[ecvKeySequences] Enabled "
                                                    "next sibling: "
                                                    "%1")
                                                    .arg(shortcut->keySequence()
                                                                 .toString()));
                        break;
                    }
                }
            }
        }
    } else {
        // No MRU shortcut set, try to find any enabled sibling
        for (auto& sibling : iter->Siblings) {
            if (sibling != shortcut && sibling) {
                QWidget* siblingParent =
                        qobject_cast<QWidget*>(sibling->parent());
                if (siblingParent && siblingParent->isEnabled() &&
                    siblingParent->isVisible()) {
                    sibling->setEnabled(true);
                    CVLog::PrintVerbose(
                            QString("[ecvKeySequences] Enabled next sibling: "
                                    "%1")
                                    .arg(shortcut->keySequence().toString()));
                    break;
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
void ecvKeySequences::removeModalShortcut() {
    auto* shortcut = qobject_cast<ecvModalShortcut*>(this->sender());
    if (!shortcut) {
        return;
    }

    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(shortcut->keySequence());
    if (iter == g_keys.Data.end()) {
        return;
    }

    // Remove the shortcut from the key sequence list
    iter->Siblings.remove(shortcut);
    QObject::disconnect(shortcut);

    CVLog::PrintVerbose(
            QString("[ecvKeySequences] Unregistered modal shortcut: %1")
                    .arg(shortcut->keySequence().toString()));

    // If no more siblings, remove the key sequence entry
    if (iter->Siblings.empty()) {
        g_keys.Data.erase(iter);
    }
    refreshCommandShortcuts();
}
