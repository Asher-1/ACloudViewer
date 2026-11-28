// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ecvKeySequences_h
#define ecvKeySequences_h

// Local
#include "CV_db.h"

// Qt
#include <QByteArray>
#include <QKeySequence>
#include <QMap>
#include <QObject>
#include <QPointer>
#include <QStringList>
#include <QWidget>

class QAction;
class ecvModalShortcut;

/**
 * @brief Manage key sequences used for shortcuts.
 *
 * ParaView-style modal shortcut manager that prevents ambiguous activation
 * by ensuring only one listener is active for each key sequence at a time.
 *
 * Key features:
 * - Automatic mutual exclusion: when one shortcut is enabled, all siblings are
 * disabled
 * - Visual feedback: widgets can show which one currently owns a shortcut
 * - Context awareness: shortcuts can be limited to specific widgets
 *
 * Usage:
 * @code
 * auto* shortcut = ecvKeySequences::instance().addModalShortcut(
 *     QKeySequence(Qt::Key_S), myAction, myWidget);
 * @endcode
 */
class CV_DB_LIB_API ecvKeySequences : public QObject {
    Q_OBJECT

public:
    /**
     * Get the singleton instance.
     */
    static ecvKeySequences& instance();

    // Configurable commands use Qt's QAction shortcut dispatcher. Instances of
    // one command may live in disjoint views; toolbar mirrors are not
    // instances.
    using Bindings = QMap<QString, QList<QKeySequence>>;
    struct Command {
        QString id;
        QString category;
        QString pluginId;
        QString pluginName;
        QString label;
        QList<QKeySequence> defaults;
        QList<QPointer<QAction>> actions;
    };
    struct Conflict {
        QString first;
        QString second;
        QKeySequence sequence;
    };

    static bool isValidSequence(const QKeySequence& sequence);
    static bool sequencesOverlap(const QKeySequence& first,
                                 const QKeySequence& second);
    bool registerCommand(const QString& id,
                         QAction* action,
                         const QString& category);
    // Plugin identities are stable IIDs; categories and names are display text.
    // Reuses an existing command ID when an exported action is already owned.
    bool registerPluginCommand(const QString& pluginId,
                               const QString& command,
                               QAction* action,
                               const QString& category,
                               const QString& pluginName = QString());
    void requestShortcutSettings(const QString& pluginId = QString()) {
        emit shortcutSettingsRequested(pluginId);
    }
    // Import a plugin's old group once through the same validated transaction;
    // canonical central values always win and the legacy group stays untouched.
    bool migrateLegacyBindings(const QString& group,
                               const QMap<QString, QString>& commandIds,
                               QStringList* errors = nullptr);
    QList<Command> commands() const;
    Bindings bindings() const { return m_bindings; }
    QList<QKeySequence> effectiveBindings(const QString& id) const;
    Bindings defaultBindings() const;
    QList<Conflict> conflicts(const Bindings& candidate) const;
    // All edits, imports, defaults and startup restoration use this
    // transaction. Validation precedes both live mutation and persistence. No
    // event-loop processing occurs while instances are rebound.
    bool applyBindings(const Bindings& candidate,
                       QStringList* errors = nullptr,
                       bool persist = true);
    bool restoreBindings(QStringList* errors = nullptr);
    QByteArray exportBindings() const;
    bool importBindings(const QByteArray& data,
                        QStringList* errors = nullptr,
                        QStringList* unavailable = nullptr);

Q_SIGNALS:
    void commandsChanged();
    void shortcutSettingsRequested(const QString& pluginId);

public:
    /**
     * Return the active shortcut for a given key sequence (if any).
     */
    ecvModalShortcut* active(const QKeySequence& keySequence) const;

    /**
     * Register a modal shortcut with the manager.
     *
     * @param keySequence The key combination to listen for
     * @param action The action to trigger (optional)
     * @param parent The context widget (optional)
     * @return The created modal shortcut (owned by the manager)
     */
    ecvModalShortcut* addModalShortcut(
            const QKeySequence& keySequence,
            QAction* action,
            QWidget* parent,
            Qt::ShortcutContext context = Qt::WindowShortcut);

    /**
     * Give a requested owner priority over overlapping temporary owners.
     */
    void reorder(ecvModalShortcut* target);

    /**
     * Dump a list of shortcuts registered for a given key sequence (debug).
     */
    void dumpShortcuts(const QKeySequence& keySequence) const;

    /**
     * Return all registered key sequences and their owner names.
     * Used by ecvShortcutDialog for comprehensive conflict detection.
     *
     * @return Map of key-sequence-string -> list of owner names
     */
    QMap<QString, QStringList> allRegisteredSequences() const;

protected Q_SLOTS:
    /**
     * Called when shortcuts are deleted to disable and unregister them.
     */
    virtual void removeModalShortcut();

protected:
    ecvKeySequences(QObject* parent);
    ~ecvKeySequences() override = default;
    bool eventFilter(QObject* watched, QEvent* event) override;

    /// Set true in slot implementations to avoid signal/slot recursion.
    bool m_silence;

private:
    QList<QKeySequence> effectiveBindings(
            const QString& id,
            const QAction* action,
            const QList<QPair<QString, QPointer<QAction>>>& listeners) const;
    void refreshCommandShortcuts();
    void reconcileModalShortcuts(ecvModalShortcut* preferred = nullptr);
    QList<QPointer<ecvModalShortcut>> m_modalPriority;
    QStringList m_runtimeConflicts;
    bool m_rebindingCommands = false;
    QMap<QString, Command> m_commands;
    Bindings m_bindings;
    Bindings m_savedBindings;
    bool m_bindingsRestored = false;
};

#endif  // ecvKeySequences_h
