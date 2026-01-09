// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionStorage.h"

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <QDateTime>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

//=============================================================================
// cvSelectionHistory Implementation
//=============================================================================

//-----------------------------------------------------------------------------
cvSelectionHistory::cvSelectionHistory(QObject* parent)
    : QObject(parent), m_maxHistorySize(50) {
    CVLog::PrintDebug("[cvSelectionHistory] Initialized with max size: 50");
}

//-----------------------------------------------------------------------------
cvSelectionHistory::~cvSelectionHistory() {}

//-----------------------------------------------------------------------------
void cvSelectionHistory::pushSelection(const cvSelectionData& selection,
                                       const QString& description) {
    // Don't push if it's the same as current
    if (!m_current.selection.isEmpty() &&
        m_current.selection.ids() == selection.ids() &&
        m_current.selection.fieldAssociation() ==
                selection.fieldAssociation()) {
        return;
    }

    // If we have a current selection, push it to undo stack
    if (!m_current.selection.isEmpty()) {
        m_undoStack.push(m_current);

        // Limit stack size
        while (m_undoStack.size() > m_maxHistorySize) {
            // Remove oldest entry (bottom of stack)
            QStack<HistoryEntry> temp;
            while (m_undoStack.size() > 1) {
                temp.push(m_undoStack.pop());
            }
            m_undoStack.clear();
            while (!temp.isEmpty()) {
                m_undoStack.push(temp.pop());
            }
        }
    }

    // Set new current
    QString desc = description.isEmpty()
                           ? QString("%1 %2")
                                     .arg(selection.count())
                                     .arg(selection.fieldTypeString())
                           : description;
    m_current = HistoryEntry(selection, desc);

    // Clear redo stack (new action invalidates redo)
    m_redoStack.clear();

    emit historyChanged();

    CVLog::PrintDebug(QString("[cvSelectionHistory] Pushed: %1").arg(desc));
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionHistory::undo() {
    if (!canUndo()) {
        CVLog::Warning("[cvSelectionHistory] No undo available");
        return cvSelectionData();
    }

    // Push current to redo stack
    if (!m_current.selection.isEmpty()) {
        m_redoStack.push(m_current);
    }

    // Pop from undo stack
    m_current = m_undoStack.pop();

    emit historyChanged();
    emit selectionRestored(m_current.selection);

    CVLog::Print(QString("[cvSelectionHistory] Undo: %1")
                         .arg(m_current.description));

    return m_current.selection;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionHistory::redo() {
    if (!canRedo()) {
        CVLog::Warning("[cvSelectionHistory] No redo available");
        return cvSelectionData();
    }

    // Push current to undo stack
    if (!m_current.selection.isEmpty()) {
        m_undoStack.push(m_current);
    }

    // Pop from redo stack
    m_current = m_redoStack.pop();

    emit historyChanged();
    emit selectionRestored(m_current.selection);

    CVLog::Print(QString("[cvSelectionHistory] Redo: %1")
                         .arg(m_current.description));

    return m_current.selection;
}

//-----------------------------------------------------------------------------
bool cvSelectionHistory::canUndo() const { return !m_undoStack.isEmpty(); }

//-----------------------------------------------------------------------------
bool cvSelectionHistory::canRedo() const { return !m_redoStack.isEmpty(); }

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionHistory::currentSelection() const {
    return m_current.selection;
}

//-----------------------------------------------------------------------------
void cvSelectionHistory::clear() {
    m_undoStack.clear();
    m_redoStack.clear();
    m_current = HistoryEntry();

    emit historyChanged();

    CVLog::Print("[cvSelectionHistory] History cleared");
}

//-----------------------------------------------------------------------------
void cvSelectionHistory::setMaxHistorySize(int maxSize) {
    if (maxSize < 1) {
        CVLog::Warning(
                "[cvSelectionHistory] Invalid max size, using minimum of 1");
        maxSize = 1;
    }

    m_maxHistorySize = maxSize;

    // Trim undo stack if needed
    while (m_undoStack.size() > m_maxHistorySize) {
        QStack<HistoryEntry> temp;
        while (m_undoStack.size() > 1) {
            temp.push(m_undoStack.pop());
        }
        m_undoStack.clear();
        while (!temp.isEmpty()) {
            m_undoStack.push(temp.pop());
        }
    }

    // Trim redo stack if needed
    while (m_redoStack.size() > m_maxHistorySize) {
        m_redoStack.pop();
    }

    CVLog::Print(QString("[cvSelectionHistory] Max history size set to: %1")
                         .arg(maxSize));
}

//-----------------------------------------------------------------------------
QString cvSelectionHistory::undoDescription(int index) const {
    if (index < 0 || index >= m_undoStack.size()) {
        return QString();
    }

    // Access from top of stack (most recent)
    QStack<HistoryEntry> temp = m_undoStack;
    for (int i = 0; i < index; ++i) {
        temp.pop();
    }

    return temp.top().description;
}

//-----------------------------------------------------------------------------
QString cvSelectionHistory::redoDescription(int index) const {
    if (index < 0 || index >= m_redoStack.size()) {
        return QString();
    }

    // Access from top of stack (next redo)
    QStack<HistoryEntry> temp = m_redoStack;
    for (int i = 0; i < index; ++i) {
        temp.pop();
    }

    return temp.top().description;
}

//=============================================================================
// cvSelectionBookmarks Implementation
//=============================================================================

//-----------------------------------------------------------------------------
cvSelectionBookmarks::cvSelectionBookmarks(QObject* parent) : QObject(parent) {
    CVLog::PrintDebug("[cvSelectionBookmarks] Initialized");
}

//-----------------------------------------------------------------------------
cvSelectionBookmarks::~cvSelectionBookmarks() {}

//-----------------------------------------------------------------------------
bool cvSelectionBookmarks::addBookmark(const QString& name,
                                       const cvSelectionData& selection,
                                       const QString& description) {
    if (name.isEmpty()) {
        CVLog::Warning(
                "[cvSelectionBookmarks] Cannot add bookmark with empty name");
        return false;
    }

    if (selection.isEmpty()) {
        CVLog::Warning(
                "[cvSelectionBookmarks] Cannot add bookmark with empty "
                "selection");
        return false;
    }

    if (m_bookmarks.contains(name)) {
        CVLog::Warning(
                QString("[cvSelectionBookmarks] Bookmark '%1' already exists")
                        .arg(name));
        return false;
    }

    m_bookmarks[name] = Bookmark(selection, name, description);

    emit bookmarkAdded(name);
    emit bookmarksChanged();

    CVLog::Print(QString("[cvSelectionBookmarks] Added bookmark: %1 (%2 %3)")
                         .arg(name)
                         .arg(selection.count())
                         .arg(selection.fieldTypeString()));

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionBookmarks::removeBookmark(const QString& name) {
    if (!m_bookmarks.contains(name)) {
        CVLog::Warning(QString("[cvSelectionBookmarks] Bookmark '%1' not found")
                               .arg(name));
        return false;
    }

    m_bookmarks.remove(name);

    emit bookmarkRemoved(name);
    emit bookmarksChanged();

    CVLog::Print(
            QString("[cvSelectionBookmarks] Removed bookmark: %1").arg(name));

    return true;
}

//-----------------------------------------------------------------------------
cvSelectionBookmarks::Bookmark cvSelectionBookmarks::getBookmark(
        const QString& name) const {
    if (!m_bookmarks.contains(name)) {
        CVLog::Warning(QString("[cvSelectionBookmarks] Bookmark '%1' not found")
                               .arg(name));
        return Bookmark();
    }

    return m_bookmarks[name];
}

//-----------------------------------------------------------------------------
bool cvSelectionBookmarks::hasBookmark(const QString& name) const {
    return m_bookmarks.contains(name);
}

//-----------------------------------------------------------------------------
QStringList cvSelectionBookmarks::bookmarkNames() const {
    return m_bookmarks.keys();
}

//-----------------------------------------------------------------------------
QList<cvSelectionBookmarks::Bookmark> cvSelectionBookmarks::allBookmarks()
        const {
    return m_bookmarks.values();
}

//-----------------------------------------------------------------------------
void cvSelectionBookmarks::clear() {
    m_bookmarks.clear();

    emit bookmarksChanged();

    CVLog::Print("[cvSelectionBookmarks] All bookmarks cleared");
}

//-----------------------------------------------------------------------------
bool cvSelectionBookmarks::renameBookmark(const QString& oldName,
                                          const QString& newName) {
    if (oldName.isEmpty() || newName.isEmpty()) {
        CVLog::Warning("[cvSelectionBookmarks] Invalid bookmark names");
        return false;
    }

    if (!m_bookmarks.contains(oldName)) {
        CVLog::Warning(QString("[cvSelectionBookmarks] Bookmark '%1' not found")
                               .arg(oldName));
        return false;
    }

    if (m_bookmarks.contains(newName)) {
        CVLog::Warning(
                QString("[cvSelectionBookmarks] Bookmark '%1' already exists")
                        .arg(newName));
        return false;
    }

    Bookmark bookmark = m_bookmarks[oldName];
    bookmark.name = newName;
    m_bookmarks.remove(oldName);
    m_bookmarks[newName] = bookmark;

    emit bookmarkRemoved(oldName);
    emit bookmarkAdded(newName);
    emit bookmarksChanged();

    CVLog::Print(QString("[cvSelectionBookmarks] Renamed bookmark: %1 -> %2")
                         .arg(oldName)
                         .arg(newName));

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionBookmarks::updateDescription(const QString& name,
                                             const QString& description) {
    if (!m_bookmarks.contains(name)) {
        CVLog::Warning(QString("[cvSelectionBookmarks] Bookmark '%1' not found")
                               .arg(name));
        return false;
    }

    m_bookmarks[name].description = description;

    emit bookmarksChanged();

    CVLog::Print(QString("[cvSelectionBookmarks] Updated description for: %1")
                         .arg(name));

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionBookmarks::exportToFile(const QString& filename) const {
    if (filename.isEmpty()) {
        CVLog::Error("[cvSelectionBookmarks] Invalid filename for export");
        return false;
    }

    QJsonArray bookmarksArray;

    for (const Bookmark& bookmark : m_bookmarks.values()) {
        QJsonObject bookmarkObj;
        bookmarkObj["name"] = bookmark.name;
        bookmarkObj["description"] = bookmark.description;
        bookmarkObj["timestamp"] = bookmark.timestamp;

        // Save selection data
        QJsonObject selectionObj;
        selectionObj["fieldAssociation"] =
                static_cast<int>(bookmark.selection.fieldAssociation());
        selectionObj["count"] = bookmark.selection.count();

        // Save IDs
        QJsonArray idsArray;
        QVector<qint64> ids = bookmark.selection.ids();
        for (qint64 id : ids) {
            idsArray.append(id);
        }
        selectionObj["ids"] = idsArray;

        bookmarkObj["selection"] = selectionObj;

        bookmarksArray.append(bookmarkObj);
    }

    QJsonObject root;
    root["version"] = 1;
    root["count"] = m_bookmarks.size();
    root["bookmarks"] = bookmarksArray;

    QJsonDocument doc(root);

    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        CVLog::Error(QString("[cvSelectionBookmarks] Failed to open file for "
                             "writing: %1")
                             .arg(filename));
        return false;
    }

    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();

    CVLog::Print(QString("[cvSelectionBookmarks] Exported %1 bookmarks to: %2")
                         .arg(m_bookmarks.size())
                         .arg(filename));

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionBookmarks::importFromFile(const QString& filename, bool merge) {
    if (filename.isEmpty()) {
        CVLog::Error("[cvSelectionBookmarks] Invalid filename for import");
        return false;
    }

    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        CVLog::Error(QString("[cvSelectionBookmarks] Failed to open file for "
                             "reading: %1")
                             .arg(filename));
        return false;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull() || !doc.isObject()) {
        CVLog::Error("[cvSelectionBookmarks] Invalid JSON format");
        return false;
    }

    QJsonObject root = doc.object();

    // Check version
    int version = root["version"].toInt();
    if (version != 1) {
        CVLog::Warning(QString("[cvSelectionBookmarks] Unsupported version: %1")
                               .arg(version));
    }

    // Clear existing bookmarks if not merging
    if (!merge) {
        m_bookmarks.clear();
    }

    QJsonArray bookmarksArray = root["bookmarks"].toArray();
    int imported = 0;
    int skipped = 0;

    for (const QJsonValue& value : bookmarksArray) {
        QJsonObject bookmarkObj = value.toObject();

        QString name = bookmarkObj["name"].toString();
        QString description = bookmarkObj["description"].toString();
        qint64 timestamp = bookmarkObj["timestamp"].toVariant().toLongLong();

        // Skip if already exists and merging
        if (merge && m_bookmarks.contains(name)) {
            CVLog::Warning(
                    QString("[cvSelectionBookmarks] Skipping duplicate: %1")
                            .arg(name));
            ++skipped;
            continue;
        }

        // Load selection data
        QJsonObject selectionObj = bookmarkObj["selection"].toObject();
        int fieldAssociation = selectionObj["fieldAssociation"].toInt();

        QJsonArray idsArray = selectionObj["ids"].toArray();
        QVector<qint64> ids;
        for (const QJsonValue& idValue : idsArray) {
            ids.append(idValue.toVariant().toLongLong());
        }

        cvSelectionData selection(
                ids, static_cast<cvSelectionData::FieldAssociation>(
                             fieldAssociation));

        // Create bookmark
        Bookmark bookmark(selection, name, description);
        bookmark.timestamp = timestamp;
        m_bookmarks[name] = bookmark;

        ++imported;
    }

    emit bookmarksChanged();

    CVLog::Print(QString("[cvSelectionBookmarks] Imported %1 bookmarks (%2 "
                         "skipped) from: %3")
                         .arg(imported)
                         .arg(skipped)
                         .arg(filename));

    return true;
}
