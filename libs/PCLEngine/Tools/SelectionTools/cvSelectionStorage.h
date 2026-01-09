// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// clang-format off
// Qt - must be included first for MOC to work correctly on Windows
#include <QtCore/QDateTime>
#include <QtCore/QList>
#include <QtCore/QMap>
#include <QtCore/QObject>
#include <QtCore/QStack>
#include <QtCore/QString>
#include <QtCore/QStringList>
// clang-format on

#include "qPCL.h"

// LOCAL
#include "cvSelectionData.h"

//=============================================================================
// Selection History (merged from cvSelectionHistory.h)
//=============================================================================

/**
 * @brief Selection history manager with undo/redo support
 *
 * Provides ParaView-style selection history:
 * - Undo/redo selection changes
 * - History limit to prevent memory issues
 * - Clear history
 * - Query history state
 *
 * Based on ParaView's pqSelectionManager history functionality.
 */
class QPCL_ENGINE_LIB_API cvSelectionHistory : public QObject {
    Q_OBJECT

public:
    explicit cvSelectionHistory(QObject* parent = nullptr);
    ~cvSelectionHistory() override;

    /**
     * @brief Push a new selection state to history
     * @param selection The selection data
     * @param description Optional description of the selection
     */
    void pushSelection(const cvSelectionData& selection,
                       const QString& description = QString());

    /**
     * @brief Undo to previous selection
     * @return The previous selection, or empty if no history
     */
    cvSelectionData undo();

    /**
     * @brief Redo to next selection
     * @return The next selection, or empty if no redo available
     */
    cvSelectionData redo();

    /**
     * @brief Check if undo is available
     */
    bool canUndo() const;

    /**
     * @brief Check if redo is available
     */
    bool canRedo() const;

    /**
     * @brief Get current selection
     */
    cvSelectionData currentSelection() const;

    /**
     * @brief Clear all history
     */
    void clear();

    /**
     * @brief Set maximum history size
     * @param maxSize Maximum number of history entries (default: 50)
     */
    void setMaxHistorySize(int maxSize);

    /**
     * @brief Get maximum history size
     */
    int maxHistorySize() const { return m_maxHistorySize; }

    /**
     * @brief Get undo stack size
     */
    int undoCount() const { return m_undoStack.size(); }

    /**
     * @brief Get redo stack size
     */
    int redoCount() const { return m_redoStack.size(); }

    /**
     * @brief Get description of selection at undo index
     * @param index Index in undo stack (0 = most recent)
     */
    QString undoDescription(int index) const;

    /**
     * @brief Get description of selection at redo index
     * @param index Index in redo stack (0 = next redo)
     */
    QString redoDescription(int index) const;

signals:
    /**
     * @brief Emitted when history state changes
     */
    void historyChanged();

    /**
     * @brief Emitted when selection is restored from history
     */
    void selectionRestored(const cvSelectionData& selection);

private:
    struct HistoryEntry {
        cvSelectionData selection;
        QString description;
        qint64 timestamp;  // Unix timestamp in milliseconds

        HistoryEntry() : timestamp(0) {}
        HistoryEntry(const cvSelectionData& sel, const QString& desc)
            : selection(sel),
              description(desc),
              timestamp(QDateTime::currentMSecsSinceEpoch()) {}
    };

    QStack<HistoryEntry> m_undoStack;
    QStack<HistoryEntry> m_redoStack;
    int m_maxHistorySize;
    HistoryEntry m_current;
};

//=============================================================================
// Selection Bookmarks (merged from cvSelectionBookmarks.h)
//=============================================================================

/**
 * @brief Selection bookmarks manager
 *
 * Allows users to save and restore named selections:
 * - Save current selection with a name
 * - Load bookmarked selection
 * - Delete bookmarks
 * - List all bookmarks
 * - Export/import bookmarks to/from file
 *
 * Based on ParaView's selection bookmarks functionality.
 */
class QPCL_ENGINE_LIB_API cvSelectionBookmarks : public QObject {
    Q_OBJECT

public:
    struct Bookmark {
        cvSelectionData selection;
        QString name;
        QString description;
        qint64 timestamp;  // Unix timestamp in milliseconds

        Bookmark() : timestamp(0) {}
        Bookmark(const cvSelectionData& sel,
                 const QString& n,
                 const QString& desc = QString())
            : selection(sel),
              name(n),
              description(desc),
              timestamp(QDateTime::currentMSecsSinceEpoch()) {}
    };

    explicit cvSelectionBookmarks(QObject* parent = nullptr);
    ~cvSelectionBookmarks() override;

    /**
     * @brief Add a bookmark
     * @param name Bookmark name (must be unique)
     * @param selection Selection data
     * @param description Optional description
     * @return True if added successfully
     */
    bool addBookmark(const QString& name,
                     const cvSelectionData& selection,
                     const QString& description = QString());

    /**
     * @brief Remove a bookmark
     * @param name Bookmark name
     * @return True if removed successfully
     */
    bool removeBookmark(const QString& name);

    /**
     * @brief Get a bookmark
     * @param name Bookmark name
     * @return Bookmark data, or empty if not found
     */
    Bookmark getBookmark(const QString& name) const;

    /**
     * @brief Check if bookmark exists
     */
    bool hasBookmark(const QString& name) const;

    /**
     * @brief Get all bookmark names
     */
    QStringList bookmarkNames() const;

    /**
     * @brief Get all bookmarks
     */
    QList<Bookmark> allBookmarks() const;

    /**
     * @brief Get bookmark count
     */
    int count() const { return m_bookmarks.size(); }

    /**
     * @brief Clear all bookmarks
     */
    void clear();

    /**
     * @brief Rename a bookmark
     * @param oldName Current name
     * @param newName New name
     * @return True if renamed successfully
     */
    bool renameBookmark(const QString& oldName, const QString& newName);

    /**
     * @brief Update bookmark description
     * @param name Bookmark name
     * @param description New description
     * @return True if updated successfully
     */
    bool updateDescription(const QString& name, const QString& description);

    /**
     * @brief Export bookmarks to JSON file
     * @param filename Output filename
     * @return True on success
     */
    bool exportToFile(const QString& filename) const;

    /**
     * @brief Import bookmarks from JSON file
     * @param filename Input filename
     * @param merge If true, merge with existing bookmarks; if false, replace
     * @return True on success
     */
    bool importFromFile(const QString& filename, bool merge = true);

signals:
    /**
     * @brief Emitted when bookmarks change
     */
    void bookmarksChanged();

    /**
     * @brief Emitted when a bookmark is added
     */
    void bookmarkAdded(const QString& name);

    /**
     * @brief Emitted when a bookmark is removed
     */
    void bookmarkRemoved(const QString& name);

    /**
     * @brief Emitted when a bookmark is loaded
     */
    void bookmarkLoaded(const QString& name, const cvSelectionData& selection);

private:
    QMap<QString, Bookmark> m_bookmarks;
};
