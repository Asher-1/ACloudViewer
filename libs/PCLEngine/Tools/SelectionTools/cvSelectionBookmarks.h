// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"

// LOCAL
#include "cvSelectionData.h"

// Qt
#include <QDateTime>
#include <QList>
#include <QMap>
#include <QObject>
#include <QString>
#include <QStringList>

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
