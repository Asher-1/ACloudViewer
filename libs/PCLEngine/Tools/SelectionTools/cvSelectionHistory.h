// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// clang-format off
// Qt - must be included first for MOC to work correctly on Windows
#include <QtCore/QObject>
#include <QtCore/QDateTime>
#include <QtCore/QStack>
#include <QtCore/QString>
// clang-format on

#include "qPCL.h"

// LOCAL
#include "cvSelectionData.h"

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
