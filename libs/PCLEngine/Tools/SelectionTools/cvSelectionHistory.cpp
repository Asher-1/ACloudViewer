// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionHistory.h"

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <QDateTime>

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
