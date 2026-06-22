// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// libs/CV_db/include/ecvUndoManager.h
#pragma once

#include <QObject>
#include <QUndoStack>

#include "CV_db.h"

class QAction;

class CV_DB_LIB_API ecvUndoManager : public QObject {
public:
    explicit ecvUndoManager(QObject* parent = nullptr);
    ~ecvUndoManager() override;

    QUndoStack* undoStack() { return &m_stack; }
    const QUndoStack* undoStack() const { return &m_stack; }

    void push(QUndoCommand* cmd);

    bool canUndo() const { return m_stack.canUndo(); }
    bool canRedo() const { return m_stack.canRedo(); }
    QString undoText() const { return m_stack.undoText(); }
    QString redoText() const { return m_stack.redoText(); }

    void undo() { m_stack.undo(); }
    void redo() { m_stack.redo(); }
    void clear();

    QAction* createUndoAction(QObject* parent,
                              const QString& prefix = QString()) const;
    QAction* createRedoAction(QObject* parent,
                              const QString& prefix = QString()) const;

    void setUndoLimit(int limit) { m_stack.setUndoLimit(limit); }

    void setMemoryBudgetMB(int mb);
    qint64 memoryBudgetBytes() const { return m_memoryBudgetBytes; }
    qint64 estimatedMemoryUsage() const { return m_estimatedMemory; }

private:
    void enforceMemoryBudget();

    QUndoStack m_stack;
    qint64 m_memoryBudgetBytes = 500LL * 1024 * 1024;
    qint64 m_estimatedMemory = 0;
    int m_currentUndoLimit = 200;
};
