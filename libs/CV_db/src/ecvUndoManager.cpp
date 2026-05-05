// libs/CV_db/src/ecvUndoManager.cpp
#include "ecvUndoManager.h"
#include "ecvEntityAddRemoveCommand.h"
#include "ecvScalarFieldEditCommand.h"

#include <CVLog.h>
#include <algorithm>

ecvUndoManager::ecvUndoManager(QObject* parent)
    : QObject(parent) {
    m_currentUndoLimit = 200;
    m_stack.setUndoLimit(m_currentUndoLimit);
}

ecvUndoManager::~ecvUndoManager() = default;

static qint64 estimateCmdMemory(const QUndoCommand* cmd) {
    if (!cmd) return 1024;

    if (auto* addRemove = dynamic_cast<const ecvEntityAddRemoveCommand*>(cmd))
        return addRemove->estimatedMemoryBytes();
    if (auto* sfEdit = dynamic_cast<const ecvScalarFieldEditCommand*>(cmd))
        return sfEdit->estimatedMemoryBytes();

    return 1024;
}

void ecvUndoManager::push(QUndoCommand* cmd) {
    if (!cmd) return;

    m_estimatedMemory += estimateCmdMemory(cmd);
    m_stack.push(cmd);
    enforceMemoryBudget();
}

void ecvUndoManager::clear() {
    m_stack.clear();
    m_estimatedMemory = 0;
    m_currentUndoLimit = 200;
    m_stack.setUndoLimit(m_currentUndoLimit);
}

void ecvUndoManager::setMemoryBudgetMB(int mb) {
    m_memoryBudgetBytes = static_cast<qint64>(mb) * 1024 * 1024;
    enforceMemoryBudget();
}

void ecvUndoManager::enforceMemoryBudget() {
    if (m_estimatedMemory <= m_memoryBudgetBytes) return;

    int newLimit = qMax(10, m_stack.count() - 1);
    if (newLimit < m_currentUndoLimit) {
        m_currentUndoLimit = newLimit;
        m_stack.setUndoLimit(m_currentUndoLimit);
        CVLog::Print(
            QString("[UndoManager] Memory budget exceeded (%1 MB / %2 MB), "
                    "reducing undo limit to %3")
                .arg(m_estimatedMemory / (1024 * 1024))
                .arg(m_memoryBudgetBytes / (1024 * 1024))
                .arg(m_currentUndoLimit));

        m_estimatedMemory = 0;
        for (int i = 0; i < m_stack.count(); ++i) {
            m_estimatedMemory += estimateCmdMemory(m_stack.command(i));
        }
    }
}

QAction* ecvUndoManager::createUndoAction(QObject* parent,
                                          const QString& prefix) const {
    return m_stack.createUndoAction(parent, prefix);
}

QAction* ecvUndoManager::createRedoAction(QObject* parent,
                                          const QString& prefix) const {
    return m_stack.createRedoAction(parent, prefix);
}

#include "ecvLayoutUndoCommand.h"
#include "ecvViewLayoutProxy.h"

void ecvLayoutUndoCommand::undo() {
    if (m_layout) {
        m_layout->loadState(m_before);
    }
}

void ecvLayoutUndoCommand::redo() {
    if (m_layout) {
        m_layout->loadState(m_after);
    }
}
