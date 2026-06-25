// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvTransformCommand.h"

ecvTransformCommand::ecvTransformCommand(ccHObject* entity,
                                         const ccGLMatrix& historyBefore,
                                         const ccGLMatrix& historyAfter,
                                         const ccGLMatrix& appliedTransform,
                                         RestoreFunc restoreFunc,
                                         RefreshFunc refreshFunc,
                                         const QString& label,
                                         QUndoCommand* parent)
    : QUndoCommand(label, parent),
      m_entity(entity),
      m_historyBefore(historyBefore),
      m_historyAfter(historyAfter),
      m_appliedTransform(appliedTransform),
      m_restore(std::move(restoreFunc)),
      m_refresh(std::move(refreshFunc)) {}

ecvTransformCommand::~ecvTransformCommand() = default;

void ecvTransformCommand::undo() {
    if (m_entity && m_restore) {
        ccGLMatrix inverseTransform = m_appliedTransform.inverse();
        m_restore(m_entity, inverseTransform);
    }
    if (m_refresh) m_refresh();
}

void ecvTransformCommand::redo() {
    if (m_firstRedo) {
        m_firstRedo = false;
        return;
    }
    if (m_entity && m_restore) {
        m_restore(m_entity, m_appliedTransform);
    }
    if (m_refresh) m_refresh();
}

int ecvTransformCommand::id() const { return 2002; }
