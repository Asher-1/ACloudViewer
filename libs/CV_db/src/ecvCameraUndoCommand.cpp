// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvCameraUndoCommand.h"

ecvCameraState::ecvCameraState() = default;
ecvCameraState::~ecvCameraState() = default;

ecvCameraUndoCommand::ecvCameraUndoCommand(const ecvCameraState& before,
                                           const ecvCameraState& after,
                                           ApplyFunc applyFunc,
                                           const QString& label,
                                           QUndoCommand* parent)
    : QUndoCommand(label, parent),
      m_before(before),
      m_after(after),
      m_apply(std::move(applyFunc)) {}

ecvCameraUndoCommand::~ecvCameraUndoCommand() = default;

void ecvCameraUndoCommand::undo() {
    if (m_apply) m_apply(m_before);
}

void ecvCameraUndoCommand::redo() {
    if (m_apply) m_apply(m_after);
}

int ecvCameraUndoCommand::id() const { return 1002; }

bool ecvCameraUndoCommand::mergeWith(const QUndoCommand* other) {
    if (other->id() != id()) return false;
    auto* o = static_cast<const ecvCameraUndoCommand*>(other);
    m_after = o->m_after;
    return true;
}
