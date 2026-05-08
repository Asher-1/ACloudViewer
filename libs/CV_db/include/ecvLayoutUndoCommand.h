// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// libs/CV_db/include/ecvLayoutUndoCommand.h
#pragma once

#include <QJsonObject>
#include <QUndoCommand>

#include "CV_db.h"

class ecvViewLayoutProxy;

class CV_DB_LIB_API ecvLayoutUndoCommand : public QUndoCommand {
public:
    ecvLayoutUndoCommand(ecvViewLayoutProxy* layout,
                         const QJsonObject& beforeState,
                         const QJsonObject& afterState,
                         const QString& label,
                         QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent),
          m_layout(layout),
          m_before(beforeState),
          m_after(afterState) {}

    void undo() override;
    void redo() override;

    int id() const override { return 1001; }

private:
    ecvViewLayoutProxy* m_layout;
    QJsonObject m_before;
    QJsonObject m_after;
};
