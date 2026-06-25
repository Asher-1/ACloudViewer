// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>
#include <QUndoCommand>
#include <functional>

#include "CV_db.h"
#include "ecvGLMatrix.h"

class ccHObject;

class CV_DB_LIB_API ecvTransformCommand : public QUndoCommand {
public:
    using RestoreFunc =
            std::function<void(ccHObject*, const ccGLMatrix& transform)>;
    using RefreshFunc = std::function<void()>;

    ecvTransformCommand(ccHObject* entity,
                        const ccGLMatrix& historyBefore,
                        const ccGLMatrix& historyAfter,
                        const ccGLMatrix& appliedTransform,
                        RestoreFunc restoreFunc,
                        RefreshFunc refreshFunc,
                        const QString& label = QStringLiteral("Transform"),
                        QUndoCommand* parent = nullptr);
    ~ecvTransformCommand() override;

    void undo() override;
    void redo() override;
    int id() const override;

private:
    ccHObject* m_entity;
    ccGLMatrix m_historyBefore;
    ccGLMatrix m_historyAfter;
    ccGLMatrix m_appliedTransform;
    RestoreFunc m_restore;
    RefreshFunc m_refresh;
    bool m_firstRedo = true;
};
