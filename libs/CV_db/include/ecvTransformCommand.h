// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

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
                        QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent),
          m_entity(entity),
          m_historyBefore(historyBefore),
          m_historyAfter(historyAfter),
          m_appliedTransform(appliedTransform),
          m_restore(std::move(restoreFunc)),
          m_refresh(std::move(refreshFunc)) {}

    void undo() override {
        if (m_entity && m_restore) {
            ccGLMatrix inverseTransform = m_appliedTransform.inverse();
            m_restore(m_entity, inverseTransform);
        }
        if (m_refresh) m_refresh();
    }

    void redo() override {
        if (m_firstRedo) {
            m_firstRedo = false;
            return;
        }
        if (m_entity && m_restore) {
            m_restore(m_entity, m_appliedTransform);
        }
        if (m_refresh) m_refresh();
    }

    int id() const override { return 2002; }

private:
    ccHObject* m_entity;
    ccGLMatrix m_historyBefore;
    ccGLMatrix m_historyAfter;
    ccGLMatrix m_appliedTransform;
    RestoreFunc m_restore;
    RefreshFunc m_refresh;
    bool m_firstRedo = true;
};
