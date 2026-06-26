// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// libs/CV_db/include/ecvCameraUndoCommand.h
#pragma once

#include <QString>
#include <QUndoCommand>
#include <functional>

#include "CV_db.h"

struct CV_DB_LIB_API ecvCameraState {
    double pos[3] = {0, 0, 0};
    double focal[3] = {0, 0, 0};
    double view[3] = {0, 0, 1};
    double clip[2] = {0.01, 1000};
    double fovy = 0.8575;
    bool parallelProjection = false;
    double parallelScale = 1.0;

    ecvCameraState();
    ~ecvCameraState();
};

class CV_DB_LIB_API ecvCameraUndoCommand : public QUndoCommand {
public:
    using ApplyFunc = std::function<void(const ecvCameraState&)>;

    ecvCameraUndoCommand(const ecvCameraState& before,
                         const ecvCameraState& after,
                         ApplyFunc applyFunc,
                         const QString& label = QStringLiteral("Camera"),
                         QUndoCommand* parent = nullptr);
    ~ecvCameraUndoCommand() override;

    void undo() override;
    void redo() override;
    int id() const override;
    bool mergeWith(const QUndoCommand* other) override;

private:
    ecvCameraState m_before;
    ecvCameraState m_after;
    ApplyFunc m_apply;
};
