// libs/CV_db/include/ecvCameraUndoCommand.h
#pragma once

#include "CV_db.h"

#include <QUndoCommand>
#include <functional>

struct CV_DB_LIB_API ecvCameraState {
    double pos[3] = {0, 0, 0};
    double focal[3] = {0, 0, 0};
    double view[3] = {0, 0, 1};
    double clip[2] = {0.01, 1000};
    double fovy = 0.8575;
    bool parallelProjection = false;
    double parallelScale = 1.0;
};

class CV_DB_LIB_API ecvCameraUndoCommand : public QUndoCommand {
public:
    using ApplyFunc = std::function<void(const ecvCameraState&)>;

    ecvCameraUndoCommand(const ecvCameraState& before,
                         const ecvCameraState& after,
                         ApplyFunc applyFunc,
                         const QString& label = QStringLiteral("Camera"),
                         QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent)
        , m_before(before)
        , m_after(after)
        , m_apply(std::move(applyFunc)) {}

    void undo() override {
        if (m_apply) m_apply(m_before);
    }

    void redo() override {
        if (m_apply) m_apply(m_after);
    }

    int id() const override { return 1002; }

    bool mergeWith(const QUndoCommand* other) override {
        if (other->id() != id()) return false;
        auto* o = static_cast<const ecvCameraUndoCommand*>(other);
        m_after = o->m_after;
        return true;
    }

private:
    ecvCameraState m_before;
    ecvCameraState m_after;
    ApplyFunc m_apply;
};
