#pragma once

#include "CV_db.h"
#include "ecvHObject.h"

#include <QTemporaryFile>
#include <QUndoCommand>
#include <functional>
#include <memory>

class CV_DB_LIB_API ecvEntityAddRemoveCommand : public QUndoCommand {
public:
    enum class Mode { Add, Remove };

    using AddFunc = std::function<void(ccHObject* entity, ccHObject* parent)>;
    using RemoveFunc = std::function<void(ccHObject* entity)>;
    using RefreshFunc = std::function<void()>;

    ecvEntityAddRemoveCommand(ccHObject* entity,
                              ccHObject* parent,
                              Mode mode,
                              AddFunc addFunc,
                              RemoveFunc removeFunc,
                              RefreshFunc refreshFunc,
                              const QString& label,
                              QUndoCommand* parentCmd = nullptr);

    ~ecvEntityAddRemoveCommand() override;

    void undo() override;
    void redo() override;

    int id() const override { return 2003; }

    qint64 estimatedMemoryBytes() const;

    static constexpr qint64 kSerializationThreshold = 10 * 1024 * 1024;

private:
    void serializeToTemp();
    ccHObject* deserializeFromTemp();
    void ensureEntityAvailable();

    ccHObject* m_entity = nullptr;
    ccHObject* m_parent = nullptr;
    Mode m_mode;
    bool m_ownsEntity = false;
    bool m_firstRedo = true;
    int m_childIndex = -1;

    std::unique_ptr<QTemporaryFile> m_tempFile;
    bool m_serialized = false;
    qint64 m_estimatedBytes = 1024;

    AddFunc m_addFunc;
    RemoveFunc m_removeFunc;
    RefreshFunc m_refreshFunc;
};
