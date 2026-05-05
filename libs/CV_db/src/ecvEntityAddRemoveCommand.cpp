#include "ecvEntityAddRemoveCommand.h"

#include <CVLog.h>
#include <QDataStream>

ecvEntityAddRemoveCommand::ecvEntityAddRemoveCommand(
        ccHObject* entity,
        ccHObject* parent,
        Mode mode,
        AddFunc addFunc,
        RemoveFunc removeFunc,
        RefreshFunc refreshFunc,
        const QString& label,
        QUndoCommand* parentCmd)
    : QUndoCommand(label, parentCmd)
    , m_entity(entity)
    , m_parent(parent)
    , m_mode(mode)
    , m_ownsEntity(mode == Mode::Remove)
    , m_addFunc(std::move(addFunc))
    , m_removeFunc(std::move(removeFunc))
    , m_refreshFunc(std::move(refreshFunc)) {
}

ecvEntityAddRemoveCommand::~ecvEntityAddRemoveCommand() {
    if (m_ownsEntity && m_entity) {
        delete m_entity;
        m_entity = nullptr;
    }
}

void ecvEntityAddRemoveCommand::undo() {
    if (m_mode == Mode::Add) {
        if (m_entity && m_removeFunc) {
            m_removeFunc(m_entity);
            m_ownsEntity = true;
        }
    } else {
        ensureEntityAvailable();
        if (m_entity && m_parent && m_addFunc) {
            m_addFunc(m_entity, m_parent);
            m_ownsEntity = false;
        }
    }
    if (m_refreshFunc) m_refreshFunc();
}

void ecvEntityAddRemoveCommand::redo() {
    if (m_firstRedo) {
        m_firstRedo = false;
        return;
    }
    if (m_mode == Mode::Add) {
        ensureEntityAvailable();
        if (m_entity && m_parent && m_addFunc) {
            m_addFunc(m_entity, m_parent);
            m_ownsEntity = false;
        }
    } else {
        if (m_entity && m_removeFunc) {
            m_removeFunc(m_entity);
            m_ownsEntity = true;
        }
    }
    if (m_refreshFunc) m_refreshFunc();
}

qint64 ecvEntityAddRemoveCommand::estimatedMemoryBytes() const {
    return m_estimatedBytes;
}

void ecvEntityAddRemoveCommand::serializeToTemp() {
    if (m_serialized || !m_entity) return;

    m_tempFile = std::make_unique<QTemporaryFile>();
    if (!m_tempFile->open()) {
        CVLog::Warning("[EntityUndoCommand] Failed to create temp file for serialization");
        return;
    }

    if (m_entity->toFile(*m_tempFile, 0)) {
        m_serialized = true;
        m_estimatedBytes = m_tempFile->size();
        CVLog::Print(QString("[EntityUndoCommand] Serialized entity '%1' to temp (%2 bytes)")
                             .arg(m_entity->getName())
                             .arg(m_tempFile->size()));
    } else {
        CVLog::Warning("[EntityUndoCommand] Serialization failed for entity");
        m_tempFile.reset();
    }
}

ccHObject* ecvEntityAddRemoveCommand::deserializeFromTemp() {
    if (!m_serialized || !m_tempFile) return nullptr;

    m_tempFile->seek(0);

    ccHObject::LoadedIDMap oldToNewIDMap;
    ccHObject* restored = new ccHObject("restored");

    if (restored->fromFile(*m_tempFile, 0, 0, oldToNewIDMap)) {
        CVLog::Print(QString("[EntityUndoCommand] Deserialized entity from temp"));
        return restored;
    }

    delete restored;
    CVLog::Warning("[EntityUndoCommand] Deserialization failed");
    return nullptr;
}

void ecvEntityAddRemoveCommand::ensureEntityAvailable() {
    if (m_entity) return;
    if (m_serialized) {
        m_entity = deserializeFromTemp();
        if (m_entity) m_ownsEntity = true;
    }
}
