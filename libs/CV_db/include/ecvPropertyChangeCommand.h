#pragma once

#include "CV_db.h"

#include <QUndoCommand>
#include <QString>
#include <functional>

template <typename T>
class ecvPropertyChangeCommand : public QUndoCommand {
public:
    using Setter = std::function<void(const T&)>;
    using RefreshFunc = std::function<void()>;

    ecvPropertyChangeCommand(unsigned int entityId,
                             const QString& propertyKey,
                             const T& before,
                             const T& after,
                             Setter setter,
                             RefreshFunc refresh,
                             const QString& label,
                             QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent)
        , m_entityId(entityId)
        , m_propertyKey(propertyKey)
        , m_before(before)
        , m_after(after)
        , m_setter(std::move(setter))
        , m_refresh(std::move(refresh))
        , m_mergeId(computeMergeId(entityId, propertyKey)) {}

    void undo() override {
        if (m_setter) m_setter(m_before);
        if (m_refresh) m_refresh();
    }

    void redo() override {
        if (m_firstRedo) {
            m_firstRedo = false;
            return;
        }
        if (m_setter) m_setter(m_after);
        if (m_refresh) m_refresh();
    }

    int id() const override { return m_mergeId; }

    bool mergeWith(const QUndoCommand* other) override {
        if (other->id() != id()) return false;
        auto* o = static_cast<const ecvPropertyChangeCommand<T>*>(other);
        if (o->m_entityId != m_entityId || o->m_propertyKey != m_propertyKey)
            return false;
        m_after = o->m_after;
        return true;
    }

private:
    static int computeMergeId(unsigned int entityId, const QString& key) {
        return static_cast<int>(qHash(key) ^ entityId) | 0x10000;
    }

    unsigned int m_entityId;
    QString m_propertyKey;
    T m_before;
    T m_after;
    Setter m_setter;
    RefreshFunc m_refresh;
    int m_mergeId;
    bool m_firstRedo = true;
};
