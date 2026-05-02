// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvRepresentationManager.h"

#include "ecvViewRepresentation.h"

ecvRepresentationManager::ecvRepresentationManager() : QObject(nullptr) {}

ecvRepresentationManager& ecvRepresentationManager::instance() {
    static ecvRepresentationManager s_instance;
    return s_instance;
}

ecvViewRepresentation* ecvRepresentationManager::getRepresentation(
        ccHObject* entity, ecvGenericGLDisplay* view) const {
    QReadLocker readLock(&m_lock);
    auto it = m_representations.find(Key(entity, view));
    if (it != m_representations.end()) {
        return it.value().get();
    }
    return nullptr;
}

ecvViewRepresentation* ecvRepresentationManager::ensureRepresentation(
        ccHObject* entity, ecvGenericGLDisplay* view) {
    {
        QReadLocker readLock(&m_lock);
        auto it = m_representations.find(Key(entity, view));
        if (it != m_representations.end()) {
            return it.value().get();
        }
    }

    QWriteLocker writeLock(&m_lock);
    auto it = m_representations.find(Key(entity, view));
    if (it != m_representations.end()) {
        return it.value().get();
    }

    auto rep = std::make_shared<ecvViewRepresentation>(entity, view);
    ecvViewRepresentation* raw = rep.get();
    m_representations.insert(Key(entity, view), std::move(rep));
    emit representationAdded(raw);
    return raw;
}

QList<ecvViewRepresentation*>
ecvRepresentationManager::getRepresentationsForEntity(ccHObject* entity) const {
    QReadLocker readLock(&m_lock);
    QList<ecvViewRepresentation*> result;
    for (auto it = m_representations.begin(); it != m_representations.end();
         ++it) {
        if (it.key().first == entity) {
            result.append(it.value().get());
        }
    }
    return result;
}

QList<ecvViewRepresentation*>
ecvRepresentationManager::getRepresentationsForView(
        ecvGenericGLDisplay* view) const {
    QReadLocker readLock(&m_lock);
    QList<ecvViewRepresentation*> result;
    for (auto it = m_representations.begin(); it != m_representations.end();
         ++it) {
        if (it.key().second == view) {
            result.append(it.value().get());
        }
    }
    return result;
}

void ecvRepresentationManager::removeRepresentationsForEntity(
        ccHObject* entity) {
    QWriteLocker writeLock(&m_lock);
    auto it = m_representations.begin();
    while (it != m_representations.end()) {
        if (it.key().first == entity) {
            ecvGenericGLDisplay* view = it.key().second;
            if (m_actorCleanup) {
                m_actorCleanup(entity, view);
            }
            it = m_representations.erase(it);
            emit representationRemoved(entity, view);
        } else {
            ++it;
        }
    }
}

void ecvRepresentationManager::removeRepresentationsForView(
        ecvGenericGLDisplay* view) {
    QWriteLocker writeLock(&m_lock);
    auto it = m_representations.begin();
    while (it != m_representations.end()) {
        if (it.key().second == view) {
            ccHObject* entity = it.key().first;
            if (m_actorCleanup) {
                m_actorCleanup(entity, view);
            }
            it = m_representations.erase(it);
            emit representationRemoved(entity, view);
        } else {
            ++it;
        }
    }
}

void ecvRepresentationManager::removeRepresentation(ccHObject* entity,
                                                    ecvGenericGLDisplay* view) {
    QWriteLocker writeLock(&m_lock);
    auto it = m_representations.find(Key(entity, view));
    if (it != m_representations.end()) {
        if (m_actorCleanup) {
            m_actorCleanup(entity, view);
        }
        m_representations.erase(it);
        emit representationRemoved(entity, view);
    }
}

int ecvRepresentationManager::count() const {
    QReadLocker readLock(&m_lock);
    return m_representations.size();
}

void ecvRepresentationManager::setActorCleanupCallback(CleanupCallback cb) {
    m_actorCleanup = std::move(cb);
}

void ecvRepresentationManager::notifyChanged(ecvViewRepresentation* rep) {
    if (rep) {
        emit representationChanged(rep);
    }
}
