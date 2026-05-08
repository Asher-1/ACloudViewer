// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvRepresentationManager.h"

#include "ecvHObject.h"
#include "ecvViewManager.h"
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

    // P6: Inherit compatible properties from an existing representation
    // of the same entity.  Prefer the active view's representation (most
    // likely what the user sees / just edited).  ParaView equivalent:
    // vtkInheritRepresentationProperties.
    //
    // Collect donor properties under the write lock, then apply them AFTER
    // releasing the lock.  This prevents deadlock: setVisible()/setProperties()
    // emit representationChanged → ecvGLView::redraw() → getRepresentation()
    // would try to acquire a read lock on the same QReadWriteLock.
    ecvViewRepresentation::Properties donorProps;
    bool donorHasVisibility = false;
    bool donorVisible = false;

    ecvViewRepresentation* raw = nullptr;
    {
        QWriteLocker writeLock(&m_lock);
        auto it = m_representations.find(Key(entity, view));
        if (it != m_representations.end()) {
            return it.value().get();
        }

        auto rep = std::make_shared<ecvViewRepresentation>(entity, view);
        raw = rep.get();

        ecvViewRepresentation* donor = nullptr;
        auto* activeView = ecvViewManager::instance().getActiveView();
        if (activeView && activeView != view) {
            auto dIt = m_representations.find(Key(entity, activeView));
            if (dIt != m_representations.end()) {
                donor = dIt.value().get();
            }
        }
        if (!donor) {
            for (auto dIt = m_representations.begin();
                 dIt != m_representations.end(); ++dIt) {
                if (dIt.key().first == entity && dIt.key().second != view) {
                    donor = dIt.value().get();
                    break;
                }
            }
        }
        if (donor) {
            donorProps = donor->properties();
            donorHasVisibility = donor->hasVisibilityOverride();
            donorVisible = donor->isVisible();
        }

        m_representations.insert(Key(entity, view), std::move(rep));
    }

    if (donorProps.opacity.has_value() || donorProps.showNormals.has_value() ||
        donorProps.pointSize.has_value() || donorProps.lineWidth.has_value() ||
        donorProps.renderMode.has_value()) {
        raw->setProperties(donorProps);
    }
    if (donorHasVisibility) {
        raw->setVisible(donorVisible);
    }

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
    // Collect removed (entity, view) pairs under the write lock, then run
    // cleanup callbacks and emit signals AFTER releasing the lock.  This
    // prevents deadlock: m_actorCleanup or a representationRemoved slot
    // could indirectly call getRepresentation() which acquires a read lock
    // on the same non-recursive QReadWriteLock.
    QList<ecvGenericGLDisplay*> removedViews;
    {
        QWriteLocker writeLock(&m_lock);
        auto it = m_representations.begin();
        while (it != m_representations.end()) {
            if (it.key().first == entity) {
                removedViews.append(it.key().second);
                it = m_representations.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto* view : removedViews) {
        if (m_actorCleanup) {
            m_actorCleanup(entity, view);
        }
        emit representationRemoved(entity, view);
    }
}

void ecvRepresentationManager::removeRepresentationsForView(
        ecvGenericGLDisplay* view) {
    QList<ccHObject*> removedEntities;
    {
        QWriteLocker writeLock(&m_lock);
        auto it = m_representations.begin();
        while (it != m_representations.end()) {
            if (it.key().second == view) {
                removedEntities.append(it.key().first);
                it = m_representations.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto* entity : removedEntities) {
        if (m_actorCleanup) {
            m_actorCleanup(entity, view);
        }
        emit representationRemoved(entity, view);
    }
}

void ecvRepresentationManager::removeRepresentation(ccHObject* entity,
                                                    ecvGenericGLDisplay* view) {
    bool removed = false;
    {
        QWriteLocker writeLock(&m_lock);
        auto it = m_representations.find(Key(entity, view));
        if (it != m_representations.end()) {
            m_representations.erase(it);
            removed = true;
        }
    }
    if (removed) {
        if (m_actorCleanup) {
            m_actorCleanup(entity, view);
        }
        emit representationRemoved(entity, view);
    }
}

int ecvRepresentationManager::count() const {
    QReadLocker readLock(&m_lock);
    return m_representations.size();
}

void ecvRepresentationManager::clear() {
    QWriteLocker writeLock(&m_lock);
    m_representations.clear();
}

void ecvRepresentationManager::setActorCleanupCallback(CleanupCallback cb) {
    m_actorCleanup = std::move(cb);
}

void ecvRepresentationManager::notifyChanged(ecvViewRepresentation* rep) {
    if (rep) {
        emit representationChanged(rep);
    }
}
