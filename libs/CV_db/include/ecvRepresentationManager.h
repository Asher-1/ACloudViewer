// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QHash>
#include <QList>
#include <QObject>
#include <QPair>
#include <QReadWriteLock>
#include <functional>
#include <memory>

#include "CV_db.h"
#include "ecvViewRepresentation.h"

class ccHObject;
class ecvGenericGLDisplay;

/// Central registry for all (entity, view) representations.
///
/// Thread-safe via QReadWriteLock for concurrent read access.
///
/// The CleanupCallback allows the VtkEngine layer to register
/// VTK-specific actor removal logic without any VTK types leaking
/// into CV_db headers (module scope isolation).
///
/// Design reference: ParaView pqServerManagerModel.
class CV_DB_LIB_API ecvRepresentationManager : public QObject {
    Q_OBJECT

public:
    static ecvRepresentationManager& instance();

    // -- Lookup --

    /// Returns the representation for (entity, view), or nullptr.
    ecvViewRepresentation* getRepresentation(ccHObject* entity,
                                             ecvGenericGLDisplay* view) const;

    /// Returns existing representation or creates a new one.
    ecvViewRepresentation* ensureRepresentation(ccHObject* entity,
                                                ecvGenericGLDisplay* view);

    // -- Batch queries --

    QList<ecvViewRepresentation*> getRepresentationsForEntity(
            ccHObject* entity) const;
    QList<ecvViewRepresentation*> getRepresentationsForView(
            ecvGenericGLDisplay* view) const;

    // -- Cleanup --

    void removeRepresentationsForEntity(ccHObject* entity);
    void removeRepresentationsForView(ecvGenericGLDisplay* view);
    void removeRepresentation(ccHObject* entity, ecvGenericGLDisplay* view);

    int count() const;

    // -- VTK actor cleanup callback --
    // Registered by VtkEngine layer at init time.
    using CleanupCallback =
            std::function<void(ccHObject* entity, ecvGenericGLDisplay* view)>;
    void setActorCleanupCallback(CleanupCallback cb);

signals:
    void representationAdded(ecvViewRepresentation* rep);
    void representationRemoved(ccHObject* entity, ecvGenericGLDisplay* view);
    void representationChanged(ecvViewRepresentation* rep);

private:
    ecvRepresentationManager();

    using Key = QPair<ccHObject*, ecvGenericGLDisplay*>;
    QHash<Key, std::shared_ptr<ecvViewRepresentation>> m_representations;
    CleanupCallback m_actorCleanup;
    mutable QReadWriteLock m_lock;
};
