// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
#include <QColor>
#include <QString>
#include <QVector>
#include <functional>

#include "PointCloudCuboid.h"

struct PointCloudDatasetImportClass {
    int id = 0;
    QString name;
    QColor color;
    bool operator==(const PointCloudDatasetImportClass& other) const {
        return id == other.id && name == other.name && color == other.color;
    }
};

struct PointCloudDatasetImportedFrame {
    QString name;
    int index = -1;
    QVector<PointCloudCuboid> cuboids;
};

struct PointCloudDatasetImportResult {
    QVector<PointCloudDatasetImportClass> classes;
    QVector<PointCloudDatasetImportedFrame> frames;
};

// A target is deliberately independent from PointCloudWindow.  The window
// can fill it from its current frame list and decide when the returned plan is
// safe to apply.
struct PointCloudDatasetImportTarget {
    QString path;
    QString labelPath;
    QString cuboidPath;
    QVector<PointCloudCuboid> cuboids;
    bool cuboidsProvided = false;
};

struct PointCloudDatasetImportFileChange {
    QString path;
    QByteArray before;
    QByteArray after;
    bool existed = false;
};

struct PointCloudDatasetImportPlan {
    QVector<PointCloudDatasetImportTarget> targets;
    QVector<PointCloudDatasetImportClass> classes;
    QVector<PointCloudDatasetImportFileChange> changes;
    int previousObjects = 0;
    int newClasses = 0;
    // Optional consumer snapshot; archive/transaction code leaves it opaque.
    QByteArray consumerState;
};

// Reads Datumaro 3D, KITTI Raw, or Supervisely point-cloud annotation ZIPs.
// No point-cloud payload is decoded; the returned frame names are matched by
// the caller to its local sequence, as in upstream import_dataset.py.
bool importPointCloudDataset(const QString& archivePath,
                             const QString& format,
                             PointCloudDatasetImportResult* result,
                             const std::function<bool(int, int)>& progress = {},
                             QString* error = nullptr);

// Stages the same frame matching, class merge, and atomic sidecar/config
// contract as upstream import_dataset.py.  It does not decode point-cloud
// payloads or mutate a live window.
bool preparePointCloudDatasetImport(
        const QString& archivePath,
        const QString& format,
        const QVector<PointCloudDatasetImportTarget>& targets,
        const QVector<PointCloudDatasetImportClass>& currentClasses,
        const QString& configPath,
        PointCloudDatasetImportPlan* plan,
        const std::function<bool(int, int)>& progress = {},
        QString* error = nullptr,
        const QVector<PointCloudDatasetImportClass>* segmentationClasses =
                nullptr);

// Applies a prepared plan with staged files and rollback on cancellation,
// source changes, or an installation failure.
bool applyPointCloudDatasetImport(
        const PointCloudDatasetImportPlan& plan,
        const std::function<bool(int, int)>& progress = {},
        QString* error = nullptr);
