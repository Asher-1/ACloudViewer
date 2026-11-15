// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QJsonArray>
#include <QStringList>
#include <functional>

enum class AnnotationShapeBatchOperation {
    DeleteAnnotations,
    DeleteImages,
    RemoveSelected,
    AddSelected
};

struct AnnotationShapeBatchResult {
    int files = 0;
    int shapes = 0;
    int skipped = 0;
    bool canceled = false;
    QStringList errors;
};

// One image/sidecar at a time, in the displayed inclusive 1-based range.
// Returning false from progress cancels before the next file is changed.
AnnotationShapeBatchResult applyAnnotationShapeBatch(
        const QStringList& images,
        const QString& outputDirectory,
        int first,
        int last,
        AnnotationShapeBatchOperation operation,
        const QJsonArray& selected,
        const std::function<bool(int, int)>& progress = {});
