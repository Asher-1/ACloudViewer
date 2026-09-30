// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QHash>
#include <QJsonObject>
#include <QString>
#include <QVector>

#include "AnnotationExporter.h"

enum class PpocrMode { Rec, Kie };

struct PpocrImport {
    QHash<QString, QJsonObject> documents;
    QString error;
    int count = 0;
};

PpocrImport importPpocr(const QByteArray& labels, PpocrMode mode);

struct PpocrConversion {
    QByteArray text;
    int count = 0;
    QString error;
};

PpocrConversion convertPpocr(const QVector<CocoImage>& images, PpocrMode mode);

struct PpocrDatasetImage {
    QString imagePath;
    QJsonObject document;
};

struct PpocrRecExport {
    int annotationCount = 0;
    int cropCount = 0;
    QString error;
};

// Writes the source-compatible Rec dataset directory. Existing Rec artifacts
// are replaced as one unit before the image list is processed.
PpocrRecExport exportPpocrRecDataset(const QVector<PpocrDatasetImage>& images,
                                     const QString& outputDirectory);
