// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
#include <QHash>
#include <QJsonObject>
#include <QStringList>

#include "AnnotationExporter.h"

struct VlmImport {
    QHash<QString, QJsonObject> documents;
    QString error;
    int count = 0;
};

struct VlmConversion {
    QByteArray annotations;
    QString error;
    int count = 0;
};

VlmImport importVlm(const QByteArray& jsonLines);
VlmConversion convertVlm(const QVector<CocoImage>& images,
                         const QStringList& classes,
                         const QString& prefix);
