// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QHash>
#include <QJsonObject>

#include "AnnotationExporter.h"

struct OdvgImport {
    QHash<QString, QJsonObject> documents;
    QString error;
    int count = 0;
};

struct OdvgConversion {
    QByteArray annotations;
    QByteArray labelMap;
    QString error;
    int count = 0;
};

OdvgImport importOdvg(const QByteArray& jsonLines);
OdvgConversion convertOdvg(const QVector<CocoImage>& images,
                           const QStringList& classes);
