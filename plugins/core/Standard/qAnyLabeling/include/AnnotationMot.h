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
#include <QSize>
#include <QStringList>
#include <QVector>

struct MotImage {
    QString fileName;
    QSize size;
    QJsonObject document;
};

struct MotImport {
    QHash<QString, QJsonObject> documents;
    QString error;
    int count = 0;
};

struct MotConversion {
    QByteArray sequence;
    QByteArray detections;
    QByteArray groundTruth;
    QString error;
    int count = 0;
};

MotImport importMot(const QByteArray& groundTruth,
                    const QStringList& classes,
                    const QVector<MotImage>& images);
MotConversion convertMot(const QVector<MotImage>& images,
                         const QStringList& classes,
                         const QString& outputFolder);
MotConversion convertMots(const QVector<MotImage>& images,
                          const QStringList& classes,
                          const QString& outputFolder);
