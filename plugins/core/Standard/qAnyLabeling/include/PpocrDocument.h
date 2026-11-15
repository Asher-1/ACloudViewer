// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QJsonObject>
#include <QPolygonF>
#include <QStringList>
#include <QVector>

namespace PpocrDocument {
struct Block {
    QString key, label, text, imagePath;
    int page = 0, index = 0, order = 0;
    QPolygonF polygon;
    bool edited = false;
};
// Accept a document JSON, page JSON or the official job's JSONL download.
QJsonObject parse(const QByteArray& input, QString* error);
QVector<Block> blocks(const QJsonObject& document);
bool edit(QJsonObject* document, const QString& key, const QString& text);
QString markdown(const QJsonObject& document);
QColor categoryColor(const QString& label);
// Upstream document bundle: doc_0.md, imgs/*.jpg, layout_det_res_N.jpg.
// Page paths are ordered, decoded one at a time. Destination changes
// atomically.
bool exportBundle(const QJsonObject& document,
                  const QString& rootDirectory,
                  const QStringList& pagePaths,
                  const QString& zipPath,
                  QString* error);
bool save(const QString& path, const QJsonObject& document, QString* error);
}  // namespace PpocrDocument
