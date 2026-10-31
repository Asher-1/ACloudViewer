// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// Detector catalog of the RelateAnything object stage. It offers the same
// rows, in the same groups, as the standalone qRelateAnything plugin (YOLOE
// prompt-free, YOLOE, YOLO-World, YOLO segment, RF-DETR, YOLO detect; F16 and
// Q8_0 rows with a pinned digest), read from the AICore catalogs.
#include <QList>
#include <QString>
#include <QStringList>

namespace RelateDetectors {

struct Info {
    QString filename, display, group, url, dir;
    bool rfdetr = false;
    // Open vocabulary model that needs the RA-4M class list and a text
    // encoder (YOLO-World and YOLOE checkpoints that are not prompt-free).
    bool vocabulary = false;
    bool world = false;
    bool valid() const { return !filename.isEmpty(); }
    QString path() const;
    QString textModelFile() const;
};

/** Official detector of the image demo (YOLOE prompt-free segmenter). */
QString defaultFilename();
/** Catalog row for a file name; invalid when unknown. */
Info info(const QString& filename);
/** All offered detectors in display order. */
QList<Info> catalog();

/** One file the detector stage needs on disk. */
struct Asset {
    QString filename, path, url;
};
/** The detector plus, for vocabulary models, the text encoder. */
QList<Asset> assets(const QString& filename);
/** The 497 object names baked into vocabulary detectors. */
QStringList vocabularyClasses();

}  // namespace RelateDetectors
