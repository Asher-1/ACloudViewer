// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RelateDetectors.h"

#include <QCoreApplication>
#include <QDir>
#include <memory>

#include "aicore/relation_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/yolo_capi.h"
#include "ecvAssetIntegrity.h"

namespace RelateDetectors {
namespace {
QString tr(const char* text) {
    return QCoreApplication::translate("RelateDetectors", text);
}

QString yoloDir() {
    std::unique_ptr<char, decltype(&aicore_yolo_free_buffer)> dir(
            aicore_yolo_model_cache_dir(), aicore_yolo_free_buffer);
    return dir ? QString::fromUtf8(dir.get()) : QString();
}

QString rfdetrDir() {
    std::unique_ptr<char, decltype(&aicore_rfdetr_free_buffer)> dir(
            aicore_rfdetr_model_cache_dir(), aicore_rfdetr_free_buffer);
    return dir ? QString::fromUtf8(dir.get()) : QString();
}

bool offered(const QString& file) {
    return !ecvAssetIntegrity::PinnedDigest(file).isEmpty() &&
           (file.endsWith(QStringLiteral("-f16.gguf")) ||
            file.endsWith(QStringLiteral("-q8_0.gguf")));
}

void addGroup(QList<Info>& out,
              const QString& group,
              enum aicore_yolo_model_role role,
              const QString& note = {},
              int promptFree = -1) {
    for (int i = 0; i < aicore_yolo_model_count(role); ++i) {
        const auto* e = aicore_yolo_model_at(i, role);
        if (!e) continue;
        const QString file = QString::fromUtf8(e->filename);
        // promptFree: 1 = only "-pf-" rows, 0 = only the others.
        if (promptFree >= 0 &&
            file.contains(QStringLiteral("-pf-")) != (promptFree == 1))
            continue;
        if (!offered(file)) continue;
        Info info;
        info.filename = file;
        info.group = group;
        info.url = QString::fromUtf8(e->download_url);
        info.dir = yoloDir();
        info.world = file.contains(QStringLiteral("-world"));
        info.vocabulary =
                e->text_input &&
                QString::fromUtf8(e->task) != QStringLiteral("text") &&
                !file.contains(QStringLiteral("-pf-"));
        QString name = QString::fromUtf8(e->display_name) +
                       QStringLiteral(" — ") + QString::fromUtf8(e->quant_note);
        name.remove(QStringLiteral(" (recommended)"));
        if (file == defaultFilename()) name += tr(" (default)");
        info.display = group + QStringLiteral(" · ") + name + note;
        out << info;
    }
}
}  // namespace

QString Info::path() const {
    return valid() && !dir.isEmpty() ? QDir(dir).filePath(filename) : QString();
}

QString Info::textModelFile() const {
    return world ? QStringLiteral("clip-ViT-B-32-f16.gguf")
                 : QStringLiteral("mobileclip2_b-f16.gguf");
}

QString defaultFilename() {
    return QStringLiteral("yoloe-26s-seg-pf-f16.gguf");
}

QList<Info> catalog() {
    QList<Info> out;
    const QString ra4m = tr(" · RA-4M 497 classes");
    addGroup(out, tr("YOLOE prompt-free (masks)"), AICORE_YOLO_ROLE_YOLOE, {},
             1);
    addGroup(out, tr("YOLOE (masks)"), AICORE_YOLO_ROLE_YOLOE, ra4m, 0);
    addGroup(out, tr("YOLO-World"), AICORE_YOLO_ROLE_WORLD, ra4m);
    addGroup(out, tr("YOLO segment"), AICORE_YOLO_ROLE_SEGMENT);
    for (int i = 0; i < aicore_rfdetr_model_count(); ++i) {
        const auto* e = aicore_rfdetr_model_at(i);
        if (!e || !offered(QString::fromUtf8(e->filename))) continue;
        Info info;
        info.filename = QString::fromUtf8(e->filename);
        info.group = tr("RF-DETR");
        info.url = QString::fromUtf8(e->download_url);
        info.dir = rfdetrDir();
        info.rfdetr = true;
        QString name = QString::fromUtf8(e->display_name);
        name.remove(QStringLiteral(" (recommended)"));
        info.display = info.group + QStringLiteral(" · ") + name;
        out << info;
    }
    addGroup(out, tr("YOLO detect (boxes)"), AICORE_YOLO_ROLE_DETECTION);
    return out;
}

Info info(const QString& filename) {
    for (const Info& item : catalog())
        if (item.filename == filename) return item;
    return {};
}

QStringList vocabularyClasses() {
    QStringList names;
    for (int i = 0; i < aicore_relation_object_vocabulary_count(); ++i)
        names << QString::fromUtf8(aicore_relation_object_vocabulary_at(i));
    return names;
}

QList<Asset> assets(const QString& filename) {
    QList<Asset> out;
    const Info detector = info(filename);
    if (!detector.valid()) return out;
    out.append({detector.filename, detector.path(), detector.url});
    if (!detector.vocabulary) return out;
    const QString textFile = detector.textModelFile();
    for (int i = 0; i < aicore_yolo_model_count(AICORE_YOLO_ROLE_TEXT); ++i) {
        const auto* e = aicore_yolo_model_at(i, AICORE_YOLO_ROLE_TEXT);
        if (e && textFile == QString::fromUtf8(e->filename))
            out.append({textFile, QDir(yoloDir()).filePath(textFile),
                        QString::fromUtf8(e->download_url)});
    }
    return out;
}

}  // namespace RelateDetectors
