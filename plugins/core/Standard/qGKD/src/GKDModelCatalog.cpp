// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "GKDModelCatalog.h"

#include <QtCompat.h>

#include <QFileInfo>
#include <QFontMetrics>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMap>
#include <QPainter>
#include <QRegularExpression>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "aicore/gkd_capi.h"
#include "aicore/yolo_capi.h"

namespace GKDHelpers {

namespace {

// Role-filtered catalog view over the AICore GKD catalog.
QVector<GKDModelEntry> catalogFromApi() {
    QVector<GKDModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_gkd_model_count();
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_gkd_model_entry* e = aicore_gkd_model_at(i);
        if (!e || !e->filename) continue;
        GKDModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        entry.sizeBytes = static_cast<qint64>(e->size_bytes);
        out.append(entry);
    }
#endif
    return out;
}

}  // namespace

QVector<GKDModelEntry> catalogModels() { return catalogFromApi(); }

namespace {

// Role-filtered catalog view over the yolo task catalog (no second model
// table: the multi-object detector and its text tower reuse the entries
// the qYOLO catalog already publishes).
QVector<GKDModelEntry> yoloRoleModels(enum aicore_yolo_model_role role) {
    QVector<GKDModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_yolo_model_count(role);
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_yolo_model_entry* e = aicore_yolo_model_at(i, role);
        if (!e || !e->filename) continue;
        GKDModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        out.append(entry);
    }
#else
    (void)role;
#endif
    return out;
}

int yoloRoleDefaultIndex(enum aicore_yolo_model_role role) {
#ifdef AICore_ENABLED
    return aicore_yolo_model_default_index(role);
#else
    (void)role;
    return -1;
#endif
}

}  // namespace

QVector<GKDModelEntry> yoloWorldModels() {
    return yoloRoleModels(AICORE_YOLO_ROLE_WORLD);
}

int yoloWorldDefaultIndex() {
#ifdef AICore_ENABLED
    // Prefer yolov8l-world for the GKD-per-box pipeline. Measured on the
    // bundled demo scenes (public C ABI probe): s-world emits near-tie
    // duplicate herd boxes once the confidence cut drops (alpaca pair at
    // IoU 0.64 survived NMS 0.7, rendered as two overlapping targets),
    // while l-world is duplicate-free with better-calibrated scores
    // (0.75-0.92) on every scene.
    constexpr const char* kPreferred = "yolov8l-world-f16.gguf";
    const int n = aicore_yolo_model_count(AICORE_YOLO_ROLE_WORLD);
    for (int i = 0; i < n; ++i) {
        const aicore_yolo_model_entry* e =
                aicore_yolo_model_at(i, AICORE_YOLO_ROLE_WORLD);
        if (e && e->filename && std::strcmp(e->filename, kPreferred) == 0) {
            return i;
        }
    }
#endif
    return yoloRoleDefaultIndex(AICORE_YOLO_ROLE_WORLD);
}

QVector<GKDModelEntry> yoloTextModels() {
    return yoloRoleModels(AICORE_YOLO_ROLE_TEXT);
}

int yoloTextDefaultIndex() {
    return yoloRoleDefaultIndex(AICORE_YOLO_ROLE_TEXT);
}

int catalogDefaultIndex() {
#ifdef AICore_ENABLED
    return aicore_gkd_model_default_index();
#else
    return -1;
#endif
}

bool findModelByFilename(const QString& filename, GKDModelEntry* out) {
    const QVector<GKDModelEntry> all = catalogModels();
    for (const GKDModelEntry& e : all) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_gkd_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_gkd_free_buffer(dir);
        return out;
    }
#endif
    return QString();
}

QString modelDisplayLabel(const GKDModelEntry& entry) {
    QString label = entry.displayName;
    if (!entry.quantNote.isEmpty() && !label.contains(entry.quantNote)) {
        label += QStringLiteral(" ") + QChar(0x2014) + QStringLiteral(" ") +
                 entry.quantNote;
    }
    return label;
}

bool imageView(QImage* image, aicore_image_view* out) {
    if (image == nullptr || out == nullptr || image->isNull()) return false;
    // Keep the borrowed view zero-copy for the formats AICore understands.
    switch (image->format()) {
        case QImage::Format_RGB888:
            break;  // RGB8, zero-copy
        case QImage::Format_Grayscale8:
            break;  // GRAY8, zero-copy
        case QImage::Format_ARGB32:
            // Little-endian ARGB32 memory layout is BGRA8.
#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
            break;
#else
            *image = image->convertToFormat(QImage::Format_RGBA8888);
            break;
#endif
        case QImage::Format_RGBA8888:
        case qtCompatQImageFormatBgr888():
            break;
        default:
            *image = image->convertToFormat(QImage::Format_ARGB32);
            if (image->format() != QImage::Format_ARGB32) return false;
#if Q_BYTE_ORDER != Q_LITTLE_ENDIAN
            *image = image->convertToFormat(QImage::Format_RGBA8888);
#endif
            break;
    }
    if (image->isNull()) return false;
    out->data = image->constBits();
    out->width = image->width();
    out->height = image->height();
    out->row_stride_bytes = static_cast<size_t>(image->bytesPerLine());
    switch (image->format()) {
        case QImage::Format_RGB888:
            out->format = AICORE_IMAGE_RGB8;
            break;
        case QImage::Format_Grayscale8:
            out->format = AICORE_IMAGE_GRAY8;
            break;
        case QImage::Format_ARGB32:
            out->format = AICORE_IMAGE_BGRA8;
            break;
        case QImage::Format_RGBA8888:
            out->format = AICORE_IMAGE_RGBA8;
            break;
        case qtCompatQImageFormatBgr888():
            out->format = AICORE_IMAGE_BGR8;
            break;
        default:
            return false;
    }
    return out->data != nullptr && out->width > 0 && out->height > 0;
}

bool parseCoordinatePairs(const QString& text, QVector<QPointF>* out) {
    if (out == nullptr) return false;
    out->clear();
    const QString trimmed = text.simplified();
    if (trimmed.isEmpty()) return true;
    static const QRegularExpression number(
            QStringLiteral("-?\\d+(?:\\.\\d+)?"));
    QRegularExpressionMatchIterator it = number.globalMatch(trimmed);
    QVector<double> values;
    QVector<QPair<int, int>> spans;
    while (it.hasNext()) {
        const auto m = it.next();
        values.append(m.captured(0).toDouble());
        spans.append({m.capturedStart(), m.capturedLength()});
    }
    if (values.size() < 2 || values.size() % 2 != 0) return false;
    // Reject leftover garbage: every character outside a number must be
    // whitespace or the pair separator (',').
    for (int i = 0; i < trimmed.size(); ++i) {
        const QChar ch = trimmed.at(i);
        if (ch.isSpace() || ch == QLatin1Char(',')) continue;
        bool insideNumber = false;
        for (const auto& span : spans) {
            if (i >= span.first && i < span.first + span.second) {
                insideNumber = true;
                break;
            }
        }
        if (!insideNumber) return false;
    }
    out->reserve(values.size() / 2);
    for (int i = 0; i + 1 < values.size(); i += 2) {
        out->append(QPointF(values[i], values[i + 1]));
    }
    return true;
}

QStringList splitPrompts(const QString& text) {
    QStringList prompts;
    QString current;
    bool inQuotes = false;
    for (const QChar ch : text) {
        if (ch == QLatin1Char('"')) {
            inQuotes = !inQuotes;
        } else if (ch == QLatin1Char(',') && !inQuotes) {
            const QString trimmed = current.trimmed();
            if (!trimmed.isEmpty()) prompts.append(trimmed);
            current.clear();
        } else {
            current.append(ch);
        }
    }
    const QString trimmed = current.trimmed();
    if (!trimmed.isEmpty()) prompts.append(trimmed);
    return prompts;
}

QVector<QPair<int, int>> parseSkeleton(const QString& text) {
    QVector<QPair<int, int>> bones;
    static const QRegularExpression pair(
            QStringLiteral("(\\d+)\\s*-\\s*(\\d+)"));
    QRegularExpressionMatchIterator it = pair.globalMatch(text);
    while (it.hasNext()) {
        const auto m = it.next();
        bones.append({m.captured(1).toInt(), m.captured(2).toInt()});
    }
    return bones;
}

QVector<QPointF> buildBones(const QStringList& prompts,
                            const QVector<GKDKeypoint>& keypoints,
                            const QVector<QPair<int, int>>& skeleton,
                            float minScore) {
    QVector<QPointF> bones;
    if (skeleton.isEmpty() || keypoints.isEmpty()) return bones;
    const bool byPrompt = !prompts.isEmpty();
    // Resolves a 1-based skeleton endpoint to a keypoint position whose
    // score passes the display cut, or a null point when absent/hidden
    // (official semantics: bones never reference invisible keypoints).
    auto endpoint = [&](int oneBasedIndex) -> QPointF {
        if (oneBasedIndex < 1) return QPointF();
        if (byPrompt) {
            if (oneBasedIndex > prompts.size()) return QPointF();
            const QString& name = prompts.at(oneBasedIndex - 1);
            for (const GKDKeypoint& kp : keypoints) {
                if (kp.prompt == name && kp.score >= minScore) {
                    return QPointF(kp.x, kp.y);
                }
            }
            return QPointF();
        }
        const int idx = oneBasedIndex - 1;
        if (idx >= keypoints.size()) return QPointF();
        const GKDKeypoint& kp = keypoints.at(idx);
        return kp.score >= minScore ? QPointF(kp.x, kp.y) : QPointF();
    };
    for (const auto& bone : skeleton) {
        const QPointF a = endpoint(bone.first);
        if (a.isNull()) continue;
        const QPointF b = endpoint(bone.second);
        if (b.isNull()) continue;
        bones.append(a);
        bones.append(b);
    }
    return bones;
}

QStringList promptModes() {
    return {QStringLiteral("text"), QStringLiteral("visual"),
            QStringLiteral("multimodal"), QStringLiteral("multi")};
}

bool modeUsesYolo(const QString& mode) {
    return mode == QStringLiteral("multi");
}

GKDModePreset modePreset(const QString& mode) {
    const QVector<GKDModePreset> presets = modePresets(mode);
    return presets.isEmpty() ? GKDModePreset{} : presets.first();
}

QVector<GKDModePreset> modePresets(const QString& mode) {
    // Values pinned from the upstream General-Keypoint-Detection README
    // demos (single-object examples 1-3 and the multi-object quadruped
    // example of section 4.3), so a one-click fill reproduces a published
    // result. All four modes share the same official demo bundle.
    // - text: 2007_007524.jpg with the five face keypoints.
    // - visual / multimodal: support image 2007_003778.jpg with its three
    //   annotated keypoints, left eye (343,166), right eye (281,158),
    //   nose (311,197) — the ONLY officially-annotated image in the
    //   bundle, so it is the fixed support for both modes. The visual
    //   query is swapped to 2008_000808.jpg (a front-facing pug) to
    //   demonstrate the open-world few-shot story (cat-face points found
    //   on a different species) instead of repeating the same cat pair;
    //   multimodal keeps the exact official pair (texts fuse with the
    //   support rows, n_kps_texts == n_support_kps backend contract).
    // - multi: every bundled multi-target scene, so successive "Try
    //   sample data" clicks walk the whole dataset (8 scenes). Keypoint
    //   texts follow the upstream predefined_keypoints.py schemas where
    //   one exists (human -> coco full body, hand -> onehand10k,
    //   car -> carfusion); measured through the public C ABI, exact
    //   in-training phrases outscore self-invented semantic ones by a
    //   wide margin (car: 0.71-0.94 vs 0.03-0.82), and YOLO-World class
    //   prompts are spelling/inflection sensitive ("pigs" 2x the boxes
    //   of "pig", "human_hand" vs "human hand" 0.11 -> 0.31 top score).
    QVector<GKDModePreset> presets;
    if (mode == QStringLiteral("text")) {
        GKDModePreset preset;
        preset.queryImage = QStringLiteral("2007_007524.jpg");
        preset.kpsTexts = QStringLiteral(
                "nose, left eye, right eye, left ear, "
                "right ear");
        // Official Example 2 --skeleton (1-based over the five texts).
        preset.skeleton = QStringLiteral("1-2 1-3 2-3 2-4 3-5");
        presets.append(preset);

        // Hand X-ray: the official obj_type 'hand_xray' retrieval path
        // (predefined_keypoints.py schema, 24 points): exact in-training
        // phrases keep 24/24 at 0.9+ on the bundled X-ray, while the
        // earlier generic "fingertip" prompt scored 0.04. Skeleton: the
        // upstream hand_xray schema leaves 'skeleton' empty, so the bone
        // topology here is self-defined over the official 24-point set
        // (per-finger root -> three knuckles -> tip chains, mirroring the
        // onehand10k wrist chain): 19 edges.
        GKDModePreset xray;
        xray.queryImage = QStringLiteral("3144.png");
        xray.kpsTexts = QStringLiteral(
                "pinky_finger's root, ring_finger's root, "
                "middle_finger's root, forefinger's root, thumb_root, "
                "thumb's first knuckle, thumb's second knuckle, thumb's tip, "
                "forefinger's first knuckle, forefinger's second knuckle, "
                "forefinger's third knuckle, forefinger's tip, "
                "middle_finger's first knuckle, middle_finger's second "
                "knuckle, middle_finger's third knuckle, middle_finger's tip, "
                "ring_finger's first knuckle, ring_finger's second knuckle, "
                "ring_finger's third knuckle, ring_finger's tip, "
                "pinky_finger's first knuckle, pinky_finger's second knuckle, "
                "pinky_finger's third knuckle, pinky_finger's tip");
        // Self-defined per-finger chains (see comment above; 1-based over
        // the 24 texts: pinky/ring/middle/forefinger roots and the thumb).
        xray.skeleton = QStringLiteral(
                "1-21 21-22 22-23 23-24 2-17 17-18 18-19 19-20 3-13 13-14 "
                "14-15 15-16 4-9 9-10 10-11 11-12 5-6 6-7 7-8");
        presets.append(xray);

        // Adelie penguin: the official obj_type 'penguin' retrieval hits
        // the 'adelie penguin' entry (animalweb schema, 9 face landmarks
        // with its own skeleton): 9/9 kept at 0.54-0.91 on the bundled
        // image, where the earlier beak/eyes trio kept 2/3.
        GKDModePreset penguin;
        penguin.queryImage = QStringLiteral("adeliepenguin_107.jpg");
        penguin.kpsTexts = QStringLiteral(
                "right corner of right eye, left corner of right eye, "
                "right corner of left eye, left corner of left eye, "
                "nose tip, right corner of lip, left corner of lip, "
                "center of upper lip, center of lower lip");
        penguin.skeleton =
                QStringLiteral("1-2 2-5 5-3 3-4 5-8 6-8 8-7 7-9 9-6");
        presets.append(penguin);

        // Chair: arbitrary-object parts (the open-world selling point).
        // Texts follow the official obj_type 'chair' retrieval path
        // (predefined_keypoints.py -> keypoint5__schema_2): the exact
        // in-training phrases score 0.83-0.93 on all ten points, while
        // self-invented part names kept only 2/4 at 0.20-0.27.
        GKDModePreset chair;
        chair.queryImage = QStringLiteral("00000016.jpg");
        chair.kpsTexts = QStringLiteral(
                "keypoint 1 of chair, keypoint 2 of chair, "
                "keypoint 3 of chair, keypoint 4 of chair, "
                "keypoint 5 of chair, keypoint 6 of chair, "
                "keypoint 7 of chair, keypoint 8 of chair, "
                "keypoint 9 of chair, keypoint 10 of chair");
        // Official keypoint5__schema_2 skeleton (verbatim, including
        // the duplicated 7-8 edge).
        chair.skeleton = QStringLiteral(
                "1-5 4-8 2-6 3-7 5-6 6-7 7-8 8-5 7-8 8-9 9-10 10-7");
        presets.append(chair);

        // Tiger: the official obj_type 'tiger' retrieval (awa_pose
        // schema) fits this walking side-profile (24/39 kept at 0.10,
        // paws/legs 0.85-0.90, in the earlier 39-point measurement). The
        // prompt list trims the four antler rows: a big cat can never
        // grow antlers, so those prompts can only produce sub-cut garbage
        // peaks (35 points below). Skeleton: the upstream awa schema
        // leaves 'skeleton' empty, so the topology here is self-defined
        // over the official point set (jaw/eye/ear chains,
        // neck-throat-back-tail spine, per-leg thai-knee-paw chains);
        // bones draw only between shown keypoints, so undetected
        // endpoints simply drop out.
        GKDModePreset tiger;
        tiger.queryImage = QStringLiteral("000002.jpg");
        tiger.kpsTexts = QStringLiteral(
                "nose, upper_jaw, lower_jaw, mouth_end_right, "
                "mouth_end_left, right_eye, right_earbase, right_earend, "
                "left_eye, left_earbase, left_earend, neck_base, neck_end, "
                "throat_base, throat_end, back_base, back_end, back_middle, "
                "tail_base, tail_end, front_left_thai, front_left_knee, "
                "front_left_paw, front_right_thai, front_right_paw, "
                "front_right_knee, back_left_knee, back_left_paw, "
                "back_left_thai, back_right_thai, back_right_paw, "
                "back_right_knee, belly_bottom, body_middle_right, "
                "body_middle_left");
        // Self-defined awa_pose topology (see the tiger comment above;
        // 1-based over the 35 texts, shared by the pigs scene).
        tiger.skeleton = QStringLiteral(
                "1-2 1-3 2-4 3-5 6-7 7-8 9-10 10-11 12-7 12-10 12-13 12-14 "
                "14-15 13-16 16-18 18-17 17-19 19-20 21-22 22-23 24-26 "
                "26-25 29-27 27-28 30-32 32-31");
        presets.append(tiger);
    } else if (mode == QStringLiteral("visual")) {
        GKDModePreset preset;
        preset.queryImage = QStringLiteral("2008_000808.jpg");
        preset.supportImage = QStringLiteral("2007_003778.jpg");
        preset.supportKps = QStringLiteral("343,166 281,158 311,197");
        // Official Example 1 --skeleton over the three support points.
        preset.skeleton = QStringLiteral("1-2 1-3 2-3");
        presets.append(preset);

        // The three support points (cat face) transfer across species:
        // tiger (same family) and a penguin. Multi-target lineup images
        // stay out of the rotation: single-object modes produce one
        // keypoint set per run, so only single-subject queries belong
        // here (the cat-&-dog lineup lives in the multi-object rotation).
        GKDModePreset tiger;
        tiger.queryImage = QStringLiteral("000002.jpg");
        tiger.supportImage = QStringLiteral("2007_003778.jpg");
        tiger.supportKps = QStringLiteral("343,166 281,158 311,197");
        tiger.skeleton = QStringLiteral("1-2 1-3 2-3");
        presets.append(tiger);

        GKDModePreset penguin;
        penguin.queryImage = QStringLiteral("adeliepenguin_107.jpg");
        penguin.supportImage = QStringLiteral("2007_003778.jpg");
        penguin.supportKps = QStringLiteral("343,166 281,158 311,197");
        penguin.skeleton = QStringLiteral("1-2 1-3 2-3");
        presets.append(penguin);

        // Official Example 4 (cross-object): the alpaca support face
        // (left eye 615,495 - right eye 483,493 - nose 521,549) drives
        // the detection on the cat pair. The upstream command passes no
        // --skeleton for this variant, so points render without bones.
        GKDModePreset crossObject;
        crossObject.queryImage = QStringLiteral("2007_007524.jpg");
        crossObject.supportImage = QStringLiteral("alpaca_150.jpg");
        crossObject.supportKps = QStringLiteral("615,495 483,493 521,549");
        presets.append(crossObject);
    } else if (mode == QStringLiteral("multimodal")) {
        GKDModePreset preset;
        preset.queryImage = QStringLiteral("2007_007524.jpg");
        preset.kpsTexts = QStringLiteral("left eye, right eye, nose");
        preset.supportImage = QStringLiteral("2007_003778.jpg");
        preset.supportKps = QStringLiteral("343,166 281,158 311,197");
        // Official Example 3 --skeleton over the three fused rows.
        preset.skeleton = QStringLiteral("1-2 1-3 2-3");
        presets.append(preset);

        // The text rows must keep matching the fixed support-point order;
        // only the query image rotates (single-subject queries only —
        // the multi-target lineup image lives in the multi-object
        // rotation).
        GKDModePreset pug;
        pug.queryImage = QStringLiteral("2008_000808.jpg");
        pug.kpsTexts = QStringLiteral("left eye, right eye, nose");
        pug.supportImage = QStringLiteral("2007_003778.jpg");
        pug.supportKps = QStringLiteral("343,166 281,158 311,197");
        pug.skeleton = QStringLiteral("1-2 1-3 2-3");
        presets.append(pug);
    } else if (mode == QStringLiteral("multi")) {
        GKDModePreset alpacas;
        alpacas.queryImage = QStringLiteral("alpaca_150.jpg");
        alpacas.kpsTexts = QStringLiteral(
                "left eye, right eye, left ear, right ear, nose, throat, "
                "withers, tail, left-front leg, right-front leg, left-back "
                "leg, right-back leg, left-front knee, right-front knee, "
                "left-back knee, right-back knee, left-front paw, "
                "right-front paw, left-back paw, right-back paw");
        // Official Example 1 --skeleton (animal_pose schema, verbatim).
        alpacas.skeleton = QStringLiteral(
                "5-1 5-2 1-3 2-4 7-8 7-9 9-13 13-17 7-10 10-14 14-18 "
                "8-11 11-15 15-19 8-12 12-16 16-20 6-7 6-5");
        alpacas.objectClasses = QStringLiteral("alpaca");
        presets.append(alpacas);

        // Bronze statues: the official multi-object "human" demo scene
        // (README section 4.3, example 3 uses this image). Keypoint texts
        // are the official coco schema (predefined_keypoints.py 'human')
        // in its underscore spelling: measured per box, the full body
        // scores 0.41-0.94 on the sculptures (shoulders/elbows/wrists
        // 0.73+, hips/knees/ankles 0.41-0.78), and the underscore form
        // beats the spaced one on occluded points (left eye 0.57 vs
        // 0.17). The class prompt stays "person" (upstream passes
        // 'human'): "person" is COCO-native for YOLO-World and finds
        // every statue, while the off-vocabulary "statue" recovers a
        // single box at any confidence.
        GKDModePreset statues;
        statues.queryImage = QStringLiteral("000000011511.jpg");
        statues.kpsTexts = QStringLiteral(
                "nose, left_eye, right_eye, left_ear, right_ear, "
                "left_shoulder, right_shoulder, left_elbow, right_elbow, "
                "left_wrist, right_wrist, left_hip, right_hip, "
                "left_knee, right_knee, left_ankle, right_ankle");
        // Official coco schema skeleton (predefined_keypoints.py).
        statues.skeleton = QStringLiteral(
                "16-14 14-12 17-15 15-13 12-13 6-12 7-13 6-7 6-8 7-9 "
                "8-10 9-11 2-3 1-2 1-3 2-4 3-5 4-6 5-7");
        statues.objectClasses = QStringLiteral("person");
        presets.append(statues);

        // Pig farm: front-facing animals. Keypoint texts follow the
        // official obj_type 'pig' retrieval (awa_pose schema; 22/39 kept
        // on the demo boxes in the earlier 39-point measurement, legs/
        // paws 0.72-0.86). The prompt list trims the four antler rows
        // (same reasoning as the tiger scene: pigs have no antlers, so
        // those prompts only produce garbage; 35 points below). The
        // skeleton is the same self-defined awa topology as the tiger
        // scene. Scene conf 0.15 (measured, not guessed): at the global
        // 0.25 only 2 boxes survive while the 0.05 sweep held 4-5
        // candidates (0.60/0.51/0.18/0.15) — the 0.15 cut recovers the
        // 0.18/0.15 pair and a PIL render of the 0.15 band confirmed
        // every box is a real pig. The class prompt is the plural
        // "pigs": the CLIP text tower embeds it with systematically
        // higher similarity on this scene (boxes at 0.68/0.43/0.39/0.31
        // vs "pig" keeping only 2 above 0.25); the remaining missed pigs
        // sit below the detector's candidate ceiling — upstream uses the
        // stronger LocateAnything detector here.
        GKDModePreset pigs;
        pigs.queryImage = QStringLiteral("pigs_stock_farming.jpg");
        pigs.kpsTexts = QStringLiteral(
                "nose, upper_jaw, lower_jaw, mouth_end_right, "
                "mouth_end_left, right_eye, right_earbase, right_earend, "
                "left_eye, left_earbase, left_earend, neck_base, neck_end, "
                "throat_base, throat_end, back_base, back_end, back_middle, "
                "tail_base, tail_end, front_left_thai, front_left_knee, "
                "front_left_paw, front_right_thai, front_right_paw, "
                "front_right_knee, back_left_knee, back_left_paw, "
                "back_left_thai, back_right_thai, back_right_paw, "
                "back_right_knee, belly_bottom, body_middle_right, "
                "body_middle_left");
        // Same self-defined awa_pose topology as the tiger scene
        // (1-based over the 35 texts).
        pigs.skeleton = QStringLiteral(
                "1-2 1-3 2-4 3-5 6-7 7-8 9-10 10-11 12-7 12-10 12-13 12-14 "
                "14-15 13-16 16-18 18-17 17-19 19-20 21-22 22-23 24-26 "
                "26-25 29-27 27-28 30-32 32-31");
        pigs.objectClasses = QStringLiteral("pigs");
        pigs.yoloConf = 0.15f;
        presets.append(pigs);

        // Fish school: the official obj_type 'fish' retrieval hits the
        // CUB bird template through the substring fallback (upstream
        // semantics, 15 points): 10-11/15 kept per box (tail 0.86,
        // legs 0.81, beak 0.79) versus 2/4 for the earlier generic fin
        // texts. Dense small targets score low but real: at the 0.25
        // default only 25 of the school is kept while the 0.05 sweep
        // holds 62 boxes — the 0.10-0.25 band rendered all-real fish —
        // so this scene fills the detector conf at 0.10 (25 -> 46
        // boxes). CUB carries no skeleton upstream.
        GKDModePreset fish;
        fish.queryImage = QStringLiteral("fish_swim.jpg");
        fish.kpsTexts = QStringLiteral(
                "back, beak, belly, breast, crown, forehead, left eye, "
                "left leg, left wing, nape, right eye, right leg, "
                "right wing, tail, throat");
        fish.objectClasses = QStringLiteral("fish");
        fish.yoloConf = 0.10f;
        presets.append(fish);

        // Cat-&-dog lineup: the official multi-object example 2 scene —
        // mixed classes; keypoint texts follow the official obj_type
        // 'cat, dog' retrieval (animal_pose_dataset schema, the same
        // 20-point set as the alpaca demo): 15/20 kept per box while the
        // earlier face-only texts hid the visible legs/paws (0.84+).
        GKDModePreset lineup;
        lineup.queryImage = QStringLiteral("cat_dog.jpg");
        lineup.kpsTexts = QStringLiteral(
                "left eye, right eye, left ear, right ear, nose, throat, "
                "withers, tail, left-front leg, right-front leg, left-back "
                "leg, right-back leg, left-front knee, right-front knee, "
                "left-back knee, right-back knee, left-front paw, "
                "right-front paw, left-back paw, right-back paw");
        // Official animal_pose schema skeleton (same topology as the
        // alpaca demo).
        lineup.skeleton = QStringLiteral(
                "5-1 5-2 1-3 2-4 7-8 7-9 9-13 13-17 7-10 10-14 14-18 "
                "8-11 11-15 15-19 8-12 12-16 16-20 6-7 6-5");
        lineup.objectClasses = QStringLiteral("cat, dog");
        presets.append(lineup);

        // Traffic intersection: the official multi-object example 5
        // scene (autonomous driving). GKDT is trained on CarFusion, so
        // the official carfusion schema texts ("car keypoint N") are
        // in-distribution and score 0.71-0.94 per box, while
        // self-invented semantic parts (headlight/wheel/plate) sit at
        // 0.03-0.82. The detector class stays the singular "car":
        // YOLO-World (unlike the upstream LocateAnything) drops to 2
        // boxes with the official 'car, bus, truck' list vs 8 with
        // "car".
        GKDModePreset traffic;
        traffic.queryImage = QStringLiteral("car_penn2_0_1931.jpg");
        traffic.kpsTexts = QStringLiteral(
                "car keypoint 1, car keypoint 2, car keypoint 3, "
                "car keypoint 4, car keypoint 5, car keypoint 6, "
                "car keypoint 7, car keypoint 8, car keypoint 9, "
                "car keypoint 10, car keypoint 11, car keypoint 12, "
                "car keypoint 13, car keypoint 14");
        // Official carfusion schema skeleton (predefined_keypoints.py).
        traffic.skeleton = QStringLiteral(
                "1-3 2-4 1-2 3-4 10-12 11-13 10-11 12-13 5-1 5-10 5-6 "
                "6-2 6-11 7-3 7-12 8-4 8-13 7-8");
        traffic.objectClasses = QStringLiteral("car");
        presets.append(traffic);

        // Egocentric dish washing (the official multi-object example 4
        // scene) was REMOVED from the rotation: the YOLO-World CLIP text
        // tower embeds 'human_hand' too weakly for reliable recall (top
        // boxes 0.31/0.29; the spaced 'human hand' tops out at 0.11),
        // and the upstream scene relies on the stronger LocateAnything
        // detector that has no GGUF asset. User decision: drop the scene
        // instead of shipping a preset that usually detects nothing.

        // Bird row: the official obj_type 'bird' retrieval (nabird
        // schema, 11 points): 7-9/11 kept per box (bill 0.93, wings/
        // breast/back 0.69-0.86) versus 3/4 for the earlier beak/eye/
        // tail trio. Skeleton: NABird leaves 'skeleton' empty upstream,
        // so the topology is self-defined over the official 11-point set
        // (bill-eye-crown head chain, nape-back-tail spine, wings off
        // the back, breast-belly line): 10 edges.
        GKDModePreset birds;
        birds.queryImage = QStringLiteral("pet_birds.jpg");
        birds.kpsTexts = QStringLiteral(
                "bill, crown, nape, left eye, right eye, belly, breast, "
                "back, tail, left wing, right wing");
        birds.skeleton =
                QStringLiteral("1-4 4-2 1-5 5-2 2-3 3-8 8-9 8-10 8-11 7-6");
        birds.objectClasses = QStringLiteral("bird");
        presets.append(birds);
    }
    return presets;
}

QString formatGkdTimings(const GkdStageTimings& t, int rois) {
    // Stable stage order so lines from different runs diff by eye.
    return QStringLiteral(
                   "preprocess %1 ms | vision %2 ms | text %3 ms | prompt "
                   "prep %4 ms | detect %5 ms (%6 ROI) | decode %7 ms | "
                   "total %8 ms")
            .arg(t.preprocessMs, 0, 'f', 1)
            .arg(t.visionMs, 0, 'f', 1)
            .arg(t.textMs, 0, 'f', 1)
            .arg(t.promptPrepMs, 0, 'f', 1)
            .arg(t.detectMs, 0, 'f', 1)
            .arg(rois)
            .arg(t.decodeMs, 0, 'f', 1)
            .arg(t.e2eMs, 0, 'f', 1);
}

QString formatYoloTimings(const YoloStageTimings& t) {
    return QStringLiteral(
                   "preprocess %1 ms | inference %2 ms | "
                   "postprocess %3 ms | total %4 ms")
            .arg(t.preprocessMs, 0, 'f', 1)
            .arg(t.inferenceMs, 0, 'f', 1)
            .arg(t.postprocessMs, 0, 'f', 1)
            .arg(t.e2eMs, 0, 'f', 1);
}

QVector<QRect> placeLabels(const QVector<QRect>& preferred,
                           const QSize& canvas) {
    // Force-directed label spreading — a C++ port of the industry-standard
    // implementation in roboflow/supervision (detection/utils/boxes.py
    // spread_out_boxes + snap_boxes, the engine behind
    // LabelAnnotator::smart_position, issue #1383). Every iteration pushes
    // each overlapping pair apart along its center line with a force
    // proportional to the pair's IoU (minimum 2 px per axis so tiny forces
    // cannot stall convergence), until no overlap remains or the
    // 100-iteration cap is hit. The cap is the termination guarantee: the
    // previous hand-rolled cascade could iterate forever when pinning
    // rewound a label into another one (the 2026-09-16 multi-minute hang).
    // O(n^2) per iteration at n ≤ ~150 labels is sub-millisecond.
    const int n = preferred.size();
    QVector<QRect> rects(preferred);
    if (n < 2) return rects;

    for (int iteration = 0; iteration < 100; ++iteration) {
        bool anyOverlap = false;
        QVector<QPointF> forces(n, QPointF(0, 0));
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                // The upstream pads every box by 1 px before the IoU so
                // near-touching labels already repel.
                const QRect paddedI = rects[i].adjusted(-1, -1, 1, 1);
                const QRect paddedJ = rects[j].adjusted(-1, -1, 1, 1);
                const QRect overlap = paddedI.intersected(paddedJ);
                if (overlap.isEmpty()) continue;
                anyOverlap = true;
                const double interArea =
                        double(overlap.width()) * overlap.height();
                const double unionArea =
                        double(paddedI.width()) * paddedI.height() +
                        double(paddedJ.width()) * paddedJ.height() - interArea;
                const double iou = unionArea > 0 ? interArea / unionArea : 0.0;
                QPointF dir =
                        QPointF(rects[i].center()) - QPointF(rects[j].center());
                double len = std::hypot(dir.x(), dir.y());
                if (len < 1e-3) {
                    // Identical centers are a force deadlock upstream
                    // (zero direction, zero push); break the symmetry
                    // upward so stacked labels still separate.
                    dir = QPointF(0, -1);
                    len = 1.0;
                }
                dir /= len;
                forces[i] += dir * iou * 10.0;
                forces[j] -= dir * iou * 10.0;
            }
        }
        if (!anyOverlap) break;
        for (int i = 0; i < n; ++i) {
            double fx = forces[i].x();
            double fy = forces[i].y();
            if (fx > 0 && fx < 2) fx = 2;
            if (fx < 0 && fx > -2) fx = -2;
            if (fy > 0 && fy < 2) fy = 2;
            if (fy < 0 && fy > -2) fy = -2;
            rects[i].translate(static_cast<int>(fx), static_cast<int>(fy));
        }
    }
    // snap_boxes: after spreading, keep every label fully on the canvas.
    // Labels that ended up outside are clamped back inside instead of
    // being cropped (the measured right-edge failure mode).
    for (QRect& r : rects) {
        r.moveLeft(std::clamp(r.left(), 2,
                              std::max(2, canvas.width() - r.width() - 2)));
        r.moveTop(std::clamp(r.top(), 2,
                             std::max(2, canvas.height() - r.height() - 2)));
    }
    return rects;
}

QColor groupColor(int groupId) {
    // COCO-consistent deterministic palette (same heritage as qYOLO).
    static const QRgb kPalette[20] = {
            qRgb(220, 20, 60),   qRgb(119, 11, 32),   qRgb(0, 0, 142),
            qRgb(0, 0, 230),     qRgb(106, 0, 228),   qRgb(0, 60, 100),
            qRgb(0, 80, 100),    qRgb(0, 0, 70),      qRgb(0, 0, 192),
            qRgb(250, 170, 30),  qRgb(100, 170, 30),  qRgb(220, 220, 0),
            qRgb(175, 116, 175), qRgb(250, 0, 30),    qRgb(165, 42, 42),
            qRgb(255, 77, 255),  qRgb(0, 226, 252),   qRgb(182, 182, 255),
            qRgb(0, 82, 0),      qRgb(120, 166, 157),
    };
    return QColor(kPalette[groupId % 20]);
}

QImage renderResult(const QImage& source,
                    const QVector<GKDKeypointSet>& sets,
                    float minScore,
                    bool pointLabels) {
    if (source.isNull()) return QImage();
    QImage out = source.format() == QImage::Format_ARGB32
                         ? source
                         : source.convertToFormat(QImage::Format_ARGB32);

    QPainter p(&out);
    p.setRenderHint(QPainter::Antialiasing, true);

    QFont font = p.font();
    // Official text sizing: upstream visualize_keypoints.py scales text
    // with image width (TEXT_SCALE = max(1, width/2000) cv2 fontScale,
    // i.e. ~1% of the width in pixels). The old height/50 read ~2x
    // larger and dominated small crops.
    font.setPixelSize(std::max(10, out.width() / 100));
    p.setFont(font);
    const QFontMetrics fm(font);

    // Optional per-point "name score" labels: default OFF (a multi-object
    // scene stacks 10+ boxes x ~20 points and the label backgrounds alone
    // hide the objects — measured on the cat_dog lineup: 63 keypoints
    // buried every animal). The user opts in per run via the dialog
    // checkbox; clustered detections then get a deterministic de-overlap
    // layout with thin leader arrows instead of raw stacking.
    struct LabelItem {
        QString text;
        QPointF point;
        QColor color;
        float score = 0;
    };
    QVector<LabelItem> labels;
    // Dot radius is mode-independent (adaptive to the image width), so it
    // hoists out of the per-set loop for reuse by the label pass.
    const qreal radius = std::max(3.0, out.width() / 250.0);

    int groupId = 0;
    for (const GKDKeypointSet& set : sets) {
        const QColor color = groupColor(groupId++);
        if (set.hasBox) {
            QPen pen(color);
            pen.setWidth(2);
            p.setPen(pen);
            p.setBrush(Qt::NoBrush);
            p.drawRect(
                    QRectF(set.x1, set.y1, set.x2 - set.x1, set.y2 - set.y1));
            if (!set.label.isEmpty()) {
                const QString label =
                        QStringLiteral("%1 (%2)").arg(set.label).arg(
                                set.keypoints.size());
                // Fit the background to the MEASURED text: the old
                // char-count x pixel-size estimate overshot by ~2x on
                // proportional fonts (labels floated far past the text).
                const int textW = fm.horizontalAdvance(label);
                const int textH = fm.height();
                QRect labelRect(static_cast<int>(set.x1),
                                static_cast<int>(set.y1) - textH - 4, textW + 8,
                                textH + 4);
                labelRect.moveLeft(std::clamp(
                        labelRect.left(), 2,
                        std::max(2, out.width() - labelRect.width() - 2)));
                if (labelRect.top() < 2) {
                    labelRect.moveTop(static_cast<int>(set.y1) + 2);
                }
                p.fillRect(labelRect.intersected(out.rect()), color);
                p.setPen(Qt::white);
                p.drawText(labelRect, Qt::AlignVCenter | Qt::AlignHCenter,
                           label);
            }
        }
        // Official-demo skeleton bones (flat x1,y1,x2,y2 segments) under
        // the keypoint dots, tinted with the group color. Line width is
        // half the official visualize_keypoints adaptive sizing
        // (max(2, 0.4% of the image width)) — at the official 0.8% the
        // bones on large images read as heavy bars and bury the dots.
        if (!set.bones.isEmpty()) {
            QPen pen(color);
            pen.setWidth(std::max(2, static_cast<int>(out.width() * 0.004)));
            p.setPen(pen);
            p.setBrush(Qt::NoBrush);
            for (int i = 0; i + 1 < set.bones.size(); i += 2) {
                p.drawLine(set.bones.at(i), set.bones.at(i + 1));
            }
        }
        // Keypoints: white-ringed dot (+ optional label pass below). Dot
        // radius follows the official visualize_keypoints adaptive sizing
        // (max(3, 0.4% of the image width)).
        for (const GKDKeypoint& kp : set.keypoints) {
            if (kp.score < minScore) continue;
            p.setPen(Qt::NoPen);
            p.setBrush(Qt::white);
            p.drawEllipse(QPointF(kp.x, kp.y), radius, radius);
            p.setBrush(color);
            p.drawEllipse(QPointF(kp.x, kp.y), radius * 0.62, radius * 0.62);

            if (pointLabels) {
                LabelItem item;
                item.text =
                        QStringLiteral("%1 %2")
                                .arg(kp.prompt.isEmpty() ? QStringLiteral("kpt")
                                                         : kp.prompt)
                                .arg(kp.score, 0, 'f', 2);
                item.point = QPointF(kp.x, kp.y);
                item.color = color;
                item.score = kp.score;
                labels.append(item);
            }
        }
    }

    // Label pass (opt-in only). Capacity governance: the official
    // visualizers never label every keypoint (category name only, or
    // score-only 6 px annotations), so dense runs here must fit the
    // labels into a bounded share of the canvas — beyond ~30% coverage
    // the scene reads as a wall of text (measured: fish-swarm 400+
    // labels, person-17 x 5 bodies 85 labels). First shrink the font
    // (official style: smaller text on crowded scenes), then drop the
    // lowest-scoring labels — the highest-confidence identities are the
    // ones worth the ink.
    if (pointLabels && !labels.isEmpty()) {
        const int budget = out.width() * out.height() * 3 / 10;
        int pixel = font.pixelSize();
        QVector<QRect> preferred(labels.size());
        auto measure = [&]() {
            QFont f = font;
            f.setPixelSize(pixel);
            const QFontMetrics m(f);
            int area = 0;
            for (int i = 0; i < labels.size(); ++i) {
                const int w = m.horizontalAdvance(labels[i].text) + 8;
                const int h = m.height() + 2;
                preferred[i].setSize(QSize(w, h));
                area += w * h;
            }
            return area;
        };
        int area = measure();
        if (area > budget) {
            pixel = std::max(
                    9,
                    static_cast<int>(pixel * std::sqrt(double(budget) / area)));
            area = measure();
            if (area > budget) {
                const int avg = std::max(1, area / labels.size());
                const int keep = std::max(1, budget / avg);
                if (keep < labels.size()) {
                    std::vector<int> order(labels.size());
                    for (int i = 0; i < labels.size(); ++i) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](int a, int b) {
                        return labels[a].score > labels[b].score;
                    });
                    order.resize(keep);
                    std::sort(order.begin(), order.end());
                    QVector<LabelItem> kept;
                    kept.reserve(keep);
                    for (int i : order) kept.append(labels[i]);
                    labels = kept;
                    preferred = QVector<QRect>(labels.size());
                    area = measure();
                }
            }
        }
        font.setPixelSize(pixel);
        p.setFont(font);

        for (int i = 0; i < labels.size(); ++i) {
            const QSize sz = preferred[i].size();
            const int x = static_cast<int>(labels[i].point.x());
            const int y = static_cast<int>(labels[i].point.y());
            QRect r(x + static_cast<int>(radius) + 2, y - sz.height() / 2,
                    sz.width(), sz.height());
            // Ultralytics-style "outside" test: when the label would run
            // off the right edge, mirror it to the LEFT of the dot
            // instead of letting it be cropped (measured: right-edge
            // keypoints on the hand-X-ray / chair scenes lost half the
            // text).
            if (r.right() > out.width() - 2)
                r.moveLeft(x - static_cast<int>(radius) - 2 - r.width());
            r.moveLeft(std::clamp(r.left(), 2,
                                  std::max(2, out.width() - r.width() - 2)));
            r.moveTop(std::clamp(r.top(), 2,
                                 std::max(2, out.height() - r.height() - 2)));
            preferred[i] = r;
        }
        const QVector<QRect> placed = placeLabels(preferred, out.size());
        for (int i = 0; i < labels.size(); ++i) {
            const LabelItem& item = labels[i];
            QRect rect =
                    placed[i].intersected(out.rect().adjusted(2, 2, -2, -2));
            if (rect.isEmpty()) rect = placed[i];
            if (placed[i] != preferred[i]) {
                QPen leader(item.color);
                leader.setWidth(1);
                p.setPen(leader);
                p.setBrush(Qt::NoBrush);
                const QPointF from =
                        rect.center();  // inner segment is covered by the
                                        // label background drawn below
                const QPointF to = item.point;
                p.drawLine(from, to);
                const QPointF dir = to - from;
                const qreal len =
                        std::sqrt(dir.x() * dir.x() + dir.y() * dir.y());
                if (len > 1e-3) {
                    const QPointF unit = dir / len;
                    const QPointF normal(-unit.y(), unit.x());
                    const qreal head = std::max(4.0, radius);
                    QPolygonF arrow;
                    arrow << to << to - unit * head + normal * head * 0.4
                          << to - unit * head - normal * head * 0.4;
                    p.drawPolygon(arrow);
                }
            }
            p.fillRect(rect.intersected(out.rect()), QColor(0, 0, 0, 160));
            p.setPen(Qt::white);
            p.drawText(rect, Qt::AlignVCenter | Qt::AlignHCenter, item.text);
        }
    }
    p.end();
    return out;
}

QString buildCocoJson(const GKDRunResult& result) {
    // Category keypoint names: the textual prompts when given, else the
    // official visual-prompt fallback naming (N_v rows).
    QStringList kpsNames = result.kpsTexts;
    if (kpsNames.isEmpty() && !result.sets.isEmpty()) {
        for (int i = 0; i < result.sets.first().keypoints.size(); ++i) {
            kpsNames << QStringLiteral("keypoint %1").arg(i);
        }
    }
    QJsonArray skeleton;
    for (const auto& edge : parseSkeleton(result.skeleton)) {
        skeleton.append(QJsonArray{edge.first, edge.second});
    }

    // One category per distinct set label ("object1" for label-less
    // single-object runs, mirroring the official writer).
    QMap<QString, int> categoryId;
    QJsonArray categories;
    auto categoryFor = [&](const QString& name) {
        auto it = categoryId.constFind(name);
        if (it != categoryId.constEnd()) return *it;
        const int id = categoryId.size() + 1;
        categoryId.insert(name, id);
        QJsonArray names;
        for (const QString& n : kpsNames) names.append(n);
        QJsonObject cat;
        cat.insert(QStringLiteral("id"), id);
        cat.insert(QStringLiteral("name"), name);
        cat.insert(QStringLiteral("supercategory"), QString());
        cat.insert(QStringLiteral("keypoints"), names);
        cat.insert(QStringLiteral("skeleton"), skeleton);
        categories.append(cat);
        return id;
    };

    QJsonArray images;
    QJsonObject image;
    image.insert(QStringLiteral("id"), 1);
    image.insert(QStringLiteral("file_name"),
                 QFileInfo(result.imageName).fileName());
    image.insert(QStringLiteral("width"), result.imageWidth);
    image.insert(QStringLiteral("height"), result.imageHeight);
    images.append(image);

    QJsonArray annotations;
    int annoId = 0;
    for (const GKDKeypointSet& set : result.sets) {
        const QString name =
                set.label.isEmpty() ? QStringLiteral("object1") : set.label;
        QJsonObject ann;
        ann.insert(QStringLiteral("id"), ++annoId);
        ann.insert(QStringLiteral("category_id"), categoryFor(name));
        ann.insert(QStringLiteral("image_id"), 1);
        // COCO stores xywh; the engine gives xyxy (whole image when the
        // run had no box, matching the official demo fallback).
        const double x = set.hasBox ? set.x1 : 0.0;
        const double y = set.hasBox ? set.y1 : 0.0;
        const double w =
                set.hasBox ? set.x2 - set.x1 : double(result.imageWidth);
        const double h =
                set.hasBox ? set.y2 - set.y1 : double(result.imageHeight);
        ann.insert(
                QStringLiteral("bbox"),
                QJsonArray{std::round(x * 100) / 100, std::round(y * 100) / 100,
                           std::round(w * 100) / 100,
                           std::round(h * 100) / 100});
        ann.insert(QStringLiteral("score"),
                   std::round(set.bboxScore * 100) / 100);
        QJsonArray kps;
        for (const GKDKeypoint& kp : set.keypoints) {
            kps.append(std::round(kp.x * 100) / 100);
            kps.append(std::round(kp.y * 100) / 100);
            kps.append(std::round(kp.score * 100) / 100);
        }
        ann.insert(QStringLiteral("keypoints"), kps);
        ann.insert(QStringLiteral("num_keypoints"), set.keypoints.size());
        annotations.append(ann);
    }

    QJsonObject root;
    root.insert(QStringLiteral("info"), QJsonObject());
    root.insert(QStringLiteral("licenses"), QJsonArray());
    root.insert(QStringLiteral("images"), images);
    root.insert(QStringLiteral("annotations"), annotations);
    root.insert(QStringLiteral("categories"), categories);
    return QJsonDocument(root).toJson(QJsonDocument::Indented);
}

}  // namespace GKDHelpers
