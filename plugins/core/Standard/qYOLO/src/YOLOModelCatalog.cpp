// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLOModelCatalog.h"

#include <QColor>
#include <QFont>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QRegularExpression>
#include <QtMath>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "aicore/yolo_capi.h"

namespace YOLOHelpers {

namespace {

// Role-filtered catalog view shared by every per-task helper.
QVector<YOLOModelEntry> roleModels(enum aicore_yolo_model_role role) {
    QVector<YOLOModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_yolo_model_count(role);
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_yolo_model_entry* e = aicore_yolo_model_at(i, role);
        if (!e || !e->filename) continue;
        YOLOModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        entry.task = QString::fromUtf8(e->task ? e->task : "detect");
        entry.depthCapable = e->depth_capable != 0;
        entry.end2end = e->end2end != 0;
        entry.textInput = e->text_input != 0;
        out.append(entry);
    }
#else
    (void)role;
#endif
    return out;
}

}  // namespace

QVector<YOLOModelEntry> catalogModels() {
#ifdef AICore_ENABLED
    return roleModels(AICORE_YOLO_ROLE_ANY);
#else
    return {};
#endif
}

QVector<YOLOModelEntry> detectionModels() {
    return roleModels(AICORE_YOLO_ROLE_DETECTION);
}

QVector<YOLOModelEntry> segmentModels() {
    return roleModels(AICORE_YOLO_ROLE_SEGMENT);
}

QVector<YOLOModelEntry> depthModels() {
    return roleModels(AICORE_YOLO_ROLE_DEPTH);
}

QVector<YOLOModelEntry> poseModels() {
    return roleModels(AICORE_YOLO_ROLE_POSE);
}

QVector<YOLOModelEntry> obbModels() { return roleModels(AICORE_YOLO_ROLE_OBB); }

QVector<YOLOModelEntry> classifyModels() {
    return roleModels(AICORE_YOLO_ROLE_CLASSIFY);
}

QVector<YOLOModelEntry> semanticModels() {
    return roleModels(AICORE_YOLO_ROLE_SEMANTIC);
}

QVector<YOLOModelEntry> worldModels() {
    return roleModels(AICORE_YOLO_ROLE_WORLD);
}

QVector<YOLOModelEntry> yoloeModels() {
    return roleModels(AICORE_YOLO_ROLE_YOLOE);
}

QVector<YOLOModelEntry> textModels() {
    return roleModels(AICORE_YOLO_ROLE_TEXT);
}

QVector<YOLOModelEntry> taskModels(const QString& task) {
    // Each tab id maps to one catalog role, so a tab only ever offers the
    // models of its own task family (world/yoloe are separated from the
    // closed-set families by their text_input flag inside the catalog).
    if (task == QStringLiteral("detect")) return detectionModels();
    if (task == QStringLiteral("segment")) return segmentModels();
    if (task == QStringLiteral("depth")) return depthModels();
    if (task == QStringLiteral("pose")) return poseModels();
    if (task == QStringLiteral("obb")) return obbModels();
    if (task == QStringLiteral("classify")) return classifyModels();
    if (task == QStringLiteral("semantic")) return semanticModels();
    if (task == QStringLiteral("world")) return worldModels();
    if (task == QStringLiteral("yoloe")) return yoloeModels();
    if (task == QStringLiteral("text")) return textModels();
    return detectionModels();  // forward-compatible fallback
}

int defaultModelIndexForTask(const QString& task) {
#ifdef AICore_ENABLED
    enum aicore_yolo_model_role role = AICORE_YOLO_ROLE_ANY;
    if (task == QStringLiteral("detect"))
        role = AICORE_YOLO_ROLE_DETECTION;
    else if (task == QStringLiteral("segment"))
        role = AICORE_YOLO_ROLE_SEGMENT;
    else if (task == QStringLiteral("depth"))
        role = AICORE_YOLO_ROLE_DEPTH;
    else if (task == QStringLiteral("pose"))
        role = AICORE_YOLO_ROLE_POSE;
    else if (task == QStringLiteral("obb"))
        role = AICORE_YOLO_ROLE_OBB;
    else if (task == QStringLiteral("classify"))
        role = AICORE_YOLO_ROLE_CLASSIFY;
    else if (task == QStringLiteral("semantic"))
        role = AICORE_YOLO_ROLE_SEMANTIC;
    else if (task == QStringLiteral("world"))
        role = AICORE_YOLO_ROLE_WORLD;
    else if (task == QStringLiteral("yoloe"))
        role = AICORE_YOLO_ROLE_YOLOE;
    else if (task == QStringLiteral("text"))
        role = AICORE_YOLO_ROLE_TEXT;
    else
        return -1;  // unknown task: let the caller's fallback decide
    return aicore_yolo_model_default_index(role);
#else
    (void)task;
    return -1;
#endif
}

bool findModelByFilename(const QString& filename, YOLOModelEntry* out) {
    const QVector<YOLOModelEntry> all = catalogModels();
    for (const YOLOModelEntry& e : all) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_yolo_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_yolo_free_buffer(dir);
        return out;
    }
#else
    (void)0;
#endif
    return QString();
}

QString modelDisplayLabel(const YOLOModelEntry& entry) {
    QString label = entry.displayName;
    if (!entry.quantNote.isEmpty() && !label.contains(entry.quantNote)) {
        label += QStringLiteral(" ") + QChar(0x2014) + QStringLiteral(" ") +
                 entry.quantNote;
    }
    return label;
}

const uchar* packedRgb888Data(const QImage& image, QByteArray* scratch) {
    if (!scratch || image.isNull() || image.format() != QImage::Format_RGB888)
        return nullptr;
    scratch->clear();
    const int rowBytes = image.width() * 3;
    if (image.bytesPerLine() == rowBytes) return image.constBits();

    scratch->resize(rowBytes * image.height());
    for (int y = 0; y < image.height(); ++y) {
        std::memcpy(scratch->data() + y * rowBytes, image.constScanLine(y),
                    static_cast<size_t>(rowBytes));
    }
    return reinterpret_cast<const uchar*>(scratch->constData());
}

bool filenameIsDepth(const QString& filename) {
    return filename.toLower().contains(QStringLiteral("depth"));
}

bool isPromptFreeFilename(const QString& filename) {
    // Prompt-free YOLOE checkpoints carry "-pf-" mid-name (e.g.
    // "yoloe-26l-seg-pf-f16.gguf") or "-pf." right before the extension
    // (e.g. "yoloe-26l-seg-pf.gguf").
    return filename.contains(QStringLiteral("-pf-")) ||
           filename.contains(QStringLiteral("-pf."));
}

QString promptFreeSiblingFilename(const QString& filename) {
    // Official no-input path for YOLOE: the -pf checkpoint of the same
    // scale (upstream trains it from the text-prompt model and fuses the
    // 4585-entry LRPC vocabulary, so it needs no prompt at all).
    if (filename.isEmpty() || isPromptFreeFilename(filename)) return {};
    if (!filename.startsWith(QStringLiteral("yoloe-"))) return {};
    const int seg = filename.indexOf(QStringLiteral("-seg"));
    if (seg < 0) return {};
    QString sibling = filename;
    sibling.insert(seg + 4, QStringLiteral("-pf"));
    YOLOModelEntry entry;
    return findModelByFilename(sibling, &entry) ? sibling : QString();
}

namespace {

const QHash<QString, QString>& zhWordMap() {
    static const QHash<QString, QString> map = {
            // colors
            {"红色", "red"},
            {"红", "red"},
            {"绿色", "green"},
            {"绿", "green"},
            {"黄色", "yellow"},
            {"黄", "yellow"},
            {"粉色", "pink"},
            {"粉红色", "pink"},
            {"粉", "pink"},
            {"蓝色", "blue"},
            {"蓝", "blue"},
            {"黑色", "black"},
            {"黑", "black"},
            {"白色", "white"},
            {"白", "white"},
            {"橙色", "orange"},
            {"橘色", "orange"},
            {"紫色", "purple"},
            {"棕色", "brown"},
            {"褐色", "brown"},
            {"灰色", "gray"},
            // people / age
            {"孩子", "child"},
            {"小孩", "child"},
            {"儿童", "child"},
            {"小朋友", "child"},
            {"成年人", "adult"},
            {"成人", "adult"},
            {"大人", "adult"},
            {"男人", "man"},
            {"男子", "man"},
            {"男士", "man"},
            {"女人", "woman"},
            {"女子", "woman"},
            {"女士", "woman"},
            {"男孩", "boy"},
            {"女孩", "girl"},
            {"人", "person"},
            {"人类", "person"},
            // wear / attributes
            {"戴", "wearing"},
            {"穿着", "wearing"},
            {"帽子", "hat"},
            {"帽", "hat"},
            {"眼镜", "glasses"},
            {"太阳镜", "sunglasses"},
            // COCO objects (common Chinese names)
            {"汽车", "car"},
            {"轿车", "car"},
            {"公交车", "bus"},
            {"巴士", "bus"},
            {"卡车", "truck"},
            {"货车", "truck"},
            {"自行车", "bicycle"},
            {"单车", "bicycle"},
            {"摩托车", "motorcycle"},
            {"飞机", "airplane"},
            {"火车", "train"},
            {"船", "boat"},
            {"轮船", "boat"},
            {"红绿灯", "traffic light"},
            {"交通灯", "traffic light"},
            {"消防栓", "fire hydrant"},
            {"停车标志", "stop sign"},
            {"长椅", "bench"},
            {"鸟", "bird"},
            {"狗", "dog"},
            {"猫", "cat"},
            {"马", "horse"},
            {"羊", "sheep"},
            {"牛", "cow"},
            {"大象", "elephant"},
            {"熊", "bear"},
            {"斑马", "zebra"},
            {"长颈鹿", "giraffe"},
            {"背包", "backpack"},
            {"雨伞", "umbrella"},
            {"伞", "umbrella"},
            {"手提包", "handbag"},
            {"领带", "tie"},
            {"行李箱", "suitcase"},
            {"飞盘", "frisbee"},
            {"滑雪板", "skis"},
            {"单板滑雪", "snowboard"},
            {"风筝", "kite"},
            {"网球拍", "tennis racket"},
            {"瓶子", "bottle"},
            {"酒杯", "wine glass"},
            {"杯子", "cup"},
            {"叉子", "fork"},
            {"刀", "knife"},
            {"勺子", "spoon"},
            {"碗", "bowl"},
            {"香蕉", "banana"},
            {"苹果", "apple"},
            {"三明治", "sandwich"},
            {"橙子", "orange"},
            {"花椰菜", "broccoli"},
            {"胡萝卜", "carrot"},
            {"热狗", "hot dog"},
            {"披萨", "pizza"},
            {"甜甜圈", "donut"},
            {"蛋糕", "cake"},
            {"椅子", "chair"},
            {"沙发", "couch"},
            {"盆栽", "potted plant"},
            {"床", "bed"},
            {"餐桌", "dining table"},
            {"马桶", "toilet"},
            {"电视", "tv"},
            {"笔记本电脑", "laptop"},
            {"鼠标", "mouse"},
            {"键盘", "keyboard"},
            {"手机", "cell phone"},
            {"微波炉", "microwave"},
            {"烤箱", "oven"},
            {"冰箱", "refrigerator"},
            {"书", "book"},
            {"时钟", "clock"},
            {"花瓶", "vase"},
            // function words (dropped in the English prompt)
            {"的", ""},
            {"一个", "a"},
            {"一位", "a"},
            {"两只", "two"},
            {"剪刀", "scissors"},
            {"泰迪熊", "teddy bear"},
            {"吹风机", "hair drier"},
            {"牙刷", "toothbrush"},
    };
    return map;
}

const QHash<QString, QString>& zhPersonMap() {
    static const QHash<QString, QString> map = {
            {"孩子", "child"},   {"小孩", "child"},   {"儿童", "child"},
            {"小朋友", "child"}, {"成年人", "adult"}, {"成人", "adult"},
            {"大人", "adult"},   {"男人", "man"},     {"男子", "man"},
            {"男士", "man"},     {"女人", "woman"},   {"女子", "woman"},
            {"女士", "woman"},   {"男孩", "boy"},     {"女孩", "girl"},
            {"人", "person"},
    };
    return map;
}

bool hasCJK(const QString& text) {
    static const QRegularExpression cjk(QStringLiteral("[\u4e00-\u9fff]"));
    return cjk.match(text).hasMatch();
}

}  // namespace

QString translatePromptToEnglish(const QString& text, bool* translated) {
    if (translated) *translated = false;
    if (!hasCJK(text)) return text;

    QString out = text;

    // Template rule: 戴<attr>帽子 的 <person>  ->  "<person> with <attr> hat"
    // (the dominant detection-prompt pattern; attr may carry colors).
    static const QRegularExpression hatPerson(
            QStringLiteral("戴([\\x{4e00}-\\x{9fff}]{0,6}?)帽子?的?"
                           "(孩子|小孩|儿童|小朋友|成年人|成人|大人|男人|男子|"
                           "女士|女人|女子|男士|男孩|女孩|人)"));
    QRegularExpressionMatchIterator it = hatPerson.globalMatch(out);
    while (it.hasNext()) {
        const QRegularExpressionMatch m = it.next();
        const QString personZh = m.captured(2);
        QString attr = m.captured(1);
        QString personEn =
                zhPersonMap().value(personZh, QStringLiteral("person"));
        QString attrEn;
        const QRegularExpression cjkWord(
                QStringLiteral("[\\x{4e00}-\\x{9fff}]+"));
        QRegularExpressionMatchIterator ait = cjkWord.globalMatch(attr);
        while (ait.hasNext()) {
            const QString w = ait.next().captured(0);
            attrEn += zhWordMap().value(w) + " ";
        }
        attrEn = attrEn.trimmed();
        QString repl =
                attrEn.isEmpty()
                        ? QStringLiteral("%1 wearing a hat").arg(personEn)
                        : QStringLiteral("%1 with %2 hat")
                                  .arg(personEn, attrEn);
        out.replace(m.capturedStart(), m.capturedLength(), repl);
        it = hatPerson.globalMatch(out);  // offsets shifted; restart scan
        if (translated) *translated = true;
    }

    // Word-level dictionary replacement for everything else (longest keys
    // first so 红 inside 红色 never wins).
    QStringList keys;
    for (auto keyIt = zhWordMap().constBegin(); keyIt != zhWordMap().constEnd();
         ++keyIt)
        keys << keyIt.key();
    std::sort(keys.begin(), keys.end(), [](const QString& a, const QString& b) {
        return a.size() > b.size();
    });
    for (const QString& key : keys) {
        const QString en = zhWordMap().value(key);
        out.replace(key, " " + en + " ");
    }
    out = out.simplified();

    if (translated && !*translated) *translated = true;
    return out;
}

QString testImageForTask(const QString& task) {
    if (task == QStringLiteral("classify")) return QStringLiteral("cat.jpg");
    if (task == QStringLiteral("obb"))
        return QStringLiteral("aerial_airport.jpg");
    if (task == QStringLiteral("pose"))
        return QStringLiteral("000000087038.jpg");
    if (task == QStringLiteral("world") || task == QStringLiteral("yoloe"))
        return QStringLiteral("party_hats.jpg");
    return QStringLiteral("000000397133.jpg");
}

bool parseDetectionsJson(const QByteArray& json, YOLORunResult* out) {
    if (out == nullptr) return false;
    out->detections.clear();
    out->resultJson = json;

    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(json, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) return false;

    const QJsonObject root = doc.object();
    out->modelVariant = root.value(QStringLiteral("model")).toString();
    out->end2end = root.value(QStringLiteral("end2end")).toInt() != 0;
    out->imageSize = root.value(QStringLiteral("image_size")).toInt();
    out->numClasses = root.value(QStringLiteral("num_classes")).toInt();

    const QJsonArray dets = root.value(QStringLiteral("detections")).toArray();
    for (const QJsonValue& v : dets) {
        if (!v.isObject()) continue;
        const QJsonObject d = v.toObject();
        YOLODetection det;
        det.classId = static_cast<uint32_t>(
                d.value(QStringLiteral("class_id")).toInt(0));
        det.className = d.value(QStringLiteral("class_name")).toString();
        det.score = static_cast<float>(
                d.value(QStringLiteral("score")).toDouble(0.0));
        const QJsonArray box = d.value(QStringLiteral("box")).toArray();
        if (box.size() == 4) {
            det.x1 = static_cast<float>(box.at(0).toDouble(0.0));
            det.y1 = static_cast<float>(box.at(1).toDouble(0.0));
            det.x2 = static_cast<float>(box.at(2).toDouble(0.0));
            det.y2 = static_cast<float>(box.at(3).toDouble(0.0));
        }
        out->detections.append(det);
    }
    out->totalDetected = out->detections.size();
    return true;
}

bool parseDepthStatsJson(const QByteArray& json, YOLODepthStats* out) {
    if (out == nullptr) return false;

    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(json, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) return false;

    const QJsonObject root = doc.object();
    out->width = root.value(QStringLiteral("depth_width")).toInt();
    out->height = root.value(QStringLiteral("depth_height")).toInt();
    out->minDepth = root.value(QStringLiteral("min_depth")).toDouble(0.0);
    out->maxDepth = root.value(QStringLiteral("max_depth")).toDouble(0.0);
    out->meanDepth = root.value(QStringLiteral("mean_depth")).toDouble(0.0);
    out->p95Depth = root.value(QStringLiteral("p95_depth")).toDouble(0.0);
    out->validPixels = static_cast<long long>(
            root.value(QStringLiteral("valid_pixels")).toDouble(0.0));
    return out->width > 0 && out->height > 0;
}

QRgb classColor(uint32_t classId) {
    // COCO-consistent deterministic palette (BGR order from OpenCV heritage).
    static const QRgb kPalette[20] = {
            qRgb(220, 20, 60),   qRgb(119, 11, 32),   qRgb(0, 0, 142),
            qRgb(0, 0, 230),     qRgb(106, 0, 228),   qRgb(0, 60, 100),
            qRgb(0, 80, 100),    qRgb(0, 0, 70),      qRgb(0, 0, 192),
            qRgb(250, 170, 30),  qRgb(100, 170, 30),  qRgb(220, 220, 0),
            qRgb(175, 116, 175), qRgb(250, 0, 30),    qRgb(165, 42, 42),
            qRgb(255, 77, 255),  qRgb(0, 226, 252),   qRgb(182, 182, 255),
            qRgb(0, 82, 0),      qRgb(120, 166, 157),
    };
    return kPalette[classId % 20];
}

namespace {

inline double turboChannel(double fourT, double offset) {
    return std::max(0.0, std::min(1.5 - std::fabs(fourT - offset), 1.0));
}

/** Shared turbo ramp: t in [0, 1] -> near (0) blue .. far (1) red. */
inline QRgb turboRgb(double t) {
    const double r = turboChannel(4.0 * t, 3.0);
    const double g = turboChannel(4.0 * t, 2.0);
    const double b = turboChannel(4.0 * t, 1.0);
    return qRgb(static_cast<int>(r * 255.0 + 0.5),
                static_cast<int>(g * 255.0 + 0.5),
                static_cast<int>(b * 255.0 + 0.5));
}

}  // namespace

void drawDetections(QImage* image,
                    const QVector<YOLODetection>& detections,
                    int thickness) {
    if (image == nullptr || image->isNull()) return;
    const int h = image->height();

    // Bind the painter to one stable ARGB32 data block for the whole call.
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }

    QPainter p(image);
    p.setRenderHint(QPainter::Antialiasing, false);

    QFont font = p.font();
    font.setPixelSize(std::max(12, h / 60));
    p.setFont(font);
    for (const YOLODetection& d : detections) {
        const QColor color(classColor(d.classId));
        QPen pen(color);
        pen.setWidth(thickness);
        p.setPen(pen);
        p.drawRect(QRectF(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1));

        const QString label = QStringLiteral("%1 %2")
                                      .arg(d.className)
                                      .arg(d.score, 0, 'f', 2);
        // Anchor the banner above the box top, then keep it fully inside
        // the image: clamp horizontally, and flip below the box top when
        // the box hugs the top edge (the painter has no clipping here, so
        // off-canvas text would simply be invisible).
        QRect labelRect(static_cast<int>(d.x1),
                        static_cast<int>(d.y1) - font.pixelSize() - 6,
                        std::max(20, label.size() * font.pixelSize()),
                        font.pixelSize() + 6);
        labelRect.setWidth(
                std::min(labelRect.width(), std::max(20, image->width() - 4)));
        labelRect.moveLeft(std::clamp(
                labelRect.left(), 2,
                std::max(2, image->width() - labelRect.width() - 2)));
        if (labelRect.top() < 2) {
            labelRect.moveTop(static_cast<int>(d.y1) + 2);
        }
        labelRect.moveTop(std::min(
                labelRect.top(),
                std::max(2, image->height() - labelRect.height() - 2)));
        const QRect bg = labelRect.adjusted(0, 0, 4, 2);
        p.fillRect(bg.intersected(image->rect()), color);
        p.setPen(Qt::white);
        p.drawText(labelRect.adjusted(2, 3, -2, -2), label);
        p.setPen(pen);
    }
    p.end();
}

/* 3-tap separable Gaussian blur [1,2,1]/4 on Grayscale8. */
static void gaussianBlurMask3(QImage& img) {
    if (img.format() != QImage::Format_Grayscale8) return;
    const int w = img.width(), h = img.height();
    if (w <= 2 || h <= 2) return;
    QImage tmp(w, h, QImage::Format_Grayscale8);
    for (int y = 0; y < h; ++y) {
        const uchar* s = img.constScanLine(y);
        uchar* d = tmp.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const int l = (x > 0) ? s[x - 1] : 0;
            const int m = s[x];
            const int r = (x < w - 1) ? s[x + 1] : 0;
            d[x] = (uint8_t)((l + m * 2 + r) / 4);
        }
    }
    for (int y = 0; y < h; ++y) {
        uchar* d = img.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const int t = (y > 0) ? tmp.constScanLine(y - 1)[x] : 0;
            const int m = tmp.constScanLine(y)[x];
            const int b = (y < h - 1) ? tmp.constScanLine(y + 1)[x] : 0;
            d[x] = (uint8_t)((t + m * 2 + b) / 4);
        }
    }
}

void drawSegmentation(QImage* image,
                      const QVector<YOLOSegMask>& masks,
                      const QVector<YOLODetection>& detections,
                      int thickness) {
    if (image == nullptr || image->isNull() || masks.isEmpty()) return;
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }

    const int imgW = image->width();
    const int imgH = image->height();

    // Masks already live in the source-image space (AICore unscales them
    // from the letterbox canvas); a straight scale to the image keeps the
    // tint aligned with the boxes.
    for (int i = 0; i < masks.size(); ++i) {
        const YOLOSegMask& mask = masks[static_cast<size_t>(i)];
        if (mask.w <= 0 || mask.h <= 0 ||
            mask.bits.size() < static_cast<qint64>(mask.w) * mask.h) {
            continue;
        }
        // Grayscale view over a COPY of the mask bytes: QImage requires the
        // backing buffer to be 32-bit aligned, and QByteArray's offset is
        // not guaranteed to be (Qt 6 debug builds assert on misalignment).
        QImage maskImage(mask.w, mask.h, QImage::Format_Grayscale8);
        // Row-by-row copy: QImage scanlines are 32-bit aligned, so for a
        // width that is not a multiple of 4 bytesPerLine > width and one
        // contiguous memcpy shears the mask.
        for (int y = 0; y < mask.h; ++y) {
            std::memcpy(maskImage.scanLine(y),
                        mask.bits.constData() + static_cast<qint64>(y) * mask.w,
                        static_cast<size_t>(mask.w));
        }
        if (maskImage.isNull()) continue;
        // YOLO mask values are {0, 1} (not {0, 255}).  Scale to full
        // uint8 range so the Gaussian blur below produces meaningful
        // intermediate values at edges instead of eroding everything
        // through integer division ((0 + 2*1 + 1) / 4 = 0).
        for (int b = 0; b < maskImage.sizeInBytes(); ++b) {
            if (maskImage.bits()[b]) maskImage.bits()[b] = 255;
        }
        // 3-tap Gaussian blur at native resolution converts the hard
        // binary mask edge into a soft gradient so SmoothTransformation
        // downscaling produces a single smooth boundary per object instead
        // of a "furry" staircase of per-pixel transitions.
        gaussianBlurMask3(maskImage);
        // The mask is already at the full image resolution (AICore's
        // unscale_masks_to_image produces image_w x image_h masks), so
        // QImage::scaled() would be a full-resolution deep copy — pure
        // waste.  Skip it and use the mask directly.
        const QColor tint = i < detections.size()
                                    ? QColor(classColor(detections[i].classId))
                                    : QColor(220, 220, 220);

        // Alpha-blend the tint over the foreground mask pixels, with the
        // blend weight proportional to the mask coverage so
        // SmoothTransformation anti-aliased edges transition smoothly instead
        // of snapping to full opacity at the first non-zero pixel.
        for (int y = 0; y < imgH; ++y) {
            const uchar* src = maskImage.constScanLine(y);
            uchar* dst = image->scanLine(y);
            for (int x = 0; x < imgW; ++x) {
                if (src[x] == 0) continue;
                const int w = src[x];  // coverage weight 1..255
                const int d = x * 4;
                // (dst * (765 - w) + tint * w) / 765  — preserves the
                // original 2/3 + 1/3 ratio when w == 255.
                dst[d] = static_cast<uchar>(
                        (dst[d] * (765 - w) + tint.blue() * w) / 765);
                dst[d + 1] = static_cast<uchar>(
                        (dst[d + 1] * (765 - w) + tint.green() * w) / 765);
                dst[d + 2] = static_cast<uchar>(
                        (dst[d + 2] * (765 - w) + tint.red() * w) / 765);
            }
        }
    }

    drawDetections(image, detections, thickness);
}

// COCO-17 skeleton edges (0-based keypoint indices: 0 nose, 1-2 eyes,
// 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees,
// 15-16 ankles).
static const int kSkeleton17[][2] = {
        {0, 1},   {0, 2},   {1, 3},   {2, 4},    // eyes/ears
        {3, 5},   {4, 6},                        // ears -> shoulders
        {5, 6},                                  // shoulder line
        {5, 7},   {7, 9},   {6, 8},   {8, 10},   // arms
        {5, 11},  {6, 12},  {11, 12},            // torso
        {11, 13}, {13, 15}, {12, 14}, {14, 16},  // legs
};

void drawPose(QImage* image,
              const QVector<YOLOKeypointSet>& keypointSets,
              int thickness) {
    if (image == nullptr || image->isNull() || keypointSets.isEmpty()) return;
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }
    QPainter p(image);
    p.setRenderHint(QPainter::Antialiasing, true);

    QFont font = p.font();
    font.setPixelSize(std::max(12, image->height() / 60));
    p.setFont(font);

    const qreal kptRadius = std::max(2.0, image->height() / 300.0);
    for (const YOLOKeypointSet& set : keypointSets) {
        const QColor color(classColor(set.det.classId));
        // Skeleton lines between visible keypoints.
        QPen linePen(color);
        linePen.setWidth(thickness);
        p.setPen(linePen);
        for (const auto& edge : kSkeleton17) {
            const int a = edge[0], b = edge[1];
            if (a >= set.kpts.size() || b >= set.kpts.size()) continue;
            const YOLOKeypoint& ka = set.kpts[a];
            const YOLOKeypoint& kb = set.kpts[b];
            // Visibility gate: skip pairs where either endpoint is invisible.
            if (ka.visibility < 0.5f || kb.visibility < 0.5f) continue;
            p.drawLine(QPointF(ka.x, ka.y), QPointF(kb.x, kb.y));
        }
        // Keypoint dots (visible only).
        p.setPen(Qt::NoPen);
        for (const YOLOKeypoint& k : set.kpts) {
            if (k.visibility < 0.5f) continue;
            p.setBrush(QColor(Qt::white));
            p.drawEllipse(QPointF(k.x, k.y), kptRadius, kptRadius);
            p.setBrush(color);
            p.drawEllipse(QPointF(k.x, k.y), kptRadius * 0.6, kptRadius * 0.6);
        }
        // Box + label, mirroring drawDetections.
        QPen pen(color);
        pen.setWidth(thickness);
        p.setPen(pen);
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(set.det.x1, set.det.y1, set.det.x2 - set.det.x1,
                          set.det.y2 - set.det.y1));
        const QString label = QStringLiteral("%1 %2")
                                      .arg(set.det.className)
                                      .arg(set.det.score, 0, 'f', 2);
        QRect labelRect(static_cast<int>(set.det.x1),
                        static_cast<int>(set.det.y1) - font.pixelSize() - 6,
                        std::max(20, label.size() * font.pixelSize()),
                        font.pixelSize() + 6);
        labelRect.setWidth(
                std::min(labelRect.width(), std::max(20, image->width() - 4)));
        labelRect.moveLeft(std::clamp(
                labelRect.left(), 2,
                std::max(2, image->width() - labelRect.width() - 2)));
        if (labelRect.top() < 2) {
            labelRect.moveTop(static_cast<int>(set.det.y1) + 2);
        }
        p.fillRect(labelRect.adjusted(0, 0, 4, 2).intersected(image->rect()),
                   color);
        p.setPen(Qt::white);
        p.drawText(labelRect.adjusted(2, 3, -2, -2), label);
    }
    p.end();
}

void drawObb(QImage* image, const QVector<YOLOObbBox>& boxes, int thickness) {
    if (image == nullptr || image->isNull() || boxes.isEmpty()) return;
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }
    QPainter p(image);
    p.setRenderHint(QPainter::Antialiasing, true);

    QFont font = p.font();
    font.setPixelSize(std::max(12, image->height() / 60));
    p.setFont(font);
    for (const YOLOObbBox& b : boxes) {
        const QColor color(classColor(b.classId));
        QPen pen(color);
        pen.setWidth(thickness);
        p.setPen(pen);
        p.setBrush(Qt::NoBrush);
        // Rotated rectangle: translate to the center, rotate by the angle,
        // draw the unrotated extent.
        p.save();
        p.translate(QPointF(b.cx, b.cy));
        p.rotate(qRadiansToDegrees(b.angle));
        p.drawRect(QRectF(-b.w / 2.0, -b.h / 2.0, b.w, b.h));
        p.restore();
        // Center mark (screen-aligned cross).
        p.drawLine(QPointF(b.cx - 4, b.cy), QPointF(b.cx + 4, b.cy));
        p.drawLine(QPointF(b.cx, b.cy - 4), QPointF(b.cx, b.cy + 4));

        const int deg = static_cast<int>(qRadiansToDegrees(b.angle) + 0.5);
        const QString full = QStringLiteral("%1 %2 %3%4")
                                     .arg(b.className)
                                     .arg(b.score, 0, 'f', 2)
                                     .arg(deg)
                                     .arg(QChar(0x00B0));
        QRect labelRect(
                static_cast<int>(b.cx - b.w / 2.0),
                static_cast<int>(b.cy - b.h / 2.0) - font.pixelSize() - 6,
                std::max(20, full.size() * font.pixelSize()),
                font.pixelSize() + 6);
        labelRect.setWidth(
                std::min(labelRect.width(), std::max(20, image->width() - 4)));
        labelRect.moveLeft(std::clamp(
                labelRect.left(), 2,
                std::max(2, image->width() - labelRect.width() - 2)));
        if (labelRect.top() < 2) {
            labelRect.moveTop(static_cast<int>(b.cy - b.h / 2.0) + 2);
        }
        p.fillRect(labelRect.adjusted(0, 0, 4, 2).intersected(image->rect()),
                   color);
        p.setPen(Qt::white);
        p.drawText(labelRect.adjusted(2, 3, -2, -2), full);
    }
    p.end();
}

// Cityscapes-19 palette (the shipped yolo26*-sem class order).
static QRgb cityscapesColor(uint32_t classId) {
    static const QRgb kCityscapes[19] = {
            qRgb(128, 64, 128),   // road
            qRgb(244, 35, 232),   // sidewalk
            qRgb(70, 70, 70),     // building
            qRgb(102, 102, 156),  // wall
            qRgb(190, 153, 153),  // fence
            qRgb(153, 153, 153),  // pole
            qRgb(250, 170, 30),   // traffic light
            qRgb(220, 220, 0),    // traffic sign
            qRgb(107, 142, 35),   // vegetation
            qRgb(152, 251, 152),  // terrain
            qRgb(70, 130, 180),   // sky
            qRgb(220, 20, 60),    // person
            qRgb(255, 0, 0),      // rider
            qRgb(0, 0, 142),      // car
            qRgb(0, 0, 70),       // truck
            qRgb(0, 60, 100),     // bus
            qRgb(0, 80, 100),     // train
            qRgb(0, 0, 230),      // motorcycle
            qRgb(119, 11, 32),    // bicycle
    };
    return classId < 19 ? kCityscapes[classId] : classColor(classId);
}

void drawSemantic(QImage* image,
                  const QByteArray& classMap,
                  int width,
                  int height,
                  int numClasses) {
    (void)numClasses;  // the palette covers ids directly
    if (image == nullptr || image->isNull() || classMap.isEmpty() ||
        width <= 0 || height <= 0 ||
        classMap.size() < static_cast<qint64>(width) * height) {
        return;
    }
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }
    // 50% alpha blend, 1:1 with the source pixels (AICore already restored
    // the argmax grid to the full source resolution).
    for (int y = 0; y < height && y < image->height(); ++y) {
        const uchar* src =
                reinterpret_cast<const uchar*>(classMap.constData()) +
                static_cast<qint64>(y) * width;
        uchar* dst = image->scanLine(y);
        for (int x = 0; x < width && x < image->width(); ++x) {
            const QRgb tint = cityscapesColor(src[x]);
            const int d = x * 4;
            dst[d + 0] = static_cast<uchar>((dst[d + 0] + qBlue(tint)) / 2);
            dst[d + 1] = static_cast<uchar>((dst[d + 1] + qGreen(tint)) / 2);
            dst[d + 2] = static_cast<uchar>((dst[d + 2] + qRed(tint)) / 2);
        }
    }
}

void drawClassifications(QImage* image,
                         const QVector<YOLOClassProb>& classifications,
                         int topK) {
    if (image == nullptr || image->isNull() || classifications.isEmpty())
        return;
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }
    // Top-k by probability (partial sort; the table is unordered).
    QVector<YOLOClassProb> sorted = classifications;
    std::partial_sort(
            sorted.begin(), sorted.begin() + std::min(topK, sorted.size()),
            sorted.end(), [](const YOLOClassProb& a, const YOLOClassProb& b) {
                return a.prob > b.prob;
            });

    QPainter p(image);
    QFont font = p.font();
    font.setPixelSize(std::max(12, image->height() / 50));
    p.setFont(font);
    const int lineH = font.pixelSize() + 6;
    const int shown = std::min(topK, sorted.size());
    for (int i = 0; i < shown; ++i) {
        const QString label = QStringLiteral("%1. %2  %3%")
                                      .arg(i + 1)
                                      .arg(sorted[i].className)
                                      .arg(sorted[i].prob * 100.0, 0, 'f', 1);
        QRect rect(4, 4 + i * lineH,
                   std::max(40, label.size() * font.pixelSize()), lineH);
        rect.setWidth(std::min(rect.width(), std::max(40, image->width() - 8)));
        p.fillRect(rect.adjusted(0, 0, 4, 2).intersected(image->rect()),
                   QColor(0, 0, 0, 170));
        p.setPen(Qt::white);
        p.drawText(rect.adjusted(3, 3, -3, -2), label);
    }
    p.end();
}

QImage depthColorImage(const float* depth,
                       int width,
                       int height,
                       double minDepth,
                       double maxDepth) {
    if (depth == nullptr || width <= 0 || height <= 0) return QImage();
    const int n = width * height;

    double lo = minDepth;
    double hi = maxDepth;
    if (!(lo < hi)) {
        // Auto range over valid pixels (finite, > 0): min .. p95 — same rule
        // as the AICore depth statistics envelope.
        std::vector<float> valid;
        valid.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const float d = depth[i];
            if (std::isfinite(d) && d > 0.0f) valid.push_back(d);
        }
        if (valid.empty()) return QImage();
        lo = *std::min_element(valid.begin(), valid.end());
        const size_t p95Idx =
                std::min((valid.size() * 95) / 100, valid.size() - 1);
        std::nth_element(valid.begin(), valid.begin() + p95Idx, valid.end());
        hi = valid[p95Idx];
    }
    if (!(lo < hi)) hi = lo + 1e-6;  // single-point range guard

    QImage out(width, height, QImage::Format_RGB888);
    const double invRange = 1.0 / (hi - lo);
    for (int y = 0; y < height; ++y) {
        uchar* row = out.scanLine(y);
        const float* src = depth + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            const float d = src[x];
            if (!(d > 0.0f) || !std::isfinite(d)) {
                // Invalid pixel (no depth): black.
                row[x * 3 + 0] = row[x * 3 + 1] = row[x * 3 + 2] = 0;
                continue;
            }
            double t = (d - lo) * invRange;
            t = std::max(0.0, std::min(1.0, t));
            const QRgb rgb = turboRgb(t);
            row[x * 3 + 0] = static_cast<uchar>(qRed(rgb));
            row[x * 3 + 1] = static_cast<uchar>(qGreen(rgb));
            row[x * 3 + 2] = static_cast<uchar>(qBlue(rgb));
        }
    }
    return out;
}

void drawDepthLegend(QImage* image, double minDepth, double maxDepth) {
    if (image == nullptr || image->isNull() || !(minDepth < maxDepth)) return;
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }

    QPainter p(image);
    const int w = image->width();
    const int h = image->height();
    const int barW = std::max(8, w / 60);
    const int barH = std::min(std::max(60, h / 2), 220);
    const int margin = 8;
    const int x0 = w - barW - margin;
    const int y0 = margin;

    // Ramp top = max (far, red) .. bottom = min (near, blue) — one mapping
    // shared with depthColorImage.
    for (int y = 0; y < barH; ++y) {
        const double t = 1.0 - static_cast<double>(y) / std::max(1, barH - 1);
        p.setPen(QPen(turboRgb(t)));
        p.drawLine(x0, y0 + y, x0 + barW - 1, y0 + y);
    }
    p.setPen(QPen(Qt::white, 1));
    p.drawRect(x0 - 1, y0 - 1, barW + 1, barH + 1);

    QFont font = p.font();
    font.setPixelSize(std::max(10, h / 70));
    p.setFont(font);
    p.setPen(Qt::white);
    const QString farLabel = QStringLiteral("%1 m").arg(maxDepth, 0, 'f', 1);
    const QString nearLabel = QStringLiteral("%1 m").arg(minDepth, 0, 'f', 1);
    const int labelWidth =
            std::max(farLabel.size(), nearLabel.size()) * font.pixelSize();
    p.drawText(QRect(x0 - labelWidth - 6, y0 - 2, labelWidth,
                     font.pixelSize() + 4),
               Qt::AlignRight | Qt::AlignVCenter, farLabel);
    p.drawText(QRect(x0 - labelWidth - 6, y0 + barH - font.pixelSize() - 2,
                     labelWidth, font.pixelSize() + 4),
               Qt::AlignRight | Qt::AlignVCenter, nearLabel);
    p.end();
}

}  // namespace YOLOHelpers
