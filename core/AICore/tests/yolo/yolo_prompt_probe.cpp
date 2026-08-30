// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// TEMPORARY PoC probe for the YOLOE linear-bridge fit: end-to-end detection
// with a chosen text tower. Delete after the Y1 PoC.

#include "aicore/runtime_capi.h"
#include "aicore/yolo_capi.h"

#include <QColor>
#include <QGuiApplication>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QPen>
#include <QFont>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void die(const char* msg) {
    std::fprintf(stderr, "probe: %s\n", msg);
    std::exit(2);
}

std::vector<std::string> splitClasses(const std::string& csv) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= csv.size()) {
        const size_t comma = csv.find(',', start);
        std::string c = csv.substr(start, comma == std::string::npos
                                            ? std::string::npos
                                            : comma - start);
        const size_t first = c.find_first_not_of(" \t");
        if (first == std::string::npos) {
            c.clear();
        } else {
            const size_t last = c.find_last_not_of(" \t");
            c = c.substr(first, last - first + 1);
        }
        if (!c.empty()) out.push_back(c);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

QColor colorFor(size_t i) {
    static const QColor kPalette[10] = {
            QColor(220, 20, 60),  QColor(0, 128, 0),    QColor(255, 165, 0),
            QColor(138, 43, 226), QColor(0, 105, 180), QColor(255, 0, 255),
            QColor(0, 128, 128),  QColor(139, 69, 19),  QColor(255, 105, 180),
            QColor(0, 0, 255),
    };
    return kPalette[i % 10];
}

struct Det {
    std::string name;
    float score = 0.0f;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

void drawAndSave(QImage image, const std::vector<Det>& dets,
                 const char* outPath) {
    if (image.format() != QImage::Format_RGB888) {
        image = image.convertToFormat(QImage::Format_RGB888);
    }
    QPainter p(&image);
    p.setRenderHint(QPainter::Antialiasing, true);
    QFont font = p.font();
    font.setPixelSize(std::max(14, image.height() / 45));
    p.setFont(font);
    for (size_t i = 0; i < dets.size(); ++i) {
        const Det& d = dets[i];
        const QColor color = colorFor(i);
        QPen pen(color);
        pen.setWidth(3);
        p.setPen(pen);
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1));
        const QString label = QStringLiteral("%1 %2")
                                      .arg(QString::fromStdString(d.name))
                                      .arg(d.score, 0, 'f', 2);
        QRect labelRect(static_cast<int>(d.x1),
                        static_cast<int>(d.y1) - font.pixelSize() - 6,
                        std::max(24, label.size() * font.pixelSize()),
                        font.pixelSize() + 6);
        labelRect.moveLeft(std::clamp(
                labelRect.left(), 2,
                std::max(2, image.width() - labelRect.width() - 2)));
        if (labelRect.top() < 2) labelRect.moveTop(static_cast<int>(d.y1) + 2);
        p.fillRect(labelRect.adjusted(0, 0, 4, 2).intersected(image.rect()),
                   color);
        p.setPen(Qt::white);
        p.drawText(labelRect.adjusted(2, 3, -2, -2), label);
    }
    p.end();
    if (!image.save(QString::fromUtf8(outPath), "JPEG", 92)) {
        die("failed to save annotated image");
    }
}

}  // namespace

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QGuiApplication app(argc, argv);
    if (argc < 6) {
        die("usage: probe <model> <image> <text_model|-> <classes_csv|-> "
            "<out.jpg> [conf] [device]");
    }
    const char* modelPath = argv[1];
    const char* imagePath = argv[2];
    const std::string textModel = argv[3];
    const std::string classesCsv = argv[4];
    const char* outPath = argv[5];
    const float conf = argc > 6 ? std::strtof(argv[6], nullptr) : 0.15f;
    const std::string device = argc > 7 ? argv[7] : "auto";

    const QImage input(QString::fromUtf8(imagePath));
    if (input.isNull()) die("failed to load image");
    const QImage rgb = input.convertToFormat(QImage::Format_RGB888);

    QByteArray packed;
    const int rowBytes = rgb.width() * 3;
    if (rgb.bytesPerLine() == rowBytes) {
        packed = QByteArray(reinterpret_cast<const char*>(rgb.constBits()),
                            rowBytes * rgb.height());
    } else {
        packed.resize(rowBytes * rgb.height());
        for (int y = 0; y < rgb.height(); ++y) {
            std::memcpy(packed.data() + y * rowBytes, rgb.constScanLine(y),
                        static_cast<size_t>(rowBytes));
        }
    }
    const uchar* rgbData = reinterpret_cast<const uchar*>(packed.constData());

    std::vector<std::string> classes =
            classesCsv == "-" ? std::vector<std::string>()
                              : splitClasses(classesCsv);
    std::vector<const char*> ptrs;
    ptrs.reserve(classes.size());
    for (const std::string& c : classes) ptrs.push_back(c.c_str());

    aicore_yolo_options* opts = aicore_yolo_options_new();
    if (!opts) die("options alloc failed");
    aicore_yolo_options_set_device(opts, device.c_str());
    aicore_yolo_options_set_conf_thres(opts, conf);
    aicore_yolo_options_set_iou_thres(opts, 0.7f);
    aicore_yolo_options_set_top_k(opts, 100);
    if (!classes.empty())
        aicore_yolo_options_set_classes(opts, ptrs.data(),
                                        static_cast<int32_t>(ptrs.size()));
    if (textModel != "-") {
        aicore_yolo_options_set_text_model(opts, textModel.c_str());
    }

    aicore_yolo_ctx* ctx = aicore_yolo_load_opts(modelPath, opts);
    aicore_yolo_options_free(opts);
    if (!ctx || !aicore_yolo_is_ready(ctx)) {
        const char* err = ctx ? aicore_yolo_last_error(ctx) : "ctx alloc";
        std::fprintf(stderr, "probe: load failed: %s\n", err ? err : "?");
        return 1;
    }
    std::printf("probe: ready device=%s classes=%zu\n",
                aicore_yolo_context_device(ctx), classes.size());
    std::fflush(stdout);

    QImage annotated = rgb;
    std::vector<Det> dets;
    if (std::strcmp(aicore_yolo_context_task(ctx), "segment") == 0) {
        aicore_yolo_segment_result* seg = aicore_yolo_seg_rgb(
                ctx, rgbData, rgb.width(), rgb.height());
        if (!seg) {
            std::fprintf(stderr, "probe: seg failed: %s\n",
                         aicore_yolo_last_error(ctx));
            return 1;
        }
        const int n = aicore_yolo_seg_det_count(seg);
        for (int i = 0; i < n; ++i) {
            const aicore_yolo_detection d = aicore_yolo_seg_det_at(seg, i);
            const char* name = aicore_yolo_seg_det_class_name(seg, i);
            Det det;
            det.name = (name && name[0]) ? name : "?";
            det.score = d.score;
            det.x1 = d.x1;
            det.y1 = d.y1;
            det.x2 = d.x2;
            det.y2 = d.y2;
            dets.push_back(det);
        }
        aicore_yolo_seg_result_free(seg);
    } else {
        char* json = aicore_yolo_detect_rgb_json(ctx, rgbData, rgb.width(),
                                                 rgb.height());
        if (!json) {
            std::fprintf(stderr, "probe: detect failed: %s\n",
                         aicore_yolo_last_error(ctx));
            return 1;
        }
        const QJsonDocument doc = QJsonDocument::fromJson(QByteArray(json));
        aicore_yolo_free_buffer(json);
        if (!doc.isObject()) die("bad detect JSON");
        const QJsonArray arr =
                doc.object().value(QStringLiteral("detections")).toArray();
        for (const QJsonValue& v : arr) {
            const QJsonObject o = v.toObject();
            Det det;
            det.name = o.value(QStringLiteral("class_name"))
                               .toString()
                               .toStdString();
            det.score = static_cast<float>(
                    o.value(QStringLiteral("score")).toDouble());
            const QJsonArray box = o.value(QStringLiteral("box")).toArray();
            if (box.size() == 4) {
                det.x1 = static_cast<float>(box.at(0).toDouble());
                det.y1 = static_cast<float>(box.at(1).toDouble());
                det.x2 = static_cast<float>(box.at(2).toDouble());
                det.y2 = static_cast<float>(box.at(3).toDouble());
            }
            dets.push_back(det);
        }
    }

    std::printf("probe: %zu detection(s)\n", dets.size());
    std::fflush(stdout);
    for (const Det& d : dets) {
        std::printf("  %s %.3f  box=[%.0f,%.0f,%.0f,%.0f]\n", d.name.c_str(),
                    d.score, d.x1, d.y1, d.x2, d.y2);
    }
    std::fflush(stdout);
    drawAndSave(annotated, dets, outPath);
    aicore_yolo_free(ctx);
    return 0;
}

