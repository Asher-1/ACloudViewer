// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "WeVisOneModelCatalog.h"

#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QObject>
#include <QRectF>
#include <QRegularExpression>
#include <algorithm>
#include <cmath>

#ifdef AICore_ENABLED
#include <aicore/wevisone_capi.h>
#endif

namespace WeVisOneHelpers {

QVector<WeVisOneModelEntry> catalogModels() {
    QVector<WeVisOneModelEntry> out;
#ifdef AICore_ENABLED
    const int count = aicore_wevisone_model_count();
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        const aicore_wevisone_model_entry* e = aicore_wevisone_model_at(i);
        if (!e) continue;
        WeVisOneModelEntry row;
        row.filename = QString::fromUtf8(e->filename);
        row.downloadUrl = QString::fromUtf8(e->download_url);
        row.displayName = QString::fromUtf8(e->display_name);
        row.family = QString::fromUtf8(e->family);
        row.role = e->role;
        row.quantization = QString::fromUtf8(e->quantization);
        row.sha256 = QString::fromUtf8(e->sha256);
        row.sizeBytes = e->size_bytes;
        out.push_back(row);
    }
#endif
    return out;
}

int catalogDefaultIndex() {
#ifdef AICore_ENABLED
    return aicore_wevisone_model_default_index();
#else
    return -1;
#endif
}

bool findModelByFilename(const QString& filename, WeVisOneModelEntry* out) {
    if (filename.isEmpty()) return false;
    for (const WeVisOneModelEntry& row : catalogModels()) {
        if (row.filename == filename) {
            if (out) *out = row;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_wevisone_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_wevisone_free_buffer(dir);
        return out;
    }
#endif
    return QString();
}

QStringList catalogFamilies() {
    QStringList families;
    for (const WeVisOneModelEntry& row : catalogModels()) {
        if (!row.family.isEmpty() && !families.contains(row.family))
            families.push_back(row.family);
    }
    return families;
}

QJsonArray catalogJson() {
    QJsonArray rows;
    for (const WeVisOneModelEntry& row : catalogModels()) {
        rows.append(QJsonObject{
                {"filename", row.filename},
                {"display_name", row.displayName},
                {"family", row.family},
                {"role", row.role},
                {"quantization", row.quantization},
                {"sha256", row.sha256},
                {"size_bytes", row.sizeBytes},
        });
    }
    return rows;
}

QVector<float> parseRefBoxes(const QString& spec, QString* error) {
    QVector<float> out;
    if (spec.trimmed().isEmpty()) return out;
    // Upstream wevisone-cli parse_ref_boxes: semicolon-separated boxes of
    // four comma-separated pixel coordinates. Plain QChar split with the
    // default KeepEmptyParts: identical on Qt5/Qt6.
    const QStringList boxes = spec.split(QLatin1Char(';'));
    for (const QString& box : boxes) {
        if (box.trimmed().isEmpty()) continue;
        const QStringList coords = box.split(QLatin1Char(','));
        QVector<float> quad;
        quad.reserve(4);
        for (const QString& coord : coords) {
            const QString trimmed = coord.trimmed();
            if (trimmed.isEmpty()) continue;
            bool ok = false;
            const float value = trimmed.toFloat(&ok);
            if (!ok) {
                if (error) {
                    *error = QObject::tr("Invalid reference-box coordinate: %1")
                                     .arg(trimmed);
                }
                return {};
            }
            quad.push_back(value);
        }
        if (quad.size() != 4) {
            if (error) {
                *error = QObject::tr(
                                 "Reference box must hold exactly "
                                 "x1,y1,x2,y2: %1")
                                 .arg(box.trimmed());
            }
            return {};
        }
        out += quad;
    }
    return out;
}

QStringList taskNames() {
    // The released prompt family (upstream prompts.py / wevisone-cli).
    return {QStringLiteral("detection"),     QStringLiteral("referring"),
            QStringLiteral("point"),         QStringLiteral("count_box"),
            QStringLiteral("count_point"),   QStringLiteral("ocr"),
            QStringLiteral("ocr_grounding"), QStringLiteral("segmentation"),
            QStringLiteral("matting"),       QStringLiteral("visual_prompt"),
            QStringLiteral("freeform")};
}

QString formatAnswerForLog(const QString& answer, const QSize& imageSize) {
    // Plain-text answers (ocr / freeform / count) carry no <ref>/<box>
    // structure.
    if (!answer.contains(QStringLiteral("<ref>")) &&
        !answer.contains(QStringLiteral("<box>")) &&
        !answer.contains(QStringLiteral("<point>"))) {
        return answer.trimmed();
    }
    // Convert the model's 0-1000 tokens to original-image pixels when the
    // size is known (same frame as the exported annotated image); otherwise
    // keep the verbatim normalized values.
    const bool px = imageSize.width() > 0 && imageSize.height() > 0;
    const auto toPx = [&](int n, int dim) {
        return px ? qRound(n / 1000.0 * dim) : n;
    };
    // One <ref>label</ref> followed by its <box>...</box> or <point>...</point>
    // block; a block may hold many comma-separated coordinate tuples.
    const QRegularExpression refRe(
            QStringLiteral("<ref>([^<]*)</ref>\\s*"
                           "(?:<box>(.*?)</box>|<point>(.*?)</point>)"),
            QRegularExpression::DotMatchesEverythingOption);
    const QRegularExpression numRe(QStringLiteral("<(\\d+)>"));
    QStringList labels;
    QStringList bodies;
    int firstStart = -1;
    int lastEnd = 0;
    auto it = refRe.globalMatch(answer);
    while (it.hasNext()) {
        const auto m = it.next();
        if (firstStart < 0) firstStart = m.capturedStart(0);
        lastEnd = m.capturedEnd(0);
        const bool isPoint = !m.captured(3).isNull();
        const QString body = isPoint ? m.captured(3) : m.captured(2);
        QVector<int> nums;
        auto nit = numRe.globalMatch(body);
        while (nit.hasNext()) nums.push_back(nit.next().captured(1).toInt());
        const int stride = isPoint ? 2 : 4;
        QStringList tuples;
        for (int o = 0; o + stride <= nums.size(); o += stride) {
            QStringList c;
            for (int k = 0; k < stride; ++k) {
                const int dim =
                        (k % 2 == 0) ? imageSize.width() : imageSize.height();
                c << QString::number(toPx(nums[o + k], dim));
            }
            tuples << (isPoint ? QStringLiteral("(") : QStringLiteral("[")) +
                              c.join(QStringLiteral(",")) +
                              (isPoint ? QStringLiteral(")")
                                       : QStringLiteral("]"));
        }
        labels << m.captured(1).trimmed();
        bodies << tuples.join(QStringLiteral(", "));
    }
    if (labels.isEmpty()) return answer.trimmed();
    // One aligned line per <ref> group: labels padded to the widest one so
    // the coordinate columns line up for human scanning. Free text before
    // the first / after the last group (e.g. the counting tail
    // "Total number: 4.") rides along as its own line, never dropped.
    int width = 0;
    for (const QString& l : labels) width = qMax(width, l.size());
    QStringList lines;
    const QString lead = answer.left(firstStart).trimmed();
    if (!lead.isEmpty()) lines << lead;
    for (int i = 0; i < labels.size(); ++i) {
        lines << labels[i].leftJustified(width) + QStringLiteral(": ") +
                         bodies[i];
    }
    const QString tail = answer.mid(lastEnd).trimmed();
    if (!tail.isEmpty()) {
        // A punctuation-only tail (e.g. the sentence-final "." after the
        // last grounding group in a freeform answer) belongs to the last
        // line, not a lonely orphan line.
        bool punctuationOnly = true;
        for (const QChar& c : tail) {
            if (c.isLetterOrNumber() || c.isSpace()) {
                punctuationOnly = false;
                break;
            }
        }
        if (punctuationOnly && !lines.isEmpty()) {
            lines.last() += tail;
        } else {
            lines << tail;
        }
    }
    return lines.join(QStringLiteral("\n"));
}

int statedTotalCount(const QString& answer) {
    // Upstream demo/app.py _COUNT_RE: the model's stated total is the
    // authoritative count for count_box/count_point; the box list is the
    // localization evidence and may be shorter than the stated total.
    const QRegularExpression re(QStringLiteral("Total number:\\s*(\\d+)"),
                                QRegularExpression::CaseInsensitiveOption);
    const auto m = re.match(answer);
    return m.hasMatch() ? m.captured(1).toInt() : -1;
}

bool readDemoSample(const QString& manifestPath,
                    const QString& task,
                    WeVisOneDemoSample* out) {
    QFile file(manifestPath);
    if (!file.open(QIODevice::ReadOnly)) return false;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isObject()) return false;
    const QJsonArray samples =
            doc.object().value(QStringLiteral("samples")).toArray();
    // Selection policy per task, in priority order:
    // 1. chess.jpg when the manifest offers it for this task — the classic
    //    chessboard case (referring "the white king", count_box "chess
    //    pieces", count_point "white chess pieces") is the canonical
    //    multi-object demo and must be reachable from the first fill.
    // 2. Otherwise the first WeVisOne-specific case-study original (an
    //    image outside the seven shared official demo photos), so tasks
    //    without a chess sample still showcase WeVisOne's own data.
    // 3. Otherwise the task's first manifest sample.
    static const QStringList kOfficial = {
            QStringLiteral("livingroom.jpg"), QStringLiteral("dogs.jpg"),
            QStringLiteral("parasol.jpg"),    QStringLiteral("seabreeze.jpg"),
            QStringLiteral("boys.jpg"),       QStringLiteral("chess.jpg"),
            QStringLiteral("coffee_sign.jpg")};
    QJsonObject first, preferred, chess;
    bool hasFirst = false, hasPreferred = false, hasChess = false;
    for (const QJsonValue& v : samples) {
        const QJsonObject o = v.toObject();
        if (o.value(QStringLiteral("task")).toString() != task) continue;
        const QString image = o.value(QStringLiteral("image")).toString();
        if (!hasFirst) {
            first = o;
            hasFirst = true;
        }
        if (!hasChess && image == QStringLiteral("chess.jpg")) {
            chess = o;
            hasChess = true;
        }
        if (!hasPreferred && !kOfficial.contains(image)) {
            preferred = o;
            hasPreferred = true;
        }
    }
    if (!hasFirst) return false;
    const QJsonObject& sel =
            hasChess ? chess : (hasPreferred ? preferred : first);
    out->image = sel.value(QStringLiteral("image")).toString();
    out->query = sel.value(QStringLiteral("query")).toString();
    QStringList boxes;
    for (const QJsonValue& b :
         sel.value(QStringLiteral("ref_boxes")).toArray()) {
        const QJsonArray q = b.toArray();
        if (q.size() != 4) continue;
        QStringList c;
        for (int i = 0; i < 4; ++i) c << QString::number(q[i].toDouble());
        boxes << c.join(QStringLiteral(","));
    }
    out->refBoxes = boxes.join(QStringLiteral(";"));
    return !out->image.isEmpty();
}

int pointMarkerRadius(int imageWidth) {
    // visualization.py: radius = max(6, round(image.width / 85)). An
    // integer width divided by 85 is never an exact .5, so lround matches
    // python round() on every reachable input.
    return std::max(6, static_cast<int>(std::lround(imageWidth / 85.0)));
}

int pointMarkerOutline(int radius) {
    // visualization.py: outline_thickness = max(2, round(radius / 3)).
    return std::max(2, static_cast<int>(std::lround(radius / 3.0)));
}

QImage annotateResult(const QImage& input,
                      const QVector<QRectF>& boxes,
                      const QVector<float>& matte,
                      int matteWidth,
                      int matteHeight) {
    QImage canvas = input.convertToFormat(QImage::Format_RGB888);
    const int w = canvas.width();
    const int h = canvas.height();
    const int stride = canvas.bytesPerLine();
    uchar* bits = canvas.bits();
    const auto blendPx = [bits, stride, w, h](int x, int y, int r, int g, int b,
                                              float alpha) {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        uchar* p = bits + static_cast<size_t>(y) * stride + x * 3;
        p[0] = static_cast<uchar>(
                std::round(p[0] * (1.0f - alpha) + r * alpha));
        p[1] = static_cast<uchar>(
                std::round(p[1] * (1.0f - alpha) + g * alpha));
        p[2] = static_cast<uchar>(
                std::round(p[2] * (1.0f - alpha) + b * alpha));
    };
    if (!matte.isEmpty() && matteWidth > 0 && matteHeight > 0) {
        // O(w*h): one nearest-neighbor matte sample per canvas pixel,
        // exactly the export helper's mapping (my = y*mh/h, mx = x*mw/w).
        for (int y = 0; y < h; ++y) {
            const int my = y * matteHeight / h;
            for (int x = 0; x < w; ++x) {
                const int mx = x * matteWidth / w;
                const float a =
                        matte[static_cast<size_t>(my) * matteWidth + mx];
                if (a > 0.01f) blendPx(x, y, 0, 220, 0, a * 0.55f);
            }
        }
    }
    const int radius = pointMarkerRadius(w);
    const int outline = pointMarkerOutline(radius);
    // cv2.circle semantics of sv.DotAnnotator: filled dot of `radius`,
    // then a white ring of `outline` thickness centered on the radius
    // circle -> fill survives inside radius - outline/2, white covers up
    // to radius + outline/2. Squared integer bounds (4*d^2 vs (2r±t)^2)
    // keep the test exact without float sqrt.
    const int outer = 2 * radius + outline;
    const int inner = 2 * radius - outline;
    const int reach = radius + outline;
    for (const QRectF& box : boxes) {
        if (box.width() < 1.0 && box.height() < 1.0) {
            // Degenerate point payload [x,y,x,y]: the export helper's
            // width-3 rectangle collapses to a 3x3 speck, invisible on
            // large images. Draw the official python overlay dot instead.
            const int cx = static_cast<int>(box.left());
            const int cy = static_cast<int>(box.top());
            for (int y = cy - reach; y <= cy + reach; ++y) {
                for (int x = cx - reach; x <= cx + reach; ++x) {
                    const int dx = x - cx, dy = y - cy;
                    const int d4 = 4 * (dx * dx + dy * dy);
                    if (d4 > outer * outer) continue;
                    if (d4 <= inner * inner) {
                        blendPx(x, y, 255, 0, 0, 1.0f);
                    } else {
                        blendPx(x, y, 255, 255, 255, 1.0f);
                    }
                }
            }
            continue;
        }
        const int x0 = std::max(0, static_cast<int>(box.left()));
        const int y0 = std::max(0, static_cast<int>(box.top()));
        const int x1 = std::min(w - 1, static_cast<int>(box.right()));
        const int y1 = std::min(h - 1, static_cast<int>(box.bottom()));
        for (int d = -1; d <= 1; ++d) {
            for (int x = x0; x <= x1; ++x) {
                blendPx(x, y0 + d, 255, 0, 0, 1.0f);
                blendPx(x, y1 + d, 255, 0, 0, 1.0f);
            }
            for (int y = y0; y <= y1; ++y) {
                blendPx(x0 + d, y, 255, 0, 0, 1.0f);
                blendPx(x1 + d, y, 255, 0, 0, 1.0f);
            }
        }
    }
    return canvas;
}

}  // namespace WeVisOneHelpers