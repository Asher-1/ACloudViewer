// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VLXSeekModelCatalog.h"

#include <QtCompat.h>

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QObject>

#ifdef AICore_ENABLED
#include <aicore/ocr_capi.h>
#include <aicore/vlx_seek_capi.h>
#endif

namespace VLXSeekHelpers {

QVector<VLXSeekModelEntry> catalogModels() {
    QVector<VLXSeekModelEntry> out;
#ifdef AICore_ENABLED
    const int count = aicore_vlx_seek_model_count();
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        const aicore_vlx_seek_model_entry* e = aicore_vlx_seek_model_at(i);
        if (!e) continue;
        VLXSeekModelEntry row;
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
    return aicore_vlx_seek_model_default_index();
#else
    return -1;
#endif
}

bool findModelByFilename(const QString& filename, VLXSeekModelEntry* out) {
    if (filename.isEmpty()) return false;
    for (const VLXSeekModelEntry& row : catalogModels()) {
        if (row.filename == filename) {
            if (out) *out = row;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_vlx_seek_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_vlx_seek_free_buffer(dir);
        return out;
    }
#endif
    return QString();
}

QStringList roleDisplayLabels(int role) {
    QStringList labels;
    for (const VLXSeekModelEntry& row : catalogModels()) {
        if (row.role != role) continue;
        labels.push_back(row.displayName);
    }
    return labels;
}

QJsonArray catalogJson() {
    QJsonArray rows;
    for (const VLXSeekModelEntry& row : catalogModels()) {
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

QVector<int32_t> parseTargetIndexes(const QString& text, QString* error) {
    QVector<int32_t> out;
    // Plain QChar split with the default KeepEmptyParts: identical on
    // Qt5/Qt6 (the SplitBehavior enum diverges across versions).
    const QStringList parts = text.split(QLatin1Char(','));
    for (const QString& part : parts) {
        const QString trimmed = part.trimmed();
        if (trimmed.isEmpty()) continue;
        bool ok = false;
        const int value = trimmed.toInt(&ok);
        if (!ok || value < 0) {
            if (error) {
                *error = QObject::tr("Invalid target region index: %1")
                                 .arg(trimmed);
            }
            return {};
        }
        out.push_back(static_cast<int32_t>(value));
    }
    return out;
}

QVector<float> parseExternalBoxes(const QString& path, QString* error) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error)
            *error = QObject::tr("Cannot open external boxes file: %1")
                             .arg(path);
        return {};
    }
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    const QJsonArray rows = doc.array();
    QVector<float> boxes;
    boxes.reserve(rows.size() * 4);
    for (const QJsonValue& row : rows) {
        const QJsonArray coords = row.toArray();
        if (coords.size() != 4) {
            if (error)
                *error = QObject::tr(
                                 "External box must hold exactly "
                                 "[x1,y1,x2,y2]: %1")
                                 .arg(path);
            return {};
        }
        for (const QJsonValue& c : coords) {
            if (!c.isDouble()) {
                if (error)
                    *error = QObject::tr(
                                     "External box coordinates must be "
                                     "numbers: %1")
                                     .arg(path);
                return {};
            }
            boxes.push_back(static_cast<float>(c.toDouble()));
        }
    }
    // The C API cap mirrors upstream (--bbox-list, <= 100 proposals).
    constexpr int kMaxBoxes = 100;
    if (boxes.size() / 4 > kMaxBoxes) {
        if (error)
            *error = QObject::tr("External box list exceeds %1 entries: %2")
                             .arg(kMaxBoxes)
                             .arg(path);
        return {};
    }
    return boxes;
}

QString ocrModelDownloadUrl(const QString& filename) {
#ifdef AICore_ENABLED
    if (filename.isEmpty()) return QString();
    const QByteArray utf8 = filename.toUtf8();
    const aicore_ocr_model_entry* entry =
            aicore_ocr_model_by_filename(utf8.constData());
    if (entry && entry->download_url)
        return QString::fromUtf8(entry->download_url);
#endif
    return QString();
}

OcrProposerModels ocrProposerModels() {
    OcrProposerModels out;
#ifdef AICore_ENABLED
    // The OCR preset owns the det/rec pairing; "ppocr-v4-ch" is the default
    // PP-OCRv4 Chinese recipe. q8_0 is the catalog's best size trade and a
    // manifest-validated quantization row. Everything resolves through the
    // public OCR catalog API — the plugin keeps no model table of its own.
    const aicore_ocr_preset_entry* preset = nullptr;
    for (int i = 0; i < aicore_ocr_preset_count(); ++i) {
        const aicore_ocr_preset_entry* p = aicore_ocr_preset_at(i);
        if (p && p->id &&
            QLatin1String(p->id) == QLatin1String("ppocr-v4-ch")) {
            preset = p;
            break;
        }
    }
    if (!preset) return out;
    out.detFileName = QStringLiteral("%1-%2.gguf")
                              .arg(QString::fromUtf8(preset->det_stem),
                                   QStringLiteral("q8_0"));
    out.recFileName = QStringLiteral("%1-%2.gguf")
                              .arg(QString::fromUtf8(preset->rec_stem),
                                   QStringLiteral("q8_0"));
    char* dir = aicore_ocr_model_cache_dir();
    if (dir) {
        out.cacheDir = QString::fromUtf8(dir);
        aicore_ocr_free_buffer(dir);
    }
#endif
    return out;
}

bool ocrQuadsToBoxes(const float* quads, int quadCount, QVector<float>* boxes) {
    if (!quads || quadCount <= 0 || !boxes) return false;
    boxes->clear();
    // The VLX-Seek C API accepts at most 100 proposals (the built-in
    // WeDetect-Uni proposer supplies exactly 100); det output beyond the cap
    // is dropped instead of failing the run.
    constexpr int kMaxProposals = 100;
    for (int i = 0; i < quadCount && boxes->size() / 4 < kMaxProposals; ++i) {
        const float* quad = quads + size_t(i) * 8;
        float x1 = quad[0], y1 = quad[1], x2 = quad[0], y2 = quad[1];
        for (int corner = 1; corner < 4; ++corner) {
            x1 = qMin(x1, quad[corner * 2 + 0]);
            y1 = qMin(y1, quad[corner * 2 + 1]);
            x2 = qMax(x2, quad[corner * 2 + 0]);
            y2 = qMax(y2, quad[corner * 2 + 1]);
        }
        if (x2 - x1 <= 0.0f || y2 - y1 <= 0.0f) continue;  // degenerate
        *boxes << x1 << y1 << x2 << y2;
    }
    return !boxes->isEmpty();
}

namespace {

// Locates the first <tag>...</tag> pair of fixed literal tags; \p content
// receives the trimmed inner text. Returns false when the pair is absent.
// (The containers are parsed with indexOf, not a regex engine: the QRegExp
// default pattern syntax rejects lazy quantifiers and QRegularExpression
// is a different engine, while literal-tag location is engine-independent.)
bool extractLiteralBlock(const QString& text,
                         const char* tag,
                         QString* content) {
    const QString open = QStringLiteral("<%1>").arg(QLatin1String(tag));
    const QString close = QStringLiteral("</%1>").arg(QLatin1String(tag));
    const int begin = text.indexOf(open);
    if (begin < 0) return false;
    const int end = text.indexOf(close, begin + open.length());
    if (end < 0) return false;
    if (content) {
        *content = text.mid(begin + open.length(), end - begin - open.length())
                           .trimmed();
    }
    return true;
}

// Removes every <tag>...</tag> block (fixed literal tags, content dropped).
QString removeLiteralBlocks(QString text, const char* tag) {
    const QString open = QStringLiteral("<%1>").arg(QLatin1String(tag));
    const QString close = QStringLiteral("</%1>").arg(QLatin1String(tag));
    int begin = text.indexOf(open);
    while (begin >= 0) {
        const int end = text.indexOf(close, begin + open.length());
        if (end < 0) break;
        text.remove(begin, end + close.length() - begin);
        begin = text.indexOf(open, begin);
    }
    return text;
}

}  // namespace

QString stripAnswerTags(const QString& answer, bool keepReasoning) {
    if (answer.isEmpty()) return answer;
    QString result = answer;
    // <answer> extraction: the <think> block is kept before the answer
    // content (console lines) or dropped (compact status lines).
    QString answerContent;
    if (extractLiteralBlock(result, "answer", &answerContent)) {
        if (keepReasoning) {
            QString thinkContent;
            if (extractLiteralBlock(result, "think", &thinkContent)) {
                result = QStringLiteral("<think>") + thinkContent +
                         QStringLiteral("</think>\n\n");
            } else {
                result = QString();
            }
            result += answerContent;
        } else {
            result = answerContent;
        }
    } else {
        // Lazy-free removal: every <think> block goes, the content between
        // distinct blocks survives.
        result = removeLiteralBlocks(result, "think");
    }
    // Replace <ground>text</ground> with just the text.
    result.remove(QStringLiteral("<ground>"));
    result.remove(QStringLiteral("</ground>"));
    // Replace EACH <objects>...</objects> group with its own [objN, ...]
    // list. The previous greedy <objects>(.*)</objects> handling spanned
    // from the first group to the last one and swallowed every interleaved
    // <ground>text</ground>: a region_ocr answer over two or more targets
    // displayed only the first text and merged all obj references into one
    // bracket, contradicting the annotated image. The obj scan mirrors the
    // runtime's remap_obj_tokens tokenizer.
    QtCompatRegExpWrapper objRe(QStringLiteral("<obj(\\d+)>"));
    QString cleaned;
    cleaned.reserve(result.size());
    int cursor = 0;
    // A bracket replacement may butt against the next ground text
    // ("[obj0]RESORTCASINO"); keep one separating space there.
    bool appendedBracket = false;
    const auto appendSegment = [&cleaned,
                                &appendedBracket](const QString& segment) {
        if (appendedBracket && !segment.isEmpty() &&
            !segment.front().isSpace()) {
            cleaned += QLatin1Char(' ');
        }
        appendedBracket = false;
        cleaned += segment;
    };
    forever {
        const int begin = result.indexOf(QStringLiteral("<objects>"), cursor);
        if (begin < 0) break;
        const int end = result.indexOf(QStringLiteral("</objects>"),
                                       begin + int(qstrlen("<objects>")));
        if (end < 0) break;
        appendSegment(result.mid(cursor, begin - cursor));
        const QString group =
                result.mid(begin + int(qstrlen("<objects>")),
                           end - begin - int(qstrlen("<objects>")));
        QStringList objList;
        int objPos = 0;
        while ((objPos = objRe.indexIn(group, objPos)) != -1) {
            objList << QStringLiteral("obj%1").arg(objRe.cap(1));
            objPos += objRe.matchedLength();
        }
        if (!objList.isEmpty()) {
            cleaned += QStringLiteral(" [%1]").arg(
                    objList.join(QStringLiteral(", ")));
            appendedBracket = true;
        }
        cursor = end + int(qstrlen("</objects>"));
    }
    appendSegment(result.mid(cursor));
    result = cleaned;
    // Clean up any remaining tags
    result.remove(QtCompatRegExpWrapper(QStringLiteral("</?[a-z]+>")));
    return result.trimmed();
}

}  // namespace VLXSeekHelpers
