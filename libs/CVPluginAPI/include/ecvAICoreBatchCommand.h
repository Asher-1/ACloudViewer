// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Shared skeleton of the headless AICore batch commands
// (`ACloudViewer -SILENT -<KEYWORD> KEY VALUE ...`).
//
// Every AI plugin's batch command is the same unit:
//
//     one model load + options  x  N input items  ->  N result rows
//
// Only the item shape (image / view set), the option table and the result
// serializer are task-specific. This header owns the rest so all commands
// share one argument grammar, one input-discovery rule, one model
// resolution/download path and one manifest schema:
//
//   * Arguments are `KEY VALUE` pairs (keys case-insensitive). Parsing stops
//     at the next `-COMMAND` token, so batch commands chain with the core
//     commands. `HELP` (alone) prints the option schema and exits.
//   * Inputs: IMAGE (repeatable) / INPUT_DIR (+RECURSIVE, PATTERN) /
//     INPUT_LIST, natural-sorted and de-duplicated. Set-shaped tasks group
//     them with SUBDIRS 1 (one set per immediate sub-directory).
//   * Models: an existing path wins; otherwise the token is a catalog
//     filename under the task cache dir. DOWNLOAD 1 fetches a missing
//     catalog file through the shared ecvModelDownloader (URL + SHA-256 are
//     read from the AICore catalog by the caller, never hard-coded here).
//   * Output: per-item files under OUT_DIR plus one RESULT_JSON manifest
//     (schema "acloudviewer.aicore.batch/1"). The command fails (non-zero
//     exit) when any item failed; CONTINUE_ON_ERROR 0 stops at the first.
//
// The option tables are plain `{"KEY", "type", "default", "help"}` rows on
// purpose: agent-integration/tests scans them to keep the CLI harness
// registry in lock-step with the binary (see the acloudviewer-aicore-plugin
// skill). Header-only; Qt-aware; never exposes task types.
// ----------------------------------------------------------------------------

#pragma once

#include <QCollator>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImage>
#include <QImageReader>
#include <QImageWriter>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QSet>
#include <QStringList>
#include <QTextStream>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <vector>

#include "aicore/model_catalog_capi.h"
#include "aicore/pipeline_timing.h"
#include "ecvAssetIntegrity.h"
#include "ecvCommandLineInterface.h"
#include "ecvModelDownloader.h"

namespace ecvAICoreBatch {

/** One accepted `KEY VALUE` option. `type` is informative (int, float,
 *  bool, string, path, list, choice:a|b|c) and drives the HELP schema. */
struct OptionSpec {
    const char* key;
    const char* type;
    const char* defaultValue;
    const char* help;
};

using OptionTable = std::vector<OptionSpec>;

/** Options every batch command accepts. */
inline const OptionTable& commonOptions() {
    static const OptionTable specs = {
            {"IMAGE", "list", "", "Input file; repeatable"},
            {"INPUT_DIR", "path", "", "Directory scanned for inputs"},
            {"INPUT_LIST", "path", "", "Text file: one input path per line"},
            {"RECURSIVE", "bool", "0", "Recurse into INPUT_DIR"},
            {"PATTERN", "string", "",
             "Glob filter, ';'-separated (default: readable image types)"},
            {"OUT_DIR", "path", "", "Output directory (required)"},
            {"RESULT_JSON", "path", "",
             "Manifest path (default OUT_DIR/_<task>_result.json)"},
            {"MODEL", "string", "", "GGUF path or catalog filename"},
            {"DEVICE", "string", "auto", "auto|cpu|cuda|vulkan|metal[:N]"},
            {"THREADS", "int", "0", "CPU threads (0 = backend default)"},
            {"DOWNLOAD", "bool", "0", "Download missing catalog models"},
            {"LIST_MODELS", "bool", "0",
             "Print the task model catalog + cache state and exit"},
            {"CONTINUE_ON_ERROR", "bool", "1",
             "Keep going after a failed item (exit code still non-zero)"},
    };
    return specs;
}

/** Extra options of set-shaped tasks (one item = several images). */
inline const OptionTable& setOptions() {
    static const OptionTable specs = {
            {"SUBDIRS", "bool", "0",
             "Each immediate sub-directory of INPUT_DIR is one set"},
    };
    return specs;
}

// ---------------------------------------------------------------------------
//  Argument walk
// ---------------------------------------------------------------------------

inline bool parseBool(const QString& value, bool* out) {
    const QString v = value.trimmed().toLower();
    if (v == "1" || v == "true" || v == "on" || v == "yes") {
        *out = true;
        return true;
    }
    if (v == "0" || v == "false" || v == "off" || v == "no") {
        *out = false;
        return true;
    }
    return false;
}

/** Parsed `KEY VALUE` pairs. Typed getters record the first conversion
 *  error; check ok() once after reading every option. */
class Args {
public:
    bool parse(ccCommandLineInterface& cmd,
               const QString& keyword,
               const std::vector<const OptionTable*>& tables) {
        QSet<QString> allowed;
        for (const OptionTable* table : tables) {
            for (const OptionSpec& spec : *table) {
                allowed.insert(QString::fromLatin1(spec.key));
            }
        }
        QStringList& tokens = cmd.arguments();
        while (!tokens.isEmpty()) {
            const QString& head = tokens.front();
            // `-WORD` starts the next command (values are taken verbatim
            // below, so negative numbers never reach this check).
            if (head.size() > 1 && head[0] == '-' && head[1].isLetter()) {
                break;
            }
            const QString key = tokens.takeFirst().toUpper();
            if (key == QStringLiteral("HELP")) {
                m_help = true;
                continue;
            }
            if (!allowed.contains(key)) {
                QStringList keys = allowed.values();
                keys.sort();
                return fail(QStringLiteral("Unknown argument after -%1: %2 "
                                           "(valid: %3)")
                                    .arg(keyword, key, keys.join(' ')));
            }
            if (tokens.isEmpty()) {
                return fail(QStringLiteral("%1 requires a value").arg(key));
            }
            m_values[key].append(tokens.takeFirst());
        }
        return true;
    }

    bool helpRequested() const { return m_help; }
    bool ok() const { return m_error.isEmpty(); }
    const QString& error() const { return m_error; }

    bool has(const char* key) const {
        return m_values.contains(QString::fromLatin1(key));
    }
    /** Last value of a key (later occurrences override earlier ones). */
    QString str(const char* key, const QString& def = QString()) const {
        const auto it = m_values.constFind(QString::fromLatin1(key));
        return it == m_values.constEnd() ? def : it->last();
    }
    /** Every occurrence verbatim (paths may legitimately contain ','). */
    QStringList list(const char* key) const {
        return m_values.value(QString::fromLatin1(key));
    }
    /** Every occurrence, each further split on ',' (ids, names, ranges):
     *  `CLASSES 0,2 CLASSES 5` -> {0, 2, 5}. */
    QStringList tokens(const char* key) const {
        QStringList out;
        for (const QString& value : list(key)) {
            for (const QString& t : value.split(',')) {
                if (!t.trimmed().isEmpty()) out << t.trimmed();
            }
        }
        return out;
    }
    int toInt(const char* key, int def) {
        if (!has(key)) return def;
        bool ok = false;
        const int v = str(key).toInt(&ok);
        if (!ok) fail(QStringLiteral("%1 expects an integer").arg(key));
        return ok ? v : def;
    }
    float toFloat(const char* key, float def) {
        if (!has(key)) return def;
        bool ok = false;
        const float v = str(key).toFloat(&ok);
        if (!ok) fail(QStringLiteral("%1 expects a number").arg(key));
        return ok ? v : def;
    }
    bool toBool(const char* key, bool def) {
        if (!has(key)) return def;
        bool v = def;
        if (!parseBool(str(key), &v)) {
            fail(QStringLiteral("%1 expects 0|1").arg(key));
        }
        return v;
    }
    /** Value restricted to a closed set (compared case-insensitively;
     *  returned lower-case). */
    QString choice(const char* key,
                   const QString& def,
                   const QStringList& choices) {
        const QString v = str(key, def).toLower();
        if (!choices.contains(v)) {
            fail(QStringLiteral("%1 must be one of %2")
                         .arg(key, choices.join('|')));
            return def;
        }
        return v;
    }
    bool fail(const QString& message) {
        if (m_error.isEmpty()) m_error = message;
        return false;
    }

    QJsonObject toJson() const {
        QJsonObject out;
        for (auto it = m_values.constBegin(); it != m_values.constEnd(); ++it) {
            if (it.key() == QStringLiteral("IMAGE")) continue;  // in items
            if (it->size() == 1) {
                out[it.key()] = it->first();
            } else {
                out[it.key()] = QJsonArray::fromStringList(*it);
            }
        }
        return out;
    }

private:
    QHash<QString, QStringList> m_values;
    QString m_error;
    bool m_help = false;
};

inline QJsonObject schemaJson(const QString& keyword,
                              const QString& task,
                              const QString& summary,
                              const std::vector<const OptionTable*>& tables) {
    QJsonArray options;
    for (const OptionTable* table : tables) {
        for (const OptionSpec& spec : *table) {
            options.append(QJsonObject{{"key", spec.key},
                                       {"type", spec.type},
                                       {"default", spec.defaultValue},
                                       {"help", spec.help}});
        }
    }
    return QJsonObject{{"schema", "acloudviewer.aicore.batch/1"},
                       {"keyword", keyword},
                       {"task", task},
                       {"summary", summary},
                       {"options", options}};
}

// ---------------------------------------------------------------------------
//  Image / JSON I/O
// ---------------------------------------------------------------------------

/** Decodes \p path (no EXIF auto-rotation, same as the GUI QImage path)
 *  and converts to \p format when given and different. */
inline QImage loadImage(const QString& path,
                        QImage::Format format,
                        QString* error) {
    QImageReader reader(path);
    QImage image = reader.read();
    if (image.isNull()) {
        *error = QStringLiteral("Cannot decode %1: %2")
                         .arg(path, reader.errorString());
        return image;
    }
    if (format != QImage::Format_Invalid && image.format() != format) {
        image = image.convertToFormat(format);
    }
    return image;
}

inline bool saveImage(const QImage& image,
                      const QString& path,
                      QString* error) {
    QImageWriter writer(path);
    if (!writer.write(image)) {
        *error = QStringLiteral("Cannot write %1: %2")
                         .arg(path, writer.errorString());
        return false;
    }
    return true;
}

inline bool saveBytes(const QByteArray& bytes,
                      const QString& path,
                      QString* error) {
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() ||
        !file.commit()) {
        *error = QStringLiteral("Cannot write %1").arg(path);
        return false;
    }
    return true;
}

/** Row-major float32 array as NumPy .npy v1.0 (numpy.load-able, no extra
 *  dependency on either side). Lossless dense maps (depth, confidence) go
 *  here; PNGs are only previews. Memory: one header + h*w*c*4 bytes copy. */
inline bool saveNpyFloat32(const float* data,
                           int height,
                           int width,
                           int channels,
                           const QString& path,
                           QString* error) {
    static_assert(Q_BYTE_ORDER == Q_LITTLE_ENDIAN,
                  "'<f4' descriptor assumes a little-endian host");
    const QString shape =
            channels > 1 ? QStringLiteral("(%1, %2, %3)")
                                   .arg(height)
                                   .arg(width)
                                   .arg(channels)
                         : QStringLiteral("(%1, %2)").arg(height).arg(width);
    QByteArray header = QStringLiteral(
                                "{'descr': '<f4', 'fortran_order': False, "
                                "'shape': %1, }")
                                .arg(shape)
                                .toLatin1();
    // magic(6) + version(2) + len(2) + header + '\n' padded to 64 bytes.
    const int unpadded = 10 + header.size() + 1;
    header.append(QByteArray((64 - unpadded % 64) % 64, ' '));
    header.append('\n');
    QByteArray bytes("\x93NUMPY\x01\x00", 8);
    const quint16 len = static_cast<quint16>(header.size());
    bytes.append(static_cast<char>(len & 0xff));
    bytes.append(static_cast<char>(len >> 8));
    bytes.append(header);
    bytes.append(reinterpret_cast<const char*>(data),
                 static_cast<int>(sizeof(float) * height * width *
                                  std::max(channels, 1)));
    return saveBytes(bytes, path, error);
}

inline bool saveJson(const QJsonObject& object,
                     const QString& path,
                     QString* error) {
    return saveBytes(QJsonDocument(object).toJson(QJsonDocument::Indented),
                     path, error);
}

/** Compatibility bridge for established batch commands that predate Args.
 *  It handles only the side-effect-free `HELP [RESULT_JSON path]` query and
 *  leaves every normal invocation untouched. New commands use prepare(). */
enum class SchemaHelp { NotRequested, Done, Failed };

inline SchemaHelp handleSchemaHelp(ccCommandLineInterface& cmd,
                                   const QString& keyword,
                                   const QString& task,
                                   const QString& summary,
                                   const OptionTable& options) {
    QStringList& tokens = cmd.arguments();
    if (tokens.isEmpty() || tokens.front().compare(QStringLiteral("HELP"),
                                                   Qt::CaseInsensitive) != 0) {
        return SchemaHelp::NotRequested;
    }
    tokens.takeFirst();
    QString resultJson;
    while (!tokens.isEmpty()) {
        const QString& head = tokens.front();
        if (head.size() > 1 && head[0] == '-' && head[1].isLetter()) break;
        const QString key = tokens.takeFirst().toUpper();
        if (key != QStringLiteral("RESULT_JSON") || tokens.isEmpty()) {
            cmd.error(QStringLiteral("HELP accepts only RESULT_JSON <path>"));
            return SchemaHelp::Failed;
        }
        resultJson = tokens.takeFirst();
    }

    const std::vector<const OptionTable*> tables{&options};
    cmd.print(QStringLiteral("[%1] %2").arg(keyword, summary));
    for (const OptionSpec& spec : options) {
        cmd.print(QStringLiteral("  %1 <%2> (default '%3'): %4")
                          .arg(spec.key, spec.type, spec.defaultValue,
                               spec.help));
    }
    const QJsonObject schema = schemaJson(keyword, task, summary, tables);
    cmd.print(QStringLiteral("[%1] schema: %2")
                      .arg(keyword,
                           QString::fromUtf8(QJsonDocument(schema).toJson(
                                   QJsonDocument::Compact))));
    if (!resultJson.isEmpty()) {
        QString error;
        if (!saveJson(schema, resultJson, &error)) {
            cmd.error(error);
            return SchemaHelp::Failed;
        }
    }
    return SchemaHelp::Done;
}

inline QJsonObject timingsJson(const aicore_pipeline_timings& t) {
    QJsonObject out;
    if (t.valid_fields & AICORE_TIMING_PREPROCESS)
        out["preprocess_ms"] = t.preprocess_ms;
    if (t.valid_fields & AICORE_TIMING_INFERENCE)
        out["inference_ms"] = t.inference_ms;
    if (t.valid_fields & AICORE_TIMING_POSTPROCESS)
        out["postprocess_ms"] = t.postprocess_ms;
    if (t.valid_fields & AICORE_TIMING_SERIALIZATION)
        out["serialization_ms"] = t.serialization_ms;
    if (t.valid_fields & AICORE_TIMING_E2E) out["e2e_ms"] = t.e2e_ms;
    return out;
}

/** Zero-initialized timing struct with the ABI version set. */
inline aicore_pipeline_timings emptyTimings() {
    aicore_pipeline_timings t{};
    t.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    return t;
}

// ---------------------------------------------------------------------------
//  Session: parse + HELP + output directory, shared by every command
// ---------------------------------------------------------------------------

struct Session {
    QString keyword;
    QString task;
    Args args;
    QString outDir;
    QString resultJson;
    QString device;
    int threads = 0;
    bool download = false;
    bool listModels = false;
    bool continueOnError = true;
};

enum class Prep { Run, Done, Failed };

/** Parses the command line into \p s. Done = HELP printed (return true);
 *  Failed = cmd.error() already reported (return false). */
inline Prep prepare(ccCommandLineInterface& cmd,
                    const QString& keyword,
                    const QString& task,
                    const QString& summary,
                    const std::vector<const OptionTable*>& taskTables,
                    Session* s,
                    const std::function<void(Args&)>& validateArgs = {}) {
    s->keyword = keyword;
    s->task = task;
    std::vector<const OptionTable*> tables{&commonOptions()};
    tables.insert(tables.end(), taskTables.begin(), taskTables.end());
    cmd.print(QStringLiteral("[%1]").arg(keyword));
    if (!s->args.parse(cmd, keyword, tables)) {
        cmd.error(s->args.error());
        return Prep::Failed;
    }
    if (s->args.helpRequested()) {
        cmd.print(QStringLiteral("[%1] %2").arg(keyword, summary));
        for (const OptionTable* table : tables) {
            for (const OptionSpec& spec : *table) {
                cmd.print(QStringLiteral("  %1 <%2> (default '%3'): %4")
                                  .arg(spec.key, spec.type, spec.defaultValue,
                                       spec.help));
            }
        }
        const QJsonObject schema = schemaJson(keyword, task, summary, tables);
        cmd.print(QStringLiteral("[%1] schema: %2")
                          .arg(keyword,
                               QString::fromUtf8(QJsonDocument(schema).toJson(
                                       QJsonDocument::Compact))));
        // Console output honours the CLI verbosity (-VERBOSE); files are the
        // machine-readable channel, so RESULT_JSON also receives the schema.
        const QString target = s->args.str("RESULT_JSON");
        QString error;
        if (!target.isEmpty() && !saveJson(schema, target, &error)) {
            cmd.error(error);
            return Prep::Failed;
        }
        return Prep::Done;
    }
    s->listModels = s->args.toBool("LIST_MODELS", false);
    if (s->listModels) {
        s->download = s->args.toBool("DOWNLOAD", false);
        if (!s->args.ok()) {
            cmd.error(s->args.error());
            return Prep::Failed;
        }
        return Prep::Run;  // catalog query: no OUT_DIR needed
    }
    // Validate task arguments before filesystem side effects; HELP/catalog
    // queries do not require task arguments. Existing consumers without a
    // callback retain their original preparation flow.
    if (validateArgs) {
        validateArgs(s->args);
        if (!s->args.ok()) {
            cmd.error(s->args.error());
            return Prep::Failed;
        }
    }
    s->outDir = s->args.str("OUT_DIR");
    if (s->outDir.isEmpty()) {
        cmd.error(QStringLiteral("-%1 needs OUT_DIR <dir>").arg(keyword));
        return Prep::Failed;
    }
    s->outDir = QDir(s->outDir).absolutePath();
    if (!QDir().mkpath(s->outDir)) {
        cmd.error(QStringLiteral("Cannot create OUT_DIR %1").arg(s->outDir));
        return Prep::Failed;
    }
    s->resultJson = s->args.str("RESULT_JSON");
    if (s->resultJson.isEmpty()) {
        s->resultJson = QDir(s->outDir).filePath(
                QStringLiteral("_%1_result.json").arg(task));
    } else if (QFileInfo(s->resultJson).isRelative()) {
        // A bare file name anchors to OUT_DIR, never to the process CWD.
        s->resultJson = QDir(s->outDir).filePath(s->resultJson);
    }
    s->device = s->args.str("DEVICE", QStringLiteral("auto"));
    s->threads = s->args.toInt("THREADS", 0);
    s->download = s->args.toBool("DOWNLOAD", false);
    s->continueOnError = s->args.toBool("CONTINUE_ON_ERROR", true);
    if (!s->args.ok()) {
        cmd.error(s->args.error());
        return Prep::Failed;
    }
    return Prep::Run;
}

// ---------------------------------------------------------------------------
//  Inputs
// ---------------------------------------------------------------------------

struct Item {
    QString path;  ///< absolute input path
    QString stem;  ///< unique, filesystem-safe output stem
};

/** Image-set item (FreeSplatter views, Lingbot sequences, ...). */
struct SetItem {
    QStringList paths;
    QString stem;
};

inline QStringList imagePatterns() {
    QStringList patterns;
    for (const QByteArray& fmt : QImageReader::supportedImageFormats()) {
        patterns << QStringLiteral("*.") + QString::fromLatin1(fmt).toLower();
    }
    patterns.removeDuplicates();
    return patterns;
}

inline void naturalSort(QStringList& paths) {
    QCollator collator;
    collator.setNumericMode(true);
    collator.setCaseSensitivity(Qt::CaseInsensitive);
    // O(n log n) comparisons; n is the batch size.
    std::sort(paths.begin(), paths.end(),
              [&collator](const QString& a, const QString& b) {
                  return collator.compare(a, b) < 0;
              });
}

inline QStringList scanDir(const QString& dir,
                           const QStringList& patterns,
                           bool recursive) {
    QStringList found;
    QDirIterator it(dir, patterns, QDir::Files | QDir::Readable,
                    recursive ? QDirIterator::Subdirectories
                              : QDirIterator::NoIteratorFlags);
    while (it.hasNext()) found << QFileInfo(it.next()).absoluteFilePath();
    naturalSort(found);
    return found;
}

inline QStringList patternsOf(const Args& args, const QStringList& defaults) {
    const QString raw = args.str("PATTERN");
    if (raw.isEmpty()) return defaults;
    QString normalized = raw;
    normalized.replace(',', ';');
    QStringList patterns;
    for (const QString& p : normalized.split(';')) {
        if (!p.trimmed().isEmpty()) patterns << p.trimmed();
    }
    return patterns;
}

inline QString sanitizeStem(QString stem) {
    for (QChar& c : stem) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' ||
            c == '"' || c == '<' || c == '>' || c == '|') {
            c = '_';
        }
    }
    return stem.isEmpty() ? QStringLiteral("item") : stem;
}

/** Makes every stem unique (`name`, `name_2`, ...). */
inline QString uniqueStem(const QString& stem, QSet<QString>* used) {
    QString candidate = sanitizeStem(stem);
    for (int n = 2; used->contains(candidate.toLower()); ++n) {
        candidate = sanitizeStem(stem) + QStringLiteral("_%1").arg(n);
    }
    used->insert(candidate.toLower());
    return candidate;
}

/** IMAGE / INPUT_LIST / INPUT_DIR -> ordered unique files. Directory hits
 *  keep their relative path in the stem (`sub__img`) so recursive scans do
 *  not collide. */
inline bool collectItems(const Args& args,
                         std::vector<Item>* items,
                         QString* error,
                         const QStringList& defaultPatterns = imagePatterns()) {
    QSet<QString> seen, stems;
    auto add = [&](const QString& path, const QString& stem) -> bool {
        const QFileInfo fi(path);
        if (!fi.isFile()) {
            *error = QStringLiteral("Input not found: %1").arg(path);
            return false;
        }
        const QString abs = fi.absoluteFilePath();
        if (seen.contains(abs)) return true;
        seen.insert(abs);
        items->push_back({abs, uniqueStem(stem, &stems)});
        return true;
    };
    for (const QString& path : args.list("IMAGE")) {
        if (!add(path, QFileInfo(path).completeBaseName())) return false;
    }
    if (args.has("INPUT_LIST")) {
        QFile file(args.str("INPUT_LIST"));
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            *error = QStringLiteral("Cannot read INPUT_LIST %1")
                             .arg(file.fileName());
            return false;
        }
        const QDir base = QFileInfo(file).absoluteDir();
        QTextStream stream(&file);
        while (!stream.atEnd()) {
            const QString line = stream.readLine().trimmed();
            if (line.isEmpty() || line.startsWith('#')) continue;
            const QString path =
                    QFileInfo(line).isRelative() ? base.filePath(line) : line;
            if (!add(path, QFileInfo(path).completeBaseName())) return false;
        }
    }
    if (args.has("INPUT_DIR")) {
        const QDir dir(args.str("INPUT_DIR"));
        if (!dir.exists()) {
            *error = QStringLiteral("INPUT_DIR not found: %1").arg(dir.path());
            return false;
        }
        bool recursive = false;
        parseBool(args.str("RECURSIVE", "0"), &recursive);
        for (const QString& path :
             scanDir(dir.absolutePath(), patternsOf(args, defaultPatterns),
                     recursive)) {
            const QFileInfo fi(path);
            QString rel = dir.relativeFilePath(fi.absolutePath());
            rel = rel == QStringLiteral(".") ? QString() : rel + "__";
            rel.replace('/', QStringLiteral("__"));
            if (!add(path, rel + fi.completeBaseName())) return false;
        }
    }
    if (items->empty()) {
        *error = QStringLiteral(
                "No inputs: pass IMAGE <file>, INPUT_DIR <dir> or "
                "INPUT_LIST <txt>");
        return false;
    }
    return true;
}

/** Set-shaped inputs. SUBDIRS 1 + INPUT_DIR: one set per immediate
 *  sub-directory (natural order); otherwise every collected file forms a
 *  single set named after INPUT_DIR (or "set"). */
inline bool collectSets(const Args& args,
                        std::vector<SetItem>* sets,
                        QString* error,
                        int minPerSet = 1) {
    bool subdirs = false;
    parseBool(args.str("SUBDIRS", "0"), &subdirs);
    QSet<QString> stems;
    if (subdirs) {
        const QDir root(args.str("INPUT_DIR"));
        if (!args.has("INPUT_DIR") || !root.exists()) {
            *error = QStringLiteral("SUBDIRS 1 needs an existing INPUT_DIR");
            return false;
        }
        QStringList dirs;
        for (const QFileInfo& fi :
             root.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot)) {
            dirs << fi.absoluteFilePath();
        }
        naturalSort(dirs);
        bool recursive = false;
        parseBool(args.str("RECURSIVE", "0"), &recursive);
        for (const QString& d : dirs) {
            const QStringList files =
                    scanDir(d, patternsOf(args, imagePatterns()), recursive);
            if (files.size() < minPerSet) continue;  // not a view set
            sets->push_back(
                    {files, uniqueStem(QFileInfo(d).fileName(), &stems)});
        }
    } else {
        std::vector<Item> items;
        if (!collectItems(args, &items, error)) return false;
        SetItem set;
        for (const Item& item : items) set.paths << item.path;
        const QString name = args.has("INPUT_DIR")
                                     ? QDir(args.str("INPUT_DIR")).dirName()
                                     : QStringLiteral("set");
        set.stem = uniqueStem(name, &stems);
        if (set.paths.size() >= minPerSet) sets->push_back(set);
    }
    if (sets->empty()) {
        *error = QStringLiteral("No input set with >= %1 image(s)")
                         .arg(minPerSet);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
//  Models
// ---------------------------------------------------------------------------

/** Converts an AICore-owned C string (e.g. *_model_cache_dir()) into a
 *  QString and releases it with the task's own free function (the buffer
 *  must go back to the allocator of libAICore, which matters on Windows
 *  where each module may link its own CRT). */
inline QString takeCString(char* owned, void (*freeFn)(void*)) {
    const QString out = QString::fromUtf8(owned != nullptr ? owned : "");
    if (owned != nullptr) freeFn(owned);
    return out;
}

/** An existing file path wins; a bare file name is looked up in the task
 *  cache dir; empty -> \p defaultFilename in the cache dir. */
inline QString resolveModelPath(const QString& token,
                                const QString& cacheDir,
                                const QString& defaultFilename = QString()) {
    const QString name = token.isEmpty() ? defaultFilename : token;
    if (name.isEmpty()) return QString();
    const QFileInfo fi(name);
    if (fi.isFile() || fi.isAbsolute() || name.contains('/') ||
        name.contains('\\')) {
        return fi.absoluteFilePath();
    }
    return QDir(cacheDir).filePath(name);
}

/** Catalog facts for a model file, read from the task catalog by the
 *  caller (empty url = not downloadable; empty sha256 = the pinned release
 *  digest registry entry for the file name). */
struct ModelSource {
    QString url;
    QString sha256;
};

/** Ensures \p path exists; with \p download set and a catalog URL, fetches
 *  it synchronously through the shared downloader (resume, mirrors and
 *  SHA-256 verification included). */
inline bool ensureModel(ccCommandLineInterface& cmd,
                        const QString& path,
                        const ModelSource& source,
                        bool download,
                        QString* error) {
    if (path.isEmpty()) {
        *error = QStringLiteral("No model selected");
        return false;
    }
    if (QFileInfo(path).isFile()) return true;
    if (!download || source.url.isEmpty()) {
        *error = QStringLiteral(
                         "Model not found: %1 (pass DOWNLOAD 1 for catalog "
                         "models, or MODEL <existing .gguf>)")
                         .arg(path);
        return false;
    }
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        *error = QStringLiteral("Cannot create %1")
                         .arg(QFileInfo(path).absolutePath());
        return false;
    }
    cmd.print(QStringLiteral("Downloading %1 -> %2").arg(source.url, path));
    ecvModelDownloader downloader;
    ecvModelDownloader::Request request;
    request.url = source.url;
    request.destPath = path;
    request.requireGgufMagic =
            path.endsWith(QStringLiteral(".gguf"), Qt::CaseInsensitive);
    request.contentAnchor = {QCryptographicHash::Sha256,
                             source.sha256.isEmpty()
                                     ? ecvAssetIntegrity::PinnedDigest(
                                               QFileInfo(path).fileName())
                                     : source.sha256.toLatin1()};
    bool ok = false;
    QEventLoop loop;
    QObject::connect(&downloader, &ecvModelDownloader::logMessage,
                     [&cmd](const QString& m) { cmd.print(m); });
    QObject::connect(&downloader, &ecvModelDownloader::finished,
                     [&](bool success, const QString&) {
                         ok = success;
                         loop.quit();
                     });
    downloader.download(request);
    if (downloader.isBusy()) loop.exec();
    if (!ok || !QFileInfo(path).isFile()) {
        *error = QStringLiteral("Download failed: %1").arg(source.url);
        return false;
    }
    return true;
}

/** One catalog row as reported by LIST_MODELS. */
struct CatalogRow {
    QString filename;
    QString displayName;
    QString url;
    QString path;  ///< resolved cache path
    QString role;  ///< task-specific (det/rec, depth/metric, ...)
};

/** LIST_MODELS 1: prints `[KW] models: {json}` with every catalog row and
 *  its cache state. With DOWNLOAD 1, first fetches the row named by MODEL
 *  (or every row for MODEL all). */
inline bool listModels(ccCommandLineInterface& cmd,
                       const Session& s,
                       const std::vector<CatalogRow>& rows) {
    const QString wanted = s.args.str("MODEL");
    if (s.download) {
        if (wanted.isEmpty()) {
            return cmd.error(QStringLiteral(
                    "LIST_MODELS + DOWNLOAD needs MODEL <filename|all>"));
        }
        bool matched = false;
        for (const CatalogRow& row : rows) {
            if (wanted != QStringLiteral("all") && row.filename != wanted) {
                continue;
            }
            matched = true;
            QString error;
            if (!ensureModel(cmd, row.path, {row.url, QString()}, true,
                             &error)) {
                return cmd.error(error);
            }
        }
        if (!matched) {
            return cmd.error(QStringLiteral("Not a %1 catalog model: %2")
                                     .arg(s.task, wanted));
        }
    }
    QJsonArray models;
    for (const CatalogRow& row : rows) {
        models.append(QJsonObject{{"filename", row.filename},
                                  {"display_name", row.displayName},
                                  {"role", row.role},
                                  {"url", row.url},
                                  {"path", row.path},
                                  {"cached", QFileInfo(row.path).isFile()}});
    }
    const QJsonObject out{{"task", s.task}, {"models", models}};
    cmd.print(
            QStringLiteral("[%1] models: %2")
                    .arg(s.keyword, QString::fromUtf8(QJsonDocument(out).toJson(
                                            QJsonDocument::Compact))));
    const QString target = s.args.str("RESULT_JSON");
    QString error;
    if (!target.isEmpty() && !saveJson(out, target, &error)) {
        return cmd.error(error);
    }
    return true;
}

/** Rows of a shared-catalog family (aicore/model_catalog_capi.h: depth,
 *  DeepLSD, LightGlue, Gaussian, ALIKED). Empty \p role keeps every row. */
inline std::vector<CatalogRow> familyRows(aicore_model_family family,
                                          const QString& cacheDir,
                                          const char* role = nullptr) {
    std::vector<CatalogRow> rows;
    const int n = aicore_model_count(family);
    for (int i = 0; i < n; ++i) {
        const aicore_model_entry* e = aicore_model_at(family, i);
        if (e == nullptr || e->filename == nullptr) continue;
        const QString entryRole = QString::fromUtf8(e->role ? e->role : "");
        if (role != nullptr && entryRole != QString::fromUtf8(role)) continue;
        const QString filename = QString::fromUtf8(e->filename);
        rows.push_back(
                {filename,
                 QString::fromUtf8(e->display_name ? e->display_name : ""),
                 QString::fromUtf8(e->download_url ? e->download_url : ""),
                 QDir(cacheDir).filePath(filename), entryRole});
    }
    return rows;
}

/** Catalog URL + digest of a shared-catalog file (empty when unknown). */
inline ModelSource familySource(aicore_model_family family,
                                const QString& filename) {
    const QByteArray name = filename.toUtf8();
    const aicore_model_entry* e =
            aicore_model_by_filename(family, name.constData());
    if (e == nullptr) return {};
    return {QString::fromUtf8(e->download_url ? e->download_url : ""),
            QString::fromLatin1(e->sha256 ? e->sha256 : "")};
}

// ---------------------------------------------------------------------------
//  Manifest + batch loop
// ---------------------------------------------------------------------------

/** OUT_DIR/<stem><suffix>. */
inline QString outPath(const Session& s,
                       const QString& stem,
                       const QString& suffix) {
    return QDir(s.outDir).filePath(stem + suffix);
}

class Manifest {
public:
    explicit Manifest(const Session& s) {
        m_root["schema"] = QStringLiteral("acloudviewer.aicore.batch/1");
        m_root["task"] = s.task;
        m_root["command"] = s.keyword;
        m_root["out_dir"] = s.outDir;
        m_root["device"] = s.device;
        m_root["options"] = s.args.toJson();
        m_timer.start();
    }
    /** Task-level facts (model, backend, resolved defaults). */
    void set(const QString& key, const QJsonValue& value) {
        m_root[key] = value;
    }
    void add(const QJsonObject& item, bool ok) {
        m_items.append(item);
        ok ? ++m_ok : ++m_failed;
    }
    int ok() const { return m_ok; }
    int failed() const { return m_failed; }
    bool write(const QString& path, QString* error) {
        m_root["ok"] = m_ok;
        m_root["failed"] = m_failed;
        m_root["total"] = m_items.size();
        m_root["elapsed_ms"] = static_cast<double>(m_timer.elapsed());
        m_root["items"] = m_items;
        return saveJson(m_root, path, error);
    }

private:
    QJsonObject m_root;
    QJsonArray m_items;
    QElapsedTimer m_timer;
    int m_ok = 0;
    int m_failed = 0;
};

/** Runs \p fn over \p inputs (Item or SetItem). \p fn fills the record
 *  (outputs, counts, timings) and returns false with \p error on failure.
 *  Records always carry index / stem / status / elapsed_ms. */
template <typename T, typename Fn>
void runBatch(ccCommandLineInterface& cmd,
              const Session& s,
              const std::vector<T>& inputs,
              Manifest& manifest,
              Fn fn) {
    const int total = static_cast<int>(inputs.size());
    for (int i = 0; i < total; ++i) {
        const T& input = inputs[static_cast<size_t>(i)];
        QJsonObject record;
        record["index"] = i;
        record["stem"] = input.stem;
        QString error;
        QElapsedTimer timer;
        timer.start();
        bool ok = false;
        try {
            ok = fn(input, record, error);
        } catch (const std::exception& e) {
            error = QString::fromUtf8(e.what());
        }
        record["elapsed_ms"] = static_cast<double>(timer.elapsed());
        record["status"] = ok ? QStringLiteral("ok") : QStringLiteral("failed");
        if (!ok) {
            record["error"] =
                    error.isEmpty() ? QStringLiteral("unknown error") : error;
        }
        manifest.add(record, ok);
        cmd.print(QStringLiteral("[%1] %2/%3 %4: %5 (%6 ms)")
                          .arg(s.keyword)
                          .arg(i + 1)
                          .arg(total)
                          .arg(input.stem,
                               ok ? QStringLiteral("ok")
                                  : QStringLiteral("FAILED - ") + error)
                          .arg(timer.elapsed()));
        if (!ok && !s.continueOnError) break;
    }
}

/** Writes the manifest and turns any failed item into a command error. */
inline bool finish(ccCommandLineInterface& cmd,
                   const Session& s,
                   Manifest& manifest) {
    QString error;
    if (!manifest.write(s.resultJson, &error)) return cmd.error(error);
    cmd.print(QStringLiteral("[%1] done: %2 ok, %3 failed. Result JSON: %4")
                      .arg(s.keyword)
                      .arg(manifest.ok())
                      .arg(manifest.failed())
                      .arg(s.resultJson));
    if (manifest.failed() > 0 || manifest.ok() == 0) {
        return cmd.error(QStringLiteral("-%1: %2 item(s) failed (see %3)")
                                 .arg(s.keyword)
                                 .arg(manifest.failed())
                                 .arg(s.resultJson));
    }
    return true;
}

}  // namespace ecvAICoreBatch
