// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch OCR (-OCR_RECOGNIZE): one PP-OCR det+rec(+layout) context,
// N images -> <stem>_ocr.json (+ <stem>_ocr.txt plain text and the official
// draw_ocr preview) each, plus the shared ecvAICoreBatch RESULT_JSON
// manifest. TABLE=1 adds the SLANet_plus table stage (layout "table"
// regions with LAYOUT=1, else the whole image) and writes
// <stem>_table<i>.html/.csv. The per-image body is OCRWorker::recognizeImage
// and the preview is OCRRender::officialVisualization, the dialog/DB-export
// code paths.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/ocr_capi.h>
#include <aicore/table_capi.h>

#include <QFile>

#include "OCRWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_OCR_RECOGNIZE "OCR_RECOGNIZE"

struct CommandOcrRecognize : public ccCommandLineInterface::Command {
    CommandOcrRecognize()
        : ccCommandLineInterface::Command(
                  QObject::tr("OCR batch text recognition"),
                  COMMAND_OCR_RECOGNIZE) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"LANG", "choice:ch|en", "ch", "Recognition script"},
                {"QUANT", "choice:f32|f16|q8_0", "f16",
                 "Catalog quantization tier for det/rec/layout/table"},
                {"LAYOUT", "bool", "0", "Run the layout-region stage"},
                {"TABLE", "bool", "0",
                 "Reconstruct tables (SLANet_plus): HTML + CSV per table"},
                {"DET_MODEL", "string", "", "Override the detector GGUF"},
                {"REC_MODEL", "string", "", "Override the recognizer GGUF"},
                {"LAYOUT_MODEL", "string", "", "Override the layout GGUF"},
                {"TABLE_MODEL", "string", "", "Override the table GGUF"},
                {"MIN_SCORE", "float", "0",
                 "Drop recognized lines below this confidence"},
                {"SAVE_VIS", "bool", "1",
                 "Write <stem>_vis.png (official draw_ocr layout)"},
                {"SAVE_TEXT", "bool", "1", "Write <stem>_ocr.txt"},
        };
        return table;
    }

    /** Catalog filename for (role, language, quant); empty when absent. */
    static QString catalogFile(const char* role,
                               const QString& lang,
                               const QString& quant) {
        const QString suffix = QStringLiteral("-%1.gguf").arg(quant);
        for (int i = 0; i < aicore_ocr_model_count(); ++i) {
            const aicore_ocr_model_entry* e = aicore_ocr_model_at(i);
            if (!e || !e->filename || !e->role) continue;
            if (qstrcmp(e->role, role) != 0) continue;
            const QString fn = QString::fromUtf8(e->filename);
            if (!fn.endsWith(suffix)) continue;
            if (qstrcmp(role, "rec") == 0 &&
                QString::fromUtf8(e->language ? e->language : "") != lang) {
                continue;
            }
            return fn;
        }
        return QString();
    }

    /** Table catalog filename for a quantization tier; empty when absent. */
    static QString tableCatalogFile(const QString& quant) {
        const QString suffix = QStringLiteral("-%1.gguf").arg(quant);
        for (int i = 0; i < aicore_table_model_count(); ++i) {
            const aicore_table_model_entry* e = aicore_table_model_at(i);
            if (!e || !e->filename) continue;
            const QString fn = QString::fromUtf8(e->filename);
            if (fn.endsWith(suffix)) return fn;
        }
        return QString();
    }

    static ecvAICoreBatch::ModelSource source(const QString& path) {
        const QByteArray name = QFileInfo(path).fileName().toUtf8();
        if (const aicore_table_model_entry* t =
                    aicore_table_model_by_filename(name.constData())) {
            return {t->download_url ? QString::fromUtf8(t->download_url)
                                    : QString(),
                    QString()};
        }
        const aicore_ocr_model_entry* e =
                aicore_ocr_model_by_filename(name.constData());
        return {e && e->download_url ? QString::fromUtf8(e->download_url)
                                     : QString(),
                QString()};
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_OCR_RECOGNIZE, QStringLiteral("ocr"),
                           QStringLiteral("Text detection + recognition "
                                          "(PP-OCRv4, optional layout and "
                                          "SLANet_plus tables)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = B::takeCString(
                aicore_ocr_model_cache_dir(),
                [](void* p) { aicore_ocr_free_buffer(static_cast<char*>(p)); });
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (int i = 0; i < aicore_ocr_model_count(); ++i) {
                const aicore_ocr_model_entry* e = aicore_ocr_model_at(i);
                if (!e || !e->filename) continue;
                const QString fn = QString::fromUtf8(e->filename);
                rows.push_back({fn, QString::fromUtf8(e->display_name),
                                QString::fromUtf8(e->download_url),
                                QDir(cacheDir).filePath(fn),
                                QString::fromUtf8(e->role ? e->role : "")});
            }
            for (int i = 0; i < aicore_table_model_count(); ++i) {
                const aicore_table_model_entry* e = aicore_table_model_at(i);
                if (!e || !e->filename) continue;
                const QString fn = QString::fromUtf8(e->filename);
                rows.push_back({fn, QString::fromUtf8(e->display_name),
                                QString::fromUtf8(e->download_url),
                                QDir(cacheDir).filePath(fn),
                                QString::fromUtf8(e->role ? e->role : "")});
            }
            return B::listModels(cmd, s, rows);
        }

        const QString lang =
                s.args.choice("LANG", QStringLiteral("ch"), {"ch", "en"});
        const QString quant = s.args.choice("QUANT", QStringLiteral("f16"),
                                            {"f32", "f16", "q8_0"});
        const bool layout = s.args.toBool("LAYOUT", false);
        const bool table =
                s.args.toBool("TABLE", false) || s.args.has("TABLE_MODEL");
        const float minScore = s.args.toFloat("MIN_SCORE", 0.0f);
        const bool saveVis = s.args.toBool("SAVE_VIS", true);
        const bool saveText = s.args.toBool("SAVE_TEXT", true);
        if (!s.args.ok()) return cmd.error(s.args.error());

        QString error;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        OCRWorker::Settings settings;
        settings.device = s.device;
        settings.threads = s.threads;
        // The DB detector is script-agnostic: one det row serves ch and en.
        settings.detPath =
                B::resolveModelPath(s.args.str("DET_MODEL"), cacheDir,
                                    catalogFile("det", lang, quant));
        settings.recPath =
                B::resolveModelPath(s.args.str("REC_MODEL"), cacheDir,
                                    catalogFile("rec", lang, quant));
        if (layout || s.args.has("LAYOUT_MODEL")) {
            settings.layoutPath =
                    B::resolveModelPath(s.args.str("LAYOUT_MODEL"), cacheDir,
                                        catalogFile("layout", lang, quant));
        }
        if (table) {
            // The table cache directory is the shared ocr_models folder.
            settings.tablePath =
                    B::resolveModelPath(s.args.str("TABLE_MODEL"), cacheDir,
                                        tableCatalogFile(quant));
        }
        for (const QString& p : {settings.detPath, settings.recPath,
                                 settings.layoutPath, settings.tablePath}) {
            if (p.isEmpty()) continue;
            if (!B::ensureModel(cmd, p, source(p), s.download, &error))
                return cmd.error(error);
        }

        aicore_ocr_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = AICORE_OCR_ABI_VERSION;
        const QByteArray det = QFile::encodeName(settings.detPath);
        const QByteArray rec = QFile::encodeName(settings.recPath);
        const QByteArray lay = QFile::encodeName(settings.layoutPath);
        const QByteArray device = s.device.toUtf8();
        opts.det_model_path = det.constData();
        opts.rec_model_path = rec.constData();
        if (!settings.layoutPath.isEmpty())
            opts.layout_model_path = lay.constData();
        opts.device = device.constData();
        opts.n_threads = s.threads;
        if (table) {
            // Official table_recognition text detection (max, 960), as in
            // the dialog's table mode.
            opts.det_limit_type = 1;
            opts.det_max_side = 960;
        }
        aicore_ocr_ctx* ctx = aicore_ocr_context_create(&opts);
        if (ctx == nullptr || !aicore_ocr_context_ready(ctx)) {
            if (ctx) aicore_ocr_context_destroy(ctx);
            return cmd.error(
                    QStringLiteral("Model load failed: %1")
                            .arg(QString::fromUtf8(aicore_ocr_last_error())));
        }
        std::unique_ptr<aicore_ocr_ctx, void (*)(aicore_ocr_ctx*)> guard(
                ctx, aicore_ocr_context_destroy);
        std::unique_ptr<aicore_table_ctx, void (*)(aicore_table_ctx*)>
                tableGuard(nullptr, aicore_table_context_destroy);
        if (table) {
            aicore_table_options topts{};
            topts.struct_size = sizeof(topts);
            topts.abi_version = AICORE_TABLE_ABI_VERSION;
            const QByteArray tablePath = QFile::encodeName(settings.tablePath);
            topts.model_path = tablePath.constData();
            topts.device = device.constData();
            topts.n_threads = s.threads;
            tableGuard.reset(aicore_table_context_create(&topts));
            if (!tableGuard || !aicore_table_context_ready(tableGuard.get())) {
                return cmd.error(QStringLiteral("Table model load failed: %1")
                                         .arg(QString::fromUtf8(
                                                 aicore_table_last_error())));
            }
        }

        B::Manifest manifest(s);
        manifest.set("det_model", settings.detPath);
        manifest.set("rec_model", settings.recPath);
        if (!settings.layoutPath.isEmpty())
            manifest.set("layout_model", settings.layoutPath);
        if (!settings.tablePath.isEmpty())
            manifest.set("table_model", settings.tablePath);
        manifest.set("language", lang);

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    OCRWorker::Settings one = settings;
                    one.imagePath = item.path;
                    OCRRunResult r;
                    if (!OCRWorker::recognizeImage(ctx, tableGuard.get(), one,
                                                   nullptr, &r, &err)) {
                        return false;
                    }
                    if (minScore > 0.0f) {
                        QVector<OCRBoxResult> kept;
                        for (const OCRBoxResult& b : r.boxes)
                            if (b.score >= minScore) kept.push_back(b);
                        r.boxes = kept;
                    }
                    auto quadJson = [](const float q[4][2]) {
                        QJsonArray a;
                        for (int c = 0; c < 4; ++c)
                            a.append(QJsonArray{q[c][0], q[c][1]});
                        return a;
                    };
                    QJsonArray lines;
                    QStringList text;
                    for (const OCRBoxResult& b : r.boxes) {
                        lines.append(QJsonObject{{"text", b.text},
                                                 {"score", b.score},
                                                 {"language", b.language},
                                                 { "quad",
                                                   quadJson(b.quad) }});
                        text << b.text;
                    }
                    QJsonArray regions;
                    for (const OCRRegionResult& g : r.regions) {
                        regions.append(
                                QJsonObject{{"class_id", g.classId},
                                            {"class_name", g.className},
                                            {"reading_order", g.readingOrder},
                                            {"score", g.score},
                                            { "quad",
                                              quadJson(g.quad) }});
                    }
                    QJsonObject json{{"image", item.path},
                                     {"width", r.sourceImage.width()},
                                     {"height", r.sourceImage.height()},
                                     { "lines",
                                       lines }};
                    if (!r.regions.isEmpty()) json["regions"] = regions;
                    QJsonObject outputs;
                    QJsonArray tables, tableFiles;
                    for (int t = 0; t < r.tables.size(); ++t) {
                        const OCRTableResult& tb = r.tables[t];
                        QJsonArray cells;
                        for (const OCRTableCell& c : tb.cells) {
                            cells.append(QJsonObject{{"row", c.row},
                                                     {"col", c.col},
                                                     {"row_span", c.rowSpan},
                                                     {"col_span", c.colSpan},
                                                     {"text", c.text},
                                                     { "quad",
                                                       quadJson(c.quad) }});
                        }
                        tables.append(QJsonObject{
                                {"box",
                                 QJsonArray {
                                     tb.box[0],
                                     tb.box[1],
                                     tb.box[2],
                                     tb.box[3]
                                 }},
                                {"layout_region", tb.layoutRegion},
                                {"rows", tb.rows},
                                {"cols", tb.cols},
                                {"structure_score", tb.structureScore},
                                {"html", tb.html},
                                { "cells",
                                  cells }});
                        const QString stem = QStringLiteral("_table%1").arg(t);
                        const QString html =
                                B::outPath(s, item.stem, stem + ".html");
                        const QString csv =
                                B::outPath(s, item.stem, stem + ".csv");
                        if (!B::saveBytes(tb.html.toUtf8(), html, &err) ||
                            !B::saveBytes(tb.csv, csv, &err)) {
                            return false;
                        }
                        tableFiles.append(QJsonObject{{"html", html},
                                                      { "csv",
                                                        csv }});
                    }
                    if (table) json["tables"] = tables;
                    const QString jsonPath =
                            B::outPath(s, item.stem, "_ocr.json");
                    if (!B::saveJson(json, jsonPath, &err)) return false;
                    outputs["result"] = jsonPath;
                    if (table) outputs["tables"] = tableFiles;
                    if (saveText) {
                        const QString p = B::outPath(s, item.stem, "_ocr.txt");
                        if (!B::saveBytes((text.join('\n') + '\n').toUtf8(), p,
                                          &err)) {
                            return false;
                        }
                        outputs["text"] = p;
                    }
                    if (saveVis) {
                        const QString p = B::outPath(s, item.stem, "_vis.png");
                        if (!B::saveImage(OCRRender::officialVisualization(r),
                                          p, &err)) {
                            return false;
                        }
                        outputs["visualization"] = p;
                    }
                    QJsonObject timings{{"preprocess_ms", r.preprocessMs},
                                        {"inference_ms", r.inferenceMs},
                                        {"postprocess_ms", r.postprocessMs},
                                        { "e2e_ms",
                                          r.e2eMs }};
                    rec["timings"] = timings;
                    rec["lines"] = r.boxes.size();
                    rec["regions"] = r.regions.size();
                    if (table) {
                        rec["tables"] = r.tables.size();
                        rec["table_ms"] = r.tableMs;
                    }
                    rec["device"] = r.resolvedDevice;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_OCR_RECOGNIZE));
    }
#endif  // AICore_ENABLED
};
