// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AICore child-process methods: aicore.batch, yolo.track, sam3d.generate,
// mapanything.reconstruct. Every child runs through runChildProcess().

#include <QtCompat.h>

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QRegularExpression>
#include <QUuid>

#include "JsonRPCChildProcess.h"
#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"

using jsonrpc::ChildOutcome;
using jsonrpc::childTimeout;
using jsonrpc::D;
using jsonrpc::headlessChildEnvironment;
using jsonrpc::runChildProcess;

// ═══════════════════════════════════════════════════════════════════════════
// Typed AICore batch bridge (aicore.batch)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcAICoreBatch(
        const QMap<QString, QVariant>& params) {
    QString command = params.value("command").toString().trimmed().toUpper();
    if (command.startsWith(QLatin1Char('-'))) command.remove(0, 1);
    // This boundary intentionally exposes inference/export commands only.
    // Arbitrary argv remains process.run_cli's separate, explicit contract.
    static const QSet<QString> kCommands = {"RMBG_REMOVE",
                                            "DEEPLSD_LINES",
                                            "RFDETR_DETECT",
                                            "YOLO_PREDICT",
                                            "YOLO_TRACK",
                                            "DA3_DEPTH",
                                            "FACE_DETECT",
                                            "GKD_KEYPOINTS",
                                            "OCR_RECOGNIZE",
                                            "SAM3_SEGMENT",
                                            "LIGHTGLUE_MATCH",
                                            "FREESPLATTER_RECONSTRUCT",
                                            "TRELLIS_GENERATE",
                                            "INSTANTMESH_GENERATE",
                                            "LINGBOT_MAP_RECONSTRUCT",
                                            "ANYLABEL_INFER",
                                            "MAP_ANYTHING_RECONSTRUCT",
                                            "SAM3D_GENERATE",
                                            "OPENPCDET_DETECT",
                                            "LLM_GENERATE"};
    if (!kCommands.contains(command)) {
        QJsonArray allowed;
        QStringList sorted = kCommands.values();
        sorted.sort();
        for (const QString& value : sorted) allowed.append(value);
        return JsonRPCResult::error(
                -32602, "Unknown or non-AICore batch command",
                QJsonObject{{"command", command}, {"allowed", allowed}});
    }
    if (!params.contains("options") ||
        !qtCompatVariantIsMap(params.value("options"))) {
        return JsonRPCResult::error(
                -32602, "Missing required parameter: options (object)",
                D("param", "options", "hint",
                  "Use uppercase command options; arrays repeat an option"));
    }
    const QVariantMap options = params.value("options").toMap();
    const QRegularExpression keyPattern(QStringLiteral("^[A-Z][A-Z0-9_]*$"));
    auto scalar = [](const QVariant& value, QString* text) -> bool {
        if (!value.isValid() || value.isNull() || qtCompatVariantIsMap(value) ||
            qtCompatVariantIsList(value))
            return false;
        if (qtCompatVariantIsBool(value)) {
            *text = value.toBool() ? QStringLiteral("1") : QStringLiteral("0");
            return true;
        }
        if (qtCompatVariantIsString(value) || qtCompatVariantIsInt(value) ||
            qtCompatVariantIsDouble(value)) {
            *text = value.toString();
            return true;
        }
        return false;
    };

    QStringList argList{"-SILENT", "-VERBOSE", "-NO_TIMESTAMP",
                        QStringLiteral("-%1").arg(command)};
    QString resultPath = params.value("result_json").toString();
    bool temporaryResult = false;
    for (auto it = options.constBegin(); it != options.constEnd(); ++it) {
        const QString key = it.key().trimmed().toUpper();
        if (!keyPattern.match(key).hasMatch()) {
            return JsonRPCResult::error(
                    -32602, "Invalid AICore option name",
                    D("option", it.key(), "hint", "Use [A-Z][A-Z0-9_]*"));
        }
        const QVariant value = it.value();
        // HELP is the command parser's sole flag (no following value).
        if (key == QLatin1String("HELP") && qtCompatVariantIsBool(value)) {
            if (value.toBool()) argList << "HELP";
            continue;
        }
        const QVariantList values = qtCompatVariantIsList(value)
                                            ? value.toList()
                                            : QVariantList{value};
        if (values.isEmpty()) {
            return JsonRPCResult::error(-32602,
                                        "AICore option array must not be empty",
                                        D("option", key));
        }
        for (const QVariant& item : values) {
            QString text;
            if (!scalar(item, &text)) {
                return JsonRPCResult::error(
                        -32602, "AICore option values must be scalar",
                        D("option", key, "hint",
                          "Use string, number, boolean, or an array of them"));
            }
            if (key == QLatin1String("RESULT_JSON")) resultPath = text;
            argList << key << text;
        }
    }
    if (resultPath.isEmpty()) {
        resultPath = QDir(QDir::tempPath())
                             .filePath(QStringLiteral("acv_aicore_rpc_%1.json")
                                               .arg(qtCompatUuidId128(
                                                       QUuid::createUuid())));
        temporaryResult = true;
        argList << "RESULT_JSON" << resultPath;
    } else if (!options.contains("RESULT_JSON") &&
               !options.contains("result_json")) {
        argList << "RESULT_JSON" << resultPath;
    }

    int timeoutMs = 0;
    JsonRPCResult invalid;
    if (!childTimeout(params, 1800000, &timeoutMs, &invalid)) return invalid;
    auto* process = new QProcess(this);
    process->setProgram(QCoreApplication::applicationFilePath());
    process->setArguments(argList);
    process->setProcessEnvironment(headlessChildEnvironment());
    return runChildProcess(
            process, timeoutMs,
            [command, resultPath, temporaryResult, timeoutMs](
                    ChildOutcome outcome, QProcess& child) {
                if (outcome == ChildOutcome::StartFailed) {
                    if (temporaryResult) QFile::remove(resultPath);
                    return JsonRPCResult::error(
                            3, "Unable to start ACloudViewer batch child",
                            D("command", command, "error",
                              child.errorString()));
                }
                if (outcome == ChildOutcome::TimedOut) {
                    if (temporaryResult) QFile::remove(resultPath);
                    return JsonRPCResult::error(
                            3, "AICore batch timed out",
                            D("command", command, "timeout_ms", timeoutMs));
                }
                const QString stdOut =
                        QString::fromUtf8(child.readAllStandardOutput())
                                .trimmed();
                const QString stdErr =
                        QString::fromUtf8(child.readAllStandardError())
                                .trimmed();
                QJsonObject manifest;
                QFile resultFile(resultPath);
                if (resultFile.open(QIODevice::ReadOnly)) {
                    const QJsonDocument document =
                            QJsonDocument::fromJson(resultFile.readAll());
                    if (document.isObject()) manifest = document.object();
                }
                if (temporaryResult) QFile::remove(resultPath);

                QJsonObject response{{"command", command},
                                     {"exit_code", child.exitCode()},
                                     {"result_json",
                                      temporaryResult ? QString() : resultPath},
                                     {"manifest", manifest},
                                     {"stdout", stdOut.left(10000)},
                                     {"stderr", stdErr.left(5000)}};
                if (outcome == ChildOutcome::Failed) {
                    return JsonRPCResult::error(3, "AICore batch failed",
                                                response);
                }
                if (manifest.isEmpty()) {
                    return JsonRPCResult::error(
                            3, "AICore batch returned no readable RESULT_JSON",
                            response);
                }
                return JsonRPCResult::success(
                        QJsonDocument(response).toVariant());
            });
}

JsonRPCResult JsonRPCPlugin::rpcYoloTrack(
        const QMap<QString, QVariant>& params) {
    const QString model = params.value("model").toString();
    const QString video = params.value("video").toString();
    const QString framesDir = params.value("frames_dir").toString();
    const QString tracksJson = params.value("tracks_json").toString();
    if (model.isEmpty() || tracksJson.isEmpty() ||
        (video.isEmpty() == framesDir.isEmpty())) {
        return JsonRPCResult::error(
                -32602,
                "Missing/ambiguous parameters: need model, tracks_json and "
                "exactly one of video / frames_dir");
    }

    int timeoutMs = 0;
    JsonRPCResult invalid;
    if (!childTimeout(params, 600000, &timeoutMs, &invalid)) return invalid;

    // Optional key/value pairs in the -YOLO_TRACK argument order.
    const QHash<QString, QString> optionKeys = {
            {"tracker", "TRACKER"}, {"gmc", "GMC"},         {"conf", "CONF"},
            {"iou", "IOU"},         {"max_det", "MAX_DET"}, {"reid", "REID"},
            {"device", "DEVICE"},   {"threads", "THREADS"},
    };

    QStringList argList;
    argList << "-SILENT" << "-NO_TIMESTAMP" << "-YOLO_TRACK"
            << "MODEL" << model;
    argList << (video.isEmpty() ? QStringLiteral("FRAMES_DIR")
                                : QStringLiteral("VIDEO"))
            << (video.isEmpty() ? framesDir : video);
    for (auto it = optionKeys.constBegin(); it != optionKeys.constEnd(); ++it) {
        const QString value = params.value(it.key()).toString().trimmed();
        if (!value.isEmpty()) {
            argList << it.value() << value;
        }
    }
    argList << "TRACKS_JSON" << tracksJson;

    auto* process = new QProcess(this);
    process->setProgram(QCoreApplication::applicationFilePath());
    process->setArguments(argList);
    process->setProcessEnvironment(headlessChildEnvironment());
    return runChildProcess(
            process, timeoutMs,
            [tracksJson, timeoutMs](ChildOutcome outcome, QProcess& child) {
                if (outcome == ChildOutcome::StartFailed) {
                    return JsonRPCResult::error(
                            3, "Unable to start YOLO tracking child",
                            D("error", child.errorString()));
                }
                if (outcome == ChildOutcome::TimedOut) {
                    return JsonRPCResult::error(3, "YOLO tracking timed out",
                                                D("timeout_ms", timeoutMs));
                }
                QJsonObject result;
                result["exit_code"] = child.exitCode();
                result["stdout"] =
                        QString::fromUtf8(child.readAllStandardOutput())
                                .left(5000);
                result["stderr"] =
                        QString::fromUtf8(child.readAllStandardError())
                                .left(2000);
                result["tracks_json"] = tracksJson;
                if (outcome == ChildOutcome::Failed) {
                    return JsonRPCResult::error(
                            3, "YOLO tracking failed",
                            D("exit_code", child.exitCode(), "stderr",
                              result["stderr"].toString()));
                }
                return JsonRPCResult::success(
                        QJsonDocument(result).toVariant());
            });
}

JsonRPCResult JsonRPCPlugin::rpcSam3dGenerate(
        const QMap<QString, QVariant>& params) {
    const QVariantList imageList = params.value("images").toList();
    const QVariantList maskList = params.value("masks").toList();
    const QString outputDir = params.value("output_dir").toString();
    if (imageList.isEmpty() || outputDir.isEmpty()) {
        return JsonRPCResult::error(
                -32602, "Missing required parameters: images[] and output_dir");
    }

    QStringList images;
    for (const QVariant& v : imageList) {
        const QString path = v.toString().trimmed();
        if (path.isEmpty()) {
            return JsonRPCResult::error(-32602,
                                        "images[] contains an empty path");
        }
        images << path;
    }
    QStringList masks;
    if (!maskList.isEmpty() && maskList.size() != images.size()) {
        return JsonRPCResult::error(
                -32602,
                "masks[] must be empty or match images[] length (ordered "
                "companion files; pass null/empty strings for images without "
                "a mask)");
    }
    for (const QVariant& v : maskList) {
        masks << v.toString().trimmed();
    }

    // Default manifest path inside the output directory so the response can
    // embed the parsed per-image results.
    QString resultJson = params.value("result_json").toString().trimmed();
    if (resultJson.isEmpty()) {
        resultJson = QDir(outputDir).filePath(
                QStringLiteral("_sam3d_batch_result.json"));
    }

    int timeoutMs = 0;
    JsonRPCResult invalid;
    if (!childTimeout(params, 3600000, &timeoutMs, &invalid)) return invalid;

    QStringList argList;
    argList << "-SILENT" << "-NO_TIMESTAMP" << "-SAM3D_GENERATE";
    for (int i = 0; i < images.size(); ++i) {
        argList << "IMAGE" << images[i];
        if (i < masks.size() && !masks[i].isEmpty()) {
            argList << "MASK" << masks[i];
        }
    }
    argList << "OUT_DIR" << outputDir;

    // Optional key/value pairs in the -SAM3D_GENERATE argument order.
    const QHash<QString, QString> optionKeys = {
            {"device", "DEVICE"},
            {"dtype", "DTYPE"},
            {"steps", "STEPS"},
            {"seed", "SEED"},
            {"mesh", "MESH"},
            {"rmbg", "RMBG"},
            {"moge_cache", "MOGE_CACHE"},
            {"threads", "THREADS"},
            {"models_dir", "MODELS_DIR"},
    };
    for (auto it = optionKeys.constBegin(); it != optionKeys.constEnd(); ++it) {
        const QString value = params.value(it.key()).toString().trimmed();
        if (!value.isEmpty()) {
            argList << it.value() << value;
        }
    }
    argList << "RESULT_JSON" << resultJson;

    QElapsedTimer timer;
    timer.start();
    auto* process = new QProcess(this);
    process->setProgram(QCoreApplication::applicationFilePath());
    process->setArguments(argList);
    process->setProcessEnvironment(headlessChildEnvironment());
    const int imagesCount = images.size();
    return runChildProcess(
            process, timeoutMs,
            [resultJson, timer, timeoutMs, imagesCount](ChildOutcome outcome,
                                                        QProcess& child) {
                if (outcome == ChildOutcome::StartFailed) {
                    return JsonRPCResult::error(
                            3, "Unable to start SAM 3D batch child",
                            D("error", child.errorString()));
                }
                if (outcome == ChildOutcome::TimedOut) {
                    return JsonRPCResult::error(
                            3, "SAM 3D batch generation timed out",
                            D("timeout_ms", timeoutMs, "images_count",
                              imagesCount));
                }

                QJsonObject result;
                result["exit_code"] = child.exitCode();
                result["stdout"] =
                        QString::fromUtf8(child.readAllStandardOutput())
                                .left(5000);
                result["stderr"] =
                        QString::fromUtf8(child.readAllStandardError())
                                .left(2000);
                result["result_json"] = resultJson;
                result["elapsed_ms"] = static_cast<qint64>(timer.elapsed());

                // Embed the parsed manifest so clients do not need a second
                // file read.
                QFile jsonFile(resultJson);
                if (jsonFile.open(QIODevice::ReadOnly)) {
                    const QJsonDocument doc =
                            QJsonDocument::fromJson(jsonFile.readAll());
                    jsonFile.close();
                    if (doc.isObject()) {
                        result["items"] = doc.object().value("items").toArray();
                        result["ok_count"] = doc.object().value("ok").toInt();
                        result["failed_count"] =
                                doc.object().value("failed").toInt();
                        result["backend"] =
                                doc.object().value("backend").toString();
                    }
                }

                if (outcome == ChildOutcome::Failed) {
                    return JsonRPCResult::error(
                            3, "SAM 3D batch generation failed",
                            D("exit_code", child.exitCode(), "stderr",
                              result["stderr"].toString(), "items",
                              result["items"]));
                }
                return JsonRPCResult::success(
                        QJsonDocument(result).toVariant());
            });
}

JsonRPCResult JsonRPCPlugin::rpcMapAnythingReconstruct(
        const QMap<QString, QVariant>& params) {
    const QVariantList imageList = params.value("images").toList();
    const QString outputDir = params.value("output_dir").toString();
    if (imageList.isEmpty() || outputDir.isEmpty()) {
        return JsonRPCResult::error(
                -32602, "Missing required parameters: images[] and output_dir");
    }
    QStringList images;
    for (const QVariant& v : imageList) {
        const QString path = v.toString().trimmed();
        if (path.isEmpty()) {
            return JsonRPCResult::error(-32602,
                                        "images[] contains an empty path");
        }
        images << path;
    }

    QString resultJson = params.value("result_json").toString().trimmed();
    if (resultJson.isEmpty()) {
        resultJson = QDir(outputDir).filePath(
                QStringLiteral("_map_anything_result.json"));
    }
    int timeoutMs = 0;
    JsonRPCResult invalid;
    if (!childTimeout(params, 3600000, &timeoutMs, &invalid)) return invalid;

    QStringList argList;
    argList << "-SILENT" << "-NO_TIMESTAMP" << "-MAP_ANYTHING_RECONSTRUCT";
    for (const QString& image : images) {
        argList << "IMAGE" << image;
    }
    argList << "OUT_DIR" << outputDir;

    const QHash<QString, QString> optionKeys = {
            {"model", "MODEL"},
            {"device", "DEVICE"},
            {"image_size", "IMAGE_SIZE"},
            {"resize_mode", "RESIZE_MODE"},
            {"min_conf", "MIN_CONF"},
            {"max_frames_per_pass", "MAX_FRAMES_PER_PASS"},
            {"threads", "THREADS"},
            {"cloud_ply", "CLOUD_PLY"},
            {"per_view", "PER_VIEW"},
            {"raw_output_dir", "RAW_OUTPUT_DIR"},
            {"stage_dump_dir", "STAGE_DUMP_DIR"},
            {"gt_meta", "GT_META"},
            {"ref_images_path", "REF_IMAGES_PATH"},
            {"ref_is_gps", "REF_IS_GPS"},
    };
    for (auto it = optionKeys.constBegin(); it != optionKeys.constEnd(); ++it) {
        const QString value = params.value(it.key()).toString().trimmed();
        if (!value.isEmpty()) {
            argList << it.value() << value;
        }
    }
    argList << "RESULT_JSON" << resultJson;

    QElapsedTimer timer;
    timer.start();
    auto* process = new QProcess(this);
    process->setProgram(QCoreApplication::applicationFilePath());
    process->setArguments(argList);
    process->setProcessEnvironment(headlessChildEnvironment());
    const int imagesCount = images.size();
    return runChildProcess(
            process, timeoutMs,
            [resultJson, timer, timeoutMs, imagesCount](ChildOutcome outcome,
                                                        QProcess& child) {
                if (outcome == ChildOutcome::StartFailed) {
                    return JsonRPCResult::error(
                            3, "Unable to start MapAnything child",
                            D("error", child.errorString()));
                }
                if (outcome == ChildOutcome::TimedOut) {
                    return JsonRPCResult::error(
                            3, "MapAnything reconstruction timed out",
                            D("timeout_ms", timeoutMs, "images_count",
                              imagesCount));
                }

                QJsonObject result;
                result["exit_code"] = child.exitCode();
                result["stdout"] =
                        QString::fromUtf8(child.readAllStandardOutput())
                                .left(5000);
                result["stderr"] =
                        QString::fromUtf8(child.readAllStandardError())
                                .left(2000);
                result["result_json"] = resultJson;
                result["elapsed_ms"] = static_cast<qint64>(timer.elapsed());

                QFile jsonFile(resultJson);
                if (jsonFile.open(QIODevice::ReadOnly)) {
                    const QJsonDocument doc =
                            QJsonDocument::fromJson(jsonFile.readAll());
                    jsonFile.close();
                    if (doc.isObject()) {
                        const QJsonObject manifest = doc.object();
                        result["architecture"] =
                                manifest.value("architecture").toString();
                        result["points"] = manifest.value("points").toInt();
                        result["views"] = manifest.value("views").toInt();
                        result["e2e_ms"] = manifest.value("e2e_ms").toDouble();
                        result["ply"] = manifest.value("ply").toString();
                        result["frames_per_pass"] =
                                manifest.value("frames_per_pass").toInt();
                    }
                }

                if (outcome == ChildOutcome::Failed) {
                    return JsonRPCResult::error(
                            3, "MapAnything reconstruction failed",
                            D("exit_code", child.exitCode(), "stderr",
                              result["stderr"].toString()));
                }
                return JsonRPCResult::success(
                        QJsonDocument(result).toVariant());
            });
}
