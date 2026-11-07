// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Child-process methods: colmap.*, process.run_cli and its csf/m3c2/ransac
// wrappers. Every child runs through runChildProcess().

#include <ecvMainAppInterface.h>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QProcess>

#include "JsonRPCChildProcess.h"
#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"

using jsonrpc::ChildOutcome;
using jsonrpc::childTimeout;
using jsonrpc::D;
using jsonrpc::headlessChildEnvironment;
using jsonrpc::runChildProcess;

// ═══════════════════════════════════════════════════════════════════════════
// colmap.reconstruct
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcColmapReconstruct(
        const QMap<QString, QVariant>& params) {
    QString imagePath = params.value("image_path").toString();
    QString workspace = params.value("workspace_path").toString();
    if (imagePath.isEmpty() || workspace.isEmpty()) {
        return JsonRPCResult::error(
                -32602,
                "Missing 'image_path' and/or 'workspace_path' parameters",
                D("image_path", imagePath, "workspace_path", workspace));
    }

    QString colmapBin = params.value("colmap_binary", "colmap").toString();
    QString quality = params.value("quality", "HIGH").toString();
    QString dataType = params.value("data_type", "INDIVIDUAL").toString();
    QString mesher = params.value("mesher", "POISSON").toString();
    QString cameraModel = params.value("camera_model", "").toString();
    bool singleCamera = params.value("single_camera", false).toBool();
    bool useGpu = params.value("use_gpu", true).toBool();

    QDir().mkpath(workspace);

    QStringList args;
    args << "automatic_reconstructor"
         << "--workspace_path" << workspace << "--image_path" << imagePath
         << "--quality" << quality << "--data_type" << dataType << "--mesher"
         << mesher;
    if (!cameraModel.isEmpty()) args << "--camera_model" << cameraModel;
    if (singleCamera) args << "--single_camera" << "1";
    if (!useGpu) args << "--use_gpu" << "0";

    int timeoutMs = 0;
    JsonRPCResult invalid;
    if (!childTimeout(params, 7200000, &timeoutMs, &invalid)) return invalid;
    const bool importResults = params.value("import_results", true).toBool();

    auto* process = new QProcess(this);
    process->setProgram(colmapBin);
    process->setArguments(args);
    // Results are imported when the child ends, so nothing in the scene is
    // held across the wait.
    return runChildProcess(
            process, timeoutMs,
            [this, colmapBin, args, timeoutMs, workspace, imagePath, quality,
             importResults](ChildOutcome outcome, QProcess& child) {
                const QString commandLine = colmapBin + " " + args.join(" ");
                if (outcome == ChildOutcome::StartFailed) {
                    return JsonRPCResult::error(3, "Unable to start Colmap",
                                                D("command", colmapBin, "error",
                                                  child.errorString()));
                }
                if (outcome == ChildOutcome::TimedOut) {
                    return JsonRPCResult::error(
                            3, "Colmap timed out",
                            D("timeout_ms", timeoutMs, "command", commandLine));
                }
                if (outcome == ChildOutcome::Failed) {
                    QString errOut = child.readAllStandardError().trimmed();
                    if (errOut.isEmpty())
                        errOut = child.readAllStandardOutput().trimmed();
                    return JsonRPCResult::error(
                            3, "Colmap failed",
                            D("exit_code", child.exitCode(), "stderr",
                              errOut.left(500), "command", commandLine));
                }
                QJsonObject result;
                result["workspace"] = workspace;
                result["image_path"] = imagePath;
                result["quality"] = quality;
                result["status"] = "completed";

                QJsonObject outputs;
                for (int idx = 0; idx < 10; ++idx) {
                    QString denseDir =
                            workspace + "/dense/" + QString::number(idx);
                    if (!QDir(denseDir).exists()) break;

                    QString fusedPly = denseDir + "/fused.ply";
                    if (QFile::exists(fusedPly)) {
                        QJsonArray arr = outputs.value("fused_ply").toArray();
                        arr.append(fusedPly);
                        outputs["fused_ply"] = arr;
                    }
                    for (const QString& meshName :
                         {"meshed-poisson.ply", "meshed-delaunay.ply"}) {
                        QString meshPath = denseDir + "/" + meshName;
                        if (QFile::exists(meshPath)) {
                            QJsonArray arr = outputs.value("mesh").toArray();
                            arr.append(meshPath);
                            outputs["mesh"] = arr;
                        }
                    }
                    QString texturedObj = denseDir + "/textured.obj";
                    if (QFile::exists(texturedObj)) {
                        QJsonArray arr =
                                outputs.value("textured_mesh").toArray();
                        arr.append(texturedObj);
                        outputs["textured_mesh"] = arr;
                    }
                }
                result["outputs"] = outputs;

                if (importResults && m_app) {
                    QJsonArray imported;
                    auto tryImport = [&](const QString& key) {
                        QJsonArray paths = outputs.value(key).toArray();
                        for (const QJsonValue& v : paths) {
                            QString path = v.toString();
                            if (!QFile::exists(path)) continue;
                            QMap<QString, QVariant> openParams;
                            openParams["filename"] = path;
                            openParams["silent"] = true;
                            auto openResult = rpcOpen(openParams);
                            if (!openResult.isError) imported.append(path);
                        }
                    };
                    tryImport("textured_mesh");
                    tryImport("mesh");
                    tryImport("fused_ply");
                    result["imported"] = imported;
                }

                return JsonRPCResult::success(
                        QJsonDocument(result).toVariant());
            });
}

// ═══════════════════════════════════════════════════════════════════════════
// colmap.run (NEW) — generic COLMAP subcommand executor
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcColmapRun(
        const QMap<QString, QVariant>& params) {
    QString command = params["command"].toString();
    if (command.isEmpty()) {
        return JsonRPCResult::error(-32602, "Missing 'command' parameter",
                                    D("param", "command"));
    }

    QString colmapBin = params.value("colmap_binary", "colmap").toString();
    int timeoutMs = 0;
    JsonRPCResult invalid;
    if (!childTimeout(params, 3600000, &timeoutMs, &invalid)) return invalid;

    QStringList argList;
    argList << command;
    QVariantList userArgs = params.value("args", QVariant()).toList();
    for (const auto& a : userArgs) {
        argList << a.toString();
    }

    QVariantMap kvArgs = params.value("kwargs", QVariant()).toMap();
    for (auto it = kvArgs.constBegin(); it != kvArgs.constEnd(); ++it) {
        argList << ("--" + it.key()) << it.value().toString();
    }

    auto* process = new QProcess(this);
    process->setProgram(colmapBin);
    process->setArguments(argList);
    return runChildProcess(
            process, timeoutMs,
            [command, colmapBin, argList, timeoutMs](ChildOutcome outcome,
                                                     QProcess& child) {
                if (outcome == ChildOutcome::StartFailed) {
                    return JsonRPCResult::error(3, "Unable to start Colmap",
                                                D("command", colmapBin, "error",
                                                  child.errorString()));
                }
                if (outcome == ChildOutcome::TimedOut) {
                    return JsonRPCResult::error(
                            3, "Colmap command timed out",
                            D("command", command, "timeout_ms", timeoutMs));
                }

                QString stdOut = child.readAllStandardOutput().trimmed();
                QString stdErr = child.readAllStandardError().trimmed();

                if (outcome == ChildOutcome::Failed) {
                    return JsonRPCResult::error(
                            3, "Colmap command failed",
                            D("command", command, "exit_code", child.exitCode(),
                              "stdout", stdOut.left(2000), "stderr",
                              stdErr.left(2000), "full_command",
                              QString(colmapBin + " " + argList.join(" "))));
                }

                QJsonObject result;
                result["command"] = command;
                result["exit_code"] = 0;
                result["stdout"] = stdOut.left(5000);
                result["stderr"] = stdErr.left(2000);
                return JsonRPCResult::success(
                        QJsonDocument(result).toVariant());
            });
}

// ═══════════════════════════════════════════════════════════════════════════
// Generic CLI Processing (process.run_cli)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcProcessRunCli(
        const QMap<QString, QVariant>& params) {
    QString inputPath = params.value("input_path").toString();
    QString outputPath = params.value("output_path").toString();
    if (inputPath.isEmpty() || outputPath.isEmpty()) {
        return JsonRPCResult::error(
                -32602, "Missing required parameters: input_path, output_path");
    }

    QVariantList cmdArgs = params.value("args", QVariant()).toList();
    if (cmdArgs.isEmpty()) {
        return JsonRPCResult::error(-32602,
                                    "Missing required parameter: args (list)");
    }

    int timeoutMs = 0;
    JsonRPCResult invalid;
    if (!childTimeout(params, 120000, &timeoutMs, &invalid)) return invalid;

    QString binary = QCoreApplication::applicationFilePath();
    QStringList argList;
    argList << "-SILENT" << "-O" << inputPath << "-AUTO_SAVE" << "OFF"
            << "-NO_TIMESTAMP";
    for (const auto& a : cmdArgs) {
        argList << a.toString();
    }
    argList << "-SAVE_CLOUDS" << "FILE" << outputPath;

    auto* process = new QProcess(this);
    process->setProgram(binary);
    process->setArguments(argList);
    process->setProcessEnvironment(headlessChildEnvironment());
    return runChildProcess(
            process, timeoutMs,
            [outputPath, timeoutMs](ChildOutcome outcome, QProcess& child) {
                if (outcome == ChildOutcome::StartFailed) {
                    return JsonRPCResult::error(
                            3, "Unable to start CLI processing child",
                            D("error", child.errorString()));
                }
                if (outcome == ChildOutcome::TimedOut) {
                    return JsonRPCResult::error(3, "CLI processing timed out",
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
                result["output_path"] = outputPath;

                if (outcome == ChildOutcome::Failed) {
                    return JsonRPCResult::error(
                            3, "CLI processing failed",
                            D("exit_code", child.exitCode(), "stderr",
                              result["stderr"].toString()));
                }
                return JsonRPCResult::success(
                        QJsonDocument(result).toVariant());
            });
}

JsonRPCResult JsonRPCPlugin::rpcProcessCsf(
        const QMap<QString, QVariant>& params) {
    QString input = params.value("input_path").toString();
    QString output = params.value("output_path").toString();
    if (input.isEmpty() || output.isEmpty()) {
        return JsonRPCResult::error(
                -32602, "Missing required: input_path, output_path");
    }
    QVariantList args;
    args << "-CSF";
    QString scene = params.value("scene", "").toString().toUpper();
    if (!scene.isEmpty()) args << "-SCENES" << scene;
    if (params.contains("cloth_resolution"))
        args << "-CLOTH_RESOLUTION"
             << QString::number(params["cloth_resolution"].toDouble());
    if (params.contains("max_iterations"))
        args << "-MAX_ITERATION"
             << QString::number(params["max_iterations"].toInt());
    if (params.contains("class_threshold"))
        args << "-CLASS_THRESHOLD"
             << QString::number(params["class_threshold"].toDouble());
    if (params.value("export_ground", false).toBool()) args << "-EXPORT_GROUND";
    if (params.value("export_offground", false).toBool())
        args << "-EXPORT_OFFGROUND";
    QMap<QString, QVariant> p;
    p["input_path"] = input;
    p["output_path"] = output;
    p["args"] = args;
    p["timeout_ms"] = params.value("timeout_ms", 120000);
    return rpcProcessRunCli(p);
}

JsonRPCResult JsonRPCPlugin::rpcProcessM3c2(
        const QMap<QString, QVariant>& params) {
    QString cloud1 = params.value("cloud1_path").toString();
    QString cloud2 = params.value("cloud2_path").toString();
    QString paramsFile = params.value("params_file").toString();
    QString output = params.value("output_path").toString();
    if (cloud1.isEmpty() || cloud2.isEmpty() || paramsFile.isEmpty() ||
        output.isEmpty()) {
        return JsonRPCResult::error(-32602,
                                    "Missing required: cloud1_path, "
                                    "cloud2_path, params_file, output_path");
    }
    int timeoutMs = 0;
    JsonRPCResult invalid;
    if (!childTimeout(params, 120000, &timeoutMs, &invalid)) return invalid;
    QString binary = QCoreApplication::applicationFilePath();
    QStringList argList;
    argList << "-SILENT" << "-O" << cloud1 << "-O" << cloud2 << "-AUTO_SAVE"
            << "OFF" << "-NO_TIMESTAMP" << "-M3C2" << paramsFile
            << "-SAVE_CLOUDS" << "FILE" << output;
    auto* process = new QProcess(this);
    process->setProgram(binary);
    process->setArguments(argList);
    process->setProcessEnvironment(headlessChildEnvironment());
    return runChildProcess(
            process, timeoutMs,
            [output](ChildOutcome outcome, QProcess& child) {
                if (outcome == ChildOutcome::StartFailed) {
                    return JsonRPCResult::error(
                            3, "Unable to start M3C2 child",
                            D("error", child.errorString()));
                }
                if (outcome == ChildOutcome::TimedOut) {
                    return JsonRPCResult::error(3, "M3C2 processing timed out");
                }
                QJsonObject result;
                result["exit_code"] = child.exitCode();
                result["stdout"] =
                        QString::fromUtf8(child.readAllStandardOutput())
                                .left(5000);
                result["stderr"] =
                        QString::fromUtf8(child.readAllStandardError())
                                .left(2000);
                result["output_path"] = output;
                if (outcome == ChildOutcome::Failed) {
                    return JsonRPCResult::error(
                            3, "M3C2 processing failed",
                            D("exit_code", child.exitCode()));
                }
                return JsonRPCResult::success(
                        QJsonDocument(result).toVariant());
            });
}

JsonRPCResult JsonRPCPlugin::rpcProcessRansac(
        const QMap<QString, QVariant>& params) {
    QString input = params.value("input_path").toString();
    QString output = params.value("output_path").toString();
    if (input.isEmpty() || output.isEmpty()) {
        return JsonRPCResult::error(
                -32602, "Missing required: input_path, output_path");
    }
    QVariantList args;
    args << "-RANSAC";
    if (params.contains("epsilon"))
        args << "-EPSILON_ABSOLUTE"
             << QString::number(params["epsilon"].toDouble());
    if (params.contains("bitmap_epsilon"))
        args << "-BITMAP_EPSILON_ABSOLUTE"
             << QString::number(params["bitmap_epsilon"].toDouble());
    if (params.contains("support_points"))
        args << "-SUPPORT_POINTS"
             << QString::number(params["support_points"].toInt());
    if (params.contains("max_normal_dev"))
        args << "-MAX_NORMAL_DEV"
             << QString::number(params["max_normal_dev"].toDouble());
    if (params.contains("probability"))
        args << "-PROBABILITY"
             << QString::number(params["probability"].toDouble());
    QVariantList prims = params.value("primitives").toList();
    for (const auto& p : prims) {
        args << "-ENABLE_PRIMITIVE" << p.toString().toUpper();
    }
    QMap<QString, QVariant> p;
    p["input_path"] = input;
    p["output_path"] = output;
    p["args"] = args;
    p["timeout_ms"] = params.value("timeout_ms", 120000);
    return rpcProcessRunCli(p);
}
