// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QStringList>

#include "SIBRViewerThread.h"
#include "ecvCommandLineInterface.h"

// Forward declarations for SIBR tool entry points
extern "C" {
int sibr_tool_alignMeshes(int argc, char** argv);
int sibr_tool_cameraConverter(int argc, char** argv);
int sibr_tool_clippingPlanes(int argc, const char** argv);
int sibr_tool_cropFromCenter(int argc, const char** argv);
int sibr_tool_distordCrop(int argc, const char* const* argv);
int sibr_tool_nvmToSIBR(int argc, const char** argv);
int sibr_tool_prepareColmap4Sibr(int argc, const char** argv);
int sibr_tool_textureMesh(int argc, char** argv);
int sibr_tool_tonemapper(int argc, char** argv);
int sibr_tool_unwrapMesh(int argc, char** argv);
}

static const char COMMAND_SIBR_TOOL[] = "SIBR_TOOL";
static const char COMMAND_SIBR_VIEWER[] = "SIBR_VIEWER";

struct CommandSIBRTool : public ccCommandLineInterface::Command {
    CommandSIBRTool()
        : ccCommandLineInterface::Command("SIBR Dataset Tool",
                                          COMMAND_SIBR_TOOL) {}

    bool process(ccCommandLineInterface& cmd) override {
        if (cmd.arguments().empty()) {
            return cmd.error(
                    QString("Missing tool name after \"-%1\". Available: "
                            "prepareColmap4Sibr, tonemapper, unwrapMesh, "
                            "textureMesh, clippingPlanes, cropFromCenter, "
                            "nvmToSIBR, distordCrop, cameraConverter, "
                            "alignMeshes")
                            .arg(COMMAND_SIBR_TOOL));
        }

        QString toolName = cmd.arguments().takeFirst();
        cmd.print(QString("[SIBR_TOOL] Running: %1").arg(toolName));

        QStringList toolArgs;
        toolArgs.append(toolName);
        while (!cmd.arguments().empty()) {
            QString arg = cmd.arguments().first();
            if (arg.startsWith("-") && arg.length() > 1 &&
                arg.at(1).isUpper()) {
                break;
            }
            toolArgs.append(cmd.arguments().takeFirst());
        }

        std::vector<std::string> storage;
        for (const auto& a : toolArgs) storage.push_back(a.toStdString());
        int argc = static_cast<int>(storage.size());
        std::vector<char*> argv_m(argc);
        std::vector<const char*> argv_c(argc);
        for (int i = 0; i < argc; ++i) {
            argv_m[i] = const_cast<char*>(storage[i].c_str());
            argv_c[i] = storage[i].c_str();
        }

        auto name = toolName.toStdString();
        int rc = -1;
        bool found = false;

        struct ToolEntry {
            const char* name;
            std::function<int()> fn;
        };
        ToolEntry table[] = {
                {"alignMeshes",
                 [&] { return sibr_tool_alignMeshes(argc, argv_m.data()); }},
                {"cameraConverter",
                 [&] {
                     return sibr_tool_cameraConverter(argc, argv_m.data());
                 }},
                {"clippingPlanes",
                 [&] { return sibr_tool_clippingPlanes(argc, argv_c.data()); }},
                {"cropFromCenter",
                 [&] { return sibr_tool_cropFromCenter(argc, argv_c.data()); }},
                {"distordCrop",
                 [&] { return sibr_tool_distordCrop(argc, argv_c.data()); }},
                {"nvmToSIBR",
                 [&] { return sibr_tool_nvmToSIBR(argc, argv_c.data()); }},
                {"prepareColmap4Sibr",
                 [&] {
                     return sibr_tool_prepareColmap4Sibr(argc, argv_c.data());
                 }},
                {"textureMesh",
                 [&] { return sibr_tool_textureMesh(argc, argv_m.data()); }},
                {"tonemapper",
                 [&] { return sibr_tool_tonemapper(argc, argv_m.data()); }},
                {"unwrapMesh",
                 [&] { return sibr_tool_unwrapMesh(argc, argv_m.data()); }},
        };

        for (auto& entry : table) {
            if (name == entry.name) {
                found = true;
                rc = entry.fn();
                break;
            }
        }

        if (!found) {
            return cmd.error(
                    QString("[SIBR_TOOL] Unknown tool: %1").arg(toolName));
        }

        if (rc != 0) {
            cmd.warning(QString("[SIBR_TOOL] %1 returned exit code %2")
                                .arg(toolName)
                                .arg(rc));
        } else {
            cmd.print(QString("[SIBR_TOOL] %1 completed successfully")
                              .arg(toolName));
        }

        return true;
    }
};

struct CommandSIBRViewer : public ccCommandLineInterface::Command {
    CommandSIBRViewer()
        : ccCommandLineInterface::Command("SIBR Viewer", COMMAND_SIBR_VIEWER) {}

    bool process(ccCommandLineInterface& cmd) override {
        if (cmd.arguments().empty()) {
            return cmd.error(
                    QString("Missing viewer name after \"-%1\".\n"
                            "Available viewers:\n"
                            "  ulr            - ULR (Unstructured Lumigraph "
                            "Rendering)\n"
                            "  ulrv2          - ULR v2/v3 with texture arrays\n"
                            "  texturedMesh   - Textured mesh viewer\n"
                            "  pointBased     - Point-based rendering\n"
                            "  gaussian       - 3D Gaussian Splatting "
                            "(requires CUDA)\n"
                            "  remoteGaussian - Remote Gaussian training "
                            "viewer\n"
                            "\n"
                            "Options (pass after viewer name):\n"
                            "  --path <dir>        Dataset directory\n"
                            "  --model-path <dir>  Gaussian model path\n"
                            "  --width <w>         Window width  (default: "
                            "1280)\n"
                            "  --height <h>        Window height (default: "
                            "720)\n"
                            "  --iteration <n>     Gaussian iteration to load\n"
                            "  --device <id>       CUDA device ID (default: "
                            "0)\n"
                            "  --no-interop        Disable CUDA-GL interop\n"
                            "  --ip <addr>         Remote server IP (default: "
                            "127.0.0.1)\n"
                            "  --port <n>          Remote server port "
                            "(default: 6009)")
                            .arg(COMMAND_SIBR_VIEWER));
        }

        QString viewerName = cmd.arguments().takeFirst();

        QStringList viewerArgs;
        while (!cmd.arguments().empty()) {
            QString arg = cmd.arguments().first();
            if (arg.startsWith("-") && arg.length() > 1 &&
                arg.at(1).isUpper()) {
                break;
            }
            viewerArgs.append(cmd.arguments().takeFirst());
        }

        auto takeOpt = [&](const QString& key) -> QString {
            int idx = viewerArgs.indexOf(key);
            if (idx >= 0 && idx + 1 < viewerArgs.size()) {
                viewerArgs.removeAt(idx);
                QString val = viewerArgs.takeAt(idx);
                return val;
            }
            return {};
        };
        auto hasFlag = [&](const QString& key) -> bool {
            int idx = viewerArgs.indexOf(key);
            if (idx >= 0) {
                viewerArgs.removeAt(idx);
                return true;
            }
            return false;
        };

        QString datasetPath = takeOpt("--path");
        QString modelPath = takeOpt("--model-path");
        int width = takeOpt("--width").toInt();
        int height = takeOpt("--height").toInt();
        QString iteration = takeOpt("--iteration");
        int device = takeOpt("--device").toInt();
        bool noInterop = hasFlag("--no-interop");
        QString ip = takeOpt("--ip");
        int port = takeOpt("--port").toInt();

        if (width <= 0) width = 1280;
        if (height <= 0) height = 720;
        if (ip.isEmpty()) ip = "127.0.0.1";
        if (port <= 0) port = 6009;

        SIBRViewerThread::ViewerMode mode;
        auto name = viewerName.toLower();

        if (name == "ulr") {
            mode = SIBRViewerThread::ViewerMode::ULR;
        } else if (name == "ulrv2") {
            mode = SIBRViewerThread::ViewerMode::ULRv2;
        } else if (name == "texturedmesh") {
            mode = SIBRViewerThread::ViewerMode::TexturedMesh;
        } else if (name == "pointbased") {
            mode = SIBRViewerThread::ViewerMode::PointBased;
        } else if (name == "gaussian") {
            mode = SIBRViewerThread::ViewerMode::GaussianSplatting;
        } else if (name == "remotegaussian") {
            mode = SIBRViewerThread::ViewerMode::RemoteGaussian;
        } else {
            return cmd.error(QString("[SIBR_VIEWER] Unknown viewer: %1")
                                     .arg(viewerName));
        }

        if (mode != SIBRViewerThread::ViewerMode::RemoteGaussian &&
            mode != SIBRViewerThread::ViewerMode::GaussianSplatting &&
            datasetPath.isEmpty()) {
            return cmd.error(
                    QString("[SIBR_VIEWER] --path is required for %1 viewer")
                            .arg(viewerName));
        }

        if (mode == SIBRViewerThread::ViewerMode::GaussianSplatting &&
            modelPath.isEmpty()) {
            return cmd.error(
                    "[SIBR_VIEWER] --model-path is required for gaussian "
                    "viewer");
        }

        cmd.print(QString("[SIBR_VIEWER] Launching %1 viewer ...")
                          .arg(viewerName));

        auto* thread = new SIBRViewerThread(mode, datasetPath, width, height);

        if (mode == SIBRViewerThread::ViewerMode::GaussianSplatting) {
            thread->setModelPath(modelPath);
            if (!iteration.isEmpty()) thread->setIteration(iteration);
            thread->setCudaDevice(device);
            thread->setNoInterop(noInterop);
        } else if (mode == SIBRViewerThread::ViewerMode::RemoteGaussian) {
            thread->setRemoteAddress(ip, port);
        }

        thread->start();
        thread->wait();
        int exitCode = thread->isFinished() ? 0 : 1;
        delete thread;

        if (exitCode != 0) {
            cmd.warning(QString("[SIBR_VIEWER] %1 returned exit code %2")
                                .arg(viewerName)
                                .arg(exitCode));
        } else {
            cmd.print(QString("[SIBR_VIEWER] %1 closed").arg(viewerName));
        }

        return true;
    }
};
