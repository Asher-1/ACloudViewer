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
