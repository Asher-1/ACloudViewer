// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PhotoscanCommands.h"

#include <QObject>

static const char COMMAND_PHOTOSCAN[] = "PHOTOSCAN";
static const char COMMAND_PS_LOAD_KEYPOINTS[] = "LOAD_KEYPOINTS";
static const char COMMAND_PS_LOAD_CAMERAS[] = "LOAD_CAMERAS";

CommandPhotoscan::CommandPhotoscan()
    : ccCommandLineInterface::Command("Photoscan", COMMAND_PHOTOSCAN) {}

bool CommandPhotoscan::process(ccCommandLineInterface& cmd) {
    cmd.print("[PHOTOSCAN]");

    bool loadKeypoints = false;
    bool loadCameras = false;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_PS_LOAD_KEYPOINTS)) {
            cmd.arguments().pop_front();
            loadKeypoints = true;
            cmd.print("[PHOTOSCAN] Load keypoints enabled");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_PS_LOAD_CAMERAS)) {
            cmd.arguments().pop_front();
            loadCameras = true;
            cmd.print("[PHOTOSCAN] Load cameras enabled");
        } else {
            break;
        }
    }

    return true;
}
