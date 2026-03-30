// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qAnimationCommands.h"

#include <QObject>

static const char COMMAND_ANIMATION[] = "ANIMATION";
static const char COMMAND_ANIM_FPS[] = "FPS";
static const char COMMAND_ANIM_TOTAL_FRAMES[] = "TOTAL_FRAMES";
static const char COMMAND_ANIM_SUPER_RES[] = "SUPER_RESOLUTION";
static const char COMMAND_ANIM_OUTPUT[] = "OUTPUT";

CommandAnimation::CommandAnimation()
        : ccCommandLineInterface::Command("Animation", COMMAND_ANIMATION) {}

bool CommandAnimation::process(ccCommandLineInterface& cmd) {
    cmd.print("[ANIMATION]");

    int fps = 30;
    int totalFrames = 0;
    int superResolution = 1;
    QString outputFile;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_ANIM_FPS)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_ANIM_FPS));
            bool ok;
            fps = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || fps < 1)
                return cmd.error("Invalid value for -FPS");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_ANIM_TOTAL_FRAMES)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_ANIM_TOTAL_FRAMES));
            bool ok;
            totalFrames = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || totalFrames < 1)
                return cmd.error("Invalid value for -TOTAL_FRAMES");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_ANIM_SUPER_RES)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_ANIM_SUPER_RES));
            bool ok;
            superResolution = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || superResolution < 1)
                return cmd.error("Invalid value for -SUPER_RESOLUTION");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_ANIM_OUTPUT)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_ANIM_OUTPUT));
            outputFile = cmd.arguments().takeFirst();
        } else {
            break;
        }
    }

    cmd.print(QObject::tr("[ANIMATION] Settings - FPS: %1, Total frames: %2, "
                          "Super resolution: %3")
                      .arg(fps)
                      .arg(totalFrames)
                      .arg(superResolution));

    if (!outputFile.isEmpty()) {
        cmd.print(QObject::tr("[ANIMATION] Output file: %1").arg(outputFile));
    }

    cmd.print(QObject::tr("[ANIMATION] Animation parameters configured "
                          "successfully. Use with GUI mode for rendering."));

    return true;
}
