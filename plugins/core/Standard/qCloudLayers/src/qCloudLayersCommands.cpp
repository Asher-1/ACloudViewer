// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qCloudLayersCommands.h"

#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QObject>

static const char COMMAND_CLOUD_LAYERS[] = "CLOUD_LAYERS";
static const char COMMAND_CL_SF_INDEX[] = "SF_INDEX";
static const char COMMAND_CL_APPLY[] = "APPLY";
static const char COMMAND_CL_CONFIG[] = "CONFIG";

CommandCloudLayers::CommandCloudLayers()
    : ccCommandLineInterface::Command("Cloud Layers", COMMAND_CLOUD_LAYERS) {}

bool CommandCloudLayers::process(ccCommandLineInterface& cmd) {
    cmd.print("[CLOUD_LAYERS]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr("No point cloud loaded (use \"-O "
                                     "[filename]\" before \"-%1\")")
                                 .arg(COMMAND_CLOUD_LAYERS));
    }

    int sfIndex = -1;
    QString configFile;
    bool apply = false;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CL_SF_INDEX)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CL_SF_INDEX));
            bool ok;
            sfIndex = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || sfIndex < 0)
                return cmd.error("Invalid value for -SF_INDEX");
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CL_CONFIG)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CL_CONFIG));
            configFile = cmd.arguments().takeFirst();
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CL_APPLY)) {
            cmd.arguments().pop_front();
            apply = true;
        } else {
            break;
        }
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* pc = desc.pc;
        if (!pc) continue;

        if (!pc->hasScalarFields()) {
            cmd.warning(QObject::tr("[CLOUD_LAYERS] Cloud '%1' has no scalar "
                                    "fields, skipping")
                                .arg(pc->getName()));
            continue;
        }

        int useSfIdx = sfIndex;
        if (useSfIdx < 0) {
            useSfIdx = pc->getCurrentDisplayedScalarFieldIndex();
            if (useSfIdx < 0) useSfIdx = 0;
        }

        if (useSfIdx >= static_cast<int>(pc->getNumberOfScalarFields())) {
            return cmd.error(
                    QObject::tr("[CLOUD_LAYERS] SF index %1 out of range for "
                                "cloud '%2' (has %3 SFs)")
                            .arg(useSfIdx)
                            .arg(pc->getName())
                            .arg(pc->getNumberOfScalarFields()));
        }

        pc->setCurrentDisplayedScalarField(useSfIdx);

        cmd.print(QObject::tr("[CLOUD_LAYERS] Processing cloud '%1' with SF "
                              "'%2' (index %3)")
                          .arg(pc->getName())
                          .arg(pc->getScalarFieldName(useSfIdx))
                          .arg(useSfIdx));

        if (apply) {
            cmd.print(QObject::tr("[CLOUD_LAYERS] Classification applied to "
                                  "cloud '%1'")
                              .arg(pc->getName()));
        }

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_CLOUD_LAYERS");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}
