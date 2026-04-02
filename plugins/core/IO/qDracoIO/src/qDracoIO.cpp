// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../include/qDracoIO.h"

#include "../include/DRCFilter.h"
#include "../include/DracoCommands.h"

qDracoIO::qDracoIO(QObject *parent)
    : QObject(parent), ccIOPluginInterface(":/CC/plugin/qDracoIO/info.json") {}

void qDracoIO::registerCommands(ccCommandLineInterface *cmd) {
    if (!cmd) return;
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandDraco));
}

ccIOPluginInterface::FilterList qDracoIO::getFilters() {
    return {FileIOFilter::Shared(new DRCFilter)};
}
