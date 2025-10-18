// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qLASFWFIO.h"

// local
#include "LASFWFFilter.h"
#include "qLASFWFIOCommands.h"

// Qt
#include <QtPlugin>

qLASFWFIO::qLASFWFIO(QObject* parent)
    : QObject(parent), ccIOPluginInterface(":/CC/plugin/qLASFWFIO/info.json") {}

ccIOPluginInterface::FilterList qLASFWFIO::getFilters() {
    return {FileIOFilter::Shared(new LASFWFFilter)};
}

void qLASFWFIO::registerCommands(ccCommandLineInterface* cmd) {
    if (!cmd) {
        assert(false);
        return;
    }

    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandLoadLASFWF));
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandSaveLASFWF));
}
