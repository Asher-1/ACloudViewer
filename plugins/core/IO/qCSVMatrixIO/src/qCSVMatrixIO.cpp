// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qCSVMatrixIO.h"

// local
#include "CSVMatrixCommands.h"
#include "CSVMatrixFilter.h"

qCSVMatrixIO::qCSVMatrixIO(QObject *parent)
    : QObject(parent),
      ccIOPluginInterface(":/CC/plugin/qCSVMatrixIO/info.json") {}

ccIOPluginInterface::FilterList qCSVMatrixIO::getFilters() {
    return {FileIOFilter::Shared(new CSVMatrixFilter)};
}

void qCSVMatrixIO::registerCommands(ccCommandLineInterface *cmd) {
    if (!cmd) return;
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandCSVMatrix));
}
