// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qFBXIO.h"

#include "FBXCommand.h"
#include "FBXFilter.h"

qFBXIO::qFBXIO(QObject *parent)
    : QObject(parent), ccIOPluginInterface(":/CC/plugin/qFBXIO/info.json") {}

void qFBXIO::registerCommands(ccCommandLineInterface *cmd) {
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new FBXCommand));
}

ccIOPluginInterface::FilterList qFBXIO::getFilters() {
    return {FileIOFilter::Shared(new FBXFilter)};
}
