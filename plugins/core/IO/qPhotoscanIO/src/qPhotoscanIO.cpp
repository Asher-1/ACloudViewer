// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qPhotoscanIO.h"

// local
#include "PhotoScanFilter.h"
#include "PhotoscanCommands.h"

qPhotoscanIO::qPhotoscanIO(QObject* parent)
    : QObject(parent),
      ccIOPluginInterface(":/CC/plugin/qPhotoscanIO/info.json") {}

ccIOPluginInterface::FilterList qPhotoscanIO::getFilters() {
    return {FileIOFilter::Shared(new PhotoScanFilter)};
}

void qPhotoscanIO::registerCommands(ccCommandLineInterface *cmd) {
    if (!cmd) return;
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandPhotoscan));
}
