// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../include/qDracoIO.h"

#include "../include/DRCFilter.h"

qDracoIO::qDracoIO(QObject *parent)
    : QObject(parent), ccIOPluginInterface(":/CC/plugin/qDracoIO/info.json") {}

void qDracoIO::registerCommands(ccCommandLineInterface *cmd) { Q_UNUSED(cmd); }

ccIOPluginInterface::FilterList qDracoIO::getFilters() {
    return {FileIOFilter::Shared(new DRCFilter)};
}
