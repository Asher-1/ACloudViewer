// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qE57IO.h"

#include "E57Filter.h"

qE57IO::qE57IO(QObject *parent)
    : QObject(parent), ccIOPluginInterface(":/CC/plugin/qE57IO/info.json") {}

void qE57IO::registerCommands(ccCommandLineInterface *cmd) { Q_UNUSED(cmd); }

ccIOPluginInterface::FilterList qE57IO::getFilters() {
    return {FileIOFilter::Shared(new E57Filter)};
}
