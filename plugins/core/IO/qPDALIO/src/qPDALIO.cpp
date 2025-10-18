// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qPDALIO.h"

#include "LASFilter.h"

qPDALIO::qPDALIO(QObject *parent)
    : QObject(parent), ccIOPluginInterface(":/CC/plugin/qPDALIO/info.json") {}

void qPDALIO::registerCommands(ccCommandLineInterface *cmd) { Q_UNUSED(cmd); }

ccIOPluginInterface::FilterList qPDALIO::getFilters() {
    return {FileIOFilter::Shared(new LASFilter)};
}
