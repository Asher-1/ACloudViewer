// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../include/qStepCADImport.h"

#include "../include/STEPFilter.h"

qStepCADImport::qStepCADImport(QObject *parent)
    : QObject(parent),
      ccIOPluginInterface(":/CC/plugin/qStepCADImport/info.json") {}

void qStepCADImport::registerCommands(ccCommandLineInterface *inCmdLine) {
    Q_UNUSED(inCmdLine);
}

ccIOPluginInterface::FilterList qStepCADImport::getFilters() {
    return {FileIOFilter::Shared(new STEPFilter)};
}
