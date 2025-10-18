// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qAdditionalIO.h"

#include "BundlerCommand.h"
#include "BundlerFilter.h"
#include "IcmFilter.h"
#include "PNFilter.h"
#include "PVFilter.h"
#include "PovFilter.h"
#include "SalomeHydroFilter.h"
#include "SinusxFilter.h"
#include "SoiFilter.h"

qAdditionalIO::qAdditionalIO(QObject* parent)
    : QObject(parent),
      ccIOPluginInterface(":/CC/plugin/qAdditionalIO/info.json") {}

void qAdditionalIO::registerCommands(ccCommandLineInterface* cmd) {
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new BundlerCommand));
}

ccIOPluginInterface::FilterList qAdditionalIO::getFilters() {
    return {
            FileIOFilter::Shared(new BundlerFilter),
            FileIOFilter::Shared(new IcmFilter),
            FileIOFilter::Shared(new PNFilter),
            FileIOFilter::Shared(new PovFilter),
            FileIOFilter::Shared(new PVFilter),
            FileIOFilter::Shared(new SalomeHydroFilter),
            FileIOFilter::Shared(new SinusxFilter),
            FileIOFilter::Shared(new SoiFilter),
    };
}
