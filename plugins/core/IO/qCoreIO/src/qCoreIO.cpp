// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qCoreIO.h"

#include "HeightProfileFilter.h"
#include "MAFilter.h"
#include "MascaretFilter.h"
#include "PDMSFilter.h"
#include "SimpleBinFilter.h"

qCoreIO::qCoreIO(QObject *parent)
    : QObject(parent), ccIOPluginInterface(":/CC/plugin/CoreIO/info.json") {}

void qCoreIO::registerCommands(ccCommandLineInterface *inCmdLine) {
    Q_UNUSED(inCmdLine);
}

ccIOPluginInterface::FilterList qCoreIO::getFilters() {
    return {
            FileIOFilter::Shared(new SimpleBinFilter),
            FileIOFilter::Shared(new PDMSFilter),
            FileIOFilter::Shared(new MAFilter),
            FileIOFilter::Shared(new MascaretFilter),
            FileIOFilter::Shared(new HeightProfileFilter),
    };
}
