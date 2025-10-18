// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qRDBIO.h"

#include "RDBFilter.h"

qRDBIO::qRDBIO(QObject* parent)
    : QObject(parent), ccIOPluginInterface(":/CC/plugin/qRDBIO/info.json") {}

ccIOPluginInterface::FilterList qRDBIO::getFilters() {
    return {
            FileIOFilter::Shared(new RDBFilter),
    };
}
