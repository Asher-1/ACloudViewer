// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qCSVMatrixIO.h"

// local
#include "CSVMatrixFilter.h"

qCSVMatrixIO::qCSVMatrixIO(QObject *parent)
    : QObject(parent),
      ccIOPluginInterface(":/CC/plugin/qCSVMatrixIO/info.json") {}

ccIOPluginInterface::FilterList qCSVMatrixIO::getFilters() {
    return {FileIOFilter::Shared(new CSVMatrixFilter)};
}
