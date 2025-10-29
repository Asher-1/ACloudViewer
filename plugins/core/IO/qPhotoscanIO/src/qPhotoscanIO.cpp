// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qPhotoscanIO.h"

// local
#include "PhotoScanFilter.h"

qPhotoscanIO::qPhotoscanIO(QObject* parent)
    : QObject(parent),
      ccIOPluginInterface(":/CC/plugin/qPhotoscanIO/info.json") {}

ccIOPluginInterface::FilterList qPhotoscanIO::getFilters() {
    return {FileIOFilter::Shared(new PhotoScanFilter)};
}
