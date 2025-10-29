// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecv2DViewportObject.h"

cc2DViewportObject::cc2DViewportObject(QString name /*=QString()*/)
    : ccHObject(name) {}

bool cc2DViewportObject::toFile_MeOnly(QFile& out) const {
    if (!ccHObject::toFile_MeOnly(out)) return false;

    // ecvViewportParameters (dataVersion>=20)
    if (!m_params.toFile(out)) return false;

    return true;
}

bool cc2DViewportObject::fromFile_MeOnly(QFile& in,
                                         short dataVersion,
                                         int flags,
                                         LoadedIDMap& oldToNewIDMap) {
    if (!ccHObject::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    // ecvViewportParameters (dataVersion>=20)
    if (!m_params.fromFile(in, dataVersion, flags, oldToNewIDMap)) return false;

    return true;
}
