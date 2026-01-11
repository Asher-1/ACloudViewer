// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecv2DViewportObject.h"

cc2DViewportObject::cc2DViewportObject(QString name /*=QString()*/)
    : ccHObject(name) {}

bool cc2DViewportObject::toFile_MeOnly(QFile& out, short dataVersion) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));
    if (dataVersion < 20) {
        assert(false);
        return false;
    }

    if (!ccHObject::toFile_MeOnly(out, dataVersion)) return false;

    // ecvViewportParameters (dataVersion>=20)
    if (!m_params.toFile(out, dataVersion)) return false;

    return true;
}

short cc2DViewportObject::minimumFileVersion_MeOnly() const {
    short minVersion = std::max(static_cast<short>(20),
                                ccHObject::minimumFileVersion_MeOnly());
    return std::max(minVersion, m_params.minimumFileVersion());
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
