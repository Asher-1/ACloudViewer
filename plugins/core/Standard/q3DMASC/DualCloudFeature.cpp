// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DualCloudFeature.h"

using namespace masc;

bool DualCloudFeature::prepare(
        const CorePoints& corePoints,
        QString& error,
        cloudViewer::GenericProgressCallback* progressCb /*=nullptr*/,
        SFCollector* generatedScalarFields /*=nullptr*/) {
    // TODO
    return false;
}

QString DualCloudFeature::toString() const {
    // use the default keyword + "_SC" + the scale
    return ToString(type) + "_SC" + QString::number(scale);
}

bool DualCloudFeature::checkValidity(QString corePointRole,
                                     QString& error) const {
    if (!Feature::checkValidity(corePointRole, error)) {
        return false;
    }

    unsigned char cloudCount = (cloud1 ? (cloud2 ? 2 : 1) : 0);
    if (cloudCount < 2) {
        error = "at least two clouds are required to compute context-based "
                "features";
        return false;
    }

    if (op != NO_OPERATION) {
        error = "math operations can't be defined on dual-cloud features";
        return false;
    }

    return true;
}
