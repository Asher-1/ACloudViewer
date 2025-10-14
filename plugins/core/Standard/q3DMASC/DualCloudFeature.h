// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ##########################################################################
// #                                                                        #
// #                     ACLOUDVIEWER PLUGIN: q3DMASC                       #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                 COPYRIGHT: Dimitri Lague / CNRS / UEB                  #
// #                                                                        #
// ##########################################################################

// Local
#include "FeaturesInterface.h"

namespace masc {
//! Dual-cloud feature
struct DualCloudFeature : public Feature {
public:  // DualCloudFeatureType
    enum DualCloudFeatureType { Invalid = 0, IDIFF };

    static QString ToString(DualCloudFeatureType type) {
        switch (type) {
            case Invalid:
                return "Invalid";
            case IDIFF:
                return "IDIFF";
            default:
                assert(false);
                break;
        }
        return "Invalid";
    }

    static inline DualCloudFeatureType FromString(const QString& token) {
        return FromUpperString(token.toUpper());
    }
    static DualCloudFeatureType FromUpperString(const QString& token) {
        if (token == "IDIFF") return IDIFF;

        return Invalid;
    }

public:  // methods
    //! Default constructor
    DualCloudFeature(DualCloudFeatureType p_type) : type(p_type) {}

    // inherited from Feature
    virtual Type getType() const override { return Type::DualCloudFeature; }
    virtual Feature::Shared clone() const override {
        return Feature::Shared(new DualCloudFeature(*this));
    }
    virtual bool prepare(
            const CorePoints& corePoints,
            QString& error,
            cloudViewer::GenericProgressCallback* progressCb = nullptr,
            SFCollector* generatedScalarFields = nullptr) override;
    virtual bool checkValidity(QString corePointRole,
                               QString& error) const override;
    virtual QString toString() const override;

    //! Dual-cloud feature type
    /** \warning different from the feature type
     **/
    DualCloudFeatureType type;
};
}  // namespace masc
