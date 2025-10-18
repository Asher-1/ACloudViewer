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
#include "ScalarFieldWrappers.h"

// Qt
#include <QSharedPointer>

namespace masc {
//! Point feature
struct PointFeature : public Feature {
public:  // PointFeatureType
    typedef QSharedPointer<PointFeature> Shared;

    enum PointFeatureType {
        Invalid = 0,
        Intensity,
        X,
        Y,
        Z,
        NbRet,
        RetNb,
        EchoRat,
        R,
        G,
        B,
        NIR,
        Dip,
        DipDir,
        M3C2,
        PCV,
        SF
    };

    static QString ToString(PointFeatureType type) {
        switch (type) {
            case Invalid:
                return "Invalid";
            case Intensity:
                return "INT";
            case X:
                return "X";
            case Y:
                return "Y";
            case Z:
                return "Z";
            case NbRet:
                return "NbRet";
            case RetNb:
                return "RetNb";
            case EchoRat:
                return "EchoRat";
            case R:
                return "R";
            case G:
                return "G";
            case B:
                return "B";
            case NIR:
                return "NIR";
            case Dip:
                return "NormDip";
            case DipDir:
                return "NormDipDir";
            case M3C2:
                return "M3C2";
            case PCV:
                return "PCV";
            case SF:
                return "SF";
            default:
                assert(false);
                break;
        }
        return "Invalid";
    }

    static inline PointFeatureType FromString(const QString& token) {
        return FromUpperString(token.toUpper());
    }

    static PointFeatureType FromUpperString(const QString& token) {
        if (token == "INT")
            return Intensity;
        else if (token == "X")
            return X;
        else if (token == "Y")
            return Y;
        else if (token == "Z")
            return Z;
        else if (token == "NBRET")
            return NbRet;
        else if (token == "RETNB")
            return RetNb;
        else if (token == "ECHORAT")
            return EchoRat;
        else if (token == "R")
            return R;
        else if (token == "G")
            return G;
        else if (token == "B")
            return B;
        else if (token == "NIR")
            return NIR;
        else if (token == "NORMDIP")
            return Dip;
        else if (token == "NORMDIPDIR")
            return DipDir;
        else if (token == "M3C2")
            return M3C2;
        else if (token == "PCV")
            return PCV;
        else if (token.startsWith("SF"))
            return SF;

        return Invalid;
    }

public:  // methods
    //! Default constructor
    PointFeature(PointFeatureType p_type)
        : type(p_type),
          sourceSFIndex(-1),
          field1(nullptr),
          field2(nullptr),
          statSF1(nullptr),
          statSF2(nullptr) {
        // auomatically set the right source for specific features
        switch (type) {
            case X:
                source = {Source::DimX, "X"};
                break;
            case Y:
                source = {Source::DimY, "Y"};
                break;
            case Z:
                source = {Source::DimZ, "Z"};
                break;
            case R:
                source = {Source::Red, "Red"};
                break;
            case G:
                source = {Source::Green, "Green"};
                break;
            case B:
                source = {Source::Blue, "Blue"};
                break;
            default:
                source = {Source::ScalarField,
                          QString()};  // source name --> TBD later
                break;
        }
    }

    //! Destructor
    ~PointFeature() override {}

    // inherited from Feature
    virtual Type getType() const override { return Type::PointFeature; }
    virtual Feature::Shared clone() const override {
        return Feature::Shared(new PointFeature(*this));
    }
    virtual bool prepare(
            const CorePoints& corePoints,
            QString& error,
            cloudViewer::GenericProgressCallback* progressCb = nullptr,
            SFCollector* generatedScalarFields = nullptr) override;
    virtual bool finish(const CorePoints& corePoints, QString& error) override;
    virtual bool checkValidity(QString corePointRole,
                               QString& error) const override;
    virtual QString toString() const override;

    //! Compute the associated 'stat' on a set of points (and with a given
    //! field)
    bool computeStat(
            const cloudViewer::DgmOctree::NeighboursSet& pointsInNeighbourhood,
            const IScalarFieldWrapper::Shared& sourceField,
            double& outputValue) const;

protected:  // methods
    //! Returns the 'source' field from a given cloud
    IScalarFieldWrapper::Shared retrieveField(ccPointCloud* cloud,
                                              QString& error);

public:  // members
    //! Point feature type
    /** \warning different from the feature type
     **/
    PointFeatureType type;

    //! Source scalar field index (if the feature source is 'ScalarField')
    int sourceSFIndex;

    //! First cloud 'source' field
    IScalarFieldWrapper::Shared field1;

    //! Second cloud 'source' field (if any)
    IScalarFieldWrapper::Shared field2;

    //! For scaled features
    cloudViewer::ScalarField* statSF1;
    cloudViewer::ScalarField* statSF2;
};
}  // namespace masc
