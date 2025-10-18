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

// qCC_db
#include <ecvPointCloud.h>
// CCLib
#include <ScalarField.h>

// Qt
#include <QSharedPointer>

class IScalarFieldWrapper {
public:
    virtual ~IScalarFieldWrapper() {}
    using Shared = QSharedPointer<IScalarFieldWrapper>;
    virtual double pointValue(unsigned index) const = 0;
    virtual bool isValid() const = 0;
    virtual QString getName() const = 0;
    virtual size_t size() const = 0;
};

class ScalarFieldWrapper : public IScalarFieldWrapper {
public:
    ScalarFieldWrapper(cloudViewer::ScalarField* sf) : m_sf(sf) {}

    virtual inline double pointValue(unsigned index) const override {
        return m_sf->at(index);
    }
    virtual inline bool isValid() const { return m_sf != nullptr; }
    virtual inline QString getName() const { return m_sf->getName(); }
    virtual size_t size() const override { return m_sf->size(); }

protected:
    cloudViewer::ScalarField* m_sf;
};

class ScalarFieldRatioWrapper : public IScalarFieldWrapper {
public:
    ScalarFieldRatioWrapper(cloudViewer::ScalarField* sfp,
                            cloudViewer::ScalarField* sfq,
                            QString name)
        : m_sfp(sfp), m_sfq(sfq), m_name(name) {}

    virtual inline double pointValue(unsigned index) const override {
        ScalarType p = m_sfp->getValue(index);
        ScalarType q = m_sfq->getValue(index);
        ScalarType ratio =
                (std::abs(q) > std::numeric_limits<ScalarType>::epsilon()
                         ? p / q
                         : NAN_VALUE);
        return ratio;
    }
    virtual inline bool isValid() const {
        return (m_sfp != nullptr && m_sfq != nullptr);
    }
    virtual inline QString getName() const { return m_name; }
    virtual inline size_t size() const override {
        return std::min(m_sfp->size(), m_sfq->size());
    }

protected:
    cloudViewer::ScalarField *m_sfp, *m_sfq;
    QString m_name;
};

class NormDipAndDipDirFieldWrapper : public IScalarFieldWrapper {
public:
    enum Mode { Dip = 0, DipDir = 1 };

    NormDipAndDipDirFieldWrapper(const ccPointCloud* cloud, Mode mode)
        : m_cloud(cloud), m_mode(mode) {}

    virtual double pointValue(unsigned index) const override {
        const CCVector3& N = m_cloud->getPointNormal(index);
        PointCoordinateType dip_deg, dipDir_deg;
        ccNormalVectors::ConvertNormalToDipAndDipDir(N, dip_deg, dipDir_deg);
        return (m_mode == Dip ? dip_deg : dipDir_deg);
    }
    virtual inline bool isValid() const {
        return m_cloud != nullptr && m_cloud->hasNormals();
    }
    virtual inline QString getName() const {
        static const char s_names[][14] = {"Norm dip", "Norm dip dir."};
        return s_names[m_mode];
    }
    virtual inline size_t size() const override { return m_cloud->size(); }

protected:
    const ccPointCloud* m_cloud;
    Mode m_mode;
};

class DimScalarFieldWrapper : public IScalarFieldWrapper {
public:
    enum Dim { DimX = 0, DimY = 1, DimZ = 2 };

    DimScalarFieldWrapper(const ccPointCloud* cloud, Dim dim)
        : m_cloud(cloud), m_dim(dim) {}

    virtual inline double pointValue(unsigned index) const override {
        return m_cloud->getPoint(index)->u[m_dim];
    }
    virtual inline bool isValid() const { return m_cloud != nullptr; }
    virtual inline QString getName() const {
        static const char s_names[][5] = {"DimX", "DimY", "DimZ"};
        return s_names[m_dim];
    }
    virtual inline size_t size() const override { return m_cloud->size(); }

protected:
    const ccPointCloud* m_cloud;
    Dim m_dim;
};

class ColorScalarFieldWrapper : public IScalarFieldWrapper {
public:
    enum Band { Red = 0, Green = 1, Blue = 2 };

    ColorScalarFieldWrapper(const ccPointCloud* cloud, Band band)
        : m_cloud(cloud), m_band(band) {}

    virtual inline double pointValue(unsigned index) const override {
        return m_cloud->getPointColor(index).rgb[m_band];
    }
    virtual inline bool isValid() const {
        return m_cloud != nullptr && m_cloud->hasColors();
    }
    virtual inline QString getName() const {
        static const char s_names[][6] = {"Red", "Green", "Blue"};
        return s_names[m_band];
    }
    virtual inline size_t size() const override { return m_cloud->size(); }

protected:
    const ccPointCloud* m_cloud;
    Band m_band;
};
