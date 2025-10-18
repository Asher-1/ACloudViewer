// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvHObject.h>
#include <ecvPointCloud.h>

#include "pybind/cloudViewer_pybind.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

// PointCloud
template <class GenericCloudBase = cloudViewer::GenericCloud>
class PyGenericCloud : public GenericCloudBase {
public:
    using GenericCloudBase::GenericCloudBase;

    unsigned size() const override {
        PYBIND11_OVERLOAD_PURE(unsigned, GenericCloudBase, );
    }
    void forEach(
            cloudViewer::GenericCloud::genericPointAction action) override {
        PYBIND11_OVERLOAD_PURE(void, GenericCloudBase,
                               cloudViewer::GenericCloud::genericPointAction);
    }
    void getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) override {
        PYBIND11_OVERLOAD_PURE(void, GenericCloudBase, bbMin, bbMax);
    }
    void placeIteratorAtBeginning() override {
        PYBIND11_OVERLOAD_PURE(void, GenericCloudBase, );
    }
    const CCVector3* getNextPoint() override {
        PYBIND11_OVERLOAD_PURE(const CCVector3*, GenericCloudBase, );
    }
    bool enableScalarField() override {
        PYBIND11_OVERLOAD_PURE(bool, GenericCloudBase, );
    }
    bool isScalarFieldEnabled() const override {
        PYBIND11_OVERLOAD_PURE(bool, GenericCloudBase, );
    }
    void setPointScalarValue(unsigned pointIndex, ScalarType value) override {
        PYBIND11_OVERLOAD_PURE(void, GenericCloudBase, pointIndex, value);
    }
    ScalarType getPointScalarValue(unsigned pointIndex) const override {
        PYBIND11_OVERLOAD_PURE(ScalarType, GenericCloudBase, pointIndex);
    }
};

template <class GenericIndexedCloudBase = cloudViewer::GenericIndexedCloud>
class PyGenericIndexedCloud : public PyGenericCloud<GenericIndexedCloudBase> {
public:
    using PyGenericCloud<GenericIndexedCloudBase>::PyGenericCloud;

    const CCVector3* getPoint(unsigned index) const override {
        PYBIND11_OVERLOAD_PURE(const CCVector3*, GenericIndexedCloudBase,
                               index);
    }
    void getPoint(unsigned index, CCVector3& P) const override {
        PYBIND11_OVERLOAD_PURE(void, GenericIndexedCloudBase, index, P);
    }
    void getPoint(unsigned index, double P[3]) const override {
        PYBIND11_OVERLOAD_PURE(void, GenericIndexedCloudBase, index, P);
    }
};

template <class GenericIndexedCloudPersistBase =
                  cloudViewer::GenericIndexedCloudPersist>
class PyGenericIndexedCloudPersist
    : public PyGenericIndexedCloud<GenericIndexedCloudPersistBase> {
public:
    using PyGenericIndexedCloud<
            GenericIndexedCloudPersistBase>::PyGenericIndexedCloud;

    const CCVector3* getPointPersistentPtr(unsigned index) override {
        PYBIND11_OVERLOAD_PURE(const CCVector3*, GenericIndexedCloudPersistBase,
                               index);
    }
};

template <class GenericPointCloudBase = ccGenericPointCloud>
class PyGenericPointCloud
    : public PyGeometry<GenericPointCloudBase>,
      public PyGenericIndexedCloudPersist<GenericPointCloudBase> {
public:
    using PyGeometry<GenericPointCloudBase>::PyGeometry;
    // using
    // PyGenericIndexedCloudPersist<GenericPointCloudBase>::PyGenericIndexedCloudPersist;

    ccGenericPointCloud* clone(ccGenericPointCloud* destCloud = nullptr,
                               bool ignoreChildren = false) override {
        PYBIND11_OVERLOAD_PURE(ccGenericPointCloud*, GenericPointCloudBase,
                               destCloud, ignoreChildren);
    }
    const ecvColor::Rgb* getScalarValueColor(ScalarType d) const override {
        PYBIND11_OVERLOAD_PURE(const ecvColor::Rgb*, GenericPointCloudBase, d);
    }
    const ecvColor::Rgb* getPointScalarValueColor(
            unsigned pointIndex) const override {
        PYBIND11_OVERLOAD_PURE(const ecvColor::Rgb*, GenericPointCloudBase,
                               pointIndex);
    }
    ScalarType getPointDisplayedDistance(unsigned pointIndex) const override {
        PYBIND11_OVERLOAD_PURE(ScalarType, GenericPointCloudBase, pointIndex);
    }
    const ecvColor::Rgb& getPointColor(unsigned pointIndex) const override {
        PYBIND11_OVERLOAD_PURE(const ecvColor::Rgb&, GenericPointCloudBase,
                               pointIndex);
    }
    const CompressedNormType& getPointNormalIndex(
            unsigned pointIndex) const override {
        PYBIND11_OVERLOAD_PURE(const CompressedNormType&, GenericPointCloudBase,
                               pointIndex);
    }
    const CCVector3& getPointNormal(unsigned pointIndex) const override {
        PYBIND11_OVERLOAD_PURE(const CCVector3&, GenericPointCloudBase,
                               pointIndex);
    }
    void refreshBB() override {
        PYBIND11_OVERLOAD_PURE(void, GenericPointCloudBase, );
    }
    // ccGenericPointCloud* createNewCloudFromVisibilitySelection(
    //	bool removeSelectedPoints = false,
    //	VisibilityTableType* visTable = nullptr,
    //	bool silent = false) override {
    //	PYBIND11_OVERLOAD_PURE(ccGenericPointCloud*, GenericPointCloudBase,
    //		removeSelectedPoints, visTable, silent);
    // }
    void applyRigidTransformation(const ccGLMatrix& trans) override {
        PYBIND11_OVERLOAD_PURE(void, GenericPointCloudBase, trans);
    }
    cloudViewer::ReferenceCloud* crop(const ccBBox& box,
                                      bool inside = true) override {
        PYBIND11_OVERLOAD_PURE(cloudViewer::ReferenceCloud*,
                               GenericPointCloudBase, box, inside);
    }
    cloudViewer::ReferenceCloud* crop(const ecvOrientedBBox& box) override {
        PYBIND11_OVERLOAD_PURE(cloudViewer::ReferenceCloud*,
                               GenericPointCloudBase, box);
    }
    void removePoints(size_t index) override {
        PYBIND11_OVERLOAD_PURE(void, GenericPointCloudBase, index);
    }
    void scale(PointCoordinateType fx,
               PointCoordinateType fy,
               PointCoordinateType fz,
               CCVector3 center = CCVector3(0, 0, 0)) override {
        PYBIND11_OVERLOAD_PURE(void, GenericPointCloudBase, fx, fy, fz, center);
    }
};

template <class GenericReferenceCloud = cloudViewer::ReferenceCloud>
class PyGenericReferenceCloud
    : public PyGenericIndexedCloudPersist<GenericReferenceCloud> {
public:
    using PyGenericIndexedCloudPersist<
            GenericReferenceCloud>::PyGenericIndexedCloudPersist;
};