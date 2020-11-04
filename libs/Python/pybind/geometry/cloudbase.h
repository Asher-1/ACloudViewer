// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <ecvHObject.h>
#include <ecvPointCloud.h>

#include "pybind/geometry/geometry.h"
#include "pybind/cloudViewer_pybind.h"
#include "pybind/geometry/geometry_trampoline.h"

// PointCloud
template <class GenericCloudBase = CVLib::GenericCloud>
class PyGenericCloud : public GenericCloudBase {
public:
	using GenericCloudBase::GenericCloudBase;

	unsigned size() const override {
		PYBIND11_OVERLOAD_PURE(unsigned, GenericCloudBase, );
	}
	void forEach(CVLib::GenericCloud::genericPointAction action) override {
		PYBIND11_OVERLOAD_PURE(void, GenericCloudBase, CVLib::GenericCloud::genericPointAction);
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

template <class GenericIndexedCloudBase = CVLib::GenericIndexedCloud>
class PyGenericIndexedCloud : public PyGenericCloud<GenericIndexedCloudBase> {
public:
	using PyGenericCloud<GenericIndexedCloudBase>::PyGenericCloud;

	const CCVector3* getPoint(unsigned index) const override {
		PYBIND11_OVERLOAD_PURE(const CCVector3*, GenericIndexedCloudBase, index);
	}
	void getPoint(unsigned index, CCVector3& P) const override {
		PYBIND11_OVERLOAD_PURE(void, GenericIndexedCloudBase, index, P);
	}
	void getPoint(unsigned index, double P[3]) const override {
		PYBIND11_OVERLOAD_PURE(void, GenericIndexedCloudBase, index, P);
	}
};

template <class GenericIndexedCloudPersistBase = CVLib::GenericIndexedCloudPersist>
class PyGenericIndexedCloudPersist : public PyGenericIndexedCloud<GenericIndexedCloudPersistBase>
{
public:
	using PyGenericIndexedCloud<GenericIndexedCloudPersistBase>::PyGenericIndexedCloud;

	const CCVector3* getPointPersistentPtr(unsigned index) override {
		PYBIND11_OVERLOAD_PURE(const CCVector3*, GenericIndexedCloudPersistBase, index);
	}
};

template <class GenericPointCloudBase = ccGenericPointCloud>
class PyGenericPointCloud :
	public PyGeometry<GenericPointCloudBase>,
	public PyGenericIndexedCloudPersist<GenericPointCloudBase>
{
public:
	using PyGeometry<GenericPointCloudBase>::PyGeometry;
	//using PyGenericIndexedCloudPersist<GenericPointCloudBase>::PyGenericIndexedCloudPersist;

	ccGenericPointCloud* clone(
		ccGenericPointCloud* destCloud = nullptr,
		bool ignoreChildren = false) override {
		PYBIND11_OVERLOAD_PURE(ccGenericPointCloud*, GenericPointCloudBase, destCloud, ignoreChildren);
	}
	const ecvColor::Rgb* getScalarValueColor(ScalarType d) const override {
		PYBIND11_OVERLOAD_PURE(const ecvColor::Rgb*, GenericPointCloudBase, d);
	}
	const ecvColor::Rgb* getPointScalarValueColor(unsigned pointIndex) const override {
		PYBIND11_OVERLOAD_PURE(const ecvColor::Rgb*, GenericPointCloudBase, pointIndex);
	}
	ScalarType getPointDisplayedDistance(unsigned pointIndex) const override {
		PYBIND11_OVERLOAD_PURE(ScalarType, GenericPointCloudBase, pointIndex);
	}
	const ecvColor::Rgb& getPointColor(unsigned pointIndex) const override {
		PYBIND11_OVERLOAD_PURE(const ecvColor::Rgb&, GenericPointCloudBase, pointIndex);
	}
	const CompressedNormType& getPointNormalIndex(unsigned pointIndex) const override {
		PYBIND11_OVERLOAD_PURE(const CompressedNormType&, GenericPointCloudBase, pointIndex);
	}
	const CCVector3& getPointNormal(unsigned pointIndex) const override {
		PYBIND11_OVERLOAD_PURE(const CCVector3&, GenericPointCloudBase, pointIndex);
	}
	void refreshBB() override {
		PYBIND11_OVERLOAD_PURE(void, GenericPointCloudBase, );
	}
	//ccGenericPointCloud* createNewCloudFromVisibilitySelection(
	//	bool removeSelectedPoints = false, 
	//	VisibilityTableType* visTable = nullptr, 
	//	bool silent = false) override {
	//	PYBIND11_OVERLOAD_PURE(ccGenericPointCloud*, GenericPointCloudBase, 
	//		removeSelectedPoints, visTable, silent);
	//}
	void applyRigidTransformation(const ccGLMatrix& trans) override {
		PYBIND11_OVERLOAD_PURE(void, GenericPointCloudBase, trans);
	}
	CVLib::ReferenceCloud* crop(const ccBBox& box, bool inside = true) override {
		PYBIND11_OVERLOAD_PURE(CVLib::ReferenceCloud*, GenericPointCloudBase, box, inside);
	}
	CVLib::ReferenceCloud* crop(const ecvOrientedBBox& box) override {
		PYBIND11_OVERLOAD_PURE(CVLib::ReferenceCloud*, GenericPointCloudBase, box);
	}
	void removePoints(size_t index) override {
		PYBIND11_OVERLOAD_PURE(void, GenericPointCloudBase, index);
	}
	void scale(PointCoordinateType fx, PointCoordinateType fy,
		PointCoordinateType fz, CCVector3 center = CCVector3(0, 0, 0)) override {
		PYBIND11_OVERLOAD_PURE(void, GenericPointCloudBase, fx, fy, fz, center);
	}
};

template <class GenericReferenceCloud = CVLib::ReferenceCloud>
class PyGenericReferenceCloud : public PyGenericIndexedCloudPersist<GenericReferenceCloud>
{
public:
	using PyGenericIndexedCloudPersist<GenericReferenceCloud>::PyGenericIndexedCloudPersist;
};