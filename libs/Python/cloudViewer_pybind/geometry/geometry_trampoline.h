// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

#include <ecvBBox.h>
#include <ecvOrientedBBox.h>
#include <ecvNormalVectors.h>
#include <ecvMaterialSet.h>
#include <ecvSubMesh.h>
#include <ecvScalarField.h>
#include <GenericTriangle.h>

#include <ecvMesh.h>
#include <ecvHObject.h>
#include <ecvPointCloud.h>
#include "cloudViewer_pybind/geometry/geometry.h"
#include "cloudViewer_pybind/cloudViewer_pybind.h"

using namespace cloudViewer;

template <class ObjectBase = ccObject>
class PyObjectBase : public ObjectBase {
public:
    using ObjectBase::ObjectBase;
    CV_CLASS_ENUM getClassID() const override {
        PYBIND11_OVERLOAD_PURE(CV_CLASS_ENUM, ObjectBase, );
    }
};

template <class DrawableObjectBase = ccDrawableObject>
class PyDrawableObjectBase : public DrawableObjectBase {
public:
	using DrawableObjectBase::DrawableObjectBase;
	void draw(CC_DRAW_CONTEXT& context) override {
		PYBIND11_OVERLOAD_PURE(void, DrawableObjectBase, context);
	}
};

template <class GeometryBase = ccHObject>
class PyGeometry : public PyObjectBase<GeometryBase>
	//public PyDrawableObjectBase<GeometryBase> 
{
public:
    using PyObjectBase<GeometryBase>::PyObjectBase;
    //using PyDrawableObjectBase<GeometryBase>::PyDrawableObjectBase;

	bool isEmpty() const override {
		PYBIND11_OVERLOAD_PURE(bool, GeometryBase, );
	}

	Eigen::Vector3d getMinBound() const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, GeometryBase, );
	}
	Eigen::Vector2d getMin2DBound() const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Vector2d, GeometryBase, );
	}
	Eigen::Vector3d getMaxBound() const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, GeometryBase, );
	}
	Eigen::Vector2d getMax2DBound() const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Vector2d, GeometryBase, );
	}
	Eigen::Vector3d getGeometryCenter() const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, GeometryBase, );
	}
	ccBBox getAxisAlignedBoundingBox() const override {
		PYBIND11_OVERLOAD_PURE(ccBBox, GeometryBase, );
	}
	ecvOrientedBBox getOrientedBoundingBox() const override {
		PYBIND11_OVERLOAD_PURE(ecvOrientedBBox, GeometryBase, );
	}
	GeometryBase& transform(const Eigen::Matrix4d& transformation) override {
		PYBIND11_OVERLOAD_PURE(GeometryBase&, GeometryBase, transformation);
	}
};

// PointCloud
template <class GenericCloudBase = CVLib::GenericCloud>
class PyGenericCloud : public GenericCloudBase {
public:
	using GenericCloudBase::GenericCloudBase;

	unsigned size() const override {
		PYBIND11_OVERLOAD_PURE(unsigned, GenericCloudBase, );
	}
	void forEach() override {
		PYBIND11_OVERLOAD_PURE(void, GenericCloudBase, );
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

	const CCVector3* getPoint(unsigned index) override {
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
	const ecvColor::Rgb* geScalarValueColor(ScalarType d) const override {
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

// MESH
template <class GenericMeshBase = CVLib::GenericMesh>
class PyGenericMesh : public GenericMeshBase {
public:
	using GenericMeshBase::GenericMeshBase;

	unsigned size() const override {
		PYBIND11_OVERLOAD_PURE(unsigned, GenericMeshBase, );
	}
	void forEach(std::function<void(CVLib::GenericTriangle &)> action) override {
		PYBIND11_OVERLOAD_PURE(void, GenericMeshBase, action);
	}
	void getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) override {
		PYBIND11_OVERLOAD_PURE(void, GenericMeshBase, bbMin, bbMax);
	}
	void placeIteratorAtBeginning() override {
		PYBIND11_OVERLOAD_PURE(void, GenericMeshBase, );
	}
	CVLib::GenericTriangle* _getNextTriangle() override {
		PYBIND11_OVERLOAD_PURE(CVLib::GenericTriangle*, GenericMeshBase, );
	}
};

template <class GenericIndexedMeshBase = CVLib::GenericIndexedMesh>
class PyGenericIndexedMesh : public PyGenericMesh<GenericIndexedMeshBase> {
public:
	using PyGenericMesh<GenericIndexedMeshBase>::PyGenericMesh;

	CVLib::GenericTriangle* _getTriangle(unsigned triangleIndex) override {
		PYBIND11_OVERLOAD_PURE(CVLib::GenericTriangle*, GenericIndexedMeshBase, triangleIndex);
	}
	CVLib::VerticesIndexes* getTriangleVertIndexes(unsigned triangleIndex) override {
		PYBIND11_OVERLOAD_PURE(CVLib::VerticesIndexes*, GenericIndexedMeshBase, triangleIndex);
	}
	void getTriangleVertices(unsigned triangleIndex, CCVector3& A, CCVector3& B, CCVector3& C) const override {
		PYBIND11_OVERLOAD_PURE(void, GenericIndexedMeshBase, triangleIndex, A, B, C);
	}
	void getTriangleVertices(unsigned triangleIndex, double A[3], double B[3], double C[3]) const override {
		PYBIND11_OVERLOAD_PURE(void, GenericIndexedMeshBase, triangleIndex, A, B, C);
	}
	CVLib::VerticesIndexes* getNextTriangleVertIndexes() override {
		PYBIND11_OVERLOAD_PURE(CVLib::VerticesIndexes*, GenericIndexedMeshBase, );
	}
};

template <class GenericTriangleMesh = ccGenericMesh>
class PyGenericTriangleMesh :
	public PyGeometry<GenericTriangleMesh>,
	public PyGenericIndexedMesh<GenericTriangleMesh>
{
public:
	using PyGeometry<GenericTriangleMesh>::PyGeometry;
	//using PyGenericIndexedMesh<GenericTriangleMesh>::PyGenericIndexedMesh;

	ccGenericPointCloud* getAssociatedCloud() const override {
		PYBIND11_OVERLOAD_PURE(ccGenericPointCloud*, GenericTriangleMesh, );
	}
	void refreshBB() override {
		PYBIND11_OVERLOAD_PURE(void, GenericTriangleMesh, );
	}
	unsigned capacity() const override {
		PYBIND11_OVERLOAD_PURE(unsigned, GenericTriangleMesh, );
	}
	bool hasMaterials() const override {
		PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, );
	}
	const ccMaterialSet* getMaterialSet() const override {
		PYBIND11_OVERLOAD_PURE(const ccMaterialSet*, GenericTriangleMesh, );
	}
	bool hasTextures() const override {
		PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, );
	}
	TextureCoordsContainer* getTexCoordinatesTable() const override {
		PYBIND11_OVERLOAD_PURE(TextureCoordsContainer*, GenericTriangleMesh, );
	}
	void getTriangleTexCoordinates(unsigned triIndex, TexCoords2D* &tx1, TexCoords2D* &tx2, TexCoords2D* &tx3) const override {
		PYBIND11_OVERLOAD_PURE(void, GenericTriangleMesh, triIndex, tx1, tx2, tx3);
	}
	void getTexCoordinates(unsigned index, TexCoords2D* &tx) const override {
		PYBIND11_OVERLOAD_PURE(void, GenericTriangleMesh, index, tx);
	}
	bool hasPerTriangleTexCoordIndexes() const override {
		PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, );
	}
	void getTriangleTexCoordinatesIndexes(unsigned triangleIndex, int& i1, int& i2, int& i3) const override {
		PYBIND11_OVERLOAD_PURE(void, GenericTriangleMesh, triangleIndex, i1, i2, i3);
	}
	bool hasTriNormals() const override {
		PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, );
	}
	void getTriangleNormalIndexes(unsigned triangleIndex, int& i1, int& i2, int& i3) const  override {
		PYBIND11_OVERLOAD_PURE(void, GenericTriangleMesh, triangleIndex, i1, i2, i3);
	}
	bool getTriangleNormals(unsigned triangleIndex, CCVector3& Na, CCVector3& Nb, CCVector3& Nc) const override {
		PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triangleIndex, Na, Nb, Nc);
	}
	NormsIndexesTableType* getTriNormsTable() const override {
		PYBIND11_OVERLOAD_PURE(NormsIndexesTableType*, GenericTriangleMesh, );
	}
	bool interpolateNormals(unsigned triIndex, const CCVector3& P, CCVector3& N) override {
		PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triIndex, P, N);
	}
	bool interpolateColors(unsigned triIndex, const CCVector3& P, ecvColor::Rgb& C) override {
		PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triIndex, P, C);
	}
	bool getColorFromMaterial(unsigned triIndex, const CCVector3& P, ecvColor::Rgb& C, bool interpolateColorIfNoTexture) override {
		PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triIndex, P, C, interpolateColorIfNoTexture);
	}
	bool getVertexColorFromMaterial(unsigned triIndex, 
		unsigned char vertIndex, ecvColor::Rgb& C,
		bool returnColorIfNoTexture) override {
		PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triIndex, vertIndex, C, returnColorIfNoTexture);
	}

};


template <class AxisBBoxBase = CVLib::BoundingBox>
class PyAxisBBoxBase : public AxisBBoxBase {
public:
	using AxisBBoxBase::AxisBBoxBase;
};

template <class OrientedBBoxBase = CVLib::OrientedBoundingBox>
class PyOrientedBBoxBase : public OrientedBBoxBase {
public:
	using OrientedBBoxBase::OrientedBBoxBase;
};