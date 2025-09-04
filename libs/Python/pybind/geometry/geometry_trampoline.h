// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <GenericTriangle.h>
#include <ecvBBox.h>
#include <ecvGenericPrimitive.h>
#include <ecvHObject.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvNormalVectors.h>
#include <ecvOrientedBBox.h>
#include <ecvPlanarEntityInterface.h>
#include <ecvScalarField.h>
#include <ecvSubMesh.h>

#include "pybind/cloudViewer_pybind.h"
#include "pybind/geometry/geometry.h"

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
class PyGeometry : public PyObjectBase<GeometryBase> {
public:
    using PyObjectBase<GeometryBase>::PyObjectBase;

    bool IsEmpty() const override {
        PYBIND11_OVERLOAD_PURE(bool, GeometryBase, );
    }

    Eigen::Vector3d GetMinBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, GeometryBase, );
    }
    Eigen::Vector2d GetMin2DBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2d, GeometryBase, );
    }
    Eigen::Vector3d GetMaxBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, GeometryBase, );
    }
    Eigen::Vector2d GetMax2DBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2d, GeometryBase, );
    }
    Eigen::Vector3d GetCenter() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, GeometryBase, );
    }
    ccBBox GetAxisAlignedBoundingBox() const override {
        PYBIND11_OVERLOAD_PURE(ccBBox, GeometryBase, );
    }
    ecvOrientedBBox GetOrientedBoundingBox() const override {
        PYBIND11_OVERLOAD_PURE(ecvOrientedBBox, GeometryBase, );
    }
    GeometryBase& Transform(const Eigen::Matrix4d& transformation) override {
        PYBIND11_OVERLOAD_PURE(GeometryBase&, GeometryBase, transformation);
    }
};

// MESH
template <class GenericMeshBase = cloudViewer::GenericMesh>
class PyGenericMesh : public GenericMeshBase {
public:
    using GenericMeshBase::GenericMeshBase;

    unsigned size() const override {
        PYBIND11_OVERLOAD_PURE(unsigned, GenericMeshBase, );
    }
    void forEach(std::function<void(cloudViewer::GenericTriangle&)> action)
            override {
        PYBIND11_OVERLOAD_PURE(void, GenericMeshBase, action);
    }
    void getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) override {
        PYBIND11_OVERLOAD_PURE(void, GenericMeshBase, bbMin, bbMax);
    }
    void placeIteratorAtBeginning() override {
        PYBIND11_OVERLOAD_PURE(void, GenericMeshBase, );
    }
    cloudViewer::GenericTriangle* _getNextTriangle() override {
        PYBIND11_OVERLOAD_PURE(cloudViewer::GenericTriangle*,
                               GenericMeshBase, );
    }
};

template <class GenericIndexedMeshBase = cloudViewer::GenericIndexedMesh>
class PyGenericIndexedMesh : public PyGenericMesh<GenericIndexedMeshBase> {
public:
    using PyGenericMesh<GenericIndexedMeshBase>::PyGenericMesh;

    cloudViewer::GenericTriangle* _getTriangle(
            unsigned triangleIndex) override {
        PYBIND11_OVERLOAD_PURE(cloudViewer::GenericTriangle*,
                               GenericIndexedMeshBase, triangleIndex);
    }
    cloudViewer::VerticesIndexes* getTriangleVertIndexes(
            unsigned triangleIndex) override {
        PYBIND11_OVERLOAD_PURE(cloudViewer::VerticesIndexes*,
                               GenericIndexedMeshBase, triangleIndex);
    }
    void getTriangleVertices(unsigned triangleIndex,
                             CCVector3& A,
                             CCVector3& B,
                             CCVector3& C) const override {
        PYBIND11_OVERLOAD_PURE(void, GenericIndexedMeshBase, triangleIndex, A,
                               B, C);
    }
    void getTriangleVertices(unsigned triangleIndex,
                             double A[3],
                             double B[3],
                             double C[3]) const override {
        PYBIND11_OVERLOAD_PURE(void, GenericIndexedMeshBase, triangleIndex, A,
                               B, C);
    }
    cloudViewer::VerticesIndexes* getNextTriangleVertIndexes() override {
        PYBIND11_OVERLOAD_PURE(cloudViewer::VerticesIndexes*,
                               GenericIndexedMeshBase, );
    }

    bool interpolateNormals(unsigned triIndex,
                            const CCVector3& P,
                            CCVector3& N) override {
        PYBIND11_OVERLOAD_PURE(bool, GenericIndexedMeshBase, triIndex, P, N);
    }

    bool normalsAvailable() const override {
        PYBIND11_OVERLOAD_PURE(bool, GenericIndexedMeshBase, );
    }
};

template <class GenericTriangleMesh = ccGenericMesh>
class PyGenericTriangleMesh : public PyGeometry<GenericTriangleMesh>,
                              public PyGenericIndexedMesh<GenericTriangleMesh> {
public:
    using PyGeometry<GenericTriangleMesh>::PyGeometry;
    // using PyGenericIndexedMesh<GenericTriangleMesh>::PyGenericIndexedMesh;

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
    void getTriangleTexCoordinates(unsigned triIndex,
                                   TexCoords2D*& tx1,
                                   TexCoords2D*& tx2,
                                   TexCoords2D*& tx3) const override {
        PYBIND11_OVERLOAD_PURE(void, GenericTriangleMesh, triIndex, tx1, tx2,
                               tx3);
    }
    void getTexCoordinates(unsigned index, TexCoords2D*& tx) const override {
        PYBIND11_OVERLOAD_PURE(void, GenericTriangleMesh, index, tx);
    }
    bool hasPerTriangleTexCoordIndexes() const override {
        PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, );
    }
    void getTriangleTexCoordinatesIndexes(unsigned triangleIndex,
                                          int& i1,
                                          int& i2,
                                          int& i3) const override {
        PYBIND11_OVERLOAD_PURE(void, GenericTriangleMesh, triangleIndex, i1, i2,
                               i3);
    }
    bool hasTriNormals() const override {
        PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, );
    }
    void getTriangleNormalIndexes(unsigned triangleIndex,
                                  int& i1,
                                  int& i2,
                                  int& i3) const override {
        PYBIND11_OVERLOAD_PURE(void, GenericTriangleMesh, triangleIndex, i1, i2,
                               i3);
    }
    bool getTriangleNormals(unsigned triangleIndex,
                            CCVector3& Na,
                            CCVector3& Nb,
                            CCVector3& Nc) const override {
        PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triangleIndex, Na, Nb,
                               Nc);
    }
    NormsIndexesTableType* getTriNormsTable() const override {
        PYBIND11_OVERLOAD_PURE(NormsIndexesTableType*, GenericTriangleMesh, );
    }
    bool interpolateColors(unsigned triIndex,
                           const CCVector3& P,
                           ecvColor::Rgb& C) override {
        PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triIndex, P, C);
    }
    bool getColorFromMaterial(unsigned triIndex,
                              const CCVector3& P,
                              ecvColor::Rgb& C,
                              bool interpolateColorIfNoTexture) override {
        PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triIndex, P, C,
                               interpolateColorIfNoTexture);
    }
    bool getVertexColorFromMaterial(unsigned triIndex,
                                    unsigned char vertIndex,
                                    ecvColor::Rgb& C,
                                    bool returnColorIfNoTexture) override {
        PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triIndex, vertIndex,
                               C, returnColorIfNoTexture);
    }

    bool interpolateNormalsBC(unsigned triIndex,
                              const CCVector3d& w,
                              CCVector3& N) override {
        PYBIND11_OVERLOAD_PURE(bool, GenericTriangleMesh, triIndex, w, N);
    }
};

template <class GenericPrimitive = ccGenericPrimitive>
class PyGenericPrimitive : public GenericPrimitive {
public:
    using GenericPrimitive::GenericPrimitive;

    //! Returns type name (sphere, cylinder, etc.)
    QString getTypeName() const override {
        PYBIND11_OVERLOAD_PURE(QString, GenericPrimitive, );
    }

    //! Clones primitive
    ccGenericPrimitive* clone() const override {
        PYBIND11_OVERLOAD_PURE(ccGenericPrimitive*, GenericPrimitive, );
    }
};

template <class GenericPlanarEntityInterface = ccPlanarEntityInterface>
class PyPlanarEntityInterface : public GenericPlanarEntityInterface {
public:
    using GenericPlanarEntityInterface::GenericPlanarEntityInterface;

    //! Returns the entity normal
    CCVector3 getNormal() const override {
        PYBIND11_OVERLOAD_PURE(CCVector3, GenericPlanarEntityInterface, );
    }
};

template <class OrientedBBoxBase = cloudViewer::OrientedBoundingBox>
class PyOrientedBBoxBase : public OrientedBBoxBase {
public:
    using OrientedBBoxBase::OrientedBBoxBase;
};
