// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file vtkutils.h
/// @brief VTK helper functions: image conversion, actor creation, export.

#include <vtkActor.h>
#include <vtkConeSource.h>
#include <vtkCubeSource.h>
#include <vtkCylinderSource.h>
#include <vtkDelaunay2D.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>

#include "point3f.h"
#include "qVTK.h"
#include "utils.h"
#include "vector4f.h"

// macroes
#ifndef VTK_CREATE
#define VTK_CREATE(TYPE, NAME) \
    vtkSmartPointer<TYPE> NAME = vtkSmartPointer<TYPE>::New()
#endif

class vtkImageData;
class QVTKOpenGLNativeWidget;
class vtkPolyDataAlgorithm;
namespace VtkUtils {

/// @param imageData VTK image to convert
/// @return QImage
QImage QVTK_ENGINE_LIB_API vtkImageDataToQImage(vtkImageData* imageData);
/// @param img QImage source
/// @param imageData VTK image output (populated)
void QVTK_ENGINE_LIB_API qImageToVtkImage(QImage& img, vtkImageData* imageData);
/// @param widget VTK OpenGL widget to capture
/// @return Screenshot as QImage
QImage QVTK_ENGINE_LIB_API vtkWidgetSnapshot(QVTKOpenGLNativeWidget* widget);

/// @brief Initialize VTK object if nullptr
template <class T>
void vtkInitOnce(T** obj) {
    if (*obj == nullptr) *obj = T::New();
}

template <class T>
void vtkInitOnce(vtkSmartPointer<T>& obj) {
    if (!obj) obj = vtkSmartPointer<T>::New();
}

template <class T>
void vtkSafeDelete(T* obj) {
    if (obj) obj->Delete();
}

/// @param algo VTK poly data algorithm
/// @return vtkActor with mapper connected to algorithm output
static inline vtkActor* createSourceActor(vtkPolyDataAlgorithm* algo) {
    vtkActor* actor = vtkActor::New();
    vtkSmartPointer<vtkPolyDataMapper> mapper(vtkPolyDataMapper::New());
    mapper->SetInputConnection(algo->GetOutputPort());
    actor->SetMapper(mapper);

    return actor;
}

/// @brief Optional configurator for VTK source algorithms
template <class T>
class SourceSetter {
public:
    void config(T* source) {
        Q_UNUSED(source)
        // no impl
    }
};

template <>
class SourceSetter<vtkSphereSource> {
public:
    void config(vtkSphereSource* source) {}
};

template <>
class SourceSetter<vtkConeSource> {
public:
    void config(vtkConeSource* source) {}
};

template <class T>
/// @param setter Optional configurator
/// @return vtkActor from source
static inline vtkActor* createSourceActor(SourceSetter<T>* setter = nullptr) {
    vtkActor* actor = vtkActor::New();

    vtkSmartPointer<T> source(T::New());

    if (setter) setter->config(source);

    vtkSmartPointer<vtkPolyDataMapper> mapper(vtkPolyDataMapper::New());
    mapper->SetInputConnection(source->GetOutputPort());
    actor->SetMapper(mapper);

    return actor;
}

/// @param points 3D points for surface
/// @param scalars Optional scalar values per point
/// @return vtkActor with Delaunay2D surface
static vtkActor* createSurfaceActor(
        const QList<Point3F>& points,
        const QList<qreal>& scalars = QList<qreal>()) {
    vtkSmartPointer<vtkPoints> vtkpoints(vtkPoints::New());
    foreach (const Point3F& p3f, points)
        vtkpoints->InsertNextPoint(p3f.x, p3f.y, p3f.z);

    vtkSmartPointer<vtkPolyData> polyData(vtkPolyData::New());
    polyData->SetPoints(vtkpoints);

    vtkSmartPointer<vtkDoubleArray> scalarArray(vtkDoubleArray::New());
    scalarArray->SetName("scalar");
    foreach (qreal scalar, scalars) scalarArray->InsertNextTuple1(scalar);

    //    vtkSmartPointer<vtkPointData> pointdata(vtkPointData::New());
    //    pointdata->SetScalars(dataArray);
    vtkpoints->SetData(scalarArray);

    vtkSmartPointer<vtkDelaunay2D> del(vtkDelaunay2D::New());
    del->SetInputData(polyData);
    del->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper(vtkPolyDataMapper::New());
    mapper->SetInputConnection(del->GetOutputPort());

    vtkActor* actor = vtkActor::New();
    actor->SetMapper(mapper);

    return actor;
}

/// @param data VTK data object
/// @return vtkActor with mapper
template <class DataObject, class Mapper = vtkPolyDataMapper>
static inline vtkActor* createActorFromData(DataObject* data) {
    vtkActor* actor = vtkActor::New();

    VTK_CREATE(Mapper, mapper);
    mapper->SetInputData(data);
    actor->SetMapper(mapper);

    return actor;
}

/// @param actor VTK actor to export
/// @param outfile Output file path
void QVTK_ENGINE_LIB_API exportActorToFile(vtkActor* actor,
                                           const QString& outfile);

}  // namespace VtkUtils
