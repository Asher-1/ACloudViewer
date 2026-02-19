// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file vtkutils.h
 * @brief VTK utility functions for CloudViewer
 *
 * Provides helper functions and templates for working with VTK in CloudViewer:
 * - Image conversion (VTK ↔ Qt QImage)
 * - Actor creation from geometric sources
 * - Smart pointer management
 * - Data visualization helpers
 * - Widget snapshots
 */

#pragma once

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
#include "qPCL.h"
#include "utils.h"
#include "vector4f.h"

/**
 * @def VTK_CREATE
 * @brief Macro for creating VTK smart pointers
 *
 * Convenient macro to create and initialize VTK smart pointers in one line.
 * Example: VTK_CREATE(vtkActor, actor);
 */
#ifndef VTK_CREATE
#define VTK_CREATE(TYPE, NAME) \
    vtkSmartPointer<TYPE> NAME = vtkSmartPointer<TYPE>::New()
#endif

class vtkImageData;
class QVTKOpenGLNativeWidget;
class vtkPolyDataAlgorithm;

/**
 * @namespace VtkUtils
 * @brief VTK utility functions for CloudViewer
 *
 * Collection of helper functions for VTK visualization and data conversion.
 * Provides bridges between VTK, Qt, and CloudViewer data structures.
 */
namespace VtkUtils {

// =====================================================================
// Image Conversion Functions
// =====================================================================

/**
 * @brief Convert VTK image to Qt QImage
 * @param imageData VTK image data to convert
 * @return Qt QImage representation
 *
 * Converts VTK's native image format to Qt's QImage for display in Qt widgets.
 */
QImage QPCL_ENGINE_LIB_API vtkImageDataToQImage(vtkImageData* imageData);

/**
 * @brief Convert Qt QImage to VTK image
 * @param img Qt image to convert
 * @param imageData Output VTK image data
 *
 * Converts Qt QImage to VTK's native image format for VTK processing.
 */
void QPCL_ENGINE_LIB_API qImageToVtkImage(QImage& img, vtkImageData* imageData);

/**
 * @brief Take snapshot of VTK widget
 * @param widget VTK OpenGL widget to capture
 * @return Qt image containing the snapshot
 *
 * Captures the current render window content as a QImage.
 */
QImage QPCL_ENGINE_LIB_API vtkWidgetSnapshot(QVTKOpenGLNativeWidget* widget);

// =====================================================================
// Smart Pointer Management
// =====================================================================

/**
 * @brief Initialize VTK object once (raw pointer version)
 * @tparam T VTK object type
 * @param obj Pointer to object pointer
 *
 * Creates VTK object only if pointer is null. Safe for multiple calls.
 */
template <class T>
void vtkInitOnce(T** obj) {
    if (*obj == nullptr) *obj = T::New();
}

/**
 * @brief Initialize VTK object once (smart pointer version)
 * @tparam T VTK object type
 * @param obj Smart pointer to object
 *
 * Creates VTK object only if smart pointer is null. Safe for multiple calls.
 */
template <class T>
void vtkInitOnce(vtkSmartPointer<T>& obj) {
    if (!obj) obj = vtkSmartPointer<T>::New();
}

/**
 * @brief Safely delete VTK object
 * @tparam T VTK object type
 * @param obj Object to delete
 *
 * Deletes VTK object if non-null. Handles reference counting properly.
 */
template <class T>
void vtkSafeDelete(T* obj) {
    if (obj) obj->Delete();
}

// =====================================================================
// Actor Creation Functions
// =====================================================================

/**
 * @brief Create actor from VTK algorithm
 * @param algo VTK polydata algorithm source
 * @return Configured VTK actor with mapper
 *
 * Creates actor with mapper connected to algorithm output.
 */
static inline vtkActor* createSourceActor(vtkPolyDataAlgorithm* algo) {
    vtkActor* actor = vtkActor::New();
    vtkSmartPointer<vtkPolyDataMapper> mapper(vtkPolyDataMapper::New());
    mapper->SetInputConnection(algo->GetOutputPort());
    actor->SetMapper(mapper);

    return actor;
}

/**
 * @class SourceSetter
 * @brief Template class for configuring VTK sources
 * @tparam T VTK source type
 *
 * Base template for source configuration. Specialized for specific sources.
 */
template <class T>
class SourceSetter {
public:
    /**
     * @brief Configure VTK source (default implementation)
     * @param source VTK source to configure
     */
    void config(T* source) {
        Q_UNUSED(source)
        // no impl
    }
};

/// Specialization for vtkSphereSource
template <>
class SourceSetter<vtkSphereSource> {
public:
    void config(vtkSphereSource* source) {}
};

/// Specialization for vtkConeSource
template <>
class SourceSetter<vtkConeSource> {
public:
    void config(vtkConeSource* source) {}
};

/**
 * @brief Create actor from VTK source with configuration
 * @tparam T VTK source type
 * @param setter Optional source configurator
 * @return Configured VTK actor
 *
 * Creates actor from VTK geometric source with optional custom configuration.
 */
template <class T>
static inline vtkActor* createSourceActor(SourceSetter<T>* setter = nullptr) {
    vtkActor* actor = vtkActor::New();

    vtkSmartPointer<T> source(T::New());

    if (setter) setter->config(source);

    vtkSmartPointer<vtkPolyDataMapper> mapper(vtkPolyDataMapper::New());
    mapper->SetInputConnection(source->GetOutputPort());
    actor->SetMapper(mapper);

    return actor;
}

/**
 * @brief Create surface actor from point cloud
 * @param points List of 3D points
 * @param scalars Optional scalar values for each point
 * @return VTK actor with Delaunay triangulation surface
 *
 * Creates a triangulated surface from points using Delaunay triangulation.
 * Optional scalar values are mapped to vertices for coloring.
 */
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

/**
 * @brief Create actor from VTK data object
 * @tparam DataObject VTK data object type
 * @tparam Mapper VTK mapper type (default: vtkPolyDataMapper)
 * @param data VTK data object to visualize
 * @return VTK actor configured with mapper
 *
 * Generic template for creating actors from any VTK data object.
 */
template <class DataObject, class Mapper = vtkPolyDataMapper>
static inline vtkActor* createActorFromData(DataObject* data) {
    vtkActor* actor = vtkActor::New();

    VTK_CREATE(Mapper, mapper);
    mapper->SetInputData(data);
    actor->SetMapper(mapper);

    return actor;
}

/**
 * @brief Export VTK actor to file
 * @param actor VTK actor to export
 * @param outfile Output file path
 *
 * Exports actor geometry to file. Format determined by file extension.
 */
void QPCL_ENGINE_LIB_API exportActorToFile(vtkActor* actor,
                                           const QString& outfile);

}  // namespace VtkUtils
