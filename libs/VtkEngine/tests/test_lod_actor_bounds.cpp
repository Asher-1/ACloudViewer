// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Bounds regression tests run without creating a render window.
#include <gtest/gtest.h>
#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkVersion.h>

#include "VTKExtensions/Views/vtkPVLODActor.h"

namespace {
class BoundsActor : public vtkPVLODActor {
public:
    static BoundsActor* New();
    vtkTypeMacro(BoundsActor, vtkPVLODActor);
    vtkMTimeType BoundsTime() { return this->BoundsMTime.GetMTime(); }
};
vtkStandardNewMacro(BoundsActor);

vtkSmartPointer<vtkPolyData> MakeVertexData(vtkPoints* points) {
    vtkNew<vtkCellArray> vertices;
    for (vtkIdType id = 0; id < points->GetNumberOfPoints(); ++id) {
        vertices->InsertNextCell(1);
        vertices->InsertCellPoint(id);
    }
    auto data = vtkSmartPointer<vtkPolyData>::New();
    data->SetPoints(points);
    data->SetVerts(vertices);
    return data;
}

void ExpectBounds(double* bounds, double xmin, double xmax) {
    ASSERT_NE(bounds, nullptr);
    EXPECT_DOUBLE_EQ(bounds[0], xmin);
    EXPECT_DOUBLE_EQ(bounds[1], xmax);
    EXPECT_DOUBLE_EQ(bounds[2], 0.0);
    EXPECT_DOUBLE_EQ(bounds[3], 1.0);
    EXPECT_DOUBLE_EQ(bounds[4], 0.0);
    EXPECT_DOUBLE_EQ(bounds[5], 1.0);
}
}  // namespace

TEST(LODActorBounds, WorldCacheInvalidatesForTransformAndMapperBounds) {
    vtkNew<vtkPoints> points;
    points->InsertNextPoint(0.0, 0.0, 0.0);
    points->InsertNextPoint(1.0, 1.0, 1.0);
    auto data = MakeVertexData(points);
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(data);
    ExpectBounds(mapper->GetBounds(), 0.0, 1.0);
    vtkNew<BoundsActor> actor;
    actor->SetMapper(mapper);

    ExpectBounds(actor->GetBounds(), 0.0, 1.0);
    auto cached = actor->BoundsTime();
    ExpectBounds(actor->GetBounds(), 0.0, 1.0);
    EXPECT_EQ(actor->BoundsTime(), cached);

    actor->SetPosition(2.0, 0.0, 0.0);
    ExpectBounds(actor->GetBounds(), 2.0, 3.0);
    EXPECT_GT(actor->BoundsTime(), cached);
    cached = actor->BoundsTime();

    points->SetPoint(1, 4.0, 1.0, 1.0);
    points->Modified();
    ExpectBounds(actor->GetBounds(), 2.0, 6.0);
    EXPECT_GT(actor->BoundsTime(), cached);
}

TEST(LODActorBounds, LODSwitchInvalidatesWithoutActorModification) {
    vtkNew<vtkPoints> fullPoints;
    fullPoints->InsertNextPoint(0.0, 0.0, 0.0);
    fullPoints->InsertNextPoint(1.0, 1.0, 1.0);
    vtkNew<vtkPoints> lodPoints;
    lodPoints->InsertNextPoint(3.0, 0.0, 0.0);
    lodPoints->InsertNextPoint(4.0, 1.0, 1.0);
    auto full = MakeVertexData(fullPoints);
    auto lod = MakeVertexData(lodPoints);
    vtkNew<vtkPolyDataMapper> fullMapper;
    fullMapper->SetInputData(full);
    vtkNew<vtkPolyDataMapper> lodMapper;
    lodMapper->SetInputData(lod);
    ExpectBounds(fullMapper->GetBounds(), 0.0, 1.0);
    ExpectBounds(lodMapper->GetBounds(), 3.0, 4.0);
    vtkNew<BoundsActor> actor;
    actor->SetMapper(fullMapper);
    actor->SetLODMapper(lodMapper);
    ExpectBounds(actor->GetBounds(), 0.0, 1.0);
    auto modified = actor->GetMTime();
    actor->SetEnableLOD(1);
    EXPECT_EQ(actor->GetMTime(), modified);
    ExpectBounds(actor->GetBounds(), 3.0, 4.0);
    actor->SetEnableLOD(0);
    ExpectBounds(actor->GetBounds(), 0.0, 1.0);
}

#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 2, 0)
TEST(LODActorBounds, NonWorldCoordinatesNeverReuseWorldBoundsCache) {
    vtkNew<vtkPoints> points;
    points->InsertNextPoint(0.0, 0.0, 0.0);
    points->InsertNextPoint(1.0, 1.0, 1.0);
    auto data = MakeVertexData(points);
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(data);
    ExpectBounds(mapper->GetBounds(), 0.0, 1.0);
    vtkNew<BoundsActor> actor;
    actor->SetMapper(mapper);
    actor->SetCoordinateSystemToPhysical();
    actor->GetBounds();
    auto cached = actor->BoundsTime();
    actor->GetBounds();
    EXPECT_GT(actor->BoundsTime(), cached);
}
#endif
