// ----------------------------------------------------------------------------
// Orthographic slice representation
// ----------------------------------------------------------------------------

#include "vtkOrthoSliceRepresentation.h"

#include <vtkActor.h>
#include <vtkAppendPolyData.h>
#include <vtkCutter.h>
#include <vtkPlane.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataPlaneClipper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkNew.h>
#include <vtkOutlineFilter.h>
#include <vtkTriangleFilter.h>

namespace {

static bool hasValidSliceGeometry(vtkPolyData* pd) {
    return pd && pd->GetNumberOfPoints() > 0 &&
           (pd->GetNumberOfPolys() > 0 || pd->GetNumberOfLines() > 0 ||
            pd->GetNumberOfVerts() > 0);
}

static vtkPolyData* emptySlicePolyData() {
    static vtkNew<vtkPolyData> empty;
    return empty.GetPointer();
}

static vtkSmartPointer<vtkPolyDataPlaneClipper> createSliceClipper(
        vtkPolyData* input, vtkPlane* plane) {
    vtkNew<vtkTriangleFilter> tri;
    tri->SetInputData(input);
    tri->Update();
    auto clipper = vtkSmartPointer<vtkPolyDataPlaneClipper>::New();
    clipper->SetInputConnection(tri->GetOutputPort());
    clipper->SetPlane(plane);
    clipper->SetCapping(true);
    clipper->SetClippingLoops(true);
    clipper->Update();
    return clipper;
}

static vtkPolyData* getSliceCapSurface(vtkPolyDataPlaneClipper* clipper) {
    if (!clipper) return nullptr;
    vtkPolyData* cap = clipper->GetCap();
    if (cap && cap->GetNumberOfPoints() > 0) return cap;
    return nullptr;
}

static vtkPolyData* getSliceSurface(vtkPolyDataPlaneClipper* clipper,
                                    vtkCutter* cutter) {
    if (auto* cap = getSliceCapSurface(clipper)) return cap;
    if (cutter && cutter->GetOutput() &&
        cutter->GetOutput()->GetNumberOfPoints() > 0)
        return cutter->GetOutput();
    return nullptr;
}

static void assignSliceMapper(vtkPolyDataMapper* mapper, vtkPolyData* slice) {
    if (!mapper) return;
    if (hasValidSliceGeometry(slice))
        mapper->SetInputData(slice);
    else
        mapper->SetInputData(emptySlicePolyData());
}

}  // namespace

struct vtkOrthoSliceRepresentation::Impl {
    vtkSmartPointer<vtkPolyData> input;
    vtkSmartPointer<vtkPlane> planes[3];
    vtkSmartPointer<vtkPolyDataPlaneClipper> clippers[3];
    vtkSmartPointer<vtkCutter> cutters[3];
    vtkSmartPointer<vtkActor> actors2D[3];
    vtkSmartPointer<vtkActor> actors3D[3];
    vtkSmartPointer<vtkActor> surfaceActor3D;
    vtkSmartPointer<vtkActor> outlineActor3D;
    double sliceOrigin[3] = {0, 0, 0};

    static const double kNormals[3][3];
};

const double vtkOrthoSliceRepresentation::Impl::kNormals[3][3] = {
        {0, 1, 0}, {1, 0, 0}, {0, 0, 1}};

vtkOrthoSliceRepresentation::vtkOrthoSliceRepresentation() : d(new Impl) {}

vtkOrthoSliceRepresentation::~vtkOrthoSliceRepresentation() { delete d; }

void vtkOrthoSliceRepresentation::setInputPolyData(vtkPolyData* input) {
    d->input = input;
    d->surfaceActor3D = nullptr;
    d->outlineActor3D = nullptr;
}

void vtkOrthoSliceRepresentation::ensureSurfaceActor() {
    if (!d->input || d->input->GetNumberOfPoints() == 0) return;
    if (d->surfaceActor3D) return;

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(d->input);
    if (d->input->GetPointData() && d->input->GetPointData()->GetScalars())
        mapper->ScalarVisibilityOn();
    else
        mapper->ScalarVisibilityOff();

    d->surfaceActor3D = vtkSmartPointer<vtkActor>::New();
    d->surfaceActor3D->SetMapper(mapper);
    const bool hasMesh = d->input->GetNumberOfPolys() > 0;
    if (hasMesh) {
        d->surfaceActor3D->GetProperty()->SetRepresentationToSurface();
        d->surfaceActor3D->GetProperty()->SetAmbient(0.2);
        d->surfaceActor3D->GetProperty()->SetDiffuse(0.8);
        d->surfaceActor3D->GetProperty()->LightingOn();
        d->surfaceActor3D->GetProperty()->SetInterpolationToPhong();
        if (!d->input->GetPointData() ||
            !d->input->GetPointData()->GetScalars()) {
            d->surfaceActor3D->GetProperty()->SetColor(0.8, 0.8, 0.8);
        }
    } else {
        d->surfaceActor3D->GetProperty()->SetRepresentationToPoints();
        d->surfaceActor3D->GetProperty()->SetPointSize(2.0);
        d->surfaceActor3D->GetProperty()->SetAmbient(1.0);
        d->surfaceActor3D->GetProperty()->SetDiffuse(0.0);
        d->surfaceActor3D->GetProperty()->LightingOff();
        d->surfaceActor3D->GetProperty()->SetColor(0.8, 0.8, 0.8);
    }
    d->surfaceActor3D->SetVisibility(0);
}

void vtkOrthoSliceRepresentation::ensureOutlineActor() {
    if (!d->input || d->input->GetNumberOfPoints() == 0) return;
    if (d->outlineActor3D) return;

    vtkNew<vtkOutlineFilter> outline;
    outline->SetInputData(d->input);
    outline->Update();
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(outline->GetOutputPort());
    d->outlineActor3D = vtkSmartPointer<vtkActor>::New();
    d->outlineActor3D->SetMapper(mapper);
    d->outlineActor3D->GetProperty()->SetColor(0.7, 0.7, 0.7);
    d->outlineActor3D->GetProperty()->SetLineWidth(1.0);
    d->outlineActor3D->GetProperty()->SetRepresentationToWireframe();
    d->outlineActor3D->GetProperty()->LightingOff();
    d->outlineActor3D->SetVisibility(0);
}

vtkPolyData* vtkOrthoSliceRepresentation::inputPolyData() const {
    return d->input;
}

void vtkOrthoSliceRepresentation::setSliceOrigin(double x, double y, double z) {
    d->sliceOrigin[0] = x;
    d->sliceOrigin[1] = y;
    d->sliceOrigin[2] = z;
}

void vtkOrthoSliceRepresentation::update() {
    if (!d->input || d->input->GetNumberOfPoints() == 0) return;

    for (int i = 0; i < 3; ++i) {
        if (!d->planes[i]) {
            d->planes[i] = vtkSmartPointer<vtkPlane>::New();
            d->planes[i]->SetNormal(Impl::kNormals[i]);
        }
        d->planes[i]->SetOrigin(d->sliceOrigin);

        d->clippers[i] = createSliceClipper(d->input, d->planes[i]);
        if (!d->cutters[i]) {
            d->cutters[i] = vtkSmartPointer<vtkCutter>::New();
            d->cutters[i]->SetInputData(d->input);
            d->cutters[i]->SetCutFunction(d->planes[i]);
        } else {
            d->cutters[i]->SetCutFunction(d->planes[i]);
            d->cutters[i]->Modified();
        }
        d->cutters[i]->Update();

        vtkPolyData* slice = getSliceSurface(d->clippers[i], d->cutters[i]);
        const bool valid = hasValidSliceGeometry(slice);

        if (!d->actors2D[i]) {
            vtkNew<vtkPolyDataMapper> mapper;
            assignSliceMapper(mapper, slice);
            mapper->ScalarVisibilityOn();
            d->actors2D[i] = vtkSmartPointer<vtkActor>::New();
            d->actors2D[i]->SetMapper(mapper);
            d->actors2D[i]->GetProperty()->SetAmbient(1.0);
            d->actors2D[i]->GetProperty()->SetDiffuse(0.0);
            d->actors2D[i]->GetProperty()->LightingOff();
        } else if (auto* mapper = vtkPolyDataMapper::SafeDownCast(
                           d->actors2D[i]->GetMapper())) {
            assignSliceMapper(mapper, slice);
            mapper->Modified();
        }
        d->actors2D[i]->SetVisibility(valid ? 1 : 0);

        if (!d->actors3D[i]) {
            vtkNew<vtkPolyDataMapper> mapper3D;
            assignSliceMapper(mapper3D, slice);
            mapper3D->ScalarVisibilityOn();
            d->actors3D[i] = vtkSmartPointer<vtkActor>::New();
            d->actors3D[i]->SetMapper(mapper3D);
            d->actors3D[i]->GetProperty()->SetAmbient(1.0);
            d->actors3D[i]->GetProperty()->SetDiffuse(0.0);
            d->actors3D[i]->GetProperty()->LightingOff();
        } else if (auto* mapper = vtkPolyDataMapper::SafeDownCast(
                           d->actors3D[i]->GetMapper())) {
            assignSliceMapper(mapper, slice);
            mapper->Modified();
        }
        d->actors3D[i]->SetVisibility(valid ? 1 : 0);
    }
}

void vtkOrthoSliceRepresentation::addToRenderers(
        vtkRenderer* orthoRenderers[3], vtkRenderer* perspectiveRenderer) {
    update();
    ensureSurfaceActor();
    ensureOutlineActor();
    for (int i = 0; i < 3; ++i) {
        if (d->actors2D[i] && orthoRenderers[i])
            orthoRenderers[i]->AddActor(d->actors2D[i]);
        if (d->actors3D[i] && perspectiveRenderer)
            perspectiveRenderer->AddActor(d->actors3D[i]);
    }
    if (d->surfaceActor3D && perspectiveRenderer)
        perspectiveRenderer->AddActor(d->surfaceActor3D);
    if (d->outlineActor3D && perspectiveRenderer)
        perspectiveRenderer->AddActor(d->outlineActor3D);
}

void vtkOrthoSliceRepresentation::removeFromRenderers(
        vtkRenderer* orthoRenderers[3], vtkRenderer* perspectiveRenderer) {
    for (int i = 0; i < 3; ++i) {
        if (d->actors2D[i] && orthoRenderers[i])
            orthoRenderers[i]->RemoveActor(d->actors2D[i]);
        if (d->actors3D[i] && perspectiveRenderer)
            perspectiveRenderer->RemoveActor(d->actors3D[i]);
    }
    if (d->surfaceActor3D && perspectiveRenderer)
        perspectiveRenderer->RemoveActor(d->surfaceActor3D);
    if (d->outlineActor3D && perspectiveRenderer)
        perspectiveRenderer->RemoveActor(d->outlineActor3D);
}

vtkActor* vtkOrthoSliceRepresentation::sliceActor2D(PlaneIndex plane) const {
    const int i = static_cast<int>(plane);
    return (i >= 0 && i < 3) ? d->actors2D[i] : nullptr;
}

vtkActor* vtkOrthoSliceRepresentation::sliceActor3D(PlaneIndex plane) const {
    const int i = static_cast<int>(plane);
    return (i >= 0 && i < 3) ? d->actors3D[i] : nullptr;
}

vtkActor* vtkOrthoSliceRepresentation::surfaceActor3D() const {
    return d->surfaceActor3D;
}

vtkActor* vtkOrthoSliceRepresentation::outlineActor3D() const {
    return d->outlineActor3D;
}

bool vtkOrthoSliceRepresentation::hasValidSlice(PlaneIndex plane) const {
    const int i = static_cast<int>(plane);
    if (i < 0 || i >= 3) return false;
    return d->actors2D[i] && d->actors2D[i]->GetVisibility() != 0;
}
