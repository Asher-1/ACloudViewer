// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file VtkVis.cpp
 * @brief Implementation of VTK/CloudViewer 3D visualization and scene
 * management.
 */

#include "VtkVis.h"

// Converters (direct CV_db -> VTK)
#include <Converters/Cc2Vtk.h>
#include <VtkRendering/Core/VtkRenderingUtils.h>

// ECV_DB
#include <ecvMesh.h>
#include <ecvNormalVectors.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>

#include "Tools/Common/ecvTools.h"
#include "VtkRendering/Rendering/MeshTextureApplier.h"
#include "VtkRendering/Rendering/TextureRenderManager.h"
#include "VtkUtils/vtkutils.h"

// SYSTEM
#include <cmath>
#include <functional>
#include <vector>

// CV_CORE_LIB
#include <CVTools.h>
#include <FileSystem.h>
#include <Helper.h>
#include <ecvGLMatrix.h>

// CV_DB_LIB
#include <LineSet.h>
#include <ecv2DLabel.h>
#include <ecvBBox.h>
#include <ecvCameraSensor.h>
#include <ecvColorScale.h>
#include <ecvDisplayTools.h>
#include <ecvGBLSensor.h>
#include <ecvGenericMesh.h>
#include <ecvHObjectCaster.h>
#include <ecvMaterial.h>
#include <ecvMaterialSet.h>
#include <ecvOrientedBBox.h>
#include <ecvScalarField.h>

// VTK Extension
#include <VTKExtensions/Core/vtkMemberFunctionCommand.h>
#include <VTKExtensions/Views/vtkPVCenterAxesActor.h>

// VTK Light
#include <VTKExtensions/Widgets/CustomVtkCaptionWidget.h>
#include <VTKExtensions/Widgets/vtkScalarBarWidgetCustom.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>

// VTK for View Properties
#include <vtkCameraOrientationRepresentation.h>
#include <vtkCameraOrientationWidget.h>
#include <vtkCubeAxesActor.h>
#include <vtkLightKit.h>

#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"
#include "VTKExtensions/InteractionStyle/vtkPVTrackballMultiRotate.h"
#include "VTKExtensions/InteractionStyle/vtkPVTrackballRoll.h"
#include "VTKExtensions/InteractionStyle/vtkPVTrackballRotate.h"
#include "VTKExtensions/InteractionStyle/vtkPVTrackballZoom.h"
#include "VTKExtensions/InteractionStyle/vtkPVTrackballZoomToMouse.h"
#include "VTKExtensions/InteractionStyle/vtkTrackballPan.h"

// VTK
#include <vtkAreaPicker.h>
#include <vtkAxes.h>
#include <vtkAxesActor.h>
#include <vtkBMPReader.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCaptionActor2D.h>
#include <vtkCaptionRepresentation.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCubeSource.h>
#include <vtkDataSetMapper.h>
#include <vtkFieldData.h>
#include <vtkFloatArray.h>
#include <vtkFollower.h>
#include <vtkJPEGReader.h>
#include <vtkLineSource.h>
#include <vtkLookupTable.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkPNGReader.h>
#include <vtkPNGWriter.h>
#include <vtkPNMReader.h>
#include <vtkPlanes.h>
#include <vtkPointData.h>
#include <vtkPointPicker.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataNormals.h>
#include <vtkPropAssembly.h>
#include <vtkPropPicker.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkQImageToImageSource.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSphereSource.h>
#include <vtkStringArray.h>
#include <vtkTIFFReader.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTexture.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTubeFilter.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVectorText.h>
#include <vtkWidgetEvent.h>
#include <vtkWidgetEventTranslator.h>
#include <vtkWindowToImageFilter.h>

// Support for VTK 7.1 upwards
#ifdef vtkGenericDataArray_h
#define SetTupleValue SetTypedTuple
#define InsertNextTupleValue InsertNextTypedTuple
#define GetTupleValue GetTypedTuple
#endif

#define ORIENT_MODE 0
#define SELECT_MODE 1

namespace Visualization {

// ============================================================================
// VtkVis Constructor and Destructor
// ============================================================================

static void SetupRenderer(vtkSmartPointer<vtkRenderer> ren) {
    ren->SetBackground(0.0, 0.0, 0.0);
    ren->GetActiveCamera()->SetParallelProjection(0);
}

VtkVis::VtkVis(vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                       interactor_style,
               const std::string& viewerName /* = ""*/,
               bool /*initIterator*/,
               int /*argc*/,
               char** /*argv*/)
    : rens_(vtkSmartPointer<vtkRendererCollection>::New()),
      win_(vtkSmartPointer<vtkRenderWindow>::New()),
      cloud_actor_map_(std::make_shared<VtkRendering::CloudActorMap>()),
      shape_actor_map_(std::make_shared<VtkRendering::ShapeActorMap>()),
      coordinate_actor_map_(
              std::make_shared<VtkRendering::CoordinateActorMap>()),
      m_widget_map(new WidgetActorMap),
      m_prop_map(new PropActorMap),
      m_currentMode(ORIENT_MODE),
      m_pointPickingEnabled(true),
      m_areaPickingEnabled(false),
      m_actorPickingEnabled(false),
      m_autoUpdateCameraPos(false),
      texture_render_manager_(
              std::make_unique<renders::TextureRenderManager>()),
      m_lightIntensity(1.0) {
    auto ren = vtkSmartPointer<vtkRenderer>::New();
    SetupRenderer(ren);
    rens_->AddItem(ren);
    win_->AddRenderer(ren);
    win_->SetWindowName(viewerName.c_str());
    win_->GlobalWarningDisplayOff();

    configCenterAxes();
    configInteractorStyle(interactor_style);
}

VtkVis::VtkVis(vtkSmartPointer<vtkRenderer> ren,
               vtkSmartPointer<vtkRenderWindow> wind,
               vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                       interactor_style,
               const std::string& viewerName,
               bool /*initIterator*/,
               int /*argc*/,
               char** /*argv*/)
    : rens_(vtkSmartPointer<vtkRendererCollection>::New()),
      win_(wind),
      cloud_actor_map_(std::make_shared<VtkRendering::CloudActorMap>()),
      shape_actor_map_(std::make_shared<VtkRendering::ShapeActorMap>()),
      coordinate_actor_map_(
              std::make_shared<VtkRendering::CoordinateActorMap>()),
      m_widget_map(new WidgetActorMap),
      m_prop_map(new PropActorMap),
      m_currentMode(ORIENT_MODE),
      m_pointPickingEnabled(true),
      m_areaPickingEnabled(false),
      m_actorPickingEnabled(false),
      m_autoUpdateCameraPos(false),
      texture_render_manager_(
              std::make_unique<renders::TextureRenderManager>()),
      m_lightIntensity(1.0) {
    SetupRenderer(ren);
    rens_->AddItem(ren);
    win_->SetWindowName(viewerName.c_str());
    win_->GlobalWarningDisplayOff();

    configCenterAxes();
    configInteractorStyle(interactor_style);
}

VtkVis::~VtkVis() {
    if (isPointPickingEnabled()) {
        setPointPickingEnabled(false);
    }

    if (isAreaPickingEnabled()) {
        setAreaPickingEnabled(false);
    }

    if (isActorPickingEnabled()) {
        setActorPickingEnabled(false);
    }

    if (m_centerAxes) {
        removeActorFromRenderer(this->m_centerAxes);
        // this->m_centerAxes->Delete();
    }

    // Clean up View Properties actors
    vtkRenderer* renderer = getCurrentRenderer();
    if (renderer) {
        // Remove all Data Axes Grids (one per object)
        for (auto& pair : m_dataAxesGridMap) {
            if (pair.second) {
                renderer->RemoveActor(pair.second);
            }
        }
        m_dataAxesGridMap.clear();
    }

    // Disable camera orientation widget
    if (m_cameraOrientationWidget) {
        m_cameraOrientationWidget->SetEnabled(0);
    }

    // vtkSmartPointer will automatically clean up the actors and widgets
}

void VtkVis::configCenterAxes() {
    this->m_centerAxes = VTKExtensions::vtkPVCenterAxesActor::New();
    this->m_centerAxes->SetComputeNormals(0);
    this->m_centerAxes->SetPickable(0);
    this->m_centerAxes->SetScale(0.25, 0.25, 0.25);
    addActorToRenderer(this->m_centerAxes);
}

void VtkVis::configInteractorStyle(
        vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                interactor_style) {
    this->TwoDInteractorStyle =
            vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>::New();
    this->ThreeDInteractorStyle = interactor_style;

    // add some default manipulators. Applications can override them without
    // much ado.
    registerInteractorStyle(false);
    // keyboard modifier configured via interactor style directly
}

void VtkVis::initialize() {
    registerMouse();
    registerKeyboard();
    registerPointPicking();
    registerAreaPicking();

    if (ThreeDInteractorStyle) {
        ThreeDInteractorStyle->setCloudActorMap(cloud_actor_map_);
        ThreeDInteractorStyle->setShapeActorMap(shape_actor_map_);
    }
}

void VtkVis::rotateWithAxis(const CCVector2i& pos,
                            const CCVector3d& axis,
                            double angle,
                            int viewport /* = 0*/) {
    vtkRenderer* ren = getCurrentRenderer(viewport);
    vtkRenderWindowInteractor* rwi = getRenderWindowInteractor();
    if (ren == nullptr || rwi == nullptr) {
        return;
    }

    vtkTransform* transform = vtkTransform::New();
    vtkCamera* camera = ren->GetActiveCamera();

    double scale = vtkMath::Norm(camera->GetPosition());
    if (scale <= 0.0) {
        scale = vtkMath::Norm(camera->GetFocalPoint());
        if (scale <= 0.0) {
            scale = 1.0;
        }
    }
    double* temp = camera->GetFocalPoint();
    camera->SetFocalPoint(temp[0] / scale, temp[1] / scale, temp[2] / scale);
    temp = camera->GetPosition();
    camera->SetPosition(temp[0] / scale, temp[1] / scale, temp[2] / scale);

    // translate to center
    transform->Identity();

    vtkCameraManipulator* currentRotateManipulator =
            this->get3DInteractorStyle()->FindManipulator(1, 0, 0);
    if (!currentRotateManipulator) {
        return;
    }

    double center[3];
    currentRotateManipulator->GetCenter(center);
    double rotationFactor = currentRotateManipulator->GetRotationFactor();
    transform->Translate(center[0] / scale, center[1] / scale,
                         center[2] / scale);

    camera->OrthogonalizeViewUp();

    transform->RotateWXYZ(angle, axis[0], axis[1], axis[2]);

    // translate back
    transform->Translate(-center[0] / scale, -center[1] / scale,
                         -center[2] / scale);

    camera->ApplyTransform(transform);
    camera->OrthogonalizeViewUp();

    // For rescale back.
    temp = camera->GetFocalPoint();
    camera->SetFocalPoint(temp[0] * scale, temp[1] * scale, temp[2] * scale);
    temp = camera->GetPosition();
    camera->SetPosition(temp[0] * scale, temp[1] * scale, temp[2] * scale);

    rwi->SetLastEventPosition(pos.x, pos.y);
    rwi->Render();
    transform->Delete();
}

void VtkVis::getCenterOfRotation(double center[3]) {
    if (this->ThreeDInteractorStyle) {
        this->ThreeDInteractorStyle->GetCenterOfRotation(center);
    }
}

void VtkVis::ExpandBounds(double bounds[6], vtkMatrix4x4* matrix) {
    if (!bounds) {
        CVLog::Warning("ERROR: Invalid bounds");
        return;
    }

    if (!matrix) {
        CVLog::Warning("ERROR: Invalid matrix");
        return;
    }

    // Expand the bounding box by model view transform matrix.
    double pt[8][4] = {{bounds[0], bounds[2], bounds[5], 1.0},
                       {bounds[1], bounds[2], bounds[5], 1.0},
                       {bounds[1], bounds[2], bounds[4], 1.0},
                       {bounds[0], bounds[2], bounds[4], 1.0},
                       {bounds[0], bounds[3], bounds[5], 1.0},
                       {bounds[1], bounds[3], bounds[5], 1.0},
                       {bounds[1], bounds[3], bounds[4], 1.0},
                       {bounds[0], bounds[3], bounds[4], 1.0}};

    // \note: Assuming that matrix doesn not have projective component. Hence
    // not dividing by the homogeneous coordinate after multiplication
    for (int i = 0; i < 8; ++i) {
        matrix->MultiplyPoint(pt[i], pt[i]);
    }

    // min = mpx = pt[0]
    double min[4], max[4];
    for (int i = 0; i < 4; ++i) {
        min[i] = pt[0][i];
        max[i] = pt[0][i];
    }

    for (int i = 1; i < 8; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (min[j] > pt[i][j]) min[j] = pt[i][j];
            if (max[j] < pt[i][j]) max[j] = pt[i][j];
        }
    }

    // Copy values back to bounds.
    bounds[0] = min[0];
    bounds[2] = min[1];
    bounds[4] = min[2];

    bounds[1] = max[0];
    bounds[3] = max[1];
    bounds[5] = max[2];
}

void VtkVis::resetCenterOfRotation(int viewport) {
    this->synchronizeGeometryBounds(viewport);
    vtkBoundingBox bbox(this->GeometryBounds);
    double center[3];
    bbox.GetCenter(center);

    setCenterOfRotation(center[0], center[1], center[2]);
}

void VtkVis::setCenterOfRotation(double x, double y, double z) {
    this->m_centerAxes->SetPosition(x, y, z);
    if (this->TwoDInteractorStyle) {
        this->TwoDInteractorStyle->SetCenterOfRotation(x, y, z);
    }
    if (this->ThreeDInteractorStyle) {
        this->ThreeDInteractorStyle->SetCenterOfRotation(x, y, z);
    }
}

//----------------------------------------------------------------------------
void VtkVis::setRotationFactor(double factor) {
    if (this->TwoDInteractorStyle) {
        this->TwoDInteractorStyle->SetRotationFactor(factor);
    }
    if (this->ThreeDInteractorStyle) {
        this->ThreeDInteractorStyle->SetRotationFactor(factor);
    }
}

double VtkVis::getRotationFactor() {
    if (this->ThreeDInteractorStyle) {
        return this->ThreeDInteractorStyle->GetRotationFactor();
    } else {
        return 1.0;
    }
}

//----------------------------------------------------------------------------
void VtkVis::setCamera2DManipulators(const int manipulators[9]) {
    this->setCameraManipulators(this->TwoDInteractorStyle, manipulators);
}

//----------------------------------------------------------------------------
void VtkVis::setCamera3DManipulators(const int manipulators[9]) {
    this->setCameraManipulators(this->ThreeDInteractorStyle, manipulators);
}

//----------------------------------------------------------------------------
void VtkVis::setCameraManipulators(
        VTKExtensions::vtkCustomInteractorStyle* style,
        const int manipulators[9]) {
    if (!style) {
        return;
    }

    style->RemoveAllManipulators();
    enum { NONE = 0, SHIFT = 1, CTRL = 2 };

    enum {
        PAN = 1,
        ZOOM = 2,
        ROLL = 3,
        ROTATE = 4,
        MULTI_ROTATE = 5,
        ZOOM_TO_MOUSE = 6
    };

    for (int manip = NONE; manip <= CTRL; manip++) {
        for (int button = 0; button < 3; button++) {
            int manipType = manipulators[3 * manip + button];
            vtkSmartPointer<vtkCameraManipulator> cameraManipulator;
            switch (manipType) {
                case PAN:
                    cameraManipulator = vtkSmartPointer<vtkTrackballPan>::New();
                    break;
                case ZOOM:
                    cameraManipulator =
                            vtkSmartPointer<vtkPVTrackballZoom>::New();
                    break;
                case ROLL:
                    cameraManipulator =
                            vtkSmartPointer<vtkPVTrackballRoll>::New();
                    break;
                case ROTATE:
                    cameraManipulator =
                            vtkSmartPointer<vtkPVTrackballRotate>::New();
                    break;
                case MULTI_ROTATE:
                    cameraManipulator =
                            vtkSmartPointer<vtkPVTrackballMultiRotate>::New();
                    break;
                case ZOOM_TO_MOUSE:
                    cameraManipulator =
                            vtkSmartPointer<vtkPVTrackballZoomToMouse>::New();
                    break;
            }
            if (cameraManipulator) {
                cameraManipulator->SetButton(button +
                                             1);  // since button starts with 1.
                cameraManipulator->SetControl(manip == CTRL ? 1 : 0);
                cameraManipulator->SetShift(manip == SHIFT ? 1 : 0);
                style->AddManipulator(cameraManipulator);
            }
        }
    }
}

//----------------------------------------------------------------------------
void VtkVis::setCamera2DMouseWheelMotionFactor(double factor) {
    if (this->TwoDInteractorStyle) {
        this->TwoDInteractorStyle->SetMouseWheelMotionFactor(factor);
    }
}

//----------------------------------------------------------------------------
void VtkVis::setCamera3DMouseWheelMotionFactor(double factor) {
    if (this->ThreeDInteractorStyle) {
        this->ThreeDInteractorStyle->SetMouseWheelMotionFactor(factor);
    }
}

//----------------------------------------------------------------------------
void VtkVis::updateCenterAxes() {
    vtkBoundingBox bbox(this->GeometryBounds);

    // include the center of rotation in the axes size determination.
    bbox.AddPoint(this->m_centerAxes->GetPosition());

    double widths[3];
    bbox.GetLengths(widths);

    // lets make some thickness in all directions
    double diameterOverTen =
            bbox.GetMaxLength() > 0 ? bbox.GetMaxLength() / 10.0 : 1.0;
    widths[0] = widths[0] < diameterOverTen ? diameterOverTen : widths[0];
    widths[1] = widths[1] < diameterOverTen ? diameterOverTen : widths[1];
    widths[2] = widths[2] < diameterOverTen ? diameterOverTen : widths[2];

    widths[0] *= 0.25;
    widths[1] *= 0.25;
    widths[2] *= 0.25;
    this->m_centerAxes->SetScale(widths);
}

//----------------------------------------------------------------------------
void VtkVis::synchronizeGeometryBounds(int viewport) {
    // get local bounds to consider 3D widgets correctly.
    // if ComputeVisiblePropBounds is called when there's no real window on the
    // local process, all vtkWidgetRepresentations return wacky Z bounds which
    // screws up the renderer and we don't see any images. Hence we skip this on
    // non-rendering nodes.
    if (!this->getCurrentRenderer(viewport)) {
        return;
    }

    this->m_centerAxes->SetUseBounds(0);
    this->GeometryBounds.Reset();

    getRendererCollection()->InitTraversal();
    vtkRenderer* renderer = nullptr;
    int i = 0;
    while ((renderer = getRendererCollection()->GetNextItem())) {
        // Modify all renderer's cameras
        if (viewport == 0 || viewport == i) {
            double prop_bounds[6];
            this->getCurrentRenderer(viewport)->ComputeVisiblePropBounds(
                    prop_bounds);
            this->GeometryBounds.AddBounds(prop_bounds);
        }
        ++i;
    }

    this->m_centerAxes->SetUseBounds(1);

    // sync up bounds across all processes when doing distributed rendering.
    if (!this->GeometryBounds.IsValid()) {
        this->GeometryBounds.SetBounds(-1, 1, -1, 1, -1, 1);
    }

    this->updateCenterAxes();
}

//*****************************************************************
// Forwarded to center axes.
//----------------------------------------------------------------------------
void VtkVis::setCenterAxesVisibility(bool v) {
    if (this->m_centerAxes) {
        this->m_centerAxes->SetVisibility(v);
    }
}

double VtkVis::getGLDepth(int x, int y) {
    // Get camera focal point and position. Convert to display (screen)
    // coordinates. We need a depth value for z-buffer.
    //
    double cameraFP[4];
    getVtkCamera()->GetFocalPoint(cameraFP);
    cameraFP[3] = 1.0;
    getCurrentRenderer()->SetWorldPoint(cameraFP);
    getCurrentRenderer()->WorldToDisplay();
    double* displayCoord = getCurrentRenderer()->GetDisplayPoint();
    double z_buffer = displayCoord[2];

    return z_buffer;
}

double VtkVis::getCameraFocalDistance(int viewport) {
    return getVtkCamera(viewport)->GetDistance();
}

void VtkVis::setCameraFocalDistance(double focal_distance, int viewport) {
    assert(focal_distance >= 0);
    vtkSmartPointer<vtkCamera> cam = getVtkCamera(viewport);
    if (cam) {
        cam->SetDistance(focal_distance);
        cam->Modified();
    }
}

void VtkVis::zoomCamera(double zoomFactor, int viewport) {
    assert(zoomFactor >= 0);
    vtkSmartPointer<vtkCamera> cam = getVtkCamera(viewport);
    if (cam) {
        cam->Zoom(zoomFactor);
        cam->Modified();
    }
}

void VtkVis::getProjectionTransformMatrix(Eigen::Matrix4d& proj, int viewport) {
    vtkRenderer* ren = getCurrentRenderer(viewport);
    vtkCamera* cam = getVtkCamera(viewport);
    if (!ren || !cam) return;
    vtkMatrix4x4* pMat = cam->GetProjectionTransformMatrix(ren);
    proj = Eigen::Matrix4d(pMat->GetData());
    proj.transposeInPlace();
}

void VtkVis::getModelViewTransformMatrix(Eigen::Matrix4d& view, int viewport) {
    vtkCamera* cam = getVtkCamera(viewport);
    if (!cam) return;
    vtkMatrix4x4* vMat = cam->GetModelViewTransformMatrix();
    view = Eigen::Matrix4d(vMat->GetData());
    view.transposeInPlace();
}

void VtkVis::resetCamera(const ccBBox* bbox) {
    if (!bbox) return;
    double bounds[6];
    bbox->getBounds(bounds);
    resetCamera(bounds);
}

void VtkVis::resetCameraClippingRange(int viewport) {
    // set all renderer to this viewpoint
    this->synchronizeGeometryBounds(viewport);
    if (this->GeometryBounds.IsValid()) {
        double originBounds[6];
        this->GeometryBounds.GetBounds(originBounds);
        this->GeometryBounds.ScaleAboutCenter(2, 2, 2);
        double bounds[6];
        this->GeometryBounds.GetBounds(bounds);
        getCurrentRenderer(viewport)->ResetCameraClippingRange(bounds);
        this->GeometryBounds.SetBounds(originBounds);
    }
    //        if (getVtkCamera()->GetParallelProjection())
    //        {
    //            double nearFar[2];
    //            this->getReasonableClippingRange(nearFar, viewport);
    //            this->setCameraClipDistances(nearFar[0] / 3, nearFar[1] * 3,
    //            viewport);
    //        }
}

void VtkVis::resetCamera(double xMin,
                         double xMax,
                         double yMin,
                         double yMax,
                         double zMin,
                         double zMax) {
    // Update the camera parameters
    getRendererCollection()->InitTraversal();
    vtkRenderer* renderer = nullptr;
    while ((renderer = getRendererCollection()->GetNextItem()) != nullptr) {
        renderer->ResetCamera(xMin, xMax, yMin, yMax, zMin, zMax);
    }
}

// reset the camera clipping range to include this entire bounding box
void VtkVis::getReasonableClippingRange(double range[2], int viewport) {
    double vn[3], position[3], a, b, c, d;
    double dist;
    int i, j, k;

    this->synchronizeGeometryBounds(viewport);

    double bounds[6];
    this->GeometryBounds.GetBounds(bounds);

    // Don't reset the clipping range when we don't have any 3D visible props
    if (!vtkMath::AreBoundsInitialized(bounds)) {
        return;
    }

    if (getVtkCamera() == nullptr) {
        CVLog::Warning("Trying to reset clipping range of non-existent camera");
        return;
    }

    double expandedBounds[6] = {bounds[0], bounds[1], bounds[2],
                                bounds[3], bounds[4], bounds[5]};
    if (!getVtkCamera()->GetUseOffAxisProjection()) {
        getVtkCamera()->GetViewPlaneNormal(vn);
        getVtkCamera()->GetPosition(position);
        this->expandBounds(expandedBounds,
                           getVtkCamera()->GetModelTransformMatrix());
    } else {
        getVtkCamera()->GetEyePosition(position);
        getVtkCamera()->GetEyePlaneNormal(vn);
        this->expandBounds(expandedBounds,
                           getVtkCamera()->GetModelViewTransformMatrix());
    }

    a = -vn[0];
    b = -vn[1];
    c = -vn[2];
    d = -(a * position[0] + b * position[1] + c * position[2]);

    // Set the max near clipping plane and the min far clipping plane
    range[0] = a * expandedBounds[0] + b * expandedBounds[2] +
               c * expandedBounds[4] + d;
    range[1] = 1e-18;

    // Find the closest / farthest bounding box vertex
    for (k = 0; k < 2; k++) {
        for (j = 0; j < 2; j++) {
            for (i = 0; i < 2; i++) {
                dist = a * expandedBounds[i] + b * expandedBounds[2 + j] +
                       c * expandedBounds[4 + k] + d;
                range[0] = (dist < range[0]) ? (dist) : (range[0]);
                range[1] = (dist > range[1]) ? (dist) : (range[1]);
            }
        }
    }

    // do not let far - near be less than 0.1 of the window height
    // this is for cases such as 2D images which may have zero range
    double minGap = 0.0;
    if (getVtkCamera()->GetParallelProjection()) {
        minGap = 0.1 * this->getParallelScale();
    } else {
        double angle =
                vtkMath::RadiansFromDegrees(getVtkCamera()->GetViewAngle());
        minGap = 0.2 * tan(angle / 2.0) * range[1];
    }
    if (range[1] - range[0] < minGap) {
        minGap = minGap - range[1] + range[0];
        range[1] += minGap / 2.0;
        range[0] -= minGap / 2.0;
    }

    // Do not let the range behind the camera throw off the calculation.
    if (range[0] < 0.0) {
        range[0] = 0.0;
    }

    // Give ourselves a little breathing room
    range[0] = 0.99 * range[0] -
               (range[1] - range[0]) *
                       getCurrentRenderer()->GetClippingRangeExpansion();
    range[1] = 1.01 * range[1] +
               (range[1] - range[0]) *
                       getCurrentRenderer()->GetClippingRangeExpansion();

    // Make sure near is not bigger than far
    range[0] = (range[0] >= range[1]) ? (0.01 * range[1]) : (range[0]);

    // Make sure near is at least some fraction of far - this prevents near
    // from being behind the camera or too close in front. How close is too
    // close depends on the resolution of the depth buffer
    if (!getCurrentRenderer()->GetNearClippingPlaneTolerance()) {
        getCurrentRenderer()->SetNearClippingPlaneTolerance(0.01);
        if (this->getRenderWindow()) {
            int ZBufferDepth = this->getRenderWindow()->GetDepthBufferSize();
            if (ZBufferDepth > 16) {
                getCurrentRenderer()->SetNearClippingPlaneTolerance(0.001);
            }
        }
    }

    // make sure the front clipping range is not too far from the far clippnig
    // range, this is to make sure that the zbuffer resolution is effectively
    // used
    if (range[0] <
        getCurrentRenderer()->GetNearClippingPlaneTolerance() * range[1]) {
        range[0] = getCurrentRenderer()->GetNearClippingPlaneTolerance() *
                   range[1];
    }
}

//----------------------------------------------------------------------------
void VtkVis::expandBounds(double bounds[6], vtkMatrix4x4* matrix) {
    if (!bounds) {
        CVLog::Error("ERROR: Invalid bounds");
        return;
    }

    if (!matrix) {
        CVLog::Error("<<ERROR: Invalid matrix");
        return;
    }

    // Expand the bounding box by model view transform matrix.
    double pt[8][4] = {{bounds[0], bounds[2], bounds[5], 1.0},
                       {bounds[1], bounds[2], bounds[5], 1.0},
                       {bounds[1], bounds[2], bounds[4], 1.0},
                       {bounds[0], bounds[2], bounds[4], 1.0},
                       {bounds[0], bounds[3], bounds[5], 1.0},
                       {bounds[1], bounds[3], bounds[5], 1.0},
                       {bounds[1], bounds[3], bounds[4], 1.0},
                       {bounds[0], bounds[3], bounds[4], 1.0}};

    // \note: Assuming that matrix doesn not have projective component. Hence
    // not dividing by the homogeneous coordinate after multiplication
    for (int i = 0; i < 8; ++i) {
        matrix->MultiplyPoint(pt[i], pt[i]);
    }

    // min = mpx = pt[0]
    double min[4], max[4];
    for (int i = 0; i < 4; ++i) {
        min[i] = pt[0][i];
        max[i] = pt[0][i];
    }

    for (int i = 1; i < 8; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (min[j] > pt[i][j]) min[j] = pt[i][j];
            if (max[j] < pt[i][j]) max[j] = pt[i][j];
        }
    }

    // Copy values back to bounds.
    bounds[0] = min[0];
    bounds[2] = min[1];
    bounds[4] = min[2];

    bounds[1] = max[0];
    bounds[3] = max[1];
    bounds[5] = max[2];
}

void VtkVis::setCameraViewAngle(double viewAngle, int viewport) {
    getRendererCollection()->InitTraversal();
    vtkRenderer* renderer = nullptr;
    int i = 0;
    while ((renderer = getRendererCollection()->GetNextItem())) {
        // Modify all renderer's cameras
        if (viewport == 0 || viewport == i) {
            vtkSmartPointer<vtkCamera> cam = renderer->GetActiveCamera();
            cam->SetViewAngle(viewAngle);
            renderer->ResetCameraClippingRange();
        }
        ++i;
    }

    this->resetCameraClippingRange(viewport);
    UpdateScreen();
}

// ============================================================================
// Methods formerly inherited from PCLVisualizer
// ============================================================================

bool VtkVis::contains(const std::string& id) const {
    return (cloud_actor_map_->find(id) != cloud_actor_map_->end() ||
            shape_actor_map_->find(id) != shape_actor_map_->end() ||
            coordinate_actor_map_->find(id) != coordinate_actor_map_->end());
}

vtkSmartPointer<vtkRendererCollection> VtkVis::getRendererCollection() {
    return rens_;
}

vtkSmartPointer<vtkRenderWindow> VtkVis::getRenderWindow() { return win_; }

void VtkVis::addCoordinateSystem(double scale,
                                 const std::string& id,
                                 int viewport) {
    if (coordinate_actor_map_->find(id) != coordinate_actor_map_->end()) return;

    auto axes = vtkSmartPointer<vtkAxes>::New();
    axes->SetOrigin(0, 0, 0);
    axes->SetScaleFactor(scale <= 0 ? 1.0 : scale);
    axes->Update();

    auto axes_colors = vtkSmartPointer<vtkFloatArray>::New();
    axes_colors->Allocate(6);
    axes_colors->InsertNextValue(0.0);
    axes_colors->InsertNextValue(0.0);
    axes_colors->InsertNextValue(0.5);
    axes_colors->InsertNextValue(0.5);
    axes_colors->InsertNextValue(1.0);
    axes_colors->InsertNextValue(1.0);

    auto axes_data = axes->GetOutput();
    axes_data->GetPointData()->SetScalars(axes_colors);

    auto tubes = vtkSmartPointer<vtkTubeFilter>::New();
    tubes->SetInputData(axes_data);
    tubes->SetRadius(axes->GetScaleFactor() / 50.0);
    tubes->SetNumberOfSides(6);

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetScalarModeToUsePointData();
    mapper->SetInputConnection(tubes->GetOutputPort());

    auto actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);

    (*coordinate_actor_map_)[id] = actor;
    addActorToRenderer(actor, viewport);
}

bool VtkVis::addText(const std::string& text,
                     int xpos,
                     int ypos,
                     int fontsize,
                     double r,
                     double g,
                     double b,
                     const std::string& id,
                     int viewport) {
    std::string tid;
    if (id.empty())
        tid = text;
    else
        tid = id;

    if (shape_actor_map_->find(tid) != shape_actor_map_->end()) return false;

    auto actor = vtkSmartPointer<vtkTextActor>::New();
    actor->SetPosition(xpos, ypos);
    actor->SetInput(text.c_str());
    auto tprop = actor->GetTextProperty();
    tprop->SetFontSize(fontsize);
    tprop->SetFontFamilyToArial();
    tprop->SetJustificationToLeft();
    tprop->BoldOn();
    tprop->SetColor(r, g, b);

    addActorToRenderer(actor, viewport);
    (*shape_actor_map_)[tid] = actor;
    return true;
}

bool VtkVis::updateText(const std::string& text,
                        int xpos,
                        int ypos,
                        const std::string& id) {
    auto it = shape_actor_map_->find(id);
    if (it == shape_actor_map_->end()) return false;
    auto textActor = vtkTextActor::SafeDownCast(it->second);
    if (!textActor) return false;
    textActor->SetInput(text.c_str());
    textActor->SetPosition(xpos, ypos);
    return true;
}

bool VtkVis::addText3D(const std::string& text,
                       double px,
                       double py,
                       double pz,
                       double text_scale,
                       double r,
                       double g,
                       double b,
                       const std::string& id,
                       int viewport) {
    std::string tid = id.empty() ? text : id;
    if (shape_actor_map_->find(tid) != shape_actor_map_->end()) return false;

    auto textSource = vtkSmartPointer<vtkVectorText>::New();
    textSource->SetText(text.c_str());
    textSource->Update();

    auto textMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    textMapper->SetInputConnection(textSource->GetOutputPort());

    rens_->InitTraversal();
    vtkRenderer* renderer;
    int i = 0;
    while ((renderer = rens_->GetNextItem())) {
        if (viewport == 0 || viewport == i) {
            auto textActor = vtkSmartPointer<vtkFollower>::New();
            textActor->SetMapper(textMapper);
            textActor->SetPosition(px, py, pz);
            textActor->SetScale(text_scale);
            textActor->GetProperty()->SetColor(r, g, b);
            textActor->SetCamera(renderer->GetActiveCamera());
            renderer->AddActor(textActor);
            const std::string uid = tid + std::string(i, '*');
            (*shape_actor_map_)[uid] = textActor;
        }
        ++i;
    }
    return true;
}

bool VtkVis::addPointCloud(vtkSmartPointer<vtkPolyData> polydata,
                           const std::string& id,
                           int viewport) {
    if (!polydata || polydata->GetNumberOfPoints() == 0) return false;

    VtkRendering::CloudActor& ca = (*cloud_actor_map_)[id];
    if (ca.actor) removeActorFromRenderer(ca.actor, viewport);

    auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(polydata);
    mapper->SetColorModeToDefault();
    mapper->SetScalarModeToUsePointData();
    mapper->ScalarVisibilityOn();

    ca.actor = vtkSmartPointer<vtkLODActor>::New();
    ca.actor->SetMapper(mapper);
    addActorToRenderer(ca.actor, viewport);
    return true;
}

bool VtkVis::updatePointCloud(vtkSmartPointer<vtkPolyData> polydata,
                              const std::string& id) {
    auto it = cloud_actor_map_->find(id);
    if (it == cloud_actor_map_->end()) return false;

    auto actor = it->second.actor;
    if (!actor || !actor->GetMapper()) return false;

    actor->GetMapper()->SetInputDataObject(polydata);
    actor->GetMapper()->Update();
    return true;
}

bool VtkVis::removePointCloud(const std::string& id, int viewport) {
    auto it = cloud_actor_map_->find(id);
    if (it == cloud_actor_map_->end()) return false;
    if (it->second.actor) removeActorFromRenderer(it->second.actor, viewport);
    cloud_actor_map_->erase(it);
    return true;
}

bool VtkVis::removeShape(const std::string& id, int viewport) {
    auto it = shape_actor_map_->find(id);
    if (it == shape_actor_map_->end()) {
        auto cit = coordinate_actor_map_->find(id);
        if (cit == coordinate_actor_map_->end()) return false;
        if (cit->second) removeActorFromRenderer(cit->second, viewport);
        coordinate_actor_map_->erase(cit);
        return true;
    }
    if (it->second) removeActorFromRenderer(it->second, viewport);
    shape_actor_map_->erase(it);
    return true;
}

bool VtkVis::removePolygonMesh(const std::string& id, int viewport) {
    return removePointCloud(id, viewport);
}

bool VtkVis::setPointCloudRenderingProperties(int property,
                                              double val,
                                              const std::string& id,
                                              int viewport) {
    auto it = cloud_actor_map_->find(id);
    if (it == cloud_actor_map_->end()) return false;
    auto* actor = vtkActor::SafeDownCast(it->second.actor);
    if (!actor) return false;
    switch (property) {
        case PCL_VISUALIZER_POINT_SIZE:
            actor->GetProperty()->SetPointSize(static_cast<float>(val));
            break;
        case PCL_VISUALIZER_OPACITY:
            actor->GetProperty()->SetOpacity(val);
            break;
        case PCL_VISUALIZER_LINE_WIDTH:
            actor->GetProperty()->SetLineWidth(static_cast<float>(val));
            break;
        case PCL_VISUALIZER_REPRESENTATION:
            if (static_cast<int>(val) == PCL_VISUALIZER_REPRESENTATION_POINTS)
                actor->GetProperty()->SetRepresentationToPoints();
            else if (static_cast<int>(val) ==
                     PCL_VISUALIZER_REPRESENTATION_WIREFRAME)
                actor->GetProperty()->SetRepresentationToWireframe();
            else
                actor->GetProperty()->SetRepresentationToSurface();
            break;
        case PCL_VISUALIZER_SHADING:
            if (static_cast<int>(val) == PCL_VISUALIZER_SHADING_FLAT)
                actor->GetProperty()->SetInterpolationToFlat();
            else if (static_cast<int>(val) == PCL_VISUALIZER_SHADING_GOURAUD)
                actor->GetProperty()->SetInterpolationToGouraud();
            else
                actor->GetProperty()->SetInterpolationToPhong();
            break;
        default:
            break;
    }
    actor->Modified();
    return true;
}

bool VtkVis::getPointCloudRenderingProperties(int property,
                                              double& val,
                                              const std::string& id,
                                              int viewport) {
    auto it = cloud_actor_map_->find(id);
    if (it == cloud_actor_map_->end()) return false;
    auto* actor = vtkActor::SafeDownCast(it->second.actor);
    if (!actor) return false;
    switch (property) {
        case PCL_VISUALIZER_POINT_SIZE:
            val = actor->GetProperty()->GetPointSize();
            break;
        case PCL_VISUALIZER_OPACITY:
            val = actor->GetProperty()->GetOpacity();
            break;
        case PCL_VISUALIZER_LINE_WIDTH:
            val = actor->GetProperty()->GetLineWidth();
            break;
        default:
            return false;
    }
    return true;
}

bool VtkVis::setPointCloudRenderingProperties(int property,
                                              double v1,
                                              double v2,
                                              double v3,
                                              const std::string& id,
                                              int viewport) {
    auto it = cloud_actor_map_->find(id);
    if (it == cloud_actor_map_->end()) return false;
    auto* actor = vtkActor::SafeDownCast(it->second.actor);
    if (!actor) return false;
    if (property == PCL_VISUALIZER_COLOR) {
        actor->GetProperty()->SetColor(v1, v2, v3);
        actor->GetMapper()->ScalarVisibilityOff();
    }
    actor->Modified();
    return true;
}

bool VtkVis::setShapeRenderingProperties(int property,
                                         double val,
                                         const std::string& id,
                                         int viewport) {
    auto it = shape_actor_map_->find(id);
    if (it == shape_actor_map_->end()) return false;
    auto* actor = vtkActor::SafeDownCast(it->second.Get());
    if (!actor) return false;
    switch (property) {
        case PCL_VISUALIZER_OPACITY:
            actor->GetProperty()->SetOpacity(val);
            break;
        case PCL_VISUALIZER_LINE_WIDTH:
            actor->GetProperty()->SetLineWidth(static_cast<float>(val));
            break;
        case PCL_VISUALIZER_REPRESENTATION:
            if (static_cast<int>(val) == PCL_VISUALIZER_REPRESENTATION_POINTS)
                actor->GetProperty()->SetRepresentationToPoints();
            else if (static_cast<int>(val) ==
                     PCL_VISUALIZER_REPRESENTATION_WIREFRAME)
                actor->GetProperty()->SetRepresentationToWireframe();
            else
                actor->GetProperty()->SetRepresentationToSurface();
            break;
        case PCL_VISUALIZER_SHADING:
            if (static_cast<int>(val) == PCL_VISUALIZER_SHADING_FLAT)
                actor->GetProperty()->SetInterpolationToFlat();
            else if (static_cast<int>(val) == PCL_VISUALIZER_SHADING_GOURAUD)
                actor->GetProperty()->SetInterpolationToGouraud();
            else
                actor->GetProperty()->SetInterpolationToPhong();
            break;
        default:
            break;
    }
    actor->Modified();
    return true;
}

bool VtkVis::setShapeRenderingProperties(int property,
                                         double v1,
                                         double v2,
                                         double v3,
                                         const std::string& id,
                                         int viewport) {
    auto it = shape_actor_map_->find(id);
    if (it == shape_actor_map_->end()) return false;
    auto* actor = vtkActor::SafeDownCast(it->second.Get());
    if (!actor) return false;
    if (property == PCL_VISUALIZER_COLOR) {
        actor->GetProperty()->SetColor(v1, v2, v3);
    }
    actor->Modified();
    return true;
}

void VtkVis::createViewPort(
        double xmin, double ymin, double xmax, double ymax, int& viewport) {
    auto ren = vtkSmartPointer<vtkRenderer>::New();
    ren->SetViewport(xmin, ymin, xmax, ymax);
    ren->SetBackground(0, 0, 0);
    rens_->AddItem(ren);
    win_->AddRenderer(ren);
    viewport = static_cast<int>(rens_->GetNumberOfItems()) - 1;
}

void VtkVis::resetCameraViewpoint(const std::string& id) {
    auto it = cloud_actor_map_->find(id);
    if (it != cloud_actor_map_->end() && it->second.viewpoint_transformation) {
        auto cam = getCurrentRenderer()->GetActiveCamera();
        cam->SetPosition(0, 0, 0);
        cam->SetFocalPoint(0, 0, 1);
        cam->SetViewUp(0, -1, 0);
        cam->ApplyTransform(static_cast<vtkTransform*>(
                vtkTransform::New()->GetLinearInverse()));
    }
    resetCamera();
}

void VtkVis::setCameraPosition(double pos_x,
                               double pos_y,
                               double pos_z,
                               double view_x,
                               double view_y,
                               double view_z,
                               double up_x,
                               double up_y,
                               double up_z,
                               int viewport) {
    auto* ren = getCurrentRenderer(viewport);
    if (!ren) return;
    auto cam = ren->GetActiveCamera();
    cam->SetPosition(pos_x, pos_y, pos_z);
    cam->SetFocalPoint(view_x, view_y, view_z);
    cam->SetViewUp(up_x, up_y, up_z);
    ren->ResetCameraClippingRange();
}

void VtkVis::setCameraPosition(double pos_x,
                               double pos_y,
                               double pos_z,
                               double up_x,
                               double up_y,
                               double up_z,
                               int viewport) {
    auto* ren = getCurrentRenderer(viewport);
    if (!ren) return;
    auto cam = ren->GetActiveCamera();
    cam->SetPosition(pos_x, pos_y, pos_z);
    cam->SetViewUp(up_x, up_y, up_z);
    ren->ResetCameraClippingRange();
}

void VtkVis::setCameraClipDistances(double znear, double zfar, int viewport) {
    auto* ren = getCurrentRenderer(viewport);
    if (!ren) return;
    ren->GetActiveCamera()->SetClippingRange(znear, zfar);
}

void VtkVis::setCameraFieldOfView(double fovy, int viewport) {
    auto* ren = getCurrentRenderer(viewport);
    if (!ren) return;
    ren->GetActiveCamera()->SetViewAngle(fovy);
}

void VtkVis::setFullScreen(bool mode) { win_->SetFullScreen(mode ? 1 : 0); }

void VtkVis::saveScreenshot(const std::string& file) {
    auto filter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    filter->SetInput(win_);
    filter->Update();
    auto writer = vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(file.c_str());
    writer->SetInputConnection(filter->GetOutputPort());
    writer->Write();
}

void VtkVis::saveCameraParameters(const std::string& /*file*/) {
    // Stub - camera parameters can be saved via getCamera()
}

void VtkVis::loadCameraParameters(const std::string& /*file*/) {
    // Stub
}

void VtkVis::setUseVbos(bool /*use_vbos*/) {
    // VBOs are the default in modern VTK - no-op
}

void VtkVis::setLookUpTableID(const std::string& /*id*/) {
    // Stub
}

void VtkVis::setBackgroundColor(double r, double g, double b, int viewport) {
    rens_->InitTraversal();
    vtkRenderer* renderer = nullptr;
    int i = 0;
    while ((renderer = rens_->GetNextItem())) {
        if (viewport == 0 || viewport == i) {
            renderer->SetBackground(r, g, b);
        }
        ++i;
    }
}

void VtkVis::resetCamera() {
    rens_->InitTraversal();
    vtkRenderer* renderer = nullptr;
    while ((renderer = rens_->GetNextItem())) {
        renderer->ResetCamera();
    }
}

// Helper: add a point cloud polydata to the renderer and cloud_actor_map_.
static bool AddPointCloudPolyData(vtkSmartPointer<vtkPolyData> polydata,
                                  const std::string& id,
                                  int viewport,
                                  Visualization::VtkVis* vis,
                                  VtkRendering::CloudActorMapPtr cloud_map) {
    if (!polydata) return false;
    vtkSmartPointer<vtkLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(polydata, actor);
    vis->addActorToRenderer(actor, viewport);
    VtkRendering::CloudActor& ca = (*cloud_map)[id];
    ca.actor = actor;
    ca.cells = polydata->GetVerts() ? polydata->GetVerts()->GetData() : nullptr;
    return true;
}

static bool UpdatePointCloudPolyData(vtkSmartPointer<vtkPolyData> polydata,
                                     const std::string& id,
                                     VtkRendering::CloudActorMapPtr cloud_map) {
    auto it = cloud_map->find(id);
    if (it == cloud_map->end()) return false;
    auto* mapper =
            vtkPolyDataMapper::SafeDownCast(it->second.actor->GetMapper());
    if (!mapper) return false;
    mapper->SetInputData(polydata);
    if (polydata->GetPointData()->GetScalars()) {
        double minmax[2];
        polydata->GetPointData()->GetScalars()->GetRange(minmax);
        mapper->SetScalarRange(minmax);
        mapper->ScalarVisibilityOn();
    }
    mapper->Update();
    it->second.actor->Modified();
    return true;
}

static bool AddPolygonMeshPolyData(vtkSmartPointer<vtkPolyData> polydata,
                                   const std::string& id,
                                   int viewport,
                                   Visualization::VtkVis* vis,
                                   VtkRendering::CloudActorMapPtr cloud_map) {
    if (!polydata) return false;
    vtkSmartPointer<vtkLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(polydata, actor);
    actor->GetProperty()->SetRepresentationToSurface();
    vis->addActorToRenderer(actor, viewport);
    VtkRendering::CloudActor& ca = (*cloud_map)[id];
    ca.actor = actor;
    return true;
}

/********************************Draw Entities*********************************/
void VtkVis::drawPointCloud(const CC_DRAW_CONTEXT& context,
                            ccPointCloud* cloud) {
    if (!cloud || cloud->size() == 0) return;

    const std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;

    bool show_sf = context.drawParam.showSF && cloud->hasScalarFields() &&
                   cloud->getCurrentDisplayedScalarFieldIndex() >= 0;
    bool show_colors = context.drawParam.showColors || show_sf;

    CVLog::PrintDebug(
            "[VtkVis::drawPointCloud] id=%s showSF=%d showColors=%d "
            "sfShown=%d hasDisplayedSF=%d sfIdx=%d hasColors=%d",
            viewID.c_str(), context.drawParam.showSF,
            context.drawParam.showColors, cloud->sfShown() ? 1 : 0,
            cloud->hasDisplayedScalarField() ? 1 : 0,
            cloud->getCurrentDisplayedScalarFieldIndex(),
            cloud->hasColors() ? 1 : 0);

    vtkSmartPointer<vtkPolyData> polydata;
    if (show_colors) {
        polydata = Converters::Cc2Vtk::PointCloudToPolyData(
                cloud, true, cloud->hasNormals(), show_sf, true);
    } else {
        polydata = Converters::Cc2Vtk::PointCloudToPolyData(
                cloud, false, cloud->hasNormals(), false, true);
        if (polydata) {
            // Apply default color as scalars
            auto nc = polydata->GetNumberOfPoints();
            auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
            colors->SetNumberOfComponents(3);
            colors->SetNumberOfTuples(nc);
            ecvColor::Rgbub col = context.pointsDefaultCol;
            for (vtkIdType i = 0; i < nc; ++i) {
                unsigned char c[3] = {col.r, col.g, col.b};
                colors->SetTypedTuple(i, c);
            }
            polydata->GetPointData()->SetScalars(colors);
        }
    }

    if (!polydata) return;

    if (!UpdatePointCloudPolyData(polydata, viewID, cloud_actor_map_)) {
        AddPointCloudPolyData(polydata, viewID, viewport, this,
                              cloud_actor_map_);
        vtkActor* actor = getActorById(viewID);
        if (actor) applyLightPropertiesToActor(actor, viewID);
    }

    updateShadingMode(context, cloud);
}

void VtkVis::drawMesh(const CC_DRAW_CONTEXT& context, ccGenericMesh* mesh) {
    if (!mesh) return;

    std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;

    ccPointCloud* pc = ccHObjectCaster::ToPointCloud(
            const_cast<ccGenericPointCloud*>(mesh->getAssociatedCloud()));
    if (!pc) return;

    auto polydata = Converters::Cc2Vtk::MeshToPolyData(pc, mesh);
    if (!polydata) return;

    if (context.visFiltering) {
        if (contains(viewID)) removeMesh(viewID, viewport);
        AddPolygonMeshPolyData(polydata, viewID, viewport, this,
                               cloud_actor_map_);
        vtkActor* actor = getActorById(viewID);
        if (actor) applyLightPropertiesToActor(actor, viewID);
    } else {
        if (!UpdatePointCloudPolyData(polydata, viewID, cloud_actor_map_)) {
            AddPolygonMeshPolyData(polydata, viewID, viewport, this,
                                   cloud_actor_map_);
            vtkActor* actor = getActorById(viewID);
            if (actor) applyLightPropertiesToActor(actor, viewID);
        }
    }

    updateShadingMode(context, pc);
}

void VtkVis::drawPolyline(const CC_DRAW_CONTEXT& context,
                          ccPolyline* polyline,
                          bool closed) {
    std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;
    Eigen::Vector3d polygonColor =
            ecvColor::Rgb::ToEigen(context.defaultPolylineColor);

    removeShape(viewID);

    auto polydata = Converters::Cc2Vtk::PolylineToPolyData(polyline);
    if (!polydata) return;

    addPolyline(polydata, polygonColor.x(), polygonColor.y(), polygonColor.z(),
                context.defaultLineWidth, viewID, viewport);
}

void VtkVis::drawSensor(const CC_DRAW_CONTEXT& context,
                        const ccSensor* sensor) {
    if (!sensor) return;

    std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;
    float lineWidth = static_cast<float>(context.currentLineWidth);
    if (contains(viewID)) {
        removeShapes(viewID, viewport);
    }

    vtkSmartPointer<vtkPolyData> linesData = nullptr;
    if (sensor->isA(CV_TYPES::CAMERA_SENSOR)) {
        // the sensor to draw
        const ccCameraSensor* camera =
                reinterpret_cast<const ccCameraSensor*>(sensor);
        if (!camera) return;
        linesData = VtkRendering::CreateCameraSensor(
                camera, context.defaultPolylineColor, context.defaultMeshColor);
    } else if (sensor->isA(CV_TYPES::GBL_SENSOR)) {
        // the sensor to draw
        const ccGBLSensor* camera =
                reinterpret_cast<const ccGBLSensor*>(sensor);
        ;
        if (!camera) return;
        linesData = VtkRendering::CreateGBLSensor(camera);
    } else {
        CVLog::Error("[VtkVis::draw] unsupported sensor type!");
    }

    if (!linesData) {
        if (sensor->isA(CV_TYPES::CAMERA_SENSOR)) {
            CVLog::Error("[VtkVis::draw] CreateCameraSensor failed!");
        } else if (sensor->isA(CV_TYPES::GBL_SENSOR)) {
            CVLog::Error("[VtkVis::draw] CreateGBLSensor failed!");
        }
        return;
    }

    // Create lines Actor
    vtkSmartPointer<vtkLODActor> linesActor;
    VtkRendering::CreateActorFromVTKDataSet(linesData, linesActor);
    linesActor->GetProperty()->SetRepresentationToSurface();
    linesActor->GetProperty()->SetLineWidth(lineWidth);
    addActorToRenderer(linesActor, viewport);

    // Save the pointer/ID pair to the global actor map
    (*getShapeActorMap())[viewID] = linesActor;

    // Apply light properties
    applyLightPropertiesToActor(linesActor, viewID);
}

void VtkVis::drawLineSet(const CC_DRAW_CONTEXT& context,
                         const cloudViewer::geometry::LineSet* lineset) {
    if (!lineset) return;

    std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;
    float lineWidth = static_cast<float>(context.currentLineWidth);
    if (contains(viewID)) {
        removeShapes(viewID, viewport);
    }

    vtkSmartPointer<vtkPolyData> linesData =
            VtkRendering::CreatePolyDataFromLineSet(*lineset, false);

    // Create lines Actor
    vtkSmartPointer<vtkLODActor> linesActor;
    VtkRendering::CreateActorFromVTKDataSet(linesData, linesActor);
    linesActor->GetProperty()->SetRepresentationToSurface();
    linesActor->GetProperty()->SetLineWidth(lineWidth);
    addActorToRenderer(linesActor, viewport);

    // Save the pointer/ID pair to the global actor map
    (*getShapeActorMap())[viewID] = linesActor;

    // Apply light properties
    applyLightPropertiesToActor(linesActor, viewID);
}

void VtkVis::transformEntities(const CC_DRAW_CONTEXT& context) {
    if (context.transformInfo.isApplyTransform()) {
        std::string viewID = CVTools::FromQString(context.viewID);
        vtkActor* actor = getActorById(viewID);
        if (actor) {
            if (context.transformInfo.isPositionChanged) {
                CCVector3d position =
                        CCVector3d::fromArray(context.transformInfo.position.u);
                actor->AddPosition(position.u);
            }

            double* origin = actor->GetOrigin();
            vtkSmartPointer<vtkTransform> trans =
                    getTransformation(context, CCVector3d::fromArray(origin));
            actor->SetUserTransform(trans);
            actor->Modified();
        }
    }
}

vtkSmartPointer<vtkTransform> VtkVis::getTransformation(
        const CC_DRAW_CONTEXT& context, const CCVector3d& origin) {
    // remove invalid bbox
    std::string bboxId =
            CVTools::FromQString(QString("BBox-") + context.viewID);
    removeShapes(bboxId, context.defaultViewPort);

    TransformInfo transInfo = context.transformInfo;

    vtkSmartPointer<vtkTransform> trans = vtkSmartPointer<vtkTransform>::New();
    trans->Identity();
    //        trans->PostMultiply();
    trans->Translate(-origin[0], -origin[1], -origin[2]);
    if (transInfo.isTranslate) {
        trans->Translate(transInfo.transVecStart.u);
        trans->Translate(transInfo.transVecEnd.u);
    }

    if (transInfo.isRotate) {
        if (transInfo.applyEuler) {
            trans->RotateZ(transInfo.eulerZYX[0]);
            trans->RotateY(transInfo.eulerZYX[1]);
            trans->RotateX(transInfo.eulerZYX[2]);
        } else {
            //                Eigen::AngleAxisd
            //                a(Eigen::Quaterniond(transInfo.quaternion));
            //                trans->RotateWXYZ
            //                (cloudViewer::RadiansToDegrees(a.angle ()), a.axis
            //                ()[0], a.axis ()[1], a.axis ()[2]);
            trans->RotateWXYZ(transInfo.rotateParam.angle,
                              transInfo.rotateParam.rotAxis.u);
        }
    }

    if (transInfo.isScale) {
        trans->Scale(transInfo.scaleXYZ.u);
    }

    trans->Translate(origin.data());

    return trans;
}

void VtkVis::updateNormals(const CC_DRAW_CONTEXT& context,
                           ccPointCloud* cloud) {
    const std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;

    if (context.drawParam.showNorms && cloud && cloud->hasNormals()) {
        const std::string normalID = viewID + "-normal";
        int normalDensity =
                context.normalDensity > 20 ? context.normalDensity : 20;
        float normalScale =
                context.normalScale > 0.02f ? context.normalScale : 0.02f;
        if (contains(normalID)) {
            removePointClouds(normalID, viewport);
        }

        // Build normal lines directly as VTK polydata
        unsigned count = cloud->size();
        unsigned step = static_cast<unsigned>(normalDensity);
        auto pts = vtkSmartPointer<vtkPoints>::New();
        auto lines = vtkSmartPointer<vtkCellArray>::New();
        vtkIdType idx = 0;
        for (unsigned i = 0; i < count; i += step) {
            const CCVector3* p = cloud->getPoint(i);
            const CCVector3& n = cloud->getPointNormal(i);
            pts->InsertNextPoint(p->x, p->y, p->z);
            pts->InsertNextPoint(p->x + n.x * normalScale,
                                 p->y + n.y * normalScale,
                                 p->z + n.z * normalScale);
            lines->InsertNextCell(2);
            lines->InsertCellPoint(idx);
            lines->InsertCellPoint(idx + 1);
            idx += 2;
        }
        auto pd = vtkSmartPointer<vtkPolyData>::New();
        pd->SetPoints(pts);
        pd->SetLines(lines);

        vtkSmartPointer<vtkLODActor> actor;
        VtkRendering::CreateActorFromVTKDataSet(pd, actor, false);
        actor->GetProperty()->SetColor(0.0, 0.0, 1.0);
        addActorToRenderer(actor, viewport);
        VtkRendering::CloudActor& ca = (*cloud_actor_map_)[normalID];
        ca.actor = actor;
    } else {
        const std::string normalID = viewID + "-normal";
        if (contains(normalID)) {
            removePointClouds(normalID, viewport);
        }
    }

    updateShadingMode(context, cloud);
}

void VtkVis::updateShadingMode(const CC_DRAW_CONTEXT& context,
                               ccPointCloud* cloud) {
    const std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;
    auto actor = getActorById(viewID);
    if (!actor) return;
    auto polydata = vtkPolyData::SafeDownCast(actor->GetMapper()->GetInput());
    if (!polydata) return;

    bool has_normal = false;
    if (cloud && cloud->hasNormals()) {
        has_normal = true;
        vtkDataArray* existingNormals = polydata->GetPointData()->GetNormals();
        bool needUpdate = context.forceRedraw || !existingNormals ||
                          existingNormals->GetNumberOfTuples() !=
                                  polydata->GetNumberOfPoints();
        if (needUpdate) {
            auto normals = vtkSmartPointer<vtkFloatArray>::New();
            normals->SetNumberOfComponents(3);
            normals->SetName("Normals");
            normals->SetNumberOfTuples(polydata->GetNumberOfPoints());
            for (vtkIdType i = 0; i < polydata->GetNumberOfPoints() &&
                                  static_cast<unsigned>(i) < cloud->size();
                 ++i) {
                const CCVector3& n =
                        cloud->getPointNormal(static_cast<unsigned>(i));
                float nf[3] = {static_cast<float>(n.x), static_cast<float>(n.y),
                               static_cast<float>(n.z)};
                normals->SetTypedTuple(i, nf);
            }
            polydata->GetPointData()->SetNormals(normals);
        }
    }

    bool has_color = false;
    if (cloud && cloud->hasColors()) {
        has_color = true;
        vtkDataArray* existingSourceRGB =
                polydata->GetPointData()->GetArray("SourceRGB");
        bool needSync = context.forceRedraw || !existingSourceRGB ||
                        existingSourceRGB->GetNumberOfTuples() !=
                                polydata->GetNumberOfPoints();
        if (needSync) {
            auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
            colors->SetName("SourceRGB");
            colors->SetNumberOfComponents(3);
            colors->SetNumberOfTuples(polydata->GetNumberOfPoints());
            for (vtkIdType i = 0; i < polydata->GetNumberOfPoints() &&
                                  static_cast<unsigned>(i) < cloud->size();
                 ++i) {
                const ecvColor::Rgb& rgb =
                        cloud->getPointColor(static_cast<unsigned>(i));
                unsigned char c[3] = {rgb.r, rgb.g, rgb.b};
                colors->SetTypedTuple(i, c);
            }
            polydata->GetPointData()->AddArray(colors);

            auto hasRGB = vtkSmartPointer<vtkIntArray>::New();
            hasRGB->SetName("HasSourceRGB");
            hasRGB->SetNumberOfTuples(1);
            hasRGB->SetValue(0, 1);
            polydata->GetFieldData()->AddArray(hasRGB);
        }
    }

    if (context.drawParam.showNorms && has_normal) {
        setMeshShadingMode(SHADING_MODE::ECV_SHADING_PHONG, viewID, viewport);
    } else {
        setMeshShadingMode(SHADING_MODE::ECV_SHADING_FLAT, viewID, viewport);
    }
    actor->GetMapper()->Update();
    actor->Modified();
}

bool VtkVis::updateScalarBar(const CC_DRAW_CONTEXT& context) {
    std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;
    vtkAbstractWidget* widget = getWidgetById(viewID);
    if (!widget || !VtkRendering::UpdateScalarBar(widget, context)) {
        this->removeWidgets(viewID, viewport);
        return false;
    }

    return true;
}

bool VtkVis::addScalarBar(const CC_DRAW_CONTEXT& context) {
    std::string viewID = CVTools::FromQString(context.viewID);
    if (containWidget(viewID)) {
        CVLog::Warning(
                "[VtkVis::addScalarBar] The id <%s> already exists! Please "
                "choose a different id and retry.",
                viewID.c_str());
        return false;
    }

    vtkSmartPointer<vtkScalarBarWidgetCustom> scalarBarWidget =
            vtkSmartPointer<vtkScalarBarWidgetCustom>::New();
    scalarBarWidget->SetInteractor(getRenderWindowInteractor());
    scalarBarWidget->CreateDefaultRepresentation();
    scalarBarWidget->CreateDefaultScalarBarActor();
    scalarBarWidget->On();

    // Save the pointer/ID pair to the global actor map
    (*m_widget_map)[viewID].widget = scalarBarWidget;

    return updateScalarBar(context);
}

bool VtkVis::updateCaption(const std::string& text,
                           const CCVector2& pos2D,
                           const CCVector3& anchorPos,
                           double r,
                           double g,
                           double b,
                           double a,
                           int fontSize,
                           const std::string& viewID,
                           int viewport) {
    Q_UNUSED(pos2D);
    Q_UNUSED(a);
    Q_UNUSED(viewport);
    vtkAbstractWidget* widget = getWidgetById(viewID);
    if (!widget) return false;

    CustomVtkCaptionWidget* captionWidget =
            CustomVtkCaptionWidget::SafeDownCast(widget);
    if (!captionWidget) return false;

    vtkCaptionRepresentation* rep = vtkCaptionRepresentation::SafeDownCast(
            captionWidget->GetRepresentation());
    if (rep) {
        // rep->SetPosition(1.0 * pos2D.x / getRenderWindow()->GetSize()[0], 1.0
        // * pos2D.y / getRenderWindow()->GetSize()[1]);
        rep->SetAnchorPosition(CCVector3d::fromArray(anchorPos.u).u);
        vtkCaptionActor2D* actor2D = rep->GetCaptionActor2D();
        actor2D->SetCaption(text.c_str());

        vtkTextProperty* textProperty =
                actor2D->GetTextActor()->GetTextProperty();
        textProperty->SetColor(r, g, b);
        textProperty->SetFontSize(fontSize);
        // Set line spacing to prevent text overlap between lines
        // 1.2 means 20% extra spacing between lines (1.0 = no extra spacing)
        textProperty->SetLineSpacing(1.2);
    }

    return true;
}

bool VtkVis::getCaptionPosition(const std::string& viewID,
                                float& posX,
                                float& posY) {
    vtkAbstractWidget* widget = getWidgetById(viewID);
    if (!widget) {
        return false;
    }

    CustomVtkCaptionWidget* captionWidget =
            CustomVtkCaptionWidget::SafeDownCast(widget);
    if (!captionWidget) {
        return false;
    }

    vtkCaptionRepresentation* rep = vtkCaptionRepresentation::SafeDownCast(
            captionWidget->GetRepresentation());
    if (!rep) {
        return false;
    }

    // GetPosition() returns a pointer to double[2] array (normalized
    // coordinates)
    const double* pos = rep->GetPosition();
    if (!pos) {
        return false;
    }

    // VTK uses normalized coordinates (0.0 to 1.0)
    // pos[0] is X (0.0 = left, 1.0 = right)
    // pos[1] is Y (0.0 = bottom, 1.0 = top in VTK coordinate system)
    posX = static_cast<float>(pos[0]);
    posY = static_cast<float>(pos[1]);

    return true;
}

bool VtkVis::addCaption(const std::string& text,
                        const CCVector2& pos2D,
                        const CCVector3& anchorPos,
                        double r,
                        double g,
                        double b,
                        double a,
                        int fontSize,
                        const std::string& viewID,
                        bool anchorDragable,
                        int viewport) {
    Q_UNUSED(viewport);

    if (containWidget(viewID)) {
        CVLog::Warning(
                "[VtkVis::addCaption] The id <%s> already exists! Please "
                "choose a different id and retry.",
                viewID.c_str());
        return false;
    }

    vtkSmartPointer<vtkCaptionRepresentation> captionRepresentation =
            vtkSmartPointer<vtkCaptionRepresentation>::New();
    {
        captionRepresentation->SetAnchorPosition(
                CCVector3d::fromArray(anchorPos.u).u);
        captionRepresentation->SetPosition(
                1.0 * pos2D.x / getRenderWindow()->GetSize()[0],
                1.0 * pos2D.y / getRenderWindow()->GetSize()[1]);
    }

    vtkCaptionActor2D* actor2D = captionRepresentation->GetCaptionActor2D();
    actor2D->SetCaption(text.c_str());
    actor2D->ThreeDimensionalLeaderOff();
    actor2D->BorderOff();
    actor2D->LeaderOn();
    actor2D->SetLeaderGlyphSize(10.0);
    actor2D->SetMaximumLeaderGlyphSize(10.0);

    const ecvColor::Rgbf& actorColor = ecvColor::FromRgb(ecvColor::red);
    actor2D->GetProperty()->SetColor(actorColor.r, actorColor.g, actorColor.b);

    vtkTextProperty* textProperty = actor2D->GetTextActor()->GetTextProperty();
    textProperty->SetColor(r, g, b);
    textProperty->SetFontSize(fontSize);
    const ecvColor::Rgbf& col = ecvColor::FromRgb(ecvColor::white);
    textProperty->SetBackgroundColor(col.r, col.g, col.b);
    textProperty->SetBackgroundOpacity(a);
    textProperty->FrameOn();
    textProperty->SetFrameColor(actorColor.r, actorColor.g, actorColor.b);
    textProperty->SetFrameWidth(2);
    textProperty->BoldOff();
    textProperty->ItalicOff();
    textProperty->SetFontFamilyToArial();
    textProperty->SetJustificationToLeft();
    textProperty->SetVerticalJustificationToCentered();
    // Set line spacing to prevent text overlap between lines
    // 1.2 means 20% extra spacing between lines (1.0 = no extra spacing)
    textProperty->SetLineSpacing(1.2);

    vtkSmartPointer<CustomVtkCaptionWidget> captionWidget =
            vtkSmartPointer<CustomVtkCaptionWidget>::New();
    captionWidget->SetHandleEnabled(anchorDragable);

    vtkRenderWindowInteractor* interactor = getRenderWindowInteractor();
    if (!interactor) {
        return false;
    }

    captionWidget->SetInteractor(interactor);
    captionWidget->SetRepresentation(captionRepresentation);
    captionWidget->On();

    // Associate the widget with the corresponding cc2DLabel for selection
    // Find the cc2DLabel by viewID in the scene database
    cc2DLabel* associatedLabel = nullptr;
    ccHObject* sceneRoot = ecvDisplayTools::GetSceneDB();
    if (sceneRoot) {
        QString viewIDStr = QString::fromStdString(viewID);

        // Recursively search for cc2DLabel with matching viewID
        std::function<cc2DLabel*(ccHObject*)> findByViewID =
                [&findByViewID, &viewIDStr](ccHObject* node) -> cc2DLabel* {
            if (!node) return nullptr;
            if (node->getViewId() == viewIDStr &&
                node->isA(CV_TYPES::LABEL_2D)) {
                return ccHObjectCaster::To2DLabel(node);
            }
            for (unsigned i = 0; i < node->getChildrenNumber(); ++i) {
                cc2DLabel* found = findByViewID(node->getChild(i));
                if (found) return found;
            }
            return nullptr;
        };
        associatedLabel = findByViewID(sceneRoot);
    }

    // Set the associated label for selection notification
    if (associatedLabel) {
        captionWidget->SetAssociatedLabel(associatedLabel);
    }

    // Save the pointer/ID pair to the global actor map
    (*m_widget_map)[viewID].widget = captionWidget;

    return true;
}

bool VtkVis::addPolyline(vtkSmartPointer<vtkPolyData> polydata,
                         double r,
                         double g,
                         double b,
                         float width,
                         const std::string& id,
                         int viewport) {
    if (!polydata || polydata->GetNumberOfPoints() < 2) {
        CVLog::Warning("[addPolyline] invalid polydata or less than 2 points.");
        return false;
    }

    auto am_it = getShapeActorMap()->find(id);
    if (am_it != getShapeActorMap()->end()) {
        CVLog::Warning(
                "[addPolyline] A shape with id <%s> already exists! Please "
                "choose a different id and retry.",
                id.c_str());
        return false;
    }

    vtkSmartPointer<vtkLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(polydata, actor, false);
    actor->GetProperty()->SetRepresentationToSurface();
    actor->GetProperty()->SetLineWidth(width);
    actor->GetProperty()->SetColor(r, g, b);
    addActorToRenderer(actor, viewport);

    (*getShapeActorMap())[id] = actor;
    applyLightPropertiesToActor(actor, id);
    return true;
}

void VtkVis::displayText(const CC_DRAW_CONTEXT& context) {
    ecvTextParam textParam = context.textParam;
    if (textParam.text.isEmpty()) {
        CVLog::Warning("empty text detected!");
        return;
    }
    std::string text = CVTools::FromQString(textParam.text);
    // std::string viewID = CVTools::FromQString(context.viewID);
    const std::string& viewID = text;

    int viewport = context.defaultViewPort;
    int xPos = static_cast<int>(textParam.textPos.x);
    int yPos = static_cast<int>(textParam.textPos.y);
    ecvColor::Rgbf textColor = ecvTools::TransFormRGB(context.textDefaultCol);
    if (textParam.display3D) {
        if (contains(viewID)) {
            removeShapes(viewID, viewport);
        }

        addText3D(text, textParam.textPos.x, textParam.textPos.y,
                  textParam.textPos.z, textParam.textScale, textColor.r,
                  textColor.g, textColor.b, viewID, viewport);
    } else {
        if (!updateText(text, xPos, yPos, viewID)) {
            addText(text, xPos, yPos, textParam.font.pointSize(), textColor.r,
                    textColor.g, textColor.b, viewID, viewport);
        }
    }
}

bool VtkVis::updateTexture(const CC_DRAW_CONTEXT& context,
                           const ccMaterialSet* materials) {
    CVLog::PrintDebug("[VtkVis::updateTexture] ENTRY: viewID=%s, materials=%zu",
                      CVTools::FromQString(context.viewID).c_str(),
                      materials ? materials->size() : 0);

    std::string viewID = CVTools::FromQString(context.viewID);
    if (!contains(viewID)) return false;
    auto actor = getActorById(viewID);
    if (!actor) return false;

    if (!materials || materials->empty()) {
        CVLog::Warning("[VtkVis::updateTexture] No materials provided");
        return false;
    }

    // Get polydata for texture coordinates
    vtkPolyData* polydata = nullptr;
    vtkPolyDataMapper* mapper =
            vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
    if (mapper) {
        polydata = vtkPolyData::SafeDownCast(mapper->GetInput());
    }

    // Get renderer
    vtkRenderer* renderer = getCurrentRenderer(context.defaultViewPort);

    // Use TextureRenderManager to update
    bool success = texture_render_manager_->Update(actor, materials, polydata,
                                                   renderer);
    if (success) {
        actor->Modified();
    }
    return success;
}

bool VtkVis::addTextureMeshFromCCMesh(ccGenericMesh* mesh,
                                      const std::string& id,
                                      int viewport) {
    CVLog::PrintDebug(
            "[VtkVis::addTextureMeshFromCCMesh] ENTRY: id=%s, viewport=%d",
            id.c_str(), viewport);

    if (!mesh) {
        CVLog::Error("[VtkVis::addTextureMeshFromCCMesh] Mesh is null!");
        return false;
    }

    // Check if actor with this ID already exists, and remove it if so
    // This allows updating/reloading meshes with the same ID (e.g., when
    // materials/textures checkbox is toggled)
    VtkRendering::CloudActorMap::iterator am_it = getCloudActorMap()->find(id);
    if (am_it != getCloudActorMap()->end()) {
        CVLog::PrintDebug(
                "[VtkVis::addTextureMeshFromCCMesh] Actor with id <%s> already "
                "exists, removing old actor before adding new one",
                id.c_str());
        vtkActor* oldActor = am_it->second.actor;
        if (oldActor) {
            removeActorFromRenderer(oldActor, viewport);
        }
        // Clean up transformation matrix from member map
        transformation_map_.erase(id);
        getCloudActorMap()->erase(am_it);
    }

    const ccMaterialSet* materials = mesh->getMaterialSet();
    if (!materials || materials->empty()) {
        CVLog::Error(
                "[VtkVis::addTextureMeshFromCCMesh] No materials found in "
                "mesh!");
        return false;
    }

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
    if (!cloud) {
        CVLog::Error(
                "[VtkVis::addTextureMeshFromCCMesh] Failed to get point cloud "
                "from mesh!");
        return false;
    }

    // Direct CV_db -> VTK conversion (no PCL intermediate)
    vtkSmartPointer<vtkPolyData> polydata;
    vtkSmartPointer<vtkMatrix4x4> transformation;
    std::vector<std::vector<Eigen::Vector2f>> tex_coordinates;
    if (!Converters::Cc2Vtk::TextureMeshToPolyData(
                cloud, mesh, polydata, transformation, tex_coordinates)) {
        CVLog::Error(
                "[VtkVis::addTextureMeshFromCCMesh] Failed to convert mesh to "
                "vtkPolyData with textures!");
        return false;
    }

    // Create mapper and actor
    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polydata);

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    addActorToRenderer(actor, viewport);

    // Apply textures using MeshTextureApplier (directly from ccMesh)
    // Now polydata already has correct texture coordinates from
    // getVtkPolyDataWithTextures
    vtkRenderer* renderer = getCurrentRenderer(viewport);
    bool success = renders::MeshTextureApplier::ApplyTexturesFromMaterialSet(
            actor, materials, tex_coordinates, polydata,
            texture_render_manager_.get(), renderer);

    if (!success) {
        CVLog::Error(
                "[VtkVis::addTextureMeshFromCCMesh] Failed to apply textures!");
        return false;
    }

    // Save the pointer/ID pair to the global actor map
    (*getCloudActorMap())[id].actor = actor;

    // Apply light properties
    applyLightPropertiesToActor(actor, id);
    // Store smart pointer in member map to ensure lifetime extends beyond
    // function scope
    transformation_map_[id] = transformation;
    (*getCloudActorMap())[id].viewpoint_transformation = transformation;

    CVLog::PrintVerbose(
            "[VtkVis::addTextureMeshFromCCMesh] Successfully added mesh "
            "with %zu materials",
            materials->size());
    return true;
}

bool VtkVis::addOrientedCube(const ccGLMatrixd& trans,
                             double width,
                             double height,
                             double depth,
                             double r,
                             double g,
                             double b,
                             const std::string& id,
                             int viewport) {
    // Check to see if this ID entry already exists (has it been already added
    // to the visualizer?)
    VtkRendering::ShapeActorMap::iterator am_it = getShapeActorMap()->find(id);
    if (am_it != getShapeActorMap()->end()) {
        CVLog::Error(
                "[VtkVis::addCube] A shape with id <%s> already exists!"
                " Please choose a different id and retry.",
                id.c_str());
        return (false);
    }

    vtkSmartPointer<vtkDataSet> data =
            VtkRendering::CreateCube(width, height, depth, trans);
    if (!data) {
        return (false);
    }

    // Create an Actor
    vtkSmartPointer<vtkLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(data, actor);
    actor->GetProperty()->SetRepresentationToSurface();
    actor->GetProperty()->SetColor(r, g, b);
    addActorToRenderer(actor, viewport);

    // Save the pointer/ID pair to the global actor map
    (*getShapeActorMap())[id] = actor;

    // Apply light properties
    applyLightPropertiesToActor(actor, id);

    return (true);
}

bool VtkVis::addOrientedCube(const Eigen::Vector3f& translation,
                             const Eigen::Quaternionf& rotation,
                             double width,
                             double height,
                             double depth,
                             double r,
                             double g,
                             double b,
                             const std::string& id,
                             int viewport) {
    // Check to see if this ID entry already exists (has it been already added
    // to the visualizer?)
    VtkRendering::ShapeActorMap::iterator am_it = getShapeActorMap()->find(id);
    if (am_it != getShapeActorMap()->end()) {
        CVLog::Error(
                "[VtkVis::addCube] A shape with id <%s> already exists!"
                " Please choose a different id and retry.",
                id.c_str());
        return (false);
    }

    auto cube = vtkSmartPointer<vtkCubeSource>::New();
    cube->SetXLength(width);
    cube->SetYLength(height);
    cube->SetZLength(depth);
    cube->Update();
    auto data = cube->GetOutput();
    if (!data) return false;

    // Apply rotation and translation
    auto transform = vtkSmartPointer<vtkTransform>::New();
    Eigen::Matrix3f rot = rotation.toRotationMatrix();
    double mat[16] = {rot(0, 0), rot(0, 1), rot(0, 2), translation.x(),
                      rot(1, 0), rot(1, 1), rot(1, 2), translation.y(),
                      rot(2, 0), rot(2, 1), rot(2, 2), translation.z(),
                      0,         0,         0,         1};
    transform->SetMatrix(mat);

    auto filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    filter->SetInputData(data);
    filter->SetTransform(transform);
    filter->Update();

    vtkSmartPointer<vtkLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(filter->GetOutput(), actor, false);
    actor->GetProperty()->SetRepresentationToSurface();
    actor->GetProperty()->SetColor(r, g, b);
    addActorToRenderer(actor, viewport);

    // Save the pointer/ID pair to the global actor map
    (*getShapeActorMap())[id] = actor;

    // Apply light properties
    applyLightPropertiesToActor(actor, id);

    return (true);
}

bool VtkVis::addOrientedCube(const ecvOrientedBBox& obb,
                             const std::string& id,
                             int viewport) {
    // Check to see if this ID entry already exists (has it been already added
    // to the visualizer?)
    VtkRendering::ShapeActorMap::iterator am_it = getShapeActorMap()->find(id);
    if (am_it != getShapeActorMap()->end()) {
        CVLog::Error(
                "[VtkVis::addCube] A shape with id <%s> already exists!"
                " Please choose a different id and retry.",
                id.c_str());
        return (false);
    }

    std::shared_ptr<cloudViewer::geometry::LineSet> linePoints =
            cloudViewer::geometry::LineSet::CreateFromOrientedBoundingBox(obb);
    vtkSmartPointer<vtkPolyData> data =
            VtkRendering::CreatePolyDataFromLineSet(*linePoints, false);
    if (!data) {
        return (false);
    }

    // Create an Actor
    vtkSmartPointer<vtkLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(data, actor);
    actor->GetProperty()->SetRepresentationToSurface();
    Eigen::Vector3d color = obb.GetColor();
    actor->GetProperty()->SetColor(color(0), color(1), color(2));
    addActorToRenderer(actor, viewport);

    // Save the pointer/ID pair to the global actor map
    (*getShapeActorMap())[id] = actor;

    // Apply light properties
    applyLightPropertiesToActor(actor, id);

    return (true);
}

bool VtkVis::addCube(double xMin,
                     double xMax,
                     double yMin,
                     double yMax,
                     double zMin,
                     double zMax,
                     double r,
                     double g,
                     double b,
                     const std::string& id,
                     int viewport) {
    if (getShapeActorMap()->find(id) != getShapeActorMap()->end()) {
        CVLog::Error(
                "[VtkVis::addCube] A shape with id <%s> already exists! Please "
                "choose a different id and retry.",
                id.c_str());
        return false;
    }

    auto cube = vtkSmartPointer<vtkCubeSource>::New();
    cube->SetBounds(xMin, xMax, yMin, yMax, zMin, zMax);
    cube->Update();
    vtkSmartPointer<vtkPolyData> data = cube->GetOutput();
    if (!data) return false;

    vtkSmartPointer<vtkLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(data, actor, false);
    actor->GetProperty()->SetRepresentationToWireframe();
    actor->GetProperty()->SetColor(r, g, b);
    addActorToRenderer(actor, viewport);

    (*getShapeActorMap())[id] = actor;
    applyLightPropertiesToActor(actor, id);
    return true;
}

bool VtkVis::addLine(double x1,
                     double y1,
                     double z1,
                     double x2,
                     double y2,
                     double z2,
                     double r,
                     double g,
                     double b,
                     const std::string& id,
                     int viewport) {
    if (getShapeActorMap()->find(id) != getShapeActorMap()->end()) {
        CVLog::Error(
                "[VtkVis::addLine] A shape with id <%s> already exists! Please "
                "choose a different id and retry.",
                id.c_str());
        return false;
    }

    auto lineSource = vtkSmartPointer<vtkLineSource>::New();
    lineSource->SetPoint1(x1, y1, z1);
    lineSource->SetPoint2(x2, y2, z2);
    lineSource->Update();
    vtkSmartPointer<vtkPolyData> data = lineSource->GetOutput();
    if (!data) return false;

    vtkSmartPointer<vtkLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(data, actor, false);
    actor->GetProperty()->SetRepresentationToSurface();
    actor->GetProperty()->SetColor(r, g, b);
    addActorToRenderer(actor, viewport);

    (*getShapeActorMap())[id] = actor;
    applyLightPropertiesToActor(actor, id);
    return true;
}

bool VtkVis::addSphere(double cx,
                       double cy,
                       double cz,
                       double radius,
                       double r,
                       double g,
                       double b,
                       const std::string& id,
                       int viewport) {
    if (getShapeActorMap()->find(id) != getShapeActorMap()->end()) {
        CVLog::Error(
                "[VtkVis::addSphere] A shape with id <%s> already exists! "
                "Please choose a different id and retry.",
                id.c_str());
        return false;
    }

    auto sphereSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetCenter(cx, cy, cz);
    sphereSource->SetRadius(radius);
    sphereSource->Update();
    vtkSmartPointer<vtkPolyData> data = sphereSource->GetOutput();
    if (!data) return false;

    vtkSmartPointer<vtkLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(data, actor, false);
    actor->GetProperty()->SetRepresentationToSurface();
    actor->GetProperty()->SetColor(r, g, b);
    addActorToRenderer(actor, viewport);

    (*getShapeActorMap())[id] = actor;
    applyLightPropertiesToActor(actor, id);
    return true;
}

/********************************Draw Entities*********************************/

/******************************** Entities Removement
 * *********************************/
void VtkVis::hideShowActors(bool visibility,
                            const std::string& viewID,
                            int viewport) {
    int opacity = visibility ? 1 : 0;

    vtkActor* actor = getActorById(viewID);
    if (actor) {
        actor->SetVisibility(opacity);
        actor->Modified();
    }
}

void VtkVis::hideShowWidgets(bool visibility,
                             const std::string& viewID,
                             int viewport) {
    // Extra step: check if there is a widget with the same ID
    WidgetActorMap::iterator wa_it = getWidgetActorMap()->find(viewID);
    if (wa_it == getWidgetActorMap()->end()) return;

    assert(wa_it->second.widget);

    visibility ? wa_it->second.widget->On() : wa_it->second.widget->Off();
}

bool VtkVis::removeEntities(const CC_DRAW_CONTEXT& context) {
    std::string removeViewID = CVTools::FromQString(context.removeViewID);

    bool removeFlag = false;

    int viewport = context.defaultViewPort;
    switch (context.removeEntityType) {
        case ENTITY_TYPE::ECV_POINT_CLOUD: {
            removePointClouds(removeViewID, viewport);
            removeFlag = true;
            break;
        }
        case ENTITY_TYPE::ECV_MESH: {
            removeMesh(removeViewID, viewport);
            removePointClouds(removeViewID, viewport);
            removeFlag = true;
        } break;
        case ENTITY_TYPE::ECV_LINES_3D:
        case ENTITY_TYPE::ECV_POLYLINE_2D:
        case ENTITY_TYPE::ECV_SHAPE:
        case ENTITY_TYPE::ECV_SENSOR: {
            removeShapes(removeViewID, viewport);
            removePointClouds(removeViewID, viewport);
            removeFlag = true;
        } break;
        case ENTITY_TYPE::ECV_TEXT3D: {
            removeText3D(removeViewID, viewport);
            removeFlag = true;
            break;
        }
        case ENTITY_TYPE::ECV_TEXT2D: {
            removeText2D(removeViewID, viewport);
            break;
        }
        case ENTITY_TYPE::ECV_CAPTION:
        case ENTITY_TYPE::ECV_SCALAR_BAR: {
            removeWidgets(removeViewID, viewport);
        } break;
        case ENTITY_TYPE::ECV_ALL: {
            removeALL(viewport);
            removeFlag = true;
            break;
        }
        default:
            break;
    }

    return removeFlag;
}

bool VtkVis::removeWidgets(const std::string& viewId, int viewport) {
    // Check to see if the given ID entry exists
    Visualization::WidgetActorMap::iterator wa_it = m_widget_map->find(viewId);

    if (wa_it == m_widget_map->end()) return (false);

    // Remove it from all renderers
    if (wa_it->second.widget) {
        // Remove the pointer/ID pair to the global actor map
        wa_it->second.widget->Off();
        m_widget_map->erase(wa_it);
        return (true);
    }
    return (false);
}

void VtkVis::removePointClouds(const std::string& viewId, int viewport) {
    if (contains(viewId)) {
        removePointCloud(viewId, viewport);
    }

    // for normals case
    std::string normalViewId = viewId + "-normal";
    if (contains(normalViewId)) {
        removePointCloud(normalViewId, viewport);
    }

    // Remove associated Data Axes Grid
    RemoveDataAxesGrid(viewId);
}

void VtkVis::removeShapes(const std::string& viewId, int viewport) {
    if (contains(viewId)) {
        removeShape(viewId, viewport);
    }

    // Remove associated Data Axes Grid
    RemoveDataAxesGrid(viewId);
}

void VtkVis::removeMesh(const std::string& viewId, int viewport) {
    if (contains(viewId)) {
        removePolygonMesh(viewId, viewport);
    }

    // Remove associated Data Axes Grid
    RemoveDataAxesGrid(viewId);
}

void VtkVis::removeText3D(const std::string& viewId, int viewport) {
    if (contains(viewId)) {
        removeText3D(viewId, viewport);
    }
}

void VtkVis::removeText2D(const std::string& viewId, int viewport) {
    if (contains(viewId)) {
        removeShapes(viewId, viewport);
    }
}

void VtkVis::removeALL(int viewport) {
    // Remove all point clouds
    std::vector<std::string> cloudIds;
    for (const auto& kv : *cloud_actor_map_) cloudIds.push_back(kv.first);
    for (const auto& id : cloudIds) removePointClouds(id, viewport);
    // Remove all shapes
    std::vector<std::string> shapeIds;
    for (const auto& kv : *shape_actor_map_) shapeIds.push_back(kv.first);
    for (const auto& id : shapeIds) removeShapes(id, viewport);
}
/******************************** Entities Removement
 * *********************************/

/******************************** Entities Setting
 * *********************************/
void VtkVis::setPointCloudUniqueColor(
        double r, double g, double b, const std::string& viewID, int viewport) {
    if (contains(viewID)) {
        setPointCloudRenderingProperties(PCL_VISUALIZER_COLOR, r, g, b, viewID,
                                         viewport);
    }
}

void VtkVis::resetScalarColor(const std::string& viewID,
                              bool flag /* = true*/,
                              int viewport /* = 0*/) {
    vtkLODActor* actor = vtkLODActor::SafeDownCast(getActorById(viewID));

    if (actor) {
        if (flag) {
            actor->GetMapper()->ScalarVisibilityOn();
        } else {
            actor->GetMapper()->ScalarVisibilityOff();
        }
        actor->Modified();
    }
}

void VtkVis::setShapeUniqueColor(
        float r, float g, float b, const std::string& viewID, int viewport) {
    if (contains(viewID)) {
        setShapeRenderingProperties(PCL_VISUALIZER_COLOR, r, g, b, viewID,
                                    viewport);
    }
}

void VtkVis::setPointSize(const unsigned char pointSize,
                          const std::string& viewID,
                          int viewport) {
    unsigned char size = pointSize > 16 ? 16 : pointSize;
    size = pointSize < 1 ? 1 : pointSize;
    if (contains(viewID)) {
        setPointCloudRenderingProperties(PCL_VISUALIZER_POINT_SIZE, size,
                                         viewID, viewport);
    }
}

void VtkVis::addScalarFieldToVTK(const std::string& viewID,
                                 ccPointCloud* cloud,
                                 int scalarFieldIndex,
                                 int viewport) {
    if (!contains(viewID) || !cloud) {
        return;
    }

    // Get scalar field
    cloudViewer::ScalarField* scalarField =
            cloud->getScalarField(scalarFieldIndex);
    if (!scalarField) {
        CVLog::Warning(
                "[VtkVis::addScalarFieldToVTK] Invalid scalar field index");
        return;
    }

    QString sfName = cloud->getScalarFieldName(scalarFieldIndex);
    std::string scalarFieldName = sfName.toStdString();

    // Get actor from cloud actor map
    VtkRendering::CloudActorMap::iterator am_it =
            getCloudActorMap()->find(viewID);
    if (am_it == getCloudActorMap()->end()) {
        return;
    }

    vtkActor* actor = am_it->second.actor;
    if (!actor) {
        return;
    }

    // Get mapper and polydata
    vtkMapper* mapper = actor->GetMapper();
    if (!mapper) {
        return;
    }

    vtkPolyData* polyData = vtkPolyData::SafeDownCast(mapper->GetInput());
    if (!polyData) {
        return;
    }

    vtkPointData* pointData = polyData->GetPointData();
    vtkDataArray* activeScalars = pointData->GetScalars();

    // Check if the correct array already exists
    vtkDataArray* existingArray = pointData->GetArray(scalarFieldName.c_str());
    if (existingArray && existingArray->GetNumberOfComponents() == 1) {
        // Correct 1-component array already exists, no need to update
        CVLog::PrintDebug(QString("[VtkVis::addScalarFieldToVTK] Scalar array "
                                  "'%1' already exists, skipping")
                                  .arg(sfName));
        return;
    }

    // NOTE: We no longer remove old scalar field arrays here.
    // This ensures all scalar fields remain available for:
    // 1. Selection/extraction operations (Find Data feature)
    // 2. Tooltip display when switching between different SFs
    // 3. Export operations that need to preserve all field data
    // The small memory overhead of keeping these arrays is acceptable.

    // Extract scalar values directly from ccPointCloud
    vtkIdType numPoints = polyData->GetNumberOfPoints();
    vtkSmartPointer<vtkFloatArray> scalarArray =
            vtkSmartPointer<vtkFloatArray>::New();
    scalarArray->SetName(scalarFieldName.c_str());
    scalarArray->SetNumberOfComponents(1);
    scalarArray->SetNumberOfTuples(numPoints);

    // Copy scalar values from ccPointCloud
    unsigned cloudSize = cloud->size();
    if (static_cast<vtkIdType>(cloudSize) != numPoints) {
        CVLog::Warning(QString("[VtkVis::addScalarFieldToVTK] Size mismatch: "
                               "ccCloud=%1, VTK=%2")
                               .arg(cloudSize)
                               .arg(numPoints));
        return;
    }

    for (vtkIdType i = 0; i < numPoints; ++i) {
        float scalarValue = static_cast<float>(
                scalarField->getValue(static_cast<unsigned>(i)));
        scalarArray->SetValue(i, scalarValue);
    }

    // Add array to polydata (but do NOT set as active scalars to avoid
    // overriding rendering) The tooltip can access this array by name without
    // it being active
    pointData->AddArray(scalarArray);
    // NOTE: Do NOT call SetActiveScalars here - it would override PCL's RGB
    // rendering!

    // Add DatasetName to field data for tooltip display (ParaView style)
    vtkFieldData* fieldData = polyData->GetFieldData();
    if (fieldData) {
        vtkStringArray* datasetNameArray = vtkStringArray::SafeDownCast(
                fieldData->GetAbstractArray("DatasetName"));

        if (!datasetNameArray) {
            // DatasetName not yet added, create it
            datasetNameArray = vtkStringArray::New();
            datasetNameArray->SetName("DatasetName");
            datasetNameArray->SetNumberOfTuples(1);

            // Use cloud name as dataset name
            QString cloudName = cloud->getName();
            if (cloudName.isEmpty()) {
                cloudName = QString::fromStdString(viewID);
            }
            datasetNameArray->SetValue(0, cloudName.toStdString());
            fieldData->AddArray(datasetNameArray);
            datasetNameArray->Delete();

            CVLog::PrintDebug(QString("[VtkVis::addScalarFieldToVTK] Added "
                                      "DatasetName: %1")
                                      .arg(cloudName));
        }

        CVLog::PrintDebug(
                QString("[VtkVis::addScalarFieldToVTK] Added scalar array "
                        "'%1' with %2 values to VTK polydata")
                        .arg(sfName)
                        .arg(numPoints));
    }
}

void VtkVis::syncAllScalarFieldsToVTK(const std::string& viewID,
                                      ccPointCloud* cloud,
                                      int viewport) {
    if (!contains(viewID) || !cloud) {
        return;
    }

    unsigned sfCount = cloud->getNumberOfScalarFields();
    if (sfCount == 0) {
        return;
    }

    CVLog::PrintDebug(
            QString("[VtkVis::syncAllScalarFieldsToVTK] Syncing %1 scalar "
                    "fields from ccPointCloud to VTK")
                    .arg(sfCount));

    // Sync each scalar field
    for (unsigned i = 0; i < sfCount; ++i) {
        addScalarFieldToVTK(viewID, cloud, static_cast<int>(i), viewport);
    }
}

void VtkVis::setCurrentSourceObject(ccHObject* obj, const std::string& viewID) {
    if (obj) {
        m_sourceObjectMap[viewID] = obj;

        CVLog::PrintDebug(QString("[VtkVis::setCurrentSourceObject] Set source "
                                  "object: '%1' (type=%2, viewID='%3')")
                                  .arg(obj->getName())
                                  .arg(obj->getClassID())
                                  .arg(QString::fromStdString(viewID)));

        // If it's a point cloud, automatically sync all scalar fields to VTK
        ccPointCloud* cloud = getSourceCloud(viewID);
        if (cloud) {
            syncAllScalarFieldsToVTK(viewID, cloud);
        }
    } else {
        // Remove from map if obj is null
        m_sourceObjectMap.erase(viewID);
    }
}

void VtkVis::removeSourceObject(const std::string& viewID) {
    m_sourceObjectMap.erase(viewID);
}

ccHObject* VtkVis::getSourceObject(const std::string& viewID) const {
    auto it = m_sourceObjectMap.find(viewID);
    if (it != m_sourceObjectMap.end()) {
        return it->second;
    }
    return nullptr;
}

ccPointCloud* VtkVis::getSourceCloud(const std::string& viewID) const {
    ccHObject* obj = getSourceObject(viewID);
    if (!obj) return nullptr;

    if (obj->isA(CV_TYPES::POINT_CLOUD)) {
        return static_cast<ccPointCloud*>(obj);
    }
    return nullptr;
}

ccMesh* VtkVis::getSourceMesh(const std::string& viewID) const {
    ccHObject* obj = getSourceObject(viewID);
    if (!obj) return nullptr;

    if (obj->isKindOf(CV_TYPES::MESH)) {
        return static_cast<ccMesh*>(obj);
    }
    return nullptr;
}

bool VtkVis::hasSourceObject(const std::string& viewID) const {
    return m_sourceObjectMap.find(viewID) != m_sourceObjectMap.end();
}

void VtkVis::setScalarFieldName(const std::string& viewID,
                                const std::string& scalarName,
                                int viewport) {
    if (!contains(viewID)) {
        return;
    }

    // Get actor from cloud actor map
    VtkRendering::CloudActorMap::iterator am_it =
            getCloudActorMap()->find(viewID);
    if (am_it == getCloudActorMap()->end()) {
        return;
    }

    vtkActor* actor = am_it->second.actor;
    if (!actor) {
        return;
    }

    // Get mapper and polydata
    vtkMapper* mapper = actor->GetMapper();
    if (!mapper) {
        return;
    }

    vtkPolyData* polyData = vtkPolyData::SafeDownCast(mapper->GetInput());
    if (!polyData) {
        return;
    }

    vtkPointData* pointData = polyData->GetPointData();

    // Try to find an array with the scalar field name (actual scalar values)
    // This should be separate from the RGB array used for rendering
    vtkDataArray* scalarArray = pointData->GetArray(scalarName.c_str());

    if (scalarArray) {
        // Found the scalar array, make it the active scalars for tooltip
        pointData->SetActiveScalars(scalarName.c_str());
        CVLog::PrintDebug(QString("[VtkVis::setScalarFieldName] Set active "
                                  "scalars to '%1' (%2 components, %3 tuples)")
                                  .arg(QString::fromStdString(scalarName))
                                  .arg(scalarArray->GetNumberOfComponents())
                                  .arg(scalarArray->GetNumberOfTuples()));
    } else {
        // Scalar array not found, try to set name on default scalars as
        // fallback
        vtkDataArray* defaultScalars = pointData->GetScalars();
        if (defaultScalars) {
            CVLog::PrintDebug(
                    QString("[VtkVis::setScalarFieldName] Scalar array '%1' "
                            "not found, using default scalars")
                            .arg(QString::fromStdString(scalarName)));
        }
    }
}

void VtkVis::setPointCloudOpacity(double opacity,
                                  const std::string& viewID,
                                  int viewport) {
    if (contains(viewID)) {
        double lastOpacity;
        getPointCloudRenderingProperties(VtkVis::PCL_VISUALIZER_OPACITY,
                                         lastOpacity, viewID);
        if (opacity != lastOpacity) {
            setPointCloudRenderingProperties(VtkVis::PCL_VISUALIZER_OPACITY,
                                             opacity, viewID, viewport);
        }
    }
}

void VtkVis::setShapeOpacity(double opacity,
                             const std::string& viewID,
                             int viewport) {
    if (contains(viewID)) {
        setShapeRenderingProperties(VtkVis::PCL_VISUALIZER_OPACITY, opacity,
                                    viewID, viewport);
    }
}

void VtkVis::setMeshOpacity(double opacity,
                            const std::string& viewID,
                            int viewport) {
    // Get the actor for this mesh - try vtkLODActor first, then vtkActor
    vtkLODActor* lodActor = vtkLODActor::SafeDownCast(getActorById(viewID));
    vtkActor* actor = lodActor;
    if (!actor) {
        // Fallback to vtkActor if not a LODActor
        actor = vtkActor::SafeDownCast(getActorById(viewID));
    }
    if (!actor) {
        CVLog::Warning("[VtkVis::setMeshOpacity] Mesh with id <%s> not found",
                       viewID.c_str());
        return;
    }

    // Check current opacity to avoid unnecessary updates
    double currentOpacity = actor->GetProperty()->GetOpacity();
    if (std::abs(currentOpacity - opacity) < 0.001) {
        return;  // No change needed
    }

    // Set the opacity on the actor's property
    // VTK automatically detects opacity < 1.0 in
    // HasTranslucentPolygonalGeometry()
    actor->GetProperty()->SetOpacity(opacity);

    // Configure transparency rendering based on opacity value
    // Following ParaView's vtkPVLODActor pattern (line 107-108)
    if (opacity < 1.0) {
        // Force actor to be treated as translucent for proper depth sorting
        // This ensures VTK renders this actor in the translucent pass
        actor->ForceTranslucentOn();
        actor->ForceOpaqueOff();

        // Configure renderer for transparency support (only once)
        vtkRenderer* renderer = getCurrentRenderer();
        if (renderer && !renderer->GetUseDepthPeeling()) {
            vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
            if (renderWindow) {
                // Enable alpha bit planes for RGBA transparency
                renderWindow->SetAlphaBitPlanes(1);
            }

            // Enable depth peeling for correct transparent object ordering
            // This is the standard VTK technique for order-independent
            // transparency
            renderer->SetUseDepthPeeling(1);
            // Quality/performance balance
            renderer->SetMaximumNumberOfPeels(4);
            // Full transparency support
            renderer->SetOcclusionRatio(0.0);
        }
    } else {
        // Opacity is 1.0 (fully opaque), render in opaque pass
        actor->ForceTranslucentOff();
        actor->ForceOpaqueOn();
    }

    // Mark the actor as modified to trigger re-render
    actor->Modified();

    CVLog::PrintVerbose("[VtkVis::setMeshOpacity] Set opacity to %.3f for <%s>",
                        opacity, viewID.c_str());
}

void VtkVis::setShapeShadingMode(SHADING_MODE mode,
                                 const std::string& viewID,
                                 int viewport) {
    if (contains(viewID)) {
        setShapeRenderingProperties(VtkVis::PCL_VISUALIZER_SHADING, mode,
                                    viewID, viewport);
    }
}

void VtkVis::setMeshShadingMode(SHADING_MODE mode,
                                const std::string& viewID,
                                int viewport) {
    vtkActor* actor = getActorById(viewID);
    if (!actor) {
        CVLog::Warning(
                "[VtkVis::SetMeshRenderingMode] Requested viewID not found, "
                "please check again...");
        return;
    }

    switch (mode) {
        case SHADING_MODE::ECV_SHADING_FLAT: {
            actor->GetProperty()->SetInterpolationToFlat();
            break;
        }
        case SHADING_MODE::ECV_SHADING_GOURAUD: {
            if (!actor->GetMapper()->GetInput()->GetPointData()->GetNormals()) {
                CVLog::Warning(
                        "[VtkVis::setMeshShadingMode] Normals do not exist in "
                        "the dataset, but Gouraud shading was requested. "
                        "Estimating normals...");
                vtkSmartPointer<vtkPolyDataNormals> normals =
                        vtkSmartPointer<vtkPolyDataNormals>::New();
                normals->SetInputConnection(actor->GetMapper()
                                                    ->GetInputAlgorithm()
                                                    ->GetOutputPort());
                vtkDataSetMapper::SafeDownCast(actor->GetMapper())
                        ->SetInputConnection(normals->GetOutputPort());
            }
            actor->GetProperty()->SetInterpolationToGouraud();
            break;
        }
        case SHADING_MODE::ECV_SHADING_PHONG: {
            if (!actor->GetMapper()->GetInput()->GetPointData()->GetNormals()) {
                CVLog::Warning(
                        "[VtkVis::setShapeRenderingProperties] "
                        "Normals do not exist in "
                        "the dataset, but Phong shading was requested. "
                        "Estimating normals...");
                vtkSmartPointer<vtkPolyDataNormals> normals =
                        vtkSmartPointer<vtkPolyDataNormals>::New();
                normals->SetInputConnection(actor->GetMapper()
                                                    ->GetInputAlgorithm()
                                                    ->GetOutputPort());
                vtkDataSetMapper::SafeDownCast(actor->GetMapper())
                        ->SetInputConnection(normals->GetOutputPort());
            }
            actor->GetProperty()->SetInterpolationToPhong();
            break;
        }
    }
    actor->Modified();
}

void VtkVis::setMeshRenderingMode(MESH_RENDERING_MODE mode,
                                  const std::string& viewID,
                                  int viewport) {
    vtkActor* actor = getActorById(viewID);
    if (!actor) {
        CVLog::Warning(
                "[VtkVis::SetMeshRenderingMode] Requested viewID not found, "
                "please check again...");
        return;
    }
    switch (mode) {
        case MESH_RENDERING_MODE::ECV_POINTS_MODE: {
            actor->GetProperty()->SetRepresentationToPoints();
            break;
        }
        case MESH_RENDERING_MODE::ECV_WIREFRAME_MODE: {
            actor->GetProperty()->SetRepresentationToWireframe();
            break;
        }
        case MESH_RENDERING_MODE::ECV_SURFACE_MODE: {
            actor->GetProperty()->SetRepresentationToSurface();
            break;
        }
    }
    actor->Modified();
}

void VtkVis::setLightMode(const std::string& viewID, int viewport) {
    vtkActor* actor = getActorById(viewID);
    if (actor) {
        // actor->GetProperty()->SetAmbient(1.0);
        actor->GetProperty()->SetLighting(false);
        actor->Modified();
    }
}

void VtkVis::setLineWidth(const unsigned char lineWidth,
                          const std::string& viewID,
                          int viewport) {
    vtkActor* actor = getActorById(viewID);
    if (actor) {
        actor->GetProperty()->SetLineWidth(float(lineWidth));
        actor->Modified();
    }
}
/******************************** Entities Setting
 * *********************************/

/******************************** Camera Tools
 * *********************************/
VtkVis::CameraParams VtkVis::getCamera(int viewport) {
    CameraParams camera;
    vtkSmartPointer<vtkCamera> vtkCam = getVtkCamera(viewport);
    if (vtkCam) {
        double pos[3], focal[3], viewUp[3], clipRange[2];
        vtkCam->GetPosition(pos);
        vtkCam->GetFocalPoint(focal);
        vtkCam->GetViewUp(viewUp);
        vtkCam->GetClippingRange(clipRange);

        std::copy(pos, pos + 3, camera.pos);
        std::copy(focal, focal + 3, camera.focal);
        std::copy(viewUp, viewUp + 3, camera.view);
        camera.clip[0] = clipRange[0];
        camera.clip[1] = clipRange[1];
        camera.fovy = vtkCam->GetViewAngle() * M_PI / 180.0;
    }
    return camera;
}

vtkSmartPointer<vtkCamera> VtkVis::getVtkCamera(int viewport) {
    vtkRenderer* ren = getCurrentRenderer(viewport);
    return ren ? ren->GetActiveCamera() : nullptr;
}

void VtkVis::setModelViewMatrix(const ccGLMatrixd& viewMat,
                                int viewport /* = 0*/) {
    getVtkCamera(viewport)->SetModelTransformMatrix(viewMat.data());
}

void VtkVis::setParallelScale(double scale, int viewport) {
    vtkSmartPointer<vtkCamera> cam = getVtkCamera(viewport);
    int flag = cam->GetParallelProjection();
    if (flag) {
        cam->SetParallelScale(scale);
        cam->Modified();
    }
}

double VtkVis::getParallelScale(int viewport) {
    return this->getVtkCamera(viewport)->GetParallelScale();
}

void VtkVis::setOrthoProjection(int viewport) {
    vtkSmartPointer<vtkCamera> cam = getVtkCamera(viewport);
    int flag = cam->GetParallelProjection();
    if (!flag) {
        cam->SetParallelProjection(true);
        getCurrentRenderer()->SetActiveCamera(cam);
        UpdateScreen();
    }
}

void VtkVis::setPerspectiveProjection(int viewport) {
    vtkSmartPointer<vtkCamera> cam = getVtkCamera(viewport);
    int flag = cam->GetParallelProjection();
    if (flag) {
        cam->SetParallelProjection(false);
        getCurrentRenderer()->SetActiveCamera(cam);
        UpdateScreen();
    }
}

bool VtkVis::getPerspectiveState(int viewport) {
    return !getVtkCamera(viewport)->GetParallelProjection();
}

/******************************** Camera Tools
 * *********************************/

/******************************** Util Tools *********************************/
void VtkVis::setAreaPickingMode(bool state) {
    if (state) {
        if (m_currentMode == ORIENT_MODE) {
            m_currentMode = SELECT_MODE;
            // Save the point picker
            m_point_picker = static_cast<vtkPointPicker*>(
                    get3DInteractorStyle()->GetInteractor()->GetPicker());
            // Switch for an area picker
            m_area_picker = vtkSmartPointer<vtkAreaPicker>::New();
            get3DInteractorStyle()->GetInteractor()->SetPicker(m_area_picker);
        }
    } else {
        if (m_currentMode == SELECT_MODE) {
            m_currentMode = ORIENT_MODE;
            // Restore point picker
            get3DInteractorStyle()->GetInteractor()->SetPicker(m_point_picker);
        }
    }

    getRendererCollection()->Render();
    get3DInteractorStyle()->GetInteractor()->Render();
}

void VtkVis::toggleAreaPicking() {
    setAreaPickingEnabled(!isAreaPickingEnabled());
    if (this->ThreeDInteractorStyle) {
        if (!this->ThreeDInteractorStyle->GetInteractor() && interactor_) {
            interactor_->SetInteractorStyle(this->ThreeDInteractorStyle);
        }
        this->ThreeDInteractorStyle->toggleAreaPicking();
    }
}

void VtkVis::exitCallbackProcess() {
    if (interactor_) interactor_->ExitCallback();
}
/******************************** Util Tools *********************************/

/********************************MarkerAxes*********************************/
void VtkVis::showPclMarkerAxes(vtkRenderWindowInteractor* interactor) {
    if (!interactor) return;
    showOrientationMarkerWidgetAxes(interactor);
    CVLog::PrintVerbose("Show Orientation Marker Widget Axes!");
}

void VtkVis::hidePclMarkerAxes() {
    // removeOrientationMarkerWidgetAxes();
    hideOrientationMarkerWidgetAxes();
    CVLog::PrintVerbose("Hide Orientation Marker Widget Axes!");
}

bool VtkVis::pclMarkerAxesShown() {
    return m_axes_widget != nullptr && m_axes_widget->GetEnabled();
}

void VtkVis::hideOrientationMarkerWidgetAxes() {
    if (m_axes_widget) {
        if (m_axes_widget->GetEnabled())
            m_axes_widget->SetEnabled(false);
        else
            CVLog::Warning(
                    "Orientation Widget Axes was already disabled, doing "
                    "nothing.");
    } else {
        CVLog::Warning(
                "Attempted to delete Orientation Widget Axes which does not "
                "exist!");
    }
}

void VtkVis::showOrientationMarkerWidgetAxes(
        vtkRenderWindowInteractor* interactor) {
    if (!m_axes_widget) {
        // Professional 3D system style with intuitive direction labels
        // Using standard directional terms instead of medical anatomy terms
        vtkSmartPointer<vtkPropAssembly> assembly =
                VtkRendering::CreateCoordinate(
                        1.8, "X", "Y", "Z", "+X",
                        "-X",         // X axis: positive/negative
                        "+Y", "-Y",   // Y axis: positive/negative
                        "+Z", "-Z");  // Z axis: positive/negative
        m_axes_widget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
        m_axes_widget->SetOutlineColor(0.9300, 0.5700, 0.1300);
        m_axes_widget->SetOutlineColor(0.2, 0.2, 0.2);  // Subtle dark outline
        m_axes_widget->SetOrientationMarker(assembly);
        m_axes_widget->SetInteractor(interactor);
        m_axes_widget->SetViewport(0.8, 0.0, 1.0, 0.2);
        // Larger viewport for better visibility, position in lower-left
        m_axes_widget->SetViewport(0.0, 0.0, 0.2, 0.2);
        m_axes_widget->SetEnabled(true);
        m_axes_widget->InteractiveOff();
    } else {
        m_axes_widget->SetEnabled(true);
        CVLog::PrintVerbose("Show Orientation Marker Widget Axes!");
    }
}

void VtkVis::toggleOrientationMarkerWidgetAxes() {
    if (m_axes_widget) {
        if (m_axes_widget->GetEnabled()) {
            m_axes_widget->SetEnabled(false);
        } else {
            m_axes_widget->SetEnabled(true);
        }
    } else {
        CVLog::Warning(
                "Attempted to delete Orientation Widget Axes which does not "
                "exist!");
    }
}
/********************************MarkerAxes*********************************/

/********************************Actor
 * Function*********************************/
vtkRenderer* VtkVis::getCurrentRenderer(int viewport) {
    auto collection = getRendererCollection();
    if (!collection) {
        return nullptr;
    }

    int itemsNumber = collection->GetNumberOfItems();
    if (itemsNumber == 0) {
        return nullptr;
    }

    if (viewport == 0 || viewport + 1 > itemsNumber) {
        return collection->GetFirstRenderer();
    }

    collection->InitTraversal();
    vtkRenderer* renderer = nullptr;
    int i = 0;
    while ((renderer = collection->GetNextItem())) {
        if (viewport == i) {
            return renderer;
        }
        ++i;
    }

    return collection->GetFirstRenderer();
}

// getPCLInteractorStyle removed - use get3DInteractorStyle() instead

vtkProp* VtkVis::getPropById(const std::string& viewId) {
    // Check to see if this ID entry already exists (has it been already added
    // to the visualizer?) Check to see if the given ID entry exists
    VtkRendering::CloudActorMap::iterator ca_it =
            getCloudActorMap()->find(viewId);
    // Extra step: check if there is a cloud with the same ID
    VtkRendering::ShapeActorMap::iterator am_it =
            getShapeActorMap()->find(viewId);

    bool shape = true;
    // Try to find a shape first
    if (am_it == getShapeActorMap()->end()) {
        // There is no cloud or shape with this ID, so just exit
        if (ca_it == getCloudActorMap()->end()) {
#ifdef QT_DEBUG
            CVLog::Error(
                    "[getActorById] Could not find any PointCloud or Shape "
                    "datasets with id <%s>!",
                    viewId.c_str());
#endif
            return nullptr;
        }

        // Cloud found, set shape to false
        shape = false;
    }

    vtkProp* prop;

    // Remove the pointer/ID pair to the global actor map
    if (shape) {
        prop = vtkLODActor::SafeDownCast(am_it->second);
        if (!prop) {
            prop = vtkActor::SafeDownCast(am_it->second);
        }
        if (!prop) {
            prop = vtkPropAssembly::SafeDownCast(am_it->second);
        }
    } else {
        prop = vtkLODActor::SafeDownCast(ca_it->second.actor);
        if (!prop) {
            prop = vtkActor::SafeDownCast(ca_it->second.actor);
        }
    }

    // Get the actor pointer

    if (!prop) return nullptr;

    return prop;
}

vtkSmartPointer<vtkPropCollection> VtkVis::getPropCollectionById(
        const std::string& viewId) {
    vtkSmartPointer<vtkPropCollection> collections =
            vtkSmartPointer<vtkPropCollection>::New();
    collections->InitTraversal();
    vtkProp* prop = getPropById(viewId);
    if (prop) {
        prop->InitPathTraversal();
        prop->GetActors(collections);
    }
    return collections;
}

vtkActor* VtkVis::getActorById(const std::string& viewId) {
    // Check to see if this ID entry already exists (has it been already added
    // to the visualizer?)
    // Check to see if the given ID entry exists
    VtkRendering::CloudActorMap::iterator ca_it =
            getCloudActorMap()->find(viewId);
    // Extra step: check if there is a cloud with the same ID
    VtkRendering::ShapeActorMap::iterator am_it =
            getShapeActorMap()->find(viewId);

    bool shape = true;
    // Try to find a shape first
    if (am_it == getShapeActorMap()->end()) {
        // There is no cloud or shape with this ID, so just exit
        if (ca_it == getCloudActorMap()->end()) {
#ifdef QT_DEBUG
            CVLog::Error(
                    "[getActorById] Could not find any PointCloud or Shape "
                    "datasets with id <%s>!",
                    viewId.c_str());
#endif
            return nullptr;
        }

        // Cloud found, set shape to false
        shape = false;
    }

    vtkActor* actor;

    // Remove the pointer/ID pair to the global actor map
    if (shape) {
        actor = vtkLODActor::SafeDownCast(am_it->second);
        if (!actor) {
            actor = vtkActor::SafeDownCast(am_it->second);
        }
    } else {
        actor = vtkLODActor::SafeDownCast(ca_it->second.actor);
        if (!actor) {
            actor = vtkActor::SafeDownCast(ca_it->second.actor);
        }
    }

    // Get the actor pointer

    if (!actor) return nullptr;

    return actor;
}

vtkAbstractWidget* VtkVis::getWidgetById(const std::string& viewId) {
    // Extra step: check if there is a widget with the same ID
    WidgetActorMap::iterator wa_it = getWidgetActorMap()->find(viewId);
    if (wa_it == getWidgetActorMap()->end()) return nullptr;
    return wa_it->second.widget;
}

std::string VtkVis::getIdByActor(vtkProp* actor) {
    // Check to see if this ID entry already exists (has it been already added
    // to the visualizer?) Check to see if the given ID entry exists
    VtkRendering::CloudActorMap::iterator cloudIt = getCloudActorMap()->begin();
    // Extra step: check if there is a cloud with the same ID
    VtkRendering::ShapeActorMap::iterator shapeIt = getShapeActorMap()->begin();

    for (; cloudIt != getCloudActorMap()->end(); cloudIt++) {
        vtkActor* tempActor = vtkLODActor::SafeDownCast(cloudIt->second.actor);
        if (tempActor && tempActor == actor) {
            return cloudIt->first;
        }
    }

    for (; shapeIt != getShapeActorMap()->end(); shapeIt++) {
        vtkActor* tempActor = vtkLODActor::SafeDownCast(shapeIt->second);
        if (tempActor && tempActor == actor) {
            return shapeIt->first;
        }
    }

    return std::string("");
}

bool VtkVis::removeActorFromRenderer(const vtkSmartPointer<vtkProp>& actor,
                                     int viewport) {
    vtkProp* actor_to_remove = vtkProp::SafeDownCast(actor);

    // Initialize traversal
    getRendererCollection()->InitTraversal();
    vtkRenderer* renderer = nullptr;
    int i = 0;
    while ((renderer = getRendererCollection()->GetNextItem()) != nullptr) {
        // Should we remove the actor from all renderers?
        if (viewport == 0) {
            renderer->RemoveActor(actor);
        } else if (viewport ==
                   i)  // add the actor only to the specified viewport
        {
            // Iterate over all actors in this renderer
            vtkPropCollection* actors = renderer->GetViewProps();
            actors->InitTraversal();
            vtkProp* current_actor = nullptr;
            while ((current_actor = actors->GetNextProp()) != nullptr) {
                if (current_actor != actor_to_remove) continue;
                renderer->RemoveActor(actor);
                // Found the correct viewport and removed the actor
                return (true);
            }
        }
        ++i;
    }
    if (viewport == 0) return (true);
    return (false);
}

void VtkVis::addActorToRenderer(const vtkSmartPointer<vtkProp>& actor,
                                int viewport) {
    // Add it to all renderers
    getRendererCollection()->InitTraversal();
    vtkRenderer* renderer = nullptr;
    int i = 0;
    while ((renderer = getRendererCollection()->GetNextItem()) != nullptr) {
        // Should we add the actor to all renderers?
        if (viewport == 0) {
            renderer->AddActor(actor);
        } else if (viewport == i) {
            // add the actor only to the specified viewport
            renderer->AddActor(actor);
        }
        ++i;
    }
}

void VtkVis::UpdateScreen() {
    // Force render window to update after actor changes
    // Similar to QVTKWidgetCustom::updateScene()
    if (getRenderWindow()) {
        getRenderWindow()->Render();
    }
}

void VtkVis::setupInteractor(vtkRenderWindowInteractor* iren,
                             vtkRenderWindow* win) {
    interactor_ = iren;
    win->AlphaBitPlanesOff();
    win->PointSmoothingOff();
    win->LineSmoothingOff();
    win->PolygonSmoothingOff();
    win->SwapBuffersOn();
    win->SetStereoTypeToAnaglyph();

    iren->SetRenderWindow(win);
    if (ThreeDInteractorStyle) {
        iren->SetInteractorStyle(ThreeDInteractorStyle);
        ThreeDInteractorStyle->Initialize();
        ThreeDInteractorStyle->setRendererCollection(rens_);
        ThreeDInteractorStyle->UseTimersOn();
    }
    iren->SetDesiredUpdateRate(30.0);
    iren->Initialize();

    auto pp = vtkSmartPointer<vtkPointPicker>::New();
    pp->SetTolerance(pp->GetTolerance() * 2);
    iren->SetPicker(pp);
}
/********************************Actor
 * Function*********************************/

/********************************Interactor
 * Function*********************************/

void VtkVis::registerKeyboard() {
    m_cloud_mutex.lock();
    auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
    cb->SetClientData(this);
    cb->SetCallback(&VtkVis::OnKeyPress);
    if (interactor_) interactor_->AddObserver(vtkCommand::KeyPressEvent, cb);
    CVLog::Print(
            "[annotation keyboard Event] press Delete to remove annotations");
    m_cloud_mutex.unlock();
}

void VtkVis::registerMouse() {
    m_cloud_mutex.lock();
    auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
    cb->SetClientData(this);
    cb->SetCallback(&VtkVis::OnRightButtonPress);
    if (interactor_)
        interactor_->AddObserver(vtkCommand::RightButtonPressEvent, cb);
    CVLog::Print(
            "[annotation mouse Event] click right button to pick annotation");
    m_cloud_mutex.unlock();
}

void VtkVis::registerPointPicking() {
    m_cloud_mutex.lock();
    auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
    cb->SetClientData(this);
    cb->SetCallback(&VtkVis::OnPointPicking);
    if (interactor_) interactor_->AddObserver(vtkCommand::EndPickEvent, cb);
    CVLog::Print("[global pointPicking] SHIFT + left click to select a point!");
    m_cloud_mutex.unlock();
}

void VtkVis::registerInteractorStyle(bool useDefault) {
    if (useDefault) {
        if (this->ThreeDInteractorStyle) {
            vtkPVTrackballRotate* manip = vtkPVTrackballRotate::New();
            manip->SetButton(1);
            this->ThreeDInteractorStyle->AddManipulator(manip);
            manip->Delete();

            vtkPVTrackballZoom* manip2 = vtkPVTrackballZoom::New();
            manip2->SetButton(3);
            this->ThreeDInteractorStyle->AddManipulator(manip2);
            manip2->Delete();

            vtkTrackballPan* manip3 = vtkTrackballPan::New();
            manip3->SetButton(2);
            this->ThreeDInteractorStyle->AddManipulator(manip3);
            manip3->Delete();
        } else {
            CVLog::Warning("register default 3D interactor styles failed!");
        }

        if (this->TwoDInteractorStyle) {
            vtkTrackballPan* manip4 = vtkTrackballPan::New();
            manip4->SetButton(1);
            this->TwoDInteractorStyle->AddManipulator(manip4);
            manip4->Delete();

            vtkPVTrackballZoom* manip5 = vtkPVTrackballZoom::New();
            manip5->SetButton(3);
            this->TwoDInteractorStyle->AddManipulator(manip5);
            manip5->Delete();
        } else {
            CVLog::Warning("register default 2D interactor styles failed!");
        }
    } else {
        int manipulators[9];
        // left button
        manipulators[0] = 4;  // no special key -> Rotate
        manipulators[3] = 1;  // shift key -> Pan
        manipulators[6] = 3;  // ctrl key -> Roll(Spin)
        // middle button
        manipulators[1] = 1;  // no special key -> Pan
        manipulators[4] = 4;  // shift key -> Rotate
        manipulators[7] = 3;  // ctrl key -> Roll(Spin)
        // right button
        manipulators[2] = 2;  // no special key -> Zoom
        manipulators[5] = 1;  // shift key -> Pan
        manipulators[8] = 6;  // ctrl key -> Zoom to Mouse
        setCamera3DManipulators(manipulators);

        // left button
        manipulators[0] = 1;  // no special key -> Pan
        manipulators[3] = 2;  // shift key -> Zoom
        manipulators[6] = 3;  // ctrl key -> Roll(Spin)
        // middle button
        manipulators[1] = 3;  // no special key -> Roll(Spin)
        manipulators[4] = 2;  // shift key -> Zoom
        manipulators[7] = 1;  // ctrl key -> Pan
        // right button
        manipulators[2] = 2;  // no special key -> Zoom
        manipulators[5] = 6;  // shift key -> Zoom to Mouse
        manipulators[8] = 4;  // ctrl key -> Rotate
        setCamera2DManipulators(manipulators);
    }
}

void VtkVis::registerAreaPicking() {
    m_cloud_mutex.lock();
    auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
    cb->SetClientData(this);
    cb->SetCallback(&VtkVis::OnAreaPicking);
    if (interactor_) interactor_->AddObserver(vtkCommand::EndPickEvent, cb);
    CVLog::Print("[global areaPicking] press A to start or ending picking!");
    m_cloud_mutex.unlock();
}

void VtkVis::OnPointPicking(vtkObject* caller,
                            unsigned long /*event_id*/,
                            void* client_data,
                            void* /*call_data*/) {
    auto* self = static_cast<VtkVis*>(client_data);
    if (!self || !self->m_pointPickingEnabled) return;

    auto* iren = vtkRenderWindowInteractor::SafeDownCast(caller);
    if (!iren) return;

    auto* picker = vtkPointPicker::SafeDownCast(iren->GetPicker());
    if (!picker) return;

    int idx = static_cast<int>(picker->GetPointId());
    if (idx == -1) return;

    double pos[3];
    picker->GetPickPosition(pos);
    CCVector3 picked_pt(static_cast<PointCoordinateType>(pos[0]),
                        static_cast<PointCoordinateType>(pos[1]),
                        static_cast<PointCoordinateType>(pos[2]));
    std::string id = self->pickItem(-1, -1, 3.0, 3.0);
    emit self->interactorPointPickedEvent(picked_pt, idx, id);
}

void VtkVis::OnAreaPicking(vtkObject* caller,
                           unsigned long /*event_id*/,
                           void* client_data,
                           void* /*call_data*/) {
    auto* self = static_cast<VtkVis*>(client_data);
    if (!self || !self->m_areaPickingEnabled) return;

    auto* iren = vtkRenderWindowInteractor::SafeDownCast(caller);
    if (!iren) return;

    auto* picker = vtkAreaPicker::SafeDownCast(iren->GetPicker());
    if (!picker) return;

    self->m_selected_slice.clear();
    auto* extract = picker->GetFrustum();
    if (!extract) return;

    vtkRenderer* ren = self->getRendererCollection()->GetFirstRenderer();
    if (!ren) return;

    auto props = ren->GetViewProps();
    props->InitTraversal();
    while (auto* prop = props->GetNextProp()) {
        auto* actor = vtkActor::SafeDownCast(prop);
        if (!actor || !actor->GetVisibility() || !actor->GetMapper()) continue;
        auto* input = actor->GetMapper()->GetInput();
        if (!input) continue;
        auto* ds = vtkDataSet::SafeDownCast(input);
        if (!ds) continue;
        for (vtkIdType i = 0; i < ds->GetNumberOfPoints(); ++i) {
            double pt[3];
            ds->GetPoint(i, pt);
            if (extract->EvaluateFunction(pt) <= 0) {
                self->m_selected_slice.push_back(static_cast<int>(i));
            }
        }
    }

    if (self->m_selected_slice.empty()) return;
    emit self->interactorAreaPickedEvent(self->m_selected_slice);
}

void VtkVis::OnKeyPress(vtkObject* caller,
                        unsigned long /*event_id*/,
                        void* client_data,
                        void* /*call_data*/) {
    auto* self = static_cast<VtkVis*>(client_data);
    if (!self) return;

    auto* iren = vtkRenderWindowInteractor::SafeDownCast(caller);
    if (!iren) return;

    std::string key = iren->GetKeySym() ? iren->GetKeySym() : "";
    if (!key.empty()) {
        emit self->interactorKeyboardEvent(key);
    }
}

void VtkVis::OnRightButtonPress(vtkObject* caller,
                                unsigned long /*event_id*/,
                                void* client_data,
                                void* /*call_data*/) {
    auto* self = static_cast<VtkVis*>(client_data);
    if (!self || !self->m_actorPickingEnabled) return;

    auto* iren = vtkRenderWindowInteractor::SafeDownCast(caller);
    if (!iren) return;

    int* pos = iren->GetEventPosition();
    vtkActor* pickedActor = self->pickActor(pos[0], pos[1]);
    if (pickedActor) {
        emit self->interactorPickedEvent(pickedActor);
    }
}

vtkActor* VtkVis::pickActor(double x, double y) {
    if (!m_propPicker) {
        m_propPicker = vtkSmartPointer<vtkPropPicker>::New();
    }

    m_propPicker->Pick(x, y, 0, getRendererCollection()->GetFirstRenderer());
    return m_propPicker->GetActor();
}

std::string VtkVis::pickItem(double x0 /* = -1*/,
                             double y0 /* = -1*/,
                             double x1 /*= 5.0*/,
                             double y1 /*= 5.0*/) {
    if (!m_area_picker) {
        m_area_picker = vtkSmartPointer<vtkAreaPicker>::New();
    }
    int* pos = getRenderWindowInteractor()->GetEventPosition();

    m_area_picker->AreaPick(pos[0], pos[1], pos[0] + x1, pos[1] + y1,
                            getRendererCollection()->GetFirstRenderer());
    vtkActor* pickedActor = m_area_picker->GetActor();
    if (pickedActor) {
        return getIdByActor(pickedActor);
    } else {
        return std::string("-1");
    }
}

QImage VtkVis::renderToImage(int zoomFactor,
                             bool renderOverlayItems,
                             bool silent,
                             int viewport) {
    bool coords_changed = false;
    bool lengend_changed = false;
    bool coords_shown = ecvDisplayTools::OrientationMarkerShown();
    bool lengend_shown = ecvDisplayTools::OverlayEntitiesAreDisplayed();
    if (lengend_shown) {
        ecvDisplayTools::DisplayOverlayEntities(false);
        lengend_changed = true;
    }
    if (renderOverlayItems) {
        if (!coords_shown) {
            ecvDisplayTools::ToggleOrientationMarker(true);
            coords_changed = true;
        }
    } else {
        if (coords_shown) {
            ecvDisplayTools::ToggleOrientationMarker(false);
            coords_changed = true;
        }
    }

    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =
            vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(getRenderWindow());
#if VTK_MAJOR_VERSION > 8 || VTK_MAJOR_VERSION == 8 && VTK_MINOR_VERSION >= 1
    windowToImageFilter->SetScale(zoomFactor);  // image quality
#else
    windowToImageFilter->SetMagnification(zoomFactor);  // image quality
#endif
    windowToImageFilter->SetInputBufferTypeToRGBA();  // also record the alpha
                                                      // (transparency) channel
    windowToImageFilter->ReadFrontBufferOff();  // read from the back buffer
    windowToImageFilter->Update();

    vtkImageData* imageData = windowToImageFilter->GetOutput();
    if (!imageData) {
        if (!silent)
            CVLog::Error("[VtkVis::renderToImage] invalid vtkImageData!");
        return QImage();
    }
    int width = imageData->GetDimensions()[0];
    int height = imageData->GetDimensions()[1];

    QImage outputImage(width, height, QImage::Format_RGB32);
    QRgb* rgbPtr =
            reinterpret_cast<QRgb*>(outputImage.bits()) + width * (height - 1);
    unsigned char* colorsPtr =
            reinterpret_cast<unsigned char*>(imageData->GetScalarPointer());
    if (!colorsPtr) {
        if (!silent)
            CVLog::Error(
                    "[VtkVis::renderToImage] invalid scalar pointer of "
                    "vtkImageData!");
        return QImage();
    }

    // Loop over the vtkImageData contents.
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            // Swap the vtkImageData RGB values with an equivalent QColor
            *(rgbPtr++) =
                    QColor(colorsPtr[0], colorsPtr[1], colorsPtr[2]).rgb();
            colorsPtr += imageData->GetNumberOfScalarComponents();
        }

        rgbPtr -= width * 2;
    }

    if (outputImage.isNull()) {
        if (!silent)
            CVLog::Error(
                    "[VtkVis::renderToImage] Direct screen capture failed! "
                    "(not enough memory?)");
    }

    if (lengend_changed) {
        ecvDisplayTools::DisplayOverlayEntities(lengend_shown);
    }

    if (coords_changed) {
        ecvDisplayTools::ToggleOrientationMarker(coords_shown);
    }

    return outputImage;
}

// ============================================================================
// PBR Material Conversion Functions
// ============================================================================

// ============================================================================
// View Properties Implementation (ParaView-compatible)
// ============================================================================

void VtkVis::applyLightPropertiesToActor(vtkActor* actor,
                                         const std::string& viewID) {
    if (!actor) return;
    vtkProperty* prop = actor->GetProperty();
    if (!prop) return;

    // Resolve the effective intensity: per-object if stored, else global
    double intensity = m_lightIntensity;
    if (!viewID.empty()) {
        auto it = m_objectLightIntensity.find(viewID);
        if (it != m_objectLightIntensity.end()) {
            intensity = it->second;
        }
    }

    // Enable lighting on the actor
    prop->SetLighting(true);

    // Determine if the actor is a point cloud or a mesh
    bool isPointCloud = false;
    vtkMapper* mapper = actor->GetMapper();
    if (mapper) {
        vtkPolyData* polydata = vtkPolyData::SafeDownCast(mapper->GetInput());
        if (polydata) {
            vtkIdType numPoints = polydata->GetNumberOfPoints();
            vtkIdType numCells = polydata->GetNumberOfCells();
            isPointCloud = (numCells == 0) || (numCells >= numPoints * 0.9);
        }
    }

    if (isPointCloud) {
        // Point clouds: scale ambient/diffuse by intensity
        double ambient = 0.1 + intensity * 0.5;  // Range: 0.1-0.6
        double diffuse = 0.3 + intensity * 0.5;  // Range: 0.3-0.8
        prop->SetAmbient(ambient);
        prop->SetDiffuse(diffuse);
        prop->SetSpecular(0.1);
    } else {
        // Meshes: ALSO scale by intensity (was previously fixed values)
        double ambient = 0.1 + intensity * 0.2;  // Range: 0.1-0.3
        double diffuse = intensity * 0.7;        // Range: 0.0-0.7
        double specular = intensity * 0.2;       // Range: 0.0-0.2
        prop->SetAmbient(ambient);
        prop->SetDiffuse(diffuse);
        prop->SetSpecular(specular);
    }

    actor->Modified();
}

void VtkVis::setObjectLightIntensity(const std::string& viewID,
                                     double intensity,
                                     int viewport) {
    intensity = std::max(0.0, std::min(1.0, intensity));
    m_objectLightIntensity[viewID] = intensity;

    // Apply to the actor immediately if it exists
    vtkActor* actor = getActorById(viewID);
    if (actor) {
        applyLightPropertiesToActor(actor, viewID);
    }

    // Trigger render update
    vtkRenderWindow* win = getRenderWindow();
    if (win) {
        win->Render();
    }
}

double VtkVis::getObjectLightIntensity(const std::string& viewID) const {
    auto it = m_objectLightIntensity.find(viewID);
    if (it != m_objectLightIntensity.end()) {
        return it->second;
    }
    return m_lightIntensity;  // Fall back to global default
}

void VtkVis::setLightIntensity(double intensity) {
    // Clamp intensity to valid range (0.0-1.0), matching ParaView
    m_lightIntensity = std::max(0.0, std::min(1.0, intensity));

    // Get the renderer
    vtkRenderer* renderer = getCurrentRenderer();
    if (!renderer) {
        CVLog::Warning("[VtkVis] No renderer available for light control");
        return;
    }

    // IMPORTANT: Do NOT call renderer->RemoveAllLights() here!
    // VTK's vtkRenderer stores an internal CreatedLight raw pointer. Calling
    // RemoveAllLights() destroys that light but leaves the dangling pointer,
    // which causes a segfault on the next Render() cycle.

    // Disable automatic light creation so VTK won't interfere
    renderer->AutomaticLightCreationOff();

    // Find or create the headlight
    vtkLightCollection* lights = renderer->GetLights();
    vtkLight* headlight = nullptr;
    if (lights) {
        lights->InitTraversal();
        vtkLight* light = nullptr;
        while ((light = lights->GetNextItem())) {
            if (light->GetLightType() == VTK_LIGHT_TYPE_HEADLIGHT) {
                headlight = light;
                break;
            }
        }
    }

    if (!headlight) {
        vtkSmartPointer<vtkLight> newLight = vtkSmartPointer<vtkLight>::New();
        newLight->SetLightTypeToHeadlight();
        newLight->SetColor(1.0, 1.0, 1.0);
        newLight->SwitchOn();
        renderer->AddLight(newLight);
        headlight = newLight;
    }

    // Update headlight intensity
    headlight->SetIntensity(m_lightIntensity);
    renderer->LightFollowCameraOn();

    // Trigger render update
    vtkRenderWindow* win = getRenderWindow();
    if (win) {
        win->Render();
    }
}

double VtkVis::getLightIntensity() const { return m_lightIntensity; }

// ============================================================================
// Axes Grid and Camera Orientation Widget (ParaView-compatible)
// ============================================================================

void VtkVis::SetDataAxesGridProperties(const std::string& viewID,
                                       const AxesGridProperties& props) {
    vtkRenderer* renderer = getCurrentRenderer();
    if (!renderer) {
        CVLog::Warning("[VtkVis] No renderer available for Data Axes Grid");
        return;
    }

    // Get or create Data Axes Grid for this viewID
    vtkSmartPointer<vtkCubeAxesActor>& dataAxesGrid = m_dataAxesGridMap[viewID];

    if (!dataAxesGrid) {
        dataAxesGrid = vtkSmartPointer<vtkCubeAxesActor>::New();

        // Set camera reference
        dataAxesGrid->SetCamera(renderer->GetActiveCamera());

        // ============ ParaView Default Configuration ============

        // Axes visibility
        dataAxesGrid->SetXAxisVisibility(1);
        dataAxesGrid->SetYAxisVisibility(1);
        dataAxesGrid->SetZAxisVisibility(1);

        // Tick marks visibility (ParaView default: true)
        dataAxesGrid->SetXAxisTickVisibility(1);
        dataAxesGrid->SetYAxisTickVisibility(1);
        dataAxesGrid->SetZAxisTickVisibility(1);

        // Minor tick marks visibility (ParaView default: true)
        dataAxesGrid->SetXAxisMinorTickVisibility(1);
        dataAxesGrid->SetYAxisMinorTickVisibility(1);
        dataAxesGrid->SetZAxisMinorTickVisibility(1);

        // Fly mode (ParaView uses outer edges for data axes)
        dataAxesGrid->SetFlyModeToOuterEdges();

        // Text properties (ParaView defaults: Arial, white color, Title: 18pt,
        // Label: 14pt)
        for (int i = 0; i < 3; i++) {
            vtkTextProperty* titleProp = dataAxesGrid->GetTitleTextProperty(i);
            if (titleProp) {
                titleProp->SetColor(1.0, 1.0, 1.0);  // White
                titleProp->SetFontFamilyToArial();
                titleProp->SetFontSize(18);
                titleProp->SetBold(0);
                titleProp->SetItalic(0);
            }

            vtkTextProperty* labelProp = dataAxesGrid->GetLabelTextProperty(i);
            if (labelProp) {
                labelProp->SetColor(1.0, 1.0, 1.0);  // White
                labelProp->SetFontFamilyToArial();
                labelProp->SetFontSize(14);
                titleProp->SetBold(0);
                titleProp->SetItalic(0);
            }
        }

        // Label format (ParaView default: "{:<#6.3g}")
        dataAxesGrid->SetXLabelFormat("{:<#6.3g}");
        dataAxesGrid->SetYLabelFormat("{:<#6.3g}");
        dataAxesGrid->SetZLabelFormat("{:<#6.3g}");

        // Add to renderer
        renderer->AddActor(dataAxesGrid);
    }

    // ============ Apply All Properties from struct ============

    // 1. Visibility
    dataAxesGrid->SetVisibility(props.visible ? 1 : 0);

    // 2. Axis Titles (Qt QString → std::string)
    dataAxesGrid->SetXTitle(props.xTitle.toStdString().c_str());
    dataAxesGrid->SetYTitle(props.yTitle.toStdString().c_str());
    dataAxesGrid->SetZTitle(props.zTitle.toStdString().c_str());

    // 3. Colors for all axes (CCVector3 [0-255] → double [0.0-1.0])
    double colorR = props.color.x / 255.0;
    double colorG = props.color.y / 255.0;
    double colorB = props.color.z / 255.0;
    dataAxesGrid->GetXAxesLinesProperty()->SetColor(colorR, colorG, colorB);
    dataAxesGrid->GetYAxesLinesProperty()->SetColor(colorR, colorG, colorB);
    dataAxesGrid->GetZAxesLinesProperty()->SetColor(colorR, colorG, colorB);

    // 4. Line width
    dataAxesGrid->GetXAxesLinesProperty()->SetLineWidth(props.lineWidth);
    dataAxesGrid->GetYAxesLinesProperty()->SetLineWidth(props.lineWidth);
    dataAxesGrid->GetZAxesLinesProperty()->SetLineWidth(props.lineWidth);

    // 5. Opacity
    dataAxesGrid->GetXAxesLinesProperty()->SetOpacity(props.opacity);
    dataAxesGrid->GetYAxesLinesProperty()->SetOpacity(props.opacity);
    dataAxesGrid->GetZAxesLinesProperty()->SetOpacity(props.opacity);

    // 6. Labels visibility
    if (props.showLabels) {
        dataAxesGrid->XAxisLabelVisibilityOn();
        dataAxesGrid->YAxisLabelVisibilityOn();
        dataAxesGrid->ZAxisLabelVisibilityOn();
    } else {
        dataAxesGrid->XAxisLabelVisibilityOff();
        dataAxesGrid->YAxisLabelVisibilityOff();
        dataAxesGrid->ZAxisLabelVisibilityOff();
    }

    // 7. Grid lines visibility (ParaView-style)
    if (props.showGrid) {
        dataAxesGrid->DrawXGridlinesOn();
        dataAxesGrid->DrawYGridlinesOn();
        dataAxesGrid->DrawZGridlinesOn();
    } else {
        dataAxesGrid->DrawXGridlinesOff();
        dataAxesGrid->DrawYGridlinesOff();
        dataAxesGrid->DrawZGridlinesOff();
        dataAxesGrid->DrawXInnerGridlinesOff();
        dataAxesGrid->DrawYInnerGridlinesOff();
        dataAxesGrid->DrawZInnerGridlinesOff();
    }

    // 8. Custom axis labels (ParaView-style, QList<QPair<double, QString>> →
    // vtkStringArray)
    if (props.xUseCustomLabels && !props.xCustomLabels.isEmpty()) {
        vtkSmartPointer<vtkStringArray> xLabelsArray =
                vtkSmartPointer<vtkStringArray>::New();
        for (const auto& label : props.xCustomLabels) {
            xLabelsArray->InsertNextValue(label.second.toStdString().c_str());
        }
        dataAxesGrid->SetAxisLabels(0, xLabelsArray);  // 0 = X axis
    } else {
        dataAxesGrid->SetAxisLabels(
                0, nullptr);  // Use default auto-generated labels
    }

    if (props.yUseCustomLabels && !props.yCustomLabels.isEmpty()) {
        vtkSmartPointer<vtkStringArray> yLabelsArray =
                vtkSmartPointer<vtkStringArray>::New();
        for (const auto& label : props.yCustomLabels) {
            yLabelsArray->InsertNextValue(label.second.toStdString().c_str());
        }
        dataAxesGrid->SetAxisLabels(1, yLabelsArray);  // 1 = Y axis
    } else {
        dataAxesGrid->SetAxisLabels(1, nullptr);
    }

    if (props.zUseCustomLabels && !props.zCustomLabels.isEmpty()) {
        vtkSmartPointer<vtkStringArray> zLabelsArray =
                vtkSmartPointer<vtkStringArray>::New();
        for (const auto& label : props.zCustomLabels) {
            zLabelsArray->InsertNextValue(label.second.toStdString().c_str());
        }
        dataAxesGrid->SetAxisLabels(2, zLabelsArray);  // 2 = Z axis
    } else {
        dataAxesGrid->SetAxisLabels(2, nullptr);
    }

    // 9. Bounds (custom or from actor or from ccHObject)
    if (props.useCustomBounds) {
        dataAxesGrid->SetBounds(props.xMin, props.xMax, props.yMin, props.yMax,
                                props.zMin, props.zMax);
    } else {
        // Get bounds from the specific actor associated with this viewID
        vtkActor* actor = getActorById(viewID);
        if (actor) {
            double bounds[6];
            actor->GetBounds(bounds);
            if (bounds[1] > bounds[0] && bounds[3] > bounds[2] &&
                bounds[5] > bounds[4]) {
                dataAxesGrid->SetBounds(bounds);
            } else {
                CVLog::Warning(
                        "[VtkVis] Invalid bounds for Data Axes Grid, axes may "
                        "not display correctly");
            }
        } else {
            // If no actor found, try to get bounds from ccHObject
            // This is especially useful for parent nodes/folders that contain
            // multiple children
            ccHObject* obj = getSourceObject(viewID);

            // If not in source object map, try to find it in the scene DB
            // This handles parent nodes/folders that aren't registered in
            // m_sourceObjectMap
            if (!obj) {
                ccHObject* sceneRoot = ecvDisplayTools::GetSceneDB();
                if (sceneRoot) {
                    QString viewIDStr = QString::fromStdString(viewID);
                    // Recursively search for object with matching viewID
                    std::function<ccHObject*(ccHObject*)> findByViewID =
                            [&findByViewID,
                             &viewIDStr](ccHObject* node) -> ccHObject* {
                        if (!node) return nullptr;
                        if (node->getViewId() == viewIDStr) {
                            return node;
                        }
                        for (unsigned i = 0; i < node->getChildrenNumber();
                             ++i) {
                            ccHObject* found = findByViewID(node->getChild(i));
                            if (found) return found;
                        }
                        return nullptr;
                    };
                    obj = findByViewID(sceneRoot);
                }
            }

            if (obj) {
                // Calculate overall bbox including all children recursively
                ccBBox overallBBox = obj->getDisplayBB_recursive(false);
                if (overallBBox.isValid()) {
                    CCVector3 minCorner = overallBBox.minCorner();
                    CCVector3 maxCorner = overallBBox.maxCorner();
                    double bounds[6] = {minCorner.x, maxCorner.x, minCorner.y,
                                        maxCorner.y, minCorner.z, maxCorner.z};
                    if (bounds[1] > bounds[0] && bounds[3] > bounds[2] &&
                        bounds[5] > bounds[4]) {
                        dataAxesGrid->SetBounds(bounds);
                        CVLog::PrintVerbose(
                                QString("[VtkVis] Set axes grid bounds from "
                                        "ccHObject '%1' (viewID: %2): "
                                        "[%.2f, %.2f] x [%.2f, %.2f] x [%.2f, "
                                        "%.2f]")
                                        .arg(obj->getName())
                                        .arg(QString::fromStdString(viewID))
                                        .arg(bounds[0])
                                        .arg(bounds[1])
                                        .arg(bounds[2])
                                        .arg(bounds[3])
                                        .arg(bounds[4])
                                        .arg(bounds[5]));
                    } else {
                        CVLog::Warning(
                                "[VtkVis] Invalid bounds calculated from "
                                "ccHObject for Data Axes Grid");
                    }
                } else {
                    CVLog::Warning(
                            QString("[VtkVis] Invalid bbox for viewID: %1, "
                                    "axes grid bounds not set")
                                    .arg(QString::fromStdString(viewID)));
                }
            } else {
                CVLog::Warning(QString("[VtkVis] No actor or source object "
                                       "found for viewID: %1, axes grid bounds "
                                       "not set")
                                       .arg(QString::fromStdString(viewID)));
            }
        }
    }

    // Trigger update
    vtkRenderWindow* win = getRenderWindow();
    if (win) {
        win->Render();
    }
}

void VtkVis::GetDataAxesGridProperties(const std::string& viewID,
                                       AxesGridProperties& props) const {
    auto it = m_dataAxesGridMap.find(viewID);
    if (it == m_dataAxesGridMap.end() || !it->second) {
        // Return default values
        props = AxesGridProperties();
        return;
    }

    const vtkSmartPointer<vtkCubeAxesActor>& dataAxesGrid = it->second;

    // Get all properties from VTK actor and convert to Qt types
    props.visible = (dataAxesGrid->GetVisibility() != 0);

    // Color: double [0.0-1.0] → CCVector3 [0-255]
    double vtkColor[3];
    dataAxesGrid->GetXAxesLinesProperty()->GetColor(vtkColor);
    props.color = CCVector3(static_cast<float>(vtkColor[0] * 255.0),
                            static_cast<float>(vtkColor[1] * 255.0),
                            static_cast<float>(vtkColor[2] * 255.0));

    props.lineWidth = dataAxesGrid->GetXAxesLinesProperty()->GetLineWidth();
    props.spacing =
            1.0;  // Conceptual - not directly supported by vtkCubeAxesActor
    props.subdivisions = 10;  // Not directly supported by vtkCubeAxesActor
    props.showLabels = (dataAxesGrid->GetXAxisLabelVisibility() != 0);
    props.opacity = dataAxesGrid->GetXAxesLinesProperty()->GetOpacity();
    props.showGrid = (dataAxesGrid->GetDrawXGridlines() != 0);

    // Titles: const char* → QString
    props.xTitle = QString::fromUtf8(dataAxesGrid->GetXTitle());
    props.yTitle = QString::fromUtf8(dataAxesGrid->GetYTitle());
    props.zTitle = QString::fromUtf8(dataAxesGrid->GetZTitle());

    // Custom labels: We can't easily retrieve them from vtkCubeAxesActor,
    // so we just check if custom labels are being used
    props.xUseCustomLabels = (dataAxesGrid->GetAxisLabels(0) != nullptr);
    props.yUseCustomLabels = (dataAxesGrid->GetAxisLabels(1) != nullptr);
    props.zUseCustomLabels = (dataAxesGrid->GetAxisLabels(2) != nullptr);

    // Custom bounds: We can get the bounds but can't determine if they were
    // custom or from actor
    double bounds[6];
    dataAxesGrid->GetBounds(bounds);
    props.xMin = bounds[0];
    props.xMax = bounds[1];
    props.yMin = bounds[2];
    props.yMax = bounds[3];
    props.zMin = bounds[4];
    props.zMax = bounds[5];
    props.useCustomBounds = false;  // Can't determine from VTK, assume false

    // Note: Custom label values (xCustomLabels, yCustomLabels, zCustomLabels)
    // cannot be retrieved from vtkCubeAxesActor API, they remain empty in the
    // returned QList
}

// ============================================================================
// Camera Orientation Widget (ParaView-compatible)
// ============================================================================

void VtkVis::ToggleCameraOrientationWidget(bool show) {
    vtkRenderer* renderer = getCurrentRenderer();
    vtkRenderWindowInteractor* interactor = getRenderWindowInteractor();

    if (!renderer || !interactor) {
        CVLog::Warning(
                "[VtkVis] No renderer or interactor available for Camera "
                "Orientation Widget");
        return;
    }

    // Create Camera Orientation Widget if it doesn't exist (ParaView-style)
    if (!m_cameraOrientationWidget) {
        m_cameraOrientationWidget =
                vtkSmartPointer<vtkCameraOrientationWidget>::New();

        // Set parent renderer (the main 3D view renderer)
        m_cameraOrientationWidget->SetParentRenderer(renderer);

        // Set interactor for event handling
        m_cameraOrientationWidget->SetInteractor(interactor);

        // ParaView disables animation when using QVTKOpenGLWidget
        m_cameraOrientationWidget->SetAnimate(false);

        // Create default representation if not already created
        m_cameraOrientationWidget->CreateDefaultRepresentation();

        // Configure the default renderer (the widget's own renderer)
        vtkRenderer* widgetRenderer =
                m_cameraOrientationWidget->GetDefaultRenderer();
        if (widgetRenderer) {
            // ParaView settings: right upper corner, 20% size
            widgetRenderer->SetViewport(0.8, 0.8, 1.0, 1.0);
            widgetRenderer->SetLayer(1);  // Render on top
            widgetRenderer->InteractiveOff();
        }

        // Configure representation
        auto* rep = vtkCameraOrientationRepresentation::SafeDownCast(
                m_cameraOrientationWidget->GetRepresentation());
        if (rep) {
            // ParaView default size
            rep->SetSize(80, 80);

            // ParaView default position: upper right
            rep->AnchorToUpperRight();

            // Set padding
            int padding[2] = {10, 10};
            rep->SetPadding(padding);

            // Square resize to maintain aspect ratio
            m_cameraOrientationWidget->SquareResize();
        }

        CVLog::PrintVerbose("[VtkVis] Camera Orientation Widget created");
    }

    // Update visibility and enabled state (ParaView behavior)
    auto* rep = m_cameraOrientationWidget->GetRepresentation();
    if (rep) {
        rep->SetVisibility(show);

        // ParaView: if we have interactor, also update enabled state
        if (interactor) {
            m_cameraOrientationWidget->SetEnabled(show ? 1 : 0);
        }
    }

    // Trigger update
    vtkRenderWindow* win = getRenderWindow();
    if (win) {
        win->Render();
    }

    CVLog::PrintDebug(QString("[VtkVis] Camera Orientation Widget: %1")
                              .arg(show ? "ON" : "OFF"));
}

bool VtkVis::IsCameraOrientationWidgetShown() const {
    if (!m_cameraOrientationWidget) {
        return false;
    }

    auto* rep = m_cameraOrientationWidget->GetRepresentation();
    return rep ? (rep->GetVisibility() != 0) : false;
}

void VtkVis::RemoveDataAxesGrid(const std::string& viewID) {
    auto it = m_dataAxesGridMap.find(viewID);
    if (it == m_dataAxesGridMap.end()) {
        return;  // No Data Axes Grid for this viewID
    }

    vtkRenderer* renderer = getCurrentRenderer();
    if (renderer && it->second) {
        renderer->RemoveActor(it->second);
    }

    m_dataAxesGridMap.erase(it);
}

}  // namespace Visualization
