// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// LOCAL
#include "vtkCustomInteractorStyle.h"

#include <FileSystem.h>

#include "vtkCameraManipulator.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <vtkAbstractPicker.h>
#include <vtkAbstractPropPicker.h>
#include <vtkActorCollection.h>
#include <vtkAreaPicker.h>
#include <vtkAssemblyPath.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCollection.h>
#include <vtkCollectionIterator.h>
#include <vtkDataSet.h>
#include <vtkExtractGeometry.h>
#include <vtkIdTypeArray.h>
#include <vtkLODActor.h>
#include <vtkLegendScaleActor.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkMapper.h>
#include <vtkObjectFactory.h>
#include <vtkPNGWriter.h>
#include <vtkPlanes.h>
#include <vtkPointData.h>
#include <vtkPointPicker.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProp3DCollection.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkScalarBarActor.h>
#include <vtkTextProperty.h>
#include <vtkVersion.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkWindowToImageFilter.h>

#include <fstream>
#include <list>
#include <sstream>

// For string splitting (replaces boost::split dependency from PCL)
static std::vector<std::string> splitString(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        if (!item.empty()) result.push_back(item);
    }
    return result;
}

#define ORIENT_MODE 0
#define SELECT_MODE 1

#define VTKISRBP_ORIENT 0
#define VTKISRBP_SELECT 1

namespace VTKExtensions {

// ============================================================================
// CVPointPickingCallback implementation
// ============================================================================
void CVPointPickingCallback::Execute(vtkObject* caller,
                                     unsigned long eventid,
                                     void*) {
    auto* style = reinterpret_cast<vtkCustomInteractorStyle*>(caller);
    vtkRenderWindowInteractor* iren = style->GetInteractor();

    if (style->CurrentMode == 0) {
        if ((eventid == vtkCommand::LeftButtonPressEvent) &&
            (iren->GetShiftKey() > 0)) {
            float x = 0, y = 0, z = 0;
            const auto idx = performSinglePick(iren, x, y, z);
            if (idx != -1) {
                PclUtils::CloudActorMapPtr cam_ptr = style->getCloudActorMap();
                std::string name;
                if (cam_ptr) {
                    for (const auto& ca : *cam_ptr) {
                        if (ca.second.actor.GetPointer() == actor_) {
                            name = ca.first;
                            break;
                        }
                    }
                }
                style->point_picking_signal_(
                        PclUtils::PointPickingEvent(idx, x, y, z, name));
            }
        } else if ((eventid == vtkCommand::LeftButtonPressEvent) &&
                   (iren->GetAltKey() == 1)) {
            pick_first_ = !pick_first_;
            float x = 0, y = 0, z = 0;
            int idx = -1;
            if (pick_first_)
                idx_ = performSinglePick(iren, x_, y_, z_);
            else
                idx = performSinglePick(iren, x, y, z);
            PclUtils::PointPickingEvent event(idx_, idx, x_, y_, z_, x, y, z);
            style->point_picking_signal_(event);
        }
        // Call the parent's class mouse events
        if (eventid == vtkCommand::LeftButtonPressEvent)
            style->OnLeftButtonDown();
        else if (eventid == vtkCommand::LeftButtonReleaseEvent)
            style->OnLeftButtonUp();
    } else {
        if (eventid == vtkCommand::LeftButtonPressEvent) {
            style->OnLeftButtonDown();
            x_ = static_cast<float>(iren->GetEventPosition()[0]);
            y_ = static_cast<float>(iren->GetEventPosition()[1]);
        } else if (eventid == vtkCommand::LeftButtonReleaseEvent) {
            style->OnLeftButtonUp();
            std::map<std::string, std::vector<int>> cloud_indices;
            performAreaPick(iren, style->getCloudActorMap(), cloud_indices);
            style->area_picking_signal_(
                    PclUtils::AreaPickingEvent(std::move(cloud_indices)));
        }
    }
}

int CVPointPickingCallback::performSinglePick(
        vtkRenderWindowInteractor* iren) {
    vtkPointPicker* point_picker =
            vtkPointPicker::SafeDownCast(iren->GetPicker());

    if (!point_picker) {
        CVLog::Error(
                "Point picker not available, not selecting any points!");
        return -1;
    }

    int mouse_x = iren->GetEventPosition()[0];
    int mouse_y = iren->GetEventPosition()[1];

    iren->StartPickCallback();
    vtkRenderer* ren = iren->FindPokedRenderer(mouse_x, mouse_y);
    point_picker->Pick(mouse_x, mouse_y, 0.0, ren);

    return static_cast<int>(point_picker->GetPointId());
}

int CVPointPickingCallback::performSinglePick(
        vtkRenderWindowInteractor* iren, float& x, float& y, float& z) {
    vtkPointPicker* point_picker =
            vtkPointPicker::SafeDownCast(iren->GetPicker());

    if (!point_picker) {
        CVLog::Error(
                "Point picker not available, not selecting any points!");
        return -1;
    }

    int mouse_x = iren->GetEventPosition()[0];
    int mouse_y = iren->GetEventPosition()[1];

    iren->StartPickCallback();
    vtkRenderer* ren = iren->FindPokedRenderer(mouse_x, mouse_y);
    point_picker->Pick(mouse_x, mouse_y, 0.0, ren);

    auto idx = static_cast<int>(point_picker->GetPointId());
    if (point_picker->GetDataSet()) {
        double p[3];
        point_picker->GetDataSet()->GetPoint(idx, p);
        x = static_cast<float>(p[0]);
        y = static_cast<float>(p[1]);
        z = static_cast<float>(p[2]);
        actor_ = point_picker->GetActor();
    }

    return idx;
}

int CVPointPickingCallback::performAreaPick(
        vtkRenderWindowInteractor* iren,
        PclUtils::CloudActorMapPtr cam_ptr,
        std::map<std::string, std::vector<int>>& cloud_indices) const {
    auto* picker = dynamic_cast<vtkAreaPicker*>(iren->GetPicker());
    vtkRenderer* ren = iren->FindPokedRenderer(iren->GetEventPosition()[0],
                                               iren->GetEventPosition()[1]);
    picker->AreaPick(x_, y_, iren->GetEventPosition()[0],
                     iren->GetEventPosition()[1], ren);

    vtkProp3DCollection* props = picker->GetProp3Ds();
    if (!props) return -1;

    int pt_numb = 0;
    vtkCollectionSimpleIterator pit;
    vtkProp3D* prop;
    for (props->InitTraversal(pit); (prop = props->GetNextProp3D(pit));) {
        vtkSmartPointer<vtkActor> actor = vtkActor::SafeDownCast(prop);
        if (!actor) continue;

        vtkPolyData* pd =
                vtkPolyData::SafeDownCast(actor->GetMapper()->GetInput());
        if (pd->GetPointData()->HasArray("Indices"))
            pd->GetPointData()->RemoveArray("Indices");

        vtkSmartPointer<vtkIdTypeArray> ids =
                vtkSmartPointer<vtkIdTypeArray>::New();
        ids->SetNumberOfComponents(1);
        ids->SetName("Indices");
        for (vtkIdType i = 0; i < pd->GetNumberOfPoints(); i++)
            ids->InsertNextValue(i);
        pd->GetPointData()->AddArray(ids);

        vtkSmartPointer<vtkExtractGeometry> extract_geometry =
                vtkSmartPointer<vtkExtractGeometry>::New();
        extract_geometry->SetImplicitFunction(picker->GetFrustum());
        extract_geometry->SetInputData(pd);
        extract_geometry->Update();

        vtkSmartPointer<vtkVertexGlyphFilter> glyph_filter =
                vtkSmartPointer<vtkVertexGlyphFilter>::New();
        glyph_filter->SetInputConnection(
                extract_geometry->GetOutputPort());
        glyph_filter->Update();

        vtkPolyData* selected = glyph_filter->GetOutput();
        vtkIdTypeArray* global_ids = vtkIdTypeArray::SafeDownCast(
                selected->GetPointData()->GetArray("Indices"));

        if (!global_ids->GetSize() || !selected->GetNumberOfPoints())
            continue;

        std::vector<int> actor_indices;
        actor_indices.reserve(selected->GetNumberOfPoints());
        for (vtkIdType i = 0; i < selected->GetNumberOfPoints(); i++)
            actor_indices.push_back(
                    static_cast<int>(global_ids->GetValue(i)));

        pt_numb += selected->GetNumberOfPoints();

        std::string name;
        if (cam_ptr) {
            for (const auto& ca : *cam_ptr) {
                if (ca.second.actor == actor) {
                    name = ca.first;
                    break;
                }
            }
        }
        cloud_indices.emplace(name, std::move(actor_indices));
    }
    return pt_numb;
}

// ============================================================================
// vtkCustomInteractorStyle implementation
// ============================================================================

vtkCustomInteractorStyle::vtkCustomInteractorStyle()
    : CameraManipulators(vtkCollection::New()),
      CurrentManipulator(nullptr),
      RotationFactor(1.0),
      lut_actor_id_("") {
    this->CenterOfRotation[0] = this->CenterOfRotation[1] =
            this->CenterOfRotation[2] = 0;
}

//-------------------------------------------------------------------------
vtkCustomInteractorStyle::~vtkCustomInteractorStyle() {
    this->CameraManipulators->Delete();
    this->CameraManipulators = nullptr;
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::Initialize() {
    modifier_ = PclUtils::INTERACTOR_KB_MOD_ALT;
    // Set windows size (width, height) to unknown (-1)
    win_height_ = win_width_ = -1;
    win_pos_x_ = win_pos_y_ = 0;
    max_win_height_ = max_win_width_ = -1;

    // Grid is disabled by default
    grid_enabled_ = false;
    grid_actor_ = vtkSmartPointer<vtkLegendScaleActor>::New();

    // LUT is disabled by default
    lut_enabled_ = false;
    lut_actor_ = vtkSmartPointer<vtkScalarBarActor>::New();
    lut_actor_->SetTitle("");
    lut_actor_->SetOrientationToHorizontal();
    lut_actor_->SetPosition(0.05, 0.01);
    lut_actor_->SetWidth(0.9);
    lut_actor_->SetHeight(0.1);
    lut_actor_->SetNumberOfLabels(lut_actor_->GetNumberOfLabels() * 2);
    vtkSmartPointer<vtkTextProperty> prop =
            lut_actor_->GetLabelTextProperty();
    prop->SetFontSize(10);
    lut_actor_->SetLabelTextProperty(prop);
    lut_actor_->SetTitleTextProperty(prop);

    // Create the image filter and PNG writer objects
    wif_ = vtkSmartPointer<vtkWindowToImageFilter>::New();
    wif_->ReadFrontBufferOff();
    snapshot_writer_ = vtkSmartPointer<vtkPNGWriter>::New();
    snapshot_writer_->SetInputConnection(wif_->GetOutputPort());

    init_ = true;

    stereo_anaglyph_mask_default_ = true;

    // Start in orient mode
    Superclass::CurrentMode = ORIENT_MODE;

    // Add our own mouse callback before any user callback. Used for accurate
    // point picking.
    mouse_callback_ = vtkSmartPointer<CVPointPickingCallback>::New();
    AddObserver(vtkCommand::LeftButtonPressEvent, mouse_callback_);
    AddObserver(vtkCommand::LeftButtonReleaseEvent, mouse_callback_);
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::RemoveAllManipulators() {
    this->CameraManipulators->RemoveAllItems();
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::AddManipulator(vtkCameraManipulator* m) {
    this->CameraManipulators->AddItem(m);
}

// ======== Event callback registration ========

PclUtils::SignalConnection vtkCustomInteractorStyle::registerMouseCallback(
        std::function<void(const PclUtils::MouseEvent&)> cb) {
    return mouse_signal_.connect(std::move(cb));
}

PclUtils::SignalConnection
vtkCustomInteractorStyle::registerKeyboardCallback(
        std::function<void(const PclUtils::KeyboardEvent&)> cb) {
    return keyboard_signal_.connect(std::move(cb));
}

PclUtils::SignalConnection
vtkCustomInteractorStyle::registerPointPickingCallback(
        std::function<void(const PclUtils::PointPickingEvent&)> cb) {
    return point_picking_signal_.connect(std::move(cb));
}

PclUtils::SignalConnection
vtkCustomInteractorStyle::registerAreaPickingCallback(
        std::function<void(const PclUtils::AreaPickingEvent&)> cb) {
    return area_picking_signal_.connect(std::move(cb));
}

// ======== Screenshot & Camera ========

void vtkCustomInteractorStyle::saveScreenshot(const std::string& file) {
    FindPokedRenderer(Interactor->GetEventPosition()[0],
                      Interactor->GetEventPosition()[1]);
    wif_->SetInput(Interactor->GetRenderWindow());
    wif_->Modified();
    snapshot_writer_->Modified();
    snapshot_writer_->SetFileName(file.c_str());
    snapshot_writer_->Write();
}

bool vtkCustomInteractorStyle::saveCameraParameters(
        const std::string& file) {
    FindPokedRenderer(Interactor->GetEventPosition()[0],
                      Interactor->GetEventPosition()[1]);

    std::ofstream ofs_cam(file.c_str());
    if (!ofs_cam.is_open()) {
        return false;
    }

    vtkSmartPointer<vtkCamera> cam = Interactor->GetRenderWindow()
                                             ->GetRenderers()
                                             ->GetFirstRenderer()
                                             ->GetActiveCamera();
    double clip[2], focal[3], pos[3], view[3];
    cam->GetClippingRange(clip);
    cam->GetFocalPoint(focal);
    cam->GetPosition(pos);
    cam->GetViewUp(view);
    int* win_pos = Interactor->GetRenderWindow()->GetPosition();
    int* win_size = Interactor->GetRenderWindow()->GetSize();
    ofs_cam << clip[0] << "," << clip[1] << "/" << focal[0] << "," << focal[1]
            << "," << focal[2] << "/" << pos[0] << "," << pos[1] << ","
            << pos[2] << "/" << view[0] << "," << view[1] << "," << view[2]
            << "/" << cam->GetViewAngle() / 180.0 * M_PI << "/" << win_size[0]
            << "," << win_size[1] << "/" << win_pos[0] << "," << win_pos[1]
            << std::endl;
    ofs_cam.close();

    return true;
}

void vtkCustomInteractorStyle::getCameraParameters(PclUtils::Camera& camera,
                                                   int viewport) const {
    rens_->InitTraversal();
    vtkRenderer* renderer = nullptr;
    int i = 0;
    while ((renderer = rens_->GetNextItem())) {
        if (viewport++ == i) {
            auto window = Interactor->GetRenderWindow();
            auto cam = renderer->GetActiveCamera();
            // Fill PclUtils::Camera from vtkCamera + vtkRenderWindow
            cam->GetFocalPoint(camera.focal);
            cam->GetPosition(camera.pos);
            cam->GetViewUp(camera.view);
            cam->GetClippingRange(camera.clip);
            camera.fovy = cam->GetViewAngle() * M_PI / 180.0;
            int* ws = window->GetSize();
            camera.window_size[0] = ws[0];
            camera.window_size[1] = ws[1];
            int* wp = window->GetPosition();
            camera.window_pos[0] = wp[0];
            camera.window_pos[1] = wp[1];
            break;
        }
    }
}

bool vtkCustomInteractorStyle::loadCameraParameters(
        const std::string& file) {
    std::ifstream fs;
    std::string line;
    std::vector<std::string> camera;
    bool ret;

    fs.open(file.c_str());
    if (!fs.is_open()) {
        return false;
    }
    while (!fs.eof()) {
        getline(fs, line);
        if (line.empty()) continue;

        camera = splitString(line, '/');
        break;
    }
    fs.close();

    ret = getCameraParameters(camera);
    if (ret) {
        camera_file_ = file;
    }

    return ret;
}

bool vtkCustomInteractorStyle::getCameraParameters(
        const std::vector<std::string>& camera) {
    PclUtils::Camera camera_temp;

    // look for '/' as a separator
    if (camera.size() != 7) {
        CVLog::Error(
                "[getCameraParameters] Camera parameters given, but with an "
                "invalid number of options (%lu vs 7)!",
                static_cast<unsigned long>(camera.size()));
        return false;
    }

    std::string clip_str = camera.at(0);
    std::string focal_str = camera.at(1);
    std::string pos_str = camera.at(2);
    std::string view_str = camera.at(3);
    std::string fovy_str = camera.at(4);
    std::string win_size_str = camera.at(5);
    std::string win_pos_str = camera.at(6);

    auto clip_st = splitString(clip_str, ',');
    if (clip_st.size() != 2) {
        CVLog::Error(
                "[getCameraParameters] Invalid parameters given for camera "
                "clipping angle!");
        return false;
    }
    camera_temp.clip[0] = atof(clip_st.at(0).c_str());
    camera_temp.clip[1] = atof(clip_st.at(1).c_str());

    auto focal_st = splitString(focal_str, ',');
    if (focal_st.size() != 3) {
        CVLog::Error(
                "[getCameraParameters] Invalid parameters given for camera "
                "focal point!");
        return false;
    }
    camera_temp.focal[0] = atof(focal_st.at(0).c_str());
    camera_temp.focal[1] = atof(focal_st.at(1).c_str());
    camera_temp.focal[2] = atof(focal_st.at(2).c_str());

    auto pos_st = splitString(pos_str, ',');
    if (pos_st.size() != 3) {
        CVLog::Error(
                "[getCameraParameters] Invalid parameters given for camera "
                "position!");
        return false;
    }
    camera_temp.pos[0] = atof(pos_st.at(0).c_str());
    camera_temp.pos[1] = atof(pos_st.at(1).c_str());
    camera_temp.pos[2] = atof(pos_st.at(2).c_str());

    auto view_st = splitString(view_str, ',');
    if (view_st.size() != 3) {
        CVLog::Error(
                "[getCameraParameters] Invalid parameters given for camera "
                "viewup!");
        return false;
    }
    camera_temp.view[0] = atof(view_st.at(0).c_str());
    camera_temp.view[1] = atof(view_st.at(1).c_str());
    camera_temp.view[2] = atof(view_st.at(2).c_str());

    auto fovy_size_st = splitString(fovy_str, ',');
    if (fovy_size_st.size() != 1) {
        CVLog::Error(
                "[getCameraParameters] Invalid parameters given for field "
                "of view angle!");
        return false;
    }
    camera_temp.fovy = atof(fovy_size_st.at(0).c_str());

    auto win_size_st = splitString(win_size_str, ',');
    if (win_size_st.size() != 2) {
        CVLog::Error(
                "[getCameraParameters] Invalid parameters given for window "
                "size!");
        return false;
    }
    camera_temp.window_size[0] = static_cast<int>(atof(win_size_st.at(0).c_str()));
    camera_temp.window_size[1] = static_cast<int>(atof(win_size_st.at(1).c_str()));

    auto win_pos_st = splitString(win_pos_str, ',');
    if (win_pos_st.size() != 2) {
        CVLog::Error(
                "[getCameraParameters] Invalid parameters given for window "
                "position!");
        return false;
    }
    camera_temp.window_pos[0] = static_cast<int>(atof(win_pos_st.at(0).c_str()));
    camera_temp.window_pos[1] = static_cast<int>(atof(win_pos_st.at(1).c_str()));

    setCameraParameters(camera_temp);

    return true;
}

void vtkCustomInteractorStyle::setCameraParameters(
        const PclUtils::Camera& camera, int viewport) {
    rens_->InitTraversal();
    vtkRenderer* renderer = nullptr;
    int i = 0;
    while ((renderer = rens_->GetNextItem())) {
        // Modify all renderer's cameras
        if (viewport == 0 || viewport == i) {
            vtkSmartPointer<vtkCamera> cam = renderer->GetActiveCamera();
            cam->SetPosition(camera.pos[0], camera.pos[1], camera.pos[2]);
            cam->SetFocalPoint(camera.focal[0], camera.focal[1],
                               camera.focal[2]);
            cam->SetViewUp(camera.view[0], camera.view[1], camera.view[2]);
            cam->SetClippingRange(camera.clip);
            cam->SetUseHorizontalViewAngle(0);
            cam->SetViewAngle(camera.fovy * 180.0 / M_PI);

            win_->SetSize(static_cast<int>(camera.window_size[0]),
                          static_cast<int>(camera.window_size[1]));
        }
        ++i;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::zoomIn() {
    FindPokedRenderer(Interactor->GetEventPosition()[0],
                      Interactor->GetEventPosition()[1]);
    // Zoom in
    StartDolly();
    double factor = 10.0 * 0.2 * .5;
    Dolly(pow(1.1, factor));
    EndDolly();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::zoomOut() {
    FindPokedRenderer(Interactor->GetEventPosition()[0],
                      Interactor->GetEventPosition()[1]);
    // Zoom out
    StartDolly();
    double factor = 10.0 * -0.2 * .5;
    Dolly(pow(1.1, factor));
    EndDolly();
}

void vtkCustomInteractorStyle::toggleAreaPicking() {
    CurrentMode = (CurrentMode == ORIENT_MODE) ? SELECT_MODE : ORIENT_MODE;
    if (CurrentMode == SELECT_MODE) {
        // Save the point picker
        point_picker_ = static_cast<vtkPointPicker*>(Interactor->GetPicker());
        // Switch for an area picker
        vtkSmartPointer<vtkAreaPicker> area_picker =
                vtkSmartPointer<vtkAreaPicker>::New();
        Interactor->SetPicker(area_picker);
    } else {
        // Restore point picker
        Interactor->SetPicker(point_picker_);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnChar() {
    // Make sure we ignore the same events we handle in OnKeyDown to avoid
    // calling things twice
    FindPokedRenderer(Interactor->GetEventPosition()[0],
                      Interactor->GetEventPosition()[1]);
    if (Interactor->GetKeyCode() >= '0' && Interactor->GetKeyCode() <= '9')
        return;
    std::string key(Interactor->GetKeySym());
    if (key.find("XF86ZoomIn") != std::string::npos)
        zoomIn();
    else if (key.find("XF86ZoomOut") != std::string::npos)
        zoomOut();

    bool keymod = false;
    switch (modifier_) {
        case PclUtils::INTERACTOR_KB_MOD_ALT: {
            keymod = Interactor->GetAltKey();
            break;
        }
        case PclUtils::INTERACTOR_KB_MOD_CTRL: {
            keymod = Interactor->GetControlKey();
            break;
        }
        case PclUtils::INTERACTOR_KB_MOD_SHIFT: {
            keymod = Interactor->GetShiftKey();
            break;
        }
    }

    switch (Interactor->GetKeyCode()) {
            // All of the options below simply exit
        case 'a':
        case 'A':
        case 'h':
        case 'H':
        case 'l':
        case 'L':
        case 'p':
        case 'P':
        case 'j':
        case 'J':
        case 'c':
        case 'C':
        // Note: KEY_PLUS (43) and KEY_MINUS (45) are removed from exit list
        // to allow grow/shrink selection shortcuts to work
        case 'f':
        case 'F':
        case 'g':
        case 'G':
        case 'o':
        case 'O':
        case 'u':
        case 'U':
        case 'q':
        case 'Q':
        case 'x':
        case 'X':
        case 'r':
        case 'R': {
            break;
        }
        // S have special !ALT case
        case 's':
        case 'S': {
            if (!keymod) vtkInteractorStyleRubberBandPick::OnChar();
            break;
        }
        default: {
            vtkInteractorStyleRubberBandPick::OnChar();
            break;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnKeyDown() {
    if (!rens_) {
        CVLog::Error(
                "[vtkCustomInteractorStyle] No renderer collection given! Use "
                "SetRendererCollection () before continuing.");
        return;
    }

    // Look for a matching camera interactor.
    this->CameraManipulators->InitTraversal();
    vtkCameraManipulator* manipulator = NULL;
    while ((manipulator = (vtkCameraManipulator*)this->CameraManipulators
                                  ->GetNextItemAsObject())) {
        manipulator->OnKeyDown(this->Interactor);
    }

    FindPokedRenderer(Interactor->GetEventPosition()[0],
                      Interactor->GetEventPosition()[1]);

    if (wif_->GetInput() == NULL) {
        wif_->SetInput(Interactor->GetRenderWindow());
        wif_->Modified();
        snapshot_writer_->Modified();
    }

    // Save the initial windows width/height
    if (win_height_ == -1 || win_width_ == -1) {
        int* win_size = Interactor->GetRenderWindow()->GetSize();
        win_height_ = win_size[0];
        win_width_ = win_size[1];
    }

    // Get the status of special keys (Cltr+Alt+Shift)
    bool shift = Interactor->GetShiftKey();
    bool ctrl = Interactor->GetControlKey();
    bool alt = Interactor->GetAltKey();

    bool keymod = false;
    switch (modifier_) {
        case PclUtils::INTERACTOR_KB_MOD_ALT: {
            keymod = alt;
            break;
        }
        case PclUtils::INTERACTOR_KB_MOD_CTRL: {
            keymod = ctrl;
            break;
        }
        case PclUtils::INTERACTOR_KB_MOD_SHIFT: {
            keymod = shift;
            break;
        }
    }

    // ---[ Check the rest of the key codes

    // Save camera parameters
    if ((Interactor->GetKeySym()[0] == 'S' ||
         Interactor->GetKeySym()[0] == 's') &&
        ctrl && !alt && !shift) {
        if (camera_file_.empty()) {
            getCameraParameters(camera_);
            camera_saved_ = true;
            CVLog::Print(
                    "Camera parameters saved, you can press CTRL + R to "
                    "restore.");
        } else {
            if (saveCameraParameters(camera_file_)) {
                CVLog::Print(
                        "Save camera parameters to %s, you can press CTRL + R "
                        "to restore.",
                        camera_file_.c_str());
            } else {
                CVLog::Error(
                        "[vtkCustomInteractorStyle] Can't save camera "
                        "parameters to file: %s.",
                        camera_file_.c_str());
            }
        }
    }

    // Restore camera parameters
    if ((Interactor->GetKeySym()[0] == 'R' ||
         Interactor->GetKeySym()[0] == 'r') &&
        ctrl && !alt && !shift) {
        if (camera_file_.empty()) {
            if (camera_saved_) {
                setCameraParameters(camera_);
                CVLog::Print("Camera parameters restored.");
            } else {
                CVLog::Print("No camera parameters saved for restoring.");
            }
        } else {
            if (cloudViewer::utility::filesystem::FileExists(camera_file_)) {
                if (loadCameraParameters(camera_file_)) {
                    CVLog::Print("Restore camera parameters from %s.",
                                 camera_file_.c_str());
                } else {
                    CVLog::Error(
                            "Can't restore camera parameters from file: %s.",
                            camera_file_.c_str());
                }
            } else {
                CVLog::Print("No camera parameters saved in %s for restoring.",
                             camera_file_.c_str());
            }
        }
    }

    // Switch between point color/geometry handlers (0-9 keys)
    // NOTE: Since we use direct VTK rendering (no PCL handlers), geometry/color
    // handler switching via 0-9 keys is simplified. The cloud_actors_ map
    // uses PclUtils::CloudActorEntry which doesn't have PCL handlers.
    // We keep the key handling for compatibility but skip handler switching.
    if (Interactor->GetKeySym() && Interactor->GetKeySym()[0] >= '0' &&
        Interactor->GetKeySym()[0] <= '9') {
        // Direct VTK path: no geometry/color handler switching needed
        // The actors are directly populated with VTK data.
        // Just fire the keyboard event and return.
        PclUtils::KeyboardEvent event(
                true, Interactor->GetKeySym(), Interactor->GetKeyCode(),
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey());
        keyboard_signal_(event);
        Interactor->Render();
        return;
    }

    std::string key(Interactor->GetKeySym());
    if (key.find("XF86ZoomIn") != std::string::npos)
        zoomIn();
    else if (key.find("XF86ZoomOut") != std::string::npos)
        zoomOut();

    switch (Interactor->GetKeyCode()) {
        case 'h':
        case 'H': {
            CVLog::Print(
                    "| Help:"
                    "-------"
                    "          CTRL + SHIFT + p, P   : switch to a point-based "
                    "representation"
                    "          CTRL + SHIFT + w, W   : switch to a "
                    "wireframe-based "
                    "representation (where available)"
                    "          CTRL + SHIFT + s, S   : switch to a "
                    "surface-based "
                    "representation (where available)"
                    ""
                    "          CTRL + ALT + j, J   : take a .PNG snapshot of "
                    "the current "
                    "window view"
                    "          CTRL + ALT + c, C   : display current "
                    "camera/window "
                    "parameters"
                    "          f, F   : fly to point mode"
                    ""
                    "          e, E   : exit the interactor"
                    "          q, Q   : stop and call VTK's TerminateApp"
                    ""
                    "          CTRL + SHIFT + +/-   : increment/decrement "
                    "overall point size"
                    "          CTRL + ALT + +/-: zoom in/out "
                    ""
                    "          CTRL + ALT + g, G   : display scale grid "
                    "(on/off)"
                    "          CTRL + ALT + u, U   : display lookup table "
                    "(on/off)"
                    ""
                    "    CTRL + ALT + o, O         : switch between "
                    "perspective/parallel "
                    "projection (default = perspective)"
                    "    r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} "
                    "-> center_{x, y, z}]"
                    "    CTRL + s, S  : save camera parameters"
                    "    CTRL + r, R  : restore camera parameters"
                    ""
                    "    CTRL + ALT + s, S   : turn stereo mode on/off"
                    "    CTRL + ALT + f, F   : switch between maximized window "
                    "mode "
                    "and original size"
                    ""
                    "          l, L           : list all available geometric "
                    "and color handlers for the current actor map"
                    "    ALT + 0..9 [+ CTRL]  : switch between different "
                    "geometric handlers (where available)"
                    "          0..9 [+ CTRL]  : switch between different color "
                    "handlers (where available)"
                    ""
                    "    SHIFT + left click   : select a point (start with "
                    "-use_point_picking)"
                    ""
                    "          a, A   : toggle rubber band selection mode for "
                    "left mouse button");
            break;
        }

        // Get the list of available handlers
        case 'l':
        case 'L': {
            // Direct VTK path: no PCL handlers to list.
            // Log a simple message instead.
                    CVLog::Print(
                    "[vtkCustomInteractorStyle] Direct VTK rendering mode: "
                    "no PCL color/geometry handlers available.");
            break;
        }

        // Switch representation to points
        case 'p':
        case 'P': {
            if (shift && ctrl) {
                vtkSmartPointer<vtkActorCollection> ac =
                        CurrentRenderer->GetActors();
                vtkCollectionSimpleIterator ait;
                for (ac->InitTraversal(ait);
                     vtkActor* actor = ac->GetNextActor(ait);) {
                    for (actor->InitPathTraversal();
                         vtkAssemblyPath* path = actor->GetNextPath();) {
                        vtkSmartPointer<vtkActor> apart =
                                reinterpret_cast<vtkActor*>(
                                        path->GetLastNode()->GetViewProp());
                        apart->GetProperty()->SetRepresentationToPoints();
                    }
                }
            }
            break;
        }

        // Switch representation to wireframe (override default behavior)
        case 'w':
        case 'W': {
            if (shift && ctrl) {
                vtkSmartPointer<vtkActorCollection> ac =
                        CurrentRenderer->GetActors();
                vtkCollectionSimpleIterator ait;
                for (ac->InitTraversal(ait);
                     vtkActor* actor = ac->GetNextActor(ait);) {
                    for (actor->InitPathTraversal();
                         vtkAssemblyPath* path = actor->GetNextPath();) {
                        vtkSmartPointer<vtkActor> apart =
                                reinterpret_cast<vtkActor*>(
                                        path->GetLastNode()->GetViewProp());
                        apart->GetProperty()->SetRepresentationToWireframe();
                        apart->GetProperty()->SetLighting(false);
                    }
                }
            }
            break;
        }

        // Save a PNG snapshot with the current screen
        case 'j':
        case 'J': {
            if (alt && ctrl) {
                char cam_fn[80], snapshot_fn[80];
                unsigned t = static_cast<unsigned>(time(0));
                sprintf(snapshot_fn, "screenshot-%d.png", t);
                saveScreenshot(snapshot_fn);

                sprintf(cam_fn, "screenshot-%d.cam", t);
                saveCameraParameters(cam_fn);

                CVLog::Print(
                        "Screenshot (%s) and camera information (%s) "
                        "successfully "
                        "captured.",
                        snapshot_fn, cam_fn);
            }
            break;
        }
        // display current camera settings/parameters
        case 'c':
        case 'C': {
            if (alt && ctrl) {
                vtkSmartPointer<vtkCamera> cam = Interactor->GetRenderWindow()
                                                         ->GetRenderers()
                                                         ->GetFirstRenderer()
                                                         ->GetActiveCamera();
                double clip[2], focal[3], pos[3], view[3];
                cam->GetClippingRange(clip);
                cam->GetFocalPoint(focal);
                cam->GetPosition(pos);
                cam->GetViewUp(view);
                int* win_pos = Interactor->GetRenderWindow()->GetPosition();
                int* win_size = Interactor->GetRenderWindow()->GetSize();
                std::cerr << "Clipping plane [near,far] " << clip[0] << ", "
                          << clip[1] << endl
                          << "Focal point [x,y,z] " << focal[0] << ", "
                          << focal[1] << ", " << focal[2] << endl
                          << "Position [x,y,z] " << pos[0] << ", " << pos[1]
                          << ", " << pos[2] << endl
                          << "View up [x,y,z] " << view[0] << ", " << view[1]
                          << ", " << view[2] << endl
                          << "Camera view angle [degrees] "
                          << cam->GetViewAngle() << endl
                          << "Window size [x,y] " << win_size[0] << ", "
                          << win_size[1] << endl
                          << "Window position [x,y] " << win_pos[0] << ", "
                          << win_pos[1] << endl;
            }
            break;
        }
        case '=': {
            // Zoom in with = key (requires modifier to avoid conflict with grow
            // selection)
            if (alt || ctrl) {
                zoomIn();
            }
            break;
        }
        case 43:  // KEY_PLUS
        {
            if (alt && ctrl) {
                zoomIn();
            } else if (shift && ctrl) {
                vtkSmartPointer<vtkActorCollection> ac =
                        CurrentRenderer->GetActors();
                vtkCollectionSimpleIterator ait;
                for (ac->InitTraversal(ait);
                     vtkActor* actor = ac->GetNextActor(ait);) {
                    for (actor->InitPathTraversal();
                         vtkAssemblyPath* path = actor->GetNextPath();) {
                        vtkSmartPointer<vtkActor> apart =
                                reinterpret_cast<vtkActor*>(
                                        path->GetLastNode()->GetViewProp());
                        float psize = apart->GetProperty()->GetPointSize();
                        if (psize < 63.0f)
                            apart->GetProperty()->SetPointSize(psize + 1.0f);
                    }
                }
            }
            break;
        }
        case 45:  // KEY_MINUS
        {
            if (alt && ctrl) {
                zoomOut();
            } else if (shift && ctrl) {
                vtkSmartPointer<vtkActorCollection> ac =
                        CurrentRenderer->GetActors();
                vtkCollectionSimpleIterator ait;
                for (ac->InitTraversal(ait);
                     vtkActor* actor = ac->GetNextActor(ait);) {
                    for (actor->InitPathTraversal();
                         vtkAssemblyPath* path = actor->GetNextPath();) {
                        vtkSmartPointer<vtkActor> apart =
                                static_cast<vtkActor*>(
                                        path->GetLastNode()->GetViewProp());
                        float psize = apart->GetProperty()->GetPointSize();
                        if (psize > 1.0f)
                            apart->GetProperty()->SetPointSize(psize - 1.0f);
                    }
                }
            }
            break;
        }
        // Switch between maximize and original window size
        case 'f':
        case 'F': {
            if (keymod) {
                if (alt && ctrl) {
                    // Get screen size
                    int* temp = Interactor->GetRenderWindow()->GetScreenSize();
                    int scr_size[2];
                    scr_size[0] = temp[0];
                    scr_size[1] = temp[1];

                    // Get window size
                    temp = Interactor->GetRenderWindow()->GetSize();
                    int win_size[2];
                    win_size[0] = temp[0];
                    win_size[1] = temp[1];
                    // Is window size = max?
                    if (win_size[0] == max_win_height_ &&
                        win_size[1] == max_win_width_) {
                        Interactor->GetRenderWindow()->SetSize(win_height_,
                                                               win_width_);
                        Interactor->GetRenderWindow()->SetPosition(win_pos_x_,
                                                                   win_pos_y_);
                        Interactor->GetRenderWindow()->Render();
                        Interactor->Render();
                    } else {
                        int* win_pos =
                                Interactor->GetRenderWindow()->GetPosition();
                        win_pos_x_ = win_pos[0];
                        win_pos_y_ = win_pos[1];
                        win_height_ = win_size[0];
                        win_width_ = win_size[1];
                        Interactor->GetRenderWindow()->SetSize(scr_size[0],
                                                               scr_size[1]);
                        Interactor->GetRenderWindow()->Render();
                        Interactor->Render();
                        int* win_size =
                                Interactor->GetRenderWindow()->GetSize();
                        max_win_height_ = win_size[0];
                        max_win_width_ = win_size[1];
                    }
                }
            } else {
                AnimState = VTKIS_ANIM_ON;
                vtkAssemblyPath* path = NULL;
                Interactor->GetPicker()->Pick(Interactor->GetEventPosition()[0],
                                              Interactor->GetEventPosition()[1],
                                              0.0, CurrentRenderer);
                vtkAbstractPropPicker* picker;
                if ((picker = vtkAbstractPropPicker::SafeDownCast(
                             Interactor->GetPicker())))
                    path = picker->GetPath();
                if (path != NULL)
                    Interactor->FlyTo(CurrentRenderer,
                                      picker->GetPickPosition());
                AnimState = VTKIS_ANIM_OFF;
            }
            break;
        }
        // 's'/'S' w/out ALT + CTRL
        case 's':
        case 'S': {
            if (alt && ctrl) {
                int stereo_render =
                        Interactor->GetRenderWindow()->GetStereoRender();
                if (!stereo_render) {
                    if (stereo_anaglyph_mask_default_) {
                        Interactor->GetRenderWindow()->SetAnaglyphColorMask(4,
                                                                            3);
                        stereo_anaglyph_mask_default_ = false;
                    } else {
                        Interactor->GetRenderWindow()->SetAnaglyphColorMask(2,
                                                                            5);
                        stereo_anaglyph_mask_default_ = true;
                    }
                }
                Interactor->GetRenderWindow()->SetStereoRender(!stereo_render);
                Interactor->GetRenderWindow()->Render();
                Interactor->Render();
            } else if (shift && ctrl) {
                vtkInteractorStyleRubberBandPick::OnKeyDown();
                vtkSmartPointer<vtkActorCollection> ac =
                        CurrentRenderer->GetActors();
                vtkCollectionSimpleIterator ait;
                for (ac->InitTraversal(ait);
                     vtkActor* actor = ac->GetNextActor(ait);) {
                    for (actor->InitPathTraversal();
                         vtkAssemblyPath* path = actor->GetNextPath();) {
                        vtkSmartPointer<vtkActor> apart =
                                reinterpret_cast<vtkActor*>(
                                        path->GetLastNode()->GetViewProp());
                        apart->GetProperty()->SetRepresentationToSurface();
                        apart->GetProperty()->SetLighting(true);
                    }
                }
            }
            break;
        }

        // Display a grid/scale over the screen
        case 'g':
        case 'G': {
            if (alt && ctrl) {
                if (!grid_enabled_) {
                    grid_actor_->TopAxisVisibilityOn();
                    CurrentRenderer->AddViewProp(grid_actor_);
                    grid_enabled_ = true;
                } else {
                    CurrentRenderer->RemoveViewProp(grid_actor_);
                    grid_enabled_ = false;
                }
            }
            break;
        }

        case 'o':
        case 'O': {
            if (alt && ctrl) {
                vtkSmartPointer<vtkCamera> cam =
                        CurrentRenderer->GetActiveCamera();
                int flag = cam->GetParallelProjection();
                cam->SetParallelProjection(!flag);

                CurrentRenderer->SetActiveCamera(cam);
                CurrentRenderer->Render();
            }
            break;
        }
        // Display a LUT actor on screen
        case 'u':
        case 'U': {
            if (alt && ctrl) {
                this->updateLookUpTableDisplay(true);
            }
            break;
        }

        // Overwrite the camera reset
        case 'r':
        case 'R': {
            if (!keymod) {
                FindPokedRenderer(Interactor->GetEventPosition()[0],
                                  Interactor->GetEventPosition()[1]);
                if (CurrentRenderer != 0)
                    CurrentRenderer->ResetCamera();
                else
                    CVLog::Warning(
                            "no current renderer on the interactor style.");

                CurrentRenderer->Render();
                break;
            }

            vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();

            // Reset camera to viewpoint from cloud actors
            if (cloud_actors_ && !cloud_actors_->empty()) {
                static PclUtils::CloudActorMap::iterator it =
                    cloud_actors_->begin();
            bool found_transformation = false;
                for (unsigned idx = 0; idx < cloud_actors_->size();
                     ++idx, ++it) {
                if (it == cloud_actors_->end()) it = cloud_actors_->begin();

                    const PclUtils::CloudActorEntry& actor = it->second;
                if (actor.viewpoint_transformation_.GetPointer()) {
                    found_transformation = true;
                    break;
                }
            }

            if (found_transformation) {
                    const PclUtils::CloudActorEntry& actor = it->second;
                cam->SetPosition(
                        actor.viewpoint_transformation_->GetElement(0, 3),
                        actor.viewpoint_transformation_->GetElement(1, 3),
                        actor.viewpoint_transformation_->GetElement(2, 3));

                cam->SetFocalPoint(
                        actor.viewpoint_transformation_->GetElement(0, 3) -
                                    actor.viewpoint_transformation_->GetElement(
                                            0, 2),
                        actor.viewpoint_transformation_->GetElement(1, 3) -
                                    actor.viewpoint_transformation_->GetElement(
                                            1, 2),
                        actor.viewpoint_transformation_->GetElement(2, 3) -
                                    actor.viewpoint_transformation_->GetElement(
                                            2, 2));

                cam->SetViewUp(
                        actor.viewpoint_transformation_->GetElement(0, 1),
                        actor.viewpoint_transformation_->GetElement(1, 1),
                        actor.viewpoint_transformation_->GetElement(2, 1));
            } else {
                cam->SetPosition(0, 0, 0);
                cam->SetFocalPoint(0, 0, 1);
                cam->SetViewUp(0, -1, 0);
            }

            if (it != cloud_actors_->end())
                ++it;
            else
                it = cloud_actors_->begin();
            } else {
                cam->SetPosition(0, 0, 0);
                cam->SetFocalPoint(0, 0, 1);
                cam->SetViewUp(0, -1, 0);
            }

            CurrentRenderer->SetActiveCamera(cam);
            CurrentRenderer->ResetCameraClippingRange();
            CurrentRenderer->Render();
            break;
        }

        case 'a':
        case 'A': {
            CurrentMode =
                    (CurrentMode == ORIENT_MODE) ? SELECT_MODE : ORIENT_MODE;
            if (CurrentMode == SELECT_MODE) {
                point_picker_ =
                        static_cast<vtkPointPicker*>(Interactor->GetPicker());
                vtkSmartPointer<vtkAreaPicker> area_picker =
                        vtkSmartPointer<vtkAreaPicker>::New();
                Interactor->SetPicker(area_picker);
            } else {
                Interactor->SetPicker(point_picker_);
            }
            break;
        }

        case 'q':
        case 'Q': {
            Interactor->ExitCallback();
            return;
        }
        default: {
            vtkInteractorStyleRubberBandPick::OnKeyDown();
            break;
        }
    }

    PclUtils::KeyboardEvent event(
            true, Interactor->GetKeySym(), Interactor->GetKeyCode(),
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey());
    keyboard_signal_(event);

    rens_->Render();
    Interactor->Render();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnKeyUp() {
    PclUtils::KeyboardEvent event(
            false, Interactor->GetKeySym(), Interactor->GetKeyCode(),
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey());
    keyboard_signal_(event);

    // Look for a matching camera interactor.
    this->CameraManipulators->InitTraversal();
    vtkCameraManipulator* manipulator = NULL;
    while ((manipulator = (vtkCameraManipulator*)this->CameraManipulators
                                  ->GetNextItemAsObject())) {
        manipulator->OnKeyUp(this->Interactor);
    }

    vtkInteractorStyleRubberBandPick::OnKeyUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnMouseMove() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    PclUtils::MouseEvent event(
            PclUtils::MouseEvent::MouseMove,
            PclUtils::MouseEvent::NoButton, x, y,
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey(),
            vtkInteractorStyleRubberBandPick::CurrentMode);
    mouse_signal_(event);

    if (this->CurrentMode != VTKISRBP_SELECT) {
        if (this->CurrentRenderer && this->CurrentManipulator) {
            // When an interaction is active, we should not change the
            // renderer being interacted with.
        } else {
            this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
                                    this->Interactor->GetEventPosition()[1]);
        }

        if (this->CurrentManipulator) {
            this->CurrentManipulator->OnMouseMove(
                    this->Interactor->GetEventPosition()[0],
                    this->Interactor->GetEventPosition()[1],
                    this->CurrentRenderer, this->Interactor);
            this->InvokeEvent(vtkCommand::InteractionEvent);
        }
    } else {
        vtkInteractorStyleRubberBandPick::OnMouseMove();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnLeftButtonDown() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];

    if (Interactor->GetRepeatCount() == 0) {
        PclUtils::MouseEvent event(
                PclUtils::MouseEvent::MouseButtonPress,
                PclUtils::MouseEvent::LeftButton, x, y,
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey(),
                vtkInteractorStyleRubberBandPick::CurrentMode);
        mouse_signal_(event);
    } else {
        PclUtils::MouseEvent event(
                PclUtils::MouseEvent::MouseDblClick,
                PclUtils::MouseEvent::LeftButton, x, y,
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey(),
                vtkInteractorStyleRubberBandPick::CurrentMode);
        mouse_signal_(event);
    }
    this->OnButtonDown(1, this->Interactor->GetShiftKey(),
                       this->Interactor->GetControlKey());
    vtkInteractorStyleRubberBandPick::OnLeftButtonDown();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnLeftButtonUp() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    PclUtils::MouseEvent event(
            PclUtils::MouseEvent::MouseButtonRelease,
            PclUtils::MouseEvent::LeftButton, x, y,
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey(),
            vtkInteractorStyleRubberBandPick::CurrentMode);
    mouse_signal_(event);
    this->OnButtonUp(1);
    vtkInteractorStyleRubberBandPick::OnLeftButtonUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnMiddleButtonDown() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    if (Interactor->GetRepeatCount() == 0) {
        PclUtils::MouseEvent event(
                PclUtils::MouseEvent::MouseButtonPress,
                PclUtils::MouseEvent::MiddleButton, x, y,
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey(),
                vtkInteractorStyleRubberBandPick::CurrentMode);
        mouse_signal_(event);
    } else {
        PclUtils::MouseEvent event(
                PclUtils::MouseEvent::MouseDblClick,
                PclUtils::MouseEvent::MiddleButton, x, y,
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey(),
                vtkInteractorStyleRubberBandPick::CurrentMode);
        mouse_signal_(event);
    }
    this->OnButtonDown(2, this->Interactor->GetShiftKey(),
                       this->Interactor->GetControlKey());
    vtkInteractorStyleRubberBandPick::OnMiddleButtonDown();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnMiddleButtonUp() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    PclUtils::MouseEvent event(
            PclUtils::MouseEvent::MouseButtonRelease,
            PclUtils::MouseEvent::MiddleButton, x, y,
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey(),
            vtkInteractorStyleRubberBandPick::CurrentMode);
    mouse_signal_(event);
    this->OnButtonUp(2);
    vtkInteractorStyleRubberBandPick::OnMiddleButtonUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnRightButtonDown() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    if (Interactor->GetRepeatCount() == 0) {
        PclUtils::MouseEvent event(
                PclUtils::MouseEvent::MouseButtonPress,
                PclUtils::MouseEvent::RightButton, x, y,
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey(),
                vtkInteractorStyleRubberBandPick::CurrentMode);
        mouse_signal_(event);
    } else {
        PclUtils::MouseEvent event(
                PclUtils::MouseEvent::MouseDblClick,
                PclUtils::MouseEvent::RightButton, x, y,
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey(),
                vtkInteractorStyleRubberBandPick::CurrentMode);
        mouse_signal_(event);
    }

    this->OnButtonDown(3, this->Interactor->GetShiftKey(),
                       this->Interactor->GetControlKey());
    vtkInteractorStyleRubberBandPick::OnRightButtonDown();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnRightButtonUp() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    PclUtils::MouseEvent event(
            PclUtils::MouseEvent::MouseButtonRelease,
            PclUtils::MouseEvent::RightButton, x, y,
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey(),
            vtkInteractorStyleRubberBandPick::CurrentMode);
    mouse_signal_(event);
    this->OnButtonUp(3);
    vtkInteractorStyleRubberBandPick::OnRightButtonUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnMouseWheelForward() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    PclUtils::MouseEvent event(
            PclUtils::MouseEvent::MouseScrollUp,
            PclUtils::MouseEvent::VScroll, x, y,
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey(),
            vtkInteractorStyleRubberBandPick::CurrentMode);
    mouse_signal_(event);
    if (Interactor->GetRepeatCount()) mouse_signal_(event);

    if (Interactor->GetAltKey()) {
        // zoom
        vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();
        double opening_angle = cam->GetViewAngle();
        if (opening_angle > 15.0) opening_angle -= 1.0;

        cam->SetViewAngle(opening_angle);
        cam->Modified();
        CurrentRenderer->SetActiveCamera(cam);
        CurrentRenderer->ResetCameraClippingRange();
        CurrentRenderer->Modified();
        CurrentRenderer->Render();
        rens_->Render();
        Interactor->Render();
    } else
        vtkInteractorStyleRubberBandPick::OnMouseWheelForward();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnMouseWheelBackward() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    PclUtils::MouseEvent event(
            PclUtils::MouseEvent::MouseScrollDown,
            PclUtils::MouseEvent::VScroll, x, y,
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey(),
            vtkInteractorStyleRubberBandPick::CurrentMode);
    mouse_signal_(event);
    if (Interactor->GetRepeatCount()) mouse_signal_(event);

    if (Interactor->GetAltKey()) {
        // zoom
        vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();
        double opening_angle = cam->GetViewAngle();
        if (opening_angle < 170.0) opening_angle += 1.0;

        cam->SetViewAngle(opening_angle);
        cam->Modified();
        CurrentRenderer->SetActiveCamera(cam);
        CurrentRenderer->ResetCameraClippingRange();
        CurrentRenderer->Modified();
        CurrentRenderer->Render();
        rens_->Render();
        Interactor->Render();
    } else
        vtkInteractorStyleRubberBandPick::OnMouseWheelBackward();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnTimer() {
    if (!init_) {
        CVLog::Error(
                "[vtkCustomInteractorStyle] Interactor style not initialized. "
                "Please call Initialize () before continuing.");
        return;
    }

    if (!rens_) {
        CVLog::Error(
                "[vtkCustomInteractorStyle] No renderer collection given! Use "
                "SetRendererCollection () before continuing.");
        return;
    }
    rens_->Render();
    Interactor->Render();
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::ResetLights() {
    if (!this->CurrentRenderer) {
        return;
    }

    vtkLight* light;

    vtkLightCollection* lights = this->CurrentRenderer->GetLights();
    vtkCamera* camera = this->CurrentRenderer->GetActiveCamera();

    lights->InitTraversal();
    light = lights->GetNextItem();
    if (!light) {
        return;
    }
    light->SetPosition(camera->GetPosition());
    light->SetFocalPoint(camera->GetFocalPoint());
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::OnButtonDown(int button,
                                            int shift,
                                            int control) {
    if (this->CurrentManipulator) {
        return;
    }

    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
                            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == NULL) {
        return;
    }

    this->CurrentManipulator = this->FindManipulator(button, shift, control);
    if (this->CurrentManipulator) {
        this->CurrentManipulator->Register(this);
        this->InvokeEvent(vtkCommand::StartInteractionEvent);
        this->CurrentManipulator->SetCenter(this->CenterOfRotation);
        this->CurrentManipulator->SetRotationFactor(this->RotationFactor);
        this->CurrentManipulator->StartInteraction();
        this->CurrentManipulator->OnButtonDown(
                this->Interactor->GetEventPosition()[0],
                this->Interactor->GetEventPosition()[1], this->CurrentRenderer,
                this->Interactor);
    }
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::OnButtonUp(int button) {
    if (this->CurrentManipulator == NULL) {
        return;
    }
    if (this->CurrentManipulator->GetButton() == button) {
        this->CurrentManipulator->OnButtonUp(
                this->Interactor->GetEventPosition()[0],
                this->Interactor->GetEventPosition()[1], this->CurrentRenderer,
                this->Interactor);
        this->CurrentManipulator->EndInteraction();
        this->InvokeEvent(vtkCommand::EndInteractionEvent);
        this->CurrentManipulator->UnRegister(this);
        this->CurrentManipulator = NULL;
    }
}

//-------------------------------------------------------------------------
vtkCameraManipulator* vtkCustomInteractorStyle::FindManipulator(int button,
                                                                int shift,
                                                                int control) {
    this->CameraManipulators->InitTraversal();
    vtkCameraManipulator* manipulator = NULL;
    while ((manipulator = (vtkCameraManipulator*)this->CameraManipulators
                                  ->GetNextItemAsObject())) {
        if (manipulator->GetButton() == button &&
            manipulator->GetShift() == shift &&
            manipulator->GetControl() == control) {
            return manipulator;
        }
    }
    return NULL;
}

void vtkCustomInteractorStyle::Dolly(double fact) {
    if (this->Interactor->GetControlKey()) {
        vtkCustomInteractorStyle::DollyToPosition(
                fact, this->Interactor->GetEventPosition(),
                this->CurrentRenderer);
    } else {
        this->vtkInteractorStyleRubberBandPick::Dolly(fact);
    }
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::DollyToPosition(double fact,
                                               int* position,
                                               vtkRenderer* renderer) {
    vtkCamera* cam = renderer->GetActiveCamera();
    if (cam->GetParallelProjection()) {
        int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        int* aSize = renderer->GetRenderWindow()->GetSize();
        int w = aSize[0];
        int h = aSize[1];
        x0 = w / 2;
        y0 = h / 2;
        x1 = position[0];
        y1 = position[1];
        vtkCustomInteractorStyle::TranslateCamera(renderer, x0, y0, x1, y1);
        cam->SetParallelScale(cam->GetParallelScale() / fact);
        vtkCustomInteractorStyle::TranslateCamera(renderer, x1, y1, x0, y0);
    } else {
        double viewFocus[4], originalViewFocus[3], cameraPos[3],
                newCameraPos[3];
        double newFocalPoint[4], norm[3];

        cam->GetPosition(cameraPos);
        cam->GetFocalPoint(viewFocus);
        cam->GetFocalPoint(originalViewFocus);
        cam->GetViewPlaneNormal(norm);

        vtkCustomInteractorStyle::ComputeWorldToDisplay(
                renderer, viewFocus[0], viewFocus[1], viewFocus[2], viewFocus);

        vtkCustomInteractorStyle::ComputeDisplayToWorld(
                renderer, double(position[0]), double(position[1]),
                viewFocus[2], newFocalPoint);

        cam->SetFocalPoint(newFocalPoint);

        cam->Dolly(fact);

        cam->GetPosition(newCameraPos);

        double newPoint[3];
        newPoint[0] = originalViewFocus[0] + newCameraPos[0] - cameraPos[0];
        newPoint[1] = originalViewFocus[1] + newCameraPos[1] - cameraPos[1];
        newPoint[2] = originalViewFocus[2] + newCameraPos[2] - cameraPos[2];

        cam->SetFocalPoint(newPoint);
    }
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::TranslateCamera(
        vtkRenderer* renderer, int toX, int toY, int fromX, int fromY) {
    vtkCamera* cam = renderer->GetActiveCamera();
    double viewFocus[4], focalDepth, viewPoint[3];
    double newPickPoint[4], oldPickPoint[4], motionVector[3];
    cam->GetFocalPoint(viewFocus);

    vtkCustomInteractorStyle::ComputeWorldToDisplay(
            renderer, viewFocus[0], viewFocus[1], viewFocus[2], viewFocus);
    focalDepth = viewFocus[2];

    vtkCustomInteractorStyle::ComputeDisplayToWorld(
            renderer, double(toX), double(toY), focalDepth, newPickPoint);
    vtkCustomInteractorStyle::ComputeDisplayToWorld(
            renderer, double(fromX), double(fromY), focalDepth, oldPickPoint);

    motionVector[0] = oldPickPoint[0] - newPickPoint[0];
    motionVector[1] = oldPickPoint[1] - newPickPoint[1];
    motionVector[2] = oldPickPoint[2] - newPickPoint[2];

    cam->GetFocalPoint(viewFocus);
    cam->GetPosition(viewPoint);
    cam->SetFocalPoint(motionVector[0] + viewFocus[0],
                       motionVector[1] + viewFocus[1],
                       motionVector[2] + viewFocus[2]);

    cam->SetPosition(motionVector[0] + viewPoint[0],
                     motionVector[1] + viewPoint[1],
                     motionVector[2] + viewPoint[2]);
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::updateLookUpTableDisplay(bool add_lut) {
    bool actor_found = false;

    if (!lut_enabled_ && !add_lut) return;

    if (lut_actor_id_ != "") {
        // Search in CloudActorMap
        if (cloud_actors_) {
            auto am_it = cloud_actors_->find(lut_actor_id_);
            if (am_it != cloud_actors_->end()) {
                PclUtils::CloudActorEntry* act = &(*am_it).second;
                if (act->actor && act->actor->GetMapper() &&
                    (act->actor->GetMapper()->GetLookupTable() ||
                     (act->actor->GetMapper()->GetInput() &&
                      act->actor->GetMapper()
                              ->GetInput()
                              ->GetPointData()
                              ->GetScalars()))) {
                    vtkScalarsToColors* lut =
                            act->actor->GetMapper()->GetLookupTable();
                    lut_actor_->SetLookupTable(lut);
                    lut_actor_->Modified();
                    actor_found = true;
                }
            }
        }
        // Search in ShapeActorMap
        if (!actor_found && shape_actors_) {
            auto sm_it = shape_actors_->find(lut_actor_id_);
            if (sm_it != shape_actors_->end()) {
                vtkSmartPointer<vtkActor> actor =
                        vtkActor::SafeDownCast(sm_it->second);
                if (actor && actor->GetMapper() &&
                    actor->GetMapper()->GetInput() &&
                    actor->GetMapper()
                         ->GetInput()
                         ->GetPointData()
                         ->GetScalars()) {
                    lut_actor_->SetLookupTable(
                            actor->GetMapper()->GetLookupTable());
                    lut_actor_->Modified();
                    actor_found = true;
                }
            }
        }

        if (!actor_found) {
            CVLog::Warning(
                    "[updateLookUpTableDisplay] Could not find any "
                    "actor with id <%s>!",
                        lut_actor_id_.c_str());
            if (lut_enabled_) {
                    CurrentRenderer->RemoveActor(lut_actor_);
                    lut_enabled_ = false;
                }
                return;
            }
    } else {
        // Search all clouds
        if (cloud_actors_) {
            for (auto& ca : *cloud_actors_) {
                auto& act = ca.second;
                if (!act.actor || !act.actor->GetMapper() ||
                    !act.actor->GetMapper()->GetLookupTable())
                    continue;
                if (!act.actor->GetMapper()->GetInput() ||
                    !act.actor->GetMapper()
                         ->GetInput()
                         ->GetPointData()
                         ->GetScalars())
                continue;

                vtkScalarsToColors* lut =
                        act.actor->GetMapper()->GetLookupTable();
            lut_actor_->SetLookupTable(lut);
            lut_actor_->Modified();
            actor_found = true;
            break;
            }
        }

        if (!actor_found && shape_actors_) {
            for (auto& sa : *shape_actors_) {
                vtkSmartPointer<vtkActor> actor =
                        vtkActor::SafeDownCast(sa.second);
                if (!actor) continue;
                if (!actor->GetMapper() || !actor->GetMapper()->GetInput() ||
                    !actor->GetMapper()
                             ->GetInput()
                             ->GetPointData()
                             ->GetScalars())
                    continue;
                lut_actor_->SetLookupTable(
                        actor->GetMapper()->GetLookupTable());
                lut_actor_->Modified();
                actor_found = true;
                break;
            }
        }
    }

    if ((!actor_found && lut_enabled_) || (lut_enabled_ && add_lut)) {
        CurrentRenderer->RemoveActor(lut_actor_);
        lut_enabled_ = false;
    } else if (!lut_enabled_ && add_lut && actor_found) {
        CurrentRenderer->AddActor(lut_actor_);
        lut_actor_->SetVisibility(true);
        lut_enabled_ = true;
    } else if (lut_enabled_) {
        CurrentRenderer->RemoveActor(lut_actor_);
        CurrentRenderer->AddActor(lut_actor_);
    } else
        return;

    CurrentRenderer->Render();
    return;
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::PrintSelf(ostream& os, vtkIndent indent) {
    this->vtkInteractorStyleRubberBandPick::PrintSelf(os, indent);
    os << indent << "CenterOfRotation: " << this->CenterOfRotation[0] << ", "
       << this->CenterOfRotation[1] << ", " << this->CenterOfRotation[2]
       << endl;
    os << indent << "RotationFactor: " << this->RotationFactor << endl;
    os << indent << "CameraManipulators: " << this->CameraManipulators << endl;
}

}  // namespace VTKExtensions

namespace VTKExtensions {
// Standard VTK macro for *New ()
vtkStandardNewMacro(vtkCustomInteractorStyle);
}  // namespace VTKExtensions
