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
#include <vtkLODActor.h>
#include <vtkLegendScaleActor.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkObjectFactory.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPointPicker.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkScalarBarActor.h>
#include <vtkTextProperty.h>
#include <vtkVersion.h>
#include <vtkWindowToImageFilter.h>

#define ORIENT_MODE 0
#define SELECT_MODE 1

#define VTKISRBP_ORIENT 0
#define VTKISRBP_SELECT 1

namespace VTKExtensions {
vtkCustomInteractorStyle::vtkCustomInteractorStyle()
    : vtkInteractorStyleBase(),
      CameraManipulators(vtkCollection::New()),
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
void vtkCustomInteractorStyle::RemoveAllManipulators() {
    this->CameraManipulators->RemoveAllItems();
}

//-------------------------------------------------------------------------
void vtkCustomInteractorStyle::AddManipulator(vtkCameraManipulator* m) {
    this->CameraManipulators->AddItem(m);
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::zoomIn() {
    if (!Interactor) return;
    FindPokedRenderer(Interactor->GetEventPosition()[0],
                      Interactor->GetEventPosition()[1]);
    StartDolly();
    double factor = 10.0 * 0.2 * .5;
    Dolly(pow(1.1, factor));
    EndDolly();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::zoomOut() {
    if (!Interactor) return;
    FindPokedRenderer(Interactor->GetEventPosition()[0],
                      Interactor->GetEventPosition()[1]);
    StartDolly();
    double factor = 10.0 * -0.2 * .5;
    Dolly(pow(1.1, factor));
    EndDolly();
}

void vtkCustomInteractorStyle::toggleAreaPicking() {
    if (!Interactor) return;
    CurrentMode = (CurrentMode == ORIENT_MODE) ? SELECT_MODE : ORIENT_MODE;
    if (CurrentMode == SELECT_MODE) {
        point_picker_ = static_cast<vtkPointPicker*>(Interactor->GetPicker());
        vtkSmartPointer<vtkAreaPicker> area_picker =
                vtkSmartPointer<vtkAreaPicker>::New();
        Interactor->SetPicker(area_picker);
    } else {
        Interactor->SetPicker(point_picker_);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnChar() {
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
        case VtkRendering::INTERACTOR_KB_MOD_ALT:
            keymod = Interactor->GetAltKey();
            break;
        case VtkRendering::INTERACTOR_KB_MOD_CTRL:
            keymod = Interactor->GetControlKey();
            break;
        case VtkRendering::INTERACTOR_KB_MOD_SHIFT:
            keymod = Interactor->GetShiftKey();
            break;
    }

    switch (Interactor->GetKeyCode()) {
        case 'a':
        case 'A':
        case 'h':
        case 'H':
        case 'p':
        case 'P':
        case 'j':
        case 'J':
        case 'c':
        case 'C':
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

    if (win_height_ == -1 || win_width_ == -1) {
        int* win_size = Interactor->GetRenderWindow()->GetSize();
        win_height_ = win_size[0];
        win_width_ = win_size[1];
    }

    bool shift = Interactor->GetShiftKey();
    bool ctrl = Interactor->GetControlKey();
    bool alt = Interactor->GetAltKey();

    bool keymod = false;
    switch (modifier_) {
        case VtkRendering::INTERACTOR_KB_MOD_ALT:
            keymod = alt;
            break;
        case VtkRendering::INTERACTOR_KB_MOD_CTRL:
            keymod = ctrl;
            break;
        case VtkRendering::INTERACTOR_KB_MOD_SHIFT:
            keymod = shift;
            break;
    }

    // Save camera parameters (Ctrl+S)
    if ((Interactor->GetKeySym()[0] == 'S' ||
         Interactor->GetKeySym()[0] == 's') &&
        ctrl && !alt && !shift) {
        if (camera_file_.empty()) {
            vtkRenderer* ren = Interactor->GetRenderWindow()
                                       ->GetRenderers()
                                       ->GetFirstRenderer();
            if (ren) {
                vtkCamera* cam = ren->GetActiveCamera();
                if (cam) {
                    cam->GetPosition(saved_cam_pos_);
                    cam->GetFocalPoint(saved_cam_focal_);
                    cam->GetViewUp(saved_cam_viewup_);
                    cam->GetClippingRange(saved_cam_clip_);
                    camera_saved_ = true;
                    CVLog::Print(
                            "Camera parameters saved, you can press CTRL + R "
                            "to restore.");
                }
            }
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

    // Restore camera parameters (Ctrl+R)
    if ((Interactor->GetKeySym()[0] == 'R' ||
         Interactor->GetKeySym()[0] == 'r') &&
        ctrl && !alt && !shift) {
        if (camera_file_.empty()) {
            if (camera_saved_) {
                vtkRenderer* ren = Interactor->GetRenderWindow()
                                           ->GetRenderers()
                                           ->GetFirstRenderer();
                if (ren) {
                    vtkCamera* cam = ren->GetActiveCamera();
                    if (cam) {
                        cam->SetPosition(saved_cam_pos_);
                        cam->SetFocalPoint(saved_cam_focal_);
                        cam->SetViewUp(saved_cam_viewup_);
                        cam->SetClippingRange(saved_cam_clip_);
                        ren->ResetCameraClippingRange();
                        ren->Render();
                    }
                }
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
                    "    SHIFT + left click   : select a point"
                    ""
                    "          a, A   : toggle rubber band selection mode for "
                    "left mouse button");
            break;
        }

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
            if (alt || ctrl) {
                zoomIn();
            }
            break;
        }
        case 43: {  // KEY_PLUS
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
        case 45: {  // KEY_MINUS
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
        case 'f':
        case 'F': {
            if (keymod) {
                if (alt && ctrl) {
                    int* temp = Interactor->GetRenderWindow()->GetScreenSize();
                    int scr_size[2];
                    scr_size[0] = temp[0];
                    scr_size[1] = temp[1];

                    temp = Interactor->GetRenderWindow()->GetSize();
                    int win_size[2];
                    win_size[0] = temp[0];
                    win_size[1] = temp[1];
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
        case 'u':
        case 'U': {
            if (alt && ctrl) {
                this->updateLookUpTableDisplay(true);
            }
            break;
        }

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

            if (cloud_actors_ && !cloud_actors_->empty()) {
                static VtkRendering::CloudActorMap::iterator it =
                        cloud_actors_->begin();
                bool found_transformation = false;
                for (unsigned idx = 0; idx < cloud_actors_->size();
                     ++idx, ++it) {
                    if (it == cloud_actors_->end()) it = cloud_actors_->begin();

                    const VtkRendering::CloudActor& actor = it->second;
                    if (actor.viewpoint_transformation) {
                        found_transformation = true;
                        break;
                    }
                }

                if (found_transformation) {
                    const VtkRendering::CloudActor& actor = it->second;
                    auto* vt = actor.viewpoint_transformation.Get();
                    cam->SetPosition(vt->GetElement(0, 3), vt->GetElement(1, 3),
                                     vt->GetElement(2, 3));
                    cam->SetFocalPoint(
                            vt->GetElement(0, 3) - vt->GetElement(0, 2),
                            vt->GetElement(1, 3) - vt->GetElement(1, 2),
                            vt->GetElement(2, 3) - vt->GetElement(2, 2));
                    cam->SetViewUp(vt->GetElement(0, 1), vt->GetElement(1, 1),
                                   vt->GetElement(2, 1));
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

    VtkRendering::KeyboardEvent event(
            true, Interactor->GetKeySym(), Interactor->GetKeyCode(),
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey());
    keyboard_signal_(event);

    rens_->Render();
    Interactor->Render();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnKeyUp() {
    VtkRendering::KeyboardEvent event(
            false, Interactor->GetKeySym(), Interactor->GetKeyCode(),
            Interactor->GetAltKey(), Interactor->GetControlKey(),
            Interactor->GetShiftKey());
    keyboard_signal_(event);

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
    VtkRendering::MouseEvent event(
            VtkRendering::MouseEvent::MouseMove,
            VtkRendering::MouseEvent::NoButton, x, y, Interactor->GetAltKey(),
            Interactor->GetControlKey(), Interactor->GetShiftKey());
    mouse_signal_(event);

    if (this->CurrentMode != VTKISRBP_SELECT) {
        if (this->CurrentRenderer && this->CurrentManipulator) {
            // Active interaction -- do not change the renderer
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
        VtkRendering::MouseEvent event(
                VtkRendering::MouseEvent::MouseButtonPress,
                VtkRendering::MouseEvent::LeftButton, x, y,
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey());
        mouse_signal_(event);
    } else {
        VtkRendering::MouseEvent event(VtkRendering::MouseEvent::MouseDblClick,
                                       VtkRendering::MouseEvent::LeftButton, x,
                                       y, Interactor->GetAltKey(),
                                       Interactor->GetControlKey(),
                                       Interactor->GetShiftKey());
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
    VtkRendering::MouseEvent event(
            VtkRendering::MouseEvent::MouseButtonRelease,
            VtkRendering::MouseEvent::LeftButton, x, y, Interactor->GetAltKey(),
            Interactor->GetControlKey(), Interactor->GetShiftKey());
    mouse_signal_(event);
    this->OnButtonUp(1);
    vtkInteractorStyleRubberBandPick::OnLeftButtonUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnMiddleButtonDown() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    if (Interactor->GetRepeatCount() == 0) {
        VtkRendering::MouseEvent event(
                VtkRendering::MouseEvent::MouseButtonPress,
                VtkRendering::MouseEvent::MiddleButton, x, y,
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey());
        mouse_signal_(event);
    } else {
        VtkRendering::MouseEvent event(VtkRendering::MouseEvent::MouseDblClick,
                                       VtkRendering::MouseEvent::MiddleButton,
                                       x, y, Interactor->GetAltKey(),
                                       Interactor->GetControlKey(),
                                       Interactor->GetShiftKey());
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
    VtkRendering::MouseEvent event(VtkRendering::MouseEvent::MouseButtonRelease,
                                   VtkRendering::MouseEvent::MiddleButton, x, y,
                                   Interactor->GetAltKey(),
                                   Interactor->GetControlKey(),
                                   Interactor->GetShiftKey());
    mouse_signal_(event);
    this->OnButtonUp(2);
    vtkInteractorStyleRubberBandPick::OnMiddleButtonUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnRightButtonDown() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    if (Interactor->GetRepeatCount() == 0) {
        VtkRendering::MouseEvent event(
                VtkRendering::MouseEvent::MouseButtonPress,
                VtkRendering::MouseEvent::RightButton, x, y,
                Interactor->GetAltKey(), Interactor->GetControlKey(),
                Interactor->GetShiftKey());
        mouse_signal_(event);
    } else {
        VtkRendering::MouseEvent event(VtkRendering::MouseEvent::MouseDblClick,
                                       VtkRendering::MouseEvent::RightButton, x,
                                       y, Interactor->GetAltKey(),
                                       Interactor->GetControlKey(),
                                       Interactor->GetShiftKey());
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
    VtkRendering::MouseEvent event(VtkRendering::MouseEvent::MouseButtonRelease,
                                   VtkRendering::MouseEvent::RightButton, x, y,
                                   Interactor->GetAltKey(),
                                   Interactor->GetControlKey(),
                                   Interactor->GetShiftKey());
    mouse_signal_(event);
    this->OnButtonUp(3);
    vtkInteractorStyleRubberBandPick::OnRightButtonUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vtkCustomInteractorStyle::OnMouseWheelForward() {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    VtkRendering::MouseEvent event(
            VtkRendering::MouseEvent::MouseScrollUp,
            VtkRendering::MouseEvent::VScroll, x, y, Interactor->GetAltKey(),
            Interactor->GetControlKey(), Interactor->GetShiftKey());
    mouse_signal_(event);
    if (Interactor->GetRepeatCount()) mouse_signal_(event);

    if (Interactor->GetAltKey()) {
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
    VtkRendering::MouseEvent event(
            VtkRendering::MouseEvent::MouseScrollDown,
            VtkRendering::MouseEvent::VScroll, x, y, Interactor->GetAltKey(),
            Interactor->GetControlKey(), Interactor->GetShiftKey());
    mouse_signal_(event);
    if (Interactor->GetRepeatCount()) mouse_signal_(event);

    if (Interactor->GetAltKey()) {
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
    if (!cloud_actors_ || !shape_actors_) return;

    VtkRendering::CloudActorMap::iterator am_it;
    VtkRendering::ShapeActorMap::iterator sm_it;
    bool actor_found = false;

    if (!lut_enabled_ && !add_lut) return;

    if (!lut_actor_id_.empty()) {
        am_it = cloud_actors_->find(lut_actor_id_);
        if (am_it == cloud_actors_->end()) {
            sm_it = shape_actors_->find(lut_actor_id_);
            if (sm_it == shape_actors_->end()) {
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

            vtkSmartPointer<vtkActor> actor =
                    vtkActor::SafeDownCast(sm_it->second);
            if (!actor || !actor->GetMapper() ||
                !actor->GetMapper()->GetInput() ||
                !actor->GetMapper()->GetInput()->GetPointData()->GetScalars()) {
                CVLog::Warning(
                        "[updateLookUpTableDisplay] id <%s> does not hold "
                        "any color information!",
                        lut_actor_id_.c_str());
                if (lut_enabled_) {
                    CurrentRenderer->RemoveActor(lut_actor_);
                    lut_enabled_ = false;
                }
                return;
            }

            lut_actor_->SetLookupTable(actor->GetMapper()->GetLookupTable());
            lut_actor_->Modified();
            actor_found = true;
        } else {
            VtkRendering::CloudActor& ca = am_it->second;
            if (!ca.actor || !ca.actor->GetMapper() ||
                (!ca.actor->GetMapper()->GetLookupTable() &&
                 !ca.actor->GetMapper()
                          ->GetInput()
                          ->GetPointData()
                          ->GetScalars())) {
                CVLog::Warning(
                        "[updateLookUpTableDisplay] id <%s> does not hold "
                        "any color information!",
                        lut_actor_id_.c_str());
                if (lut_enabled_) {
                    CurrentRenderer->RemoveActor(lut_actor_);
                    lut_enabled_ = false;
                }
                return;
            }

            vtkScalarsToColors* lut = ca.actor->GetMapper()->GetLookupTable();
            lut_actor_->SetLookupTable(lut);
            lut_actor_->Modified();
            actor_found = true;
        }
    } else {
        for (am_it = cloud_actors_->begin(); am_it != cloud_actors_->end();
             ++am_it) {
            VtkRendering::CloudActor& ca = am_it->second;
            if (!ca.actor || !ca.actor->GetMapper()) continue;
            if (!ca.actor->GetMapper()->GetLookupTable()) continue;
            if (!ca.actor->GetMapper()
                         ->GetInput()
                         ->GetPointData()
                         ->GetScalars())
                continue;

            vtkScalarsToColors* lut = ca.actor->GetMapper()->GetLookupTable();
            lut_actor_->SetLookupTable(lut);
            lut_actor_->Modified();
            actor_found = true;
            break;
        }

        if (!actor_found) {
            for (sm_it = shape_actors_->begin(); sm_it != shape_actors_->end();
                 ++sm_it) {
                vtkSmartPointer<vtkActor> actor =
                        vtkActor::SafeDownCast(sm_it->second);
                if (!actor || !actor->GetMapper()) continue;
                if (!actor->GetMapper()
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
vtkStandardNewMacro(vtkCustomInteractorStyle);
}  // namespace VTKExtensions
