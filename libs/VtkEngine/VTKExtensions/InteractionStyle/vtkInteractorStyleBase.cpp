// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkInteractorStyleBase.h"

#include <CVLog.h>
#include <FileSystem.h>
#include <vtkCamera.h>
#include <vtkPNGWriter.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkWindowToImageFilter.h>

#include <fstream>

namespace VTKExtensions {

vtkInteractorStyleBase::vtkInteractorStyleBase() {
    grid_actor_ = vtkSmartPointer<vtkLegendScaleActor>::New();
    lut_actor_ = vtkSmartPointer<vtkScalarBarActor>::New();
    snapshot_writer_ = vtkSmartPointer<vtkPNGWriter>::New();
    wif_ = vtkSmartPointer<vtkWindowToImageFilter>::New();
}

void vtkInteractorStyleBase::Initialize() { init_ = true; }

void vtkInteractorStyleBase::saveScreenshot(const std::string& file) {
    FindPokedRenderer(Interactor->GetEventPosition()[0],
                      Interactor->GetEventPosition()[1]);

    wif_->SetInput(Interactor->GetRenderWindow());
    wif_->Modified();
    wif_->ReadFrontBufferOff();
    wif_->Update();

    snapshot_writer_->Modified();
    snapshot_writer_->SetFileName(file.c_str());
    snapshot_writer_->SetInputConnection(wif_->GetOutputPort());
    snapshot_writer_->Write();
}

bool vtkInteractorStyleBase::saveCameraParameters(const std::string& file) {
    if (!Interactor || !Interactor->GetRenderWindow()) return false;

    vtkRenderer* ren =
            Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer();
    if (!ren) return false;

    vtkCamera* cam = ren->GetActiveCamera();
    if (!cam) return false;

    std::ofstream ofs(file);
    if (!ofs.is_open()) return false;

    double clip[2], focal[3], pos[3], view[3];
    cam->GetClippingRange(clip);
    cam->GetFocalPoint(focal);
    cam->GetPosition(pos);
    cam->GetViewUp(view);

    int* win_size = Interactor->GetRenderWindow()->GetSize();
    int* win_pos = Interactor->GetRenderWindow()->GetPosition();

    ofs << clip[0] << "," << clip[1] << "/" << focal[0] << "," << focal[1]
        << "," << focal[2] << "/" << pos[0] << "," << pos[1] << "," << pos[2]
        << "/" << view[0] << "," << view[1] << "," << view[2] << "/"
        << cam->GetViewAngle() << "/" << win_size[0] << "," << win_size[1]
        << "/" << win_pos[0] << "," << win_pos[1] << std::endl;

    return ofs.good();
}

bool vtkInteractorStyleBase::loadCameraParameters(const std::string& file) {
    if (!cloudViewer::utility::filesystem::FileExists(file)) return false;
    if (!Interactor || !Interactor->GetRenderWindow()) return false;

    vtkRenderer* ren =
            Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer();
    if (!ren) return false;

    vtkCamera* cam = ren->GetActiveCamera();
    if (!cam) return false;

    std::ifstream ifs(file);
    if (!ifs.is_open()) return false;

    std::string line;
    if (!std::getline(ifs, line)) return false;

    double clip[2], focal[3], pos[3], view[3], angle;
    int win_size[2], win_pos[2];

    int n = sscanf(line.c_str(),
                   "%lf,%lf/%lf,%lf,%lf/%lf,%lf,%lf/%lf,%lf,%lf/%lf/%d,%d/"
                   "%d,%d",
                   &clip[0], &clip[1], &focal[0], &focal[1], &focal[2], &pos[0],
                   &pos[1], &pos[2], &view[0], &view[1], &view[2], &angle,
                   &win_size[0], &win_size[1], &win_pos[0], &win_pos[1]);

    if (n < 12) return false;

    cam->SetClippingRange(clip);
    cam->SetFocalPoint(focal);
    cam->SetPosition(pos);
    cam->SetViewUp(view);
    cam->SetViewAngle(angle);

    if (n >= 14)
        Interactor->GetRenderWindow()->SetSize(win_size[0], win_size[1]);
    if (n >= 16)
        Interactor->GetRenderWindow()->SetPosition(win_pos[0], win_pos[1]);

    ren->ResetCameraClippingRange();
    ren->Render();
    Interactor->Render();
    return true;
}

}  // namespace VTKExtensions
