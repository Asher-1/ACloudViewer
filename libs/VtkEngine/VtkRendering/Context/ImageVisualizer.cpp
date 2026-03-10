// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file ImageVisualizer.cpp
 * @brief Implementation of image viewer interactor style and camera adjustment.
 */

#include "ImageVisualizer.h"

#include <vtkCamera.h>
#include <vtkContext2D.h>
#include <vtkContextScene.h>
#include <vtkImageData.h>
#include <vtkObjectFactory.h>
#include <vtkRendererCollection.h>

#include "ContextItems.h"

namespace VtkRendering {

vtkStandardNewMacro(ImageViewerInteractorStyle);

void ImageViewerInteractorStyle::OnLeftButtonDown() {
    this->Interactor->GetRenderWindow()->SetDesiredUpdateRate(0.5);
    this->Interactor->GetRenderWindow()->SetCurrentCursor(VTK_CURSOR_HAND);
    this->Interactor->GetRenderWindow()->Render();
}

void ImageViewerInteractorStyle::OnChar() {
    this->Interactor->GetRenderWindow()->SetDesiredUpdateRate(0.001);
    this->Interactor->GetRenderWindow()->SetCurrentCursor(VTK_CURSOR_DEFAULT);
    this->Interactor->GetRenderWindow()->Render();
}

void ImageViewerInteractorStyle::adjustCamera(vtkImageData* image,
                                              vtkRenderer* ren) {
    if (!image || !ren) return;
    double origin[3];
    int extent[6];
    double spacing[3];
    image->GetOrigin(origin);
    image->GetExtent(extent);
    image->GetSpacing(spacing);

    vtkCamera* camera = ren->GetActiveCamera();
    camera->ParallelProjectionOn();

    double xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0];
    double yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1];
    double yd = (extent[3] - extent[2] + 1) * spacing[1];
    double d = camera->GetDistance();
    camera->SetParallelScale(0.5 * yd);
    camera->SetFocalPoint(xc, yc, 0.0);
    camera->SetPosition(xc, yc, d);
}

void ImageViewerInteractorStyle::adjustCamera(vtkRenderer* ren) {
    if (!ren) return;
    vtkCamera* camera = ren->GetActiveCamera();
    camera->ParallelProjectionOn();
}

ImageVisualizer::ImageVisualizer(const std::string& window_title)
    : timer_id_(-1), stopped_(false) {
    win_ = vtkSmartPointer<vtkRenderWindow>::New();
    ren_ = vtkSmartPointer<vtkRenderer>::New();
    slice_ = vtkSmartPointer<vtkImageSlice>::New();

    win_->AddRenderer(ren_);
    if (!window_title.empty()) {
        win_->SetWindowName(window_title.c_str());
    }
    ren_->AddViewProp(slice_);
}

ImageVisualizer::~ImageVisualizer() { close(); }

void ImageVisualizer::setPosition(int x, int y) {
    if (win_) win_->SetPosition(x, y);
}

void ImageVisualizer::setSize(int xw, int yw) {
    if (win_) win_->SetSize(xw, yw);
}

int* ImageVisualizer::getSize() { return win_ ? win_->GetSize() : nullptr; }

void ImageVisualizer::close() {
    stopped_ = true;
    if (interactor_) {
        interactor_->TerminateApp();
    }
}

void ImageVisualizer::removeLayer(const std::string& layer_id) {
    auto it = std::find_if(layer_map_.begin(), layer_map_.end(),
                           LayerComparator(layer_id));
    if (it != layer_map_.end()) {
        if (it->actor && ren_) {
            ren_->RemoveActor(it->actor);
        }
        layer_map_.erase(it);
    }
}

ImageVisualizer::LayerMap::iterator ImageVisualizer::createLayer(
        const std::string& layer_id,
        int width,
        int height,
        double opacity,
        bool fill_box) {
    Layer l;
    l.layer_name = layer_id;
    l.actor = vtkSmartPointer<vtkContextActor>::New();
    l.actor->PickableOff();
    l.actor->DragableOff();
    if (fill_box) {
        auto rect = vtkSmartPointer<
                VtkRendering::context_items::FilledRectangle>::New();
        rect->setColors(0, 0, 0);
        rect->setOpacity(opacity);
        rect->set(0, 0, static_cast<float>(width), static_cast<float>(height));
        l.actor->GetScene()->AddItem(rect);
    }
    if (ren_) {
        ren_->AddActor(l.actor);
    }
    layer_map_.push_back(l);
    return (layer_map_.end() - 1);
}

VtkRendering::Connection ImageVisualizer::registerMouseCallback(
        std::function<void(const VtkRendering::MouseEvent&)> cb) {
    return mouse_signal_.connect(cb);
}

void ImageVisualizer::emitMouseEvent(unsigned long event_id) {
    if (!interactor_) return;
    int x = interactor_->GetEventPosition()[0];
    int y = interactor_->GetEventPosition()[1];

    MouseEvent::Type type;
    MouseEvent::MouseButton button = MouseEvent::NoButton;

    switch (event_id) {
        case vtkCommand::MouseMoveEvent:
            type = MouseEvent::MouseMove;
            break;
        case vtkCommand::LeftButtonPressEvent:
            type = MouseEvent::MouseButtonPress;
            button = MouseEvent::LeftButton;
            break;
        case vtkCommand::LeftButtonReleaseEvent:
            type = MouseEvent::MouseButtonRelease;
            button = MouseEvent::LeftButton;
            break;
        case vtkCommand::RightButtonPressEvent:
            type = MouseEvent::MouseButtonPress;
            button = MouseEvent::RightButton;
            break;
        case vtkCommand::RightButtonReleaseEvent:
            type = MouseEvent::MouseButtonRelease;
            button = MouseEvent::RightButton;
            break;
        case vtkCommand::MiddleButtonPressEvent:
            type = MouseEvent::MouseButtonPress;
            button = MouseEvent::MiddleButton;
            break;
        case vtkCommand::MiddleButtonReleaseEvent:
            type = MouseEvent::MouseButtonRelease;
            button = MouseEvent::MiddleButton;
            break;
        case vtkCommand::MouseWheelForwardEvent:
            type = MouseEvent::MouseScrollUp;
            button = MouseEvent::VScroll;
            break;
        case vtkCommand::MouseWheelBackwardEvent:
            type = MouseEvent::MouseScrollDown;
            button = MouseEvent::VScroll;
            break;
        default:
            return;
    }

    bool alt = interactor_->GetAltKey() != 0;
    bool ctrl = interactor_->GetControlKey() != 0;
    bool shift = interactor_->GetShiftKey() != 0;

    MouseEvent event(type, button, x, y, alt, ctrl, shift);
    mouse_signal_(event);
}

void ImageVisualizer::MouseCallback(vtkObject*,
                                    unsigned long event_id,
                                    void* clientdata,
                                    void*) {
    auto* self = reinterpret_cast<ImageVisualizer*>(clientdata);
    if (self) self->emitMouseEvent(event_id);
}

void ImageVisualizer::ExitMainLoopTimerCallback::Execute(vtkObject*,
                                                         unsigned long event_id,
                                                         void* call_data) {
    if (event_id != vtkCommand::TimerEvent) return;
    int timer_id = *static_cast<int*>(call_data);
    if (timer_id != right_timer_id) return;
    if (window) {
        window->interactor_->TerminateApp();
    }
}

void ImageVisualizer::ExitCallback::Execute(vtkObject*,
                                            unsigned long event_id,
                                            void*) {
    if (event_id != vtkCommand::ExitEvent) return;
    if (window) {
        window->stopped_ = true;
        window->interactor_->TerminateApp();
    }
}

}  // namespace VtkRendering
