// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file ImageVisualizer.h
 *  @brief Image viewer with overlay layers and mouse callback support
 */

#include <VtkRendering/Core/EventCallbacks.h>
#include <VtkRendering/Core/MouseEvent.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkContextActor.h>
#include <vtkImageSlice.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "qVTK.h"

class vtkImageViewer;
class vtkImageFlip;

namespace VtkRendering {

/** @class ImageViewerInteractorStyle
 *  @brief Custom interactor style for image viewer (mouse/keyboard handling)
 */
class QVTK_ENGINE_LIB_API ImageViewerInteractorStyle
    : public vtkInteractorStyleImage {
public:
    static ImageViewerInteractorStyle* New();
    vtkTypeMacro(ImageViewerInteractorStyle, vtkInteractorStyleImage);

    void OnMouseWheelForward() override {}
    void OnMouseWheelBackward() override {}
    void OnMiddleButtonDown() override {}
    void OnRightButtonDown() override {}
    void OnLeftButtonDown() override;
    void OnChar() override;

    /// @param image Image data for bounds
    /// @param ren Renderer to adjust
    void adjustCamera(vtkImageData* image, vtkRenderer* ren);
    /// @param ren Renderer to adjust
    void adjustCamera(vtkRenderer* ren);
};

/** @class ImageVisualizer
 *  @brief Image viewer window with overlay layers and mouse callback
 * registration
 */
class QVTK_ENGINE_LIB_API ImageVisualizer {
public:
    using Ptr = std::shared_ptr<ImageVisualizer>;
    using ConstPtr = std::shared_ptr<const ImageVisualizer>;

    struct Layer {
        vtkSmartPointer<vtkContextActor> actor;
        std::string layer_name;
    };

    using LayerMap = std::vector<Layer>;

    struct LayerComparator {
        explicit LayerComparator(const std::string& str) : str_(str) {}
        const std::string& str_;
        bool operator()(const Layer& layer) {
            return (layer.layer_name == str_);
        }
    };

    /// @param window_title Window title string
    explicit ImageVisualizer(const std::string& window_title = "");
    virtual ~ImageVisualizer();

    /// @param x Window X position
    /// @param y Window Y position
    void setPosition(int x, int y);
    /// @param xw Window width
    /// @param yw Window height
    void setSize(int xw, int yw);
    /// @return Pointer to [width, height] array
    int* getSize();

    /// @return true if window was closed/stopped
    bool wasStopped() const { return stopped_; }
    void close();

    /// @param layer_id Layer name to remove
    void removeLayer(const std::string& layer_id);

    /// @param callback Function to call on mouse events
    /// @param cookie User data passed to callback
    /// @return Connection handle for disconnect
    VtkRendering::Connection registerMouseCallback(
            void (*callback)(const VtkRendering::MouseEvent&, void*),
            void* cookie = nullptr) {
        return registerMouseCallback([=](const VtkRendering::MouseEvent& e) {
            (*callback)(e, cookie);
        });
    }

    /// @param callback Member function to call on mouse events
    /// @param instance Object instance
    /// @param cookie User data passed to callback
    /// @return Connection handle for disconnect
    template <typename T>
    VtkRendering::Connection registerMouseCallback(
            void (T::*callback)(const VtkRendering::MouseEvent&, void*),
            T& instance,
            void* cookie = nullptr) {
        return registerMouseCallback(
                [=, &instance](const VtkRendering::MouseEvent& e) {
                    (instance.*callback)(e, cookie);
                });
    }

    /// @param cb Callback function
    /// @return Connection handle for disconnect
    VtkRendering::Connection registerMouseCallback(
            std::function<void(const VtkRendering::MouseEvent&)> cb);

protected:
    LayerMap::iterator createLayer(const std::string& layer_id,
                                   int width,
                                   int height,
                                   double opacity = 0.5,
                                   bool fill_box = true);

    void resetStoppedFlag() { stopped_ = false; }
    void emitMouseEvent(unsigned long event_id);

    static void MouseCallback(vtkObject* caller,
                              unsigned long event_id,
                              void* clientdata,
                              void* calldata);

    struct ExitMainLoopTimerCallback : public vtkCommand {
        ExitMainLoopTimerCallback() : right_timer_id(-1), window(nullptr) {}
        static ExitMainLoopTimerCallback* New() {
            return new ExitMainLoopTimerCallback;
        }
        void Execute(vtkObject*,
                     unsigned long event_id,
                     void* call_data) override;
        int right_timer_id;
        ImageVisualizer* window;
    };

    struct ExitCallback : public vtkCommand {
        ExitCallback() : window(nullptr) {}
        static ExitCallback* New() { return new ExitCallback; }
        void Execute(vtkObject*, unsigned long event_id, void*) override;
        ImageVisualizer* window;
    };

    VtkRendering::Signal<const VtkRendering::MouseEvent&> mouse_signal_;

    vtkSmartPointer<vtkRenderWindowInteractor> interactor_;
    vtkSmartPointer<vtkCallbackCommand> mouse_command_;

    vtkSmartPointer<ExitMainLoopTimerCallback> exit_main_loop_timer_callback_;
    vtkSmartPointer<ExitCallback> exit_callback_;

    vtkSmartPointer<vtkRenderWindow> win_;
    vtkSmartPointer<vtkRenderer> ren_;
    vtkSmartPointer<vtkImageSlice> slice_;

    vtkSmartPointer<ImageViewerInteractorStyle> interactor_style_;

    LayerMap layer_map_;

    int timer_id_;
    bool stopped_;
};

}  // namespace VtkRendering
