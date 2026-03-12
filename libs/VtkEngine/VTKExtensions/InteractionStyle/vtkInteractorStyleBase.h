// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkInteractorStyleBase.h
 * @brief Base interactor style for CloudViewer, providing actor maps and camera
 * helpers.
 */

#include <VtkRendering/Core/ActorMap.h>
#include <VtkRendering/Core/EventCallbacks.h>
#include <VtkRendering/Core/KeyboardEvent.h>
#include <VtkRendering/Core/MouseEvent.h>
#include <vtkInteractorStyleRubberBandPick.h>
#include <vtkLegendScaleActor.h>
#include <vtkPNGWriter.h>
#include <vtkPointPicker.h>
#include <vtkRenderWindow.h>
#include <vtkRendererCollection.h>
#include <vtkScalarBarActor.h>
#include <vtkSmartPointer.h>
#include <vtkWindowToImageFilter.h>

#include <string>

#include "qVTK.h"

namespace VTKExtensions {

/**
 * @class vtkInteractorStyleBase
 * @brief Base interactor style replacing PCLVisualizerInteractorStyle.
 *
 * Provides all the member variables and camera helpers that the PCL base class
 * used to supply, but without any PCL dependency.
 */
class QVTK_ENGINE_LIB_API vtkInteractorStyleBase
    : public vtkInteractorStyleRubberBandPick {
public:
    vtkTypeMacro(vtkInteractorStyleBase, vtkInteractorStyleRubberBandPick);

    /// @param actors Cloud actor map to set
    inline void setCloudActorMap(const VtkRendering::CloudActorMapPtr& actors) {
        cloud_actors_ = actors;
    }
    /// @param actors Shape actor map to set
    inline void setShapeActorMap(const VtkRendering::ShapeActorMapPtr& actors) {
        shape_actors_ = actors;
    }
    /// @return Cloud actor map
    inline VtkRendering::CloudActorMapPtr getCloudActorMap() {
        return cloud_actors_;
    }
    /// @return Shape actor map
    inline VtkRendering::ShapeActorMapPtr getShapeActorMap() {
        return shape_actors_;
    }

    /// @param rens Renderer collection to set
    void setRendererCollection(vtkSmartPointer<vtkRendererCollection>& rens) {
        rens_ = rens;
    }

    /// @param use_vbos Whether to use VBOs
    inline void setUseVbos(bool use_vbos) { use_vbos_ = use_vbos; }

    /// @param modifier Keyboard modifier for interactor
    inline void setKeyboardModifier(
            VtkRendering::InteractorKeyboardModifier modifier) {
        modifier_ = modifier;
    }

    /// @param file Path to camera file
    void setCameraFile(const std::string& file) { camera_file_ = file; }
    /// @return Path to camera file
    std::string getCameraFile() const { return camera_file_; }

    virtual void Initialize();

    /// @param file Output path for screenshot
    void saveScreenshot(const std::string& file);
    /// @param file Output path for camera parameters
    /// @return true if saved successfully
    bool saveCameraParameters(const std::string& file);
    /// @param file Path to camera parameters file
    /// @return true if loaded successfully
    bool loadCameraParameters(const std::string& file);

protected:
    vtkInteractorStyleBase();
    ~vtkInteractorStyleBase() override = default;

    bool init_{false};

    vtkSmartPointer<vtkRendererCollection> rens_;

    VtkRendering::CloudActorMapPtr cloud_actors_;
    VtkRendering::ShapeActorMapPtr shape_actors_;

    int win_height_{0}, win_width_{0};
    int win_pos_x_{0}, win_pos_y_{0};
    int max_win_height_{0}, max_win_width_{0};

    bool use_vbos_{false};

    bool grid_enabled_{false};
    vtkSmartPointer<vtkLegendScaleActor> grid_actor_;

    bool lut_enabled_{false};
    vtkSmartPointer<vtkScalarBarActor> lut_actor_;

    vtkSmartPointer<vtkPNGWriter> snapshot_writer_;
    vtkSmartPointer<vtkWindowToImageFilter> wif_;
    vtkSmartPointer<vtkPointPicker> point_picker_;

    VtkRendering::Signal<const VtkRendering::MouseEvent&> mouse_signal_;
    VtkRendering::Signal<const VtkRendering::KeyboardEvent&> keyboard_signal_;

    bool stereo_anaglyph_mask_default_{false};

    VtkRendering::InteractorKeyboardModifier modifier_{
            VtkRendering::INTERACTOR_KB_MOD_ALT};

    std::string camera_file_;
    bool camera_saved_{false};

    /// Saved camera state (position, focal, viewup, clipping range)
    double saved_cam_pos_[3]{0, 0, 0};
    double saved_cam_focal_[3]{0, 0, 1};
    double saved_cam_viewup_[3]{0, -1, 0};
    double saved_cam_clip_[2]{0.01, 1000.0};

    vtkSmartPointer<vtkRenderWindow> win_;

    void setRenderWindow(const vtkSmartPointer<vtkRenderWindow>& win) {
        win_ = win;
    }

private:
    vtkInteractorStyleBase(const vtkInteractorStyleBase&) = delete;
    void operator=(const vtkInteractorStyleBase&) = delete;
};

}  // namespace VTKExtensions
