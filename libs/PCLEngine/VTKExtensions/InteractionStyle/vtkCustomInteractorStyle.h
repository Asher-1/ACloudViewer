// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// PclUtils types (replaces pcl::visualization types)
#include "base/CVVisualizerTypes.h"

#include <vtkCommand.h>
#include <vtkInteractorStyleRubberBandPick.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>

#include "qPCL.h"

class vtkRendererCollection;
class vtkLegendScaleActor;
class vtkScalarBarActor;
class vtkPNGWriter;
class vtkWindowToImageFilter;
class vtkPointPicker;
class vtkCameraManipulator;
class vtkCollection;

namespace VTKExtensions {

// ============================================================================
// CVPointPickingCallback - replaces pcl::visualization::PointPickingCallback
// ============================================================================
class QPCL_ENGINE_LIB_API CVPointPickingCallback : public vtkCommand {
public:
    static CVPointPickingCallback* New() {
        return new CVPointPickingCallback;
    }

    ~CVPointPickingCallback() override = default;

    void Execute(vtkObject* caller,
                 unsigned long eventid,
                 void* calldata) override;

    int performSinglePick(vtkRenderWindowInteractor* iren);

    int performSinglePick(vtkRenderWindowInteractor* iren,
                          float& x,
                          float& y,
                          float& z);

    int performAreaPick(
            vtkRenderWindowInteractor* iren,
            PclUtils::CloudActorMapPtr cam_ptr,
            std::map<std::string, std::vector<int>>& cloud_indices) const;

private:
    float x_{0.0f}, y_{0.0f}, z_{0.0f};
    int idx_{-1};
    bool pick_first_{false};
    const vtkActor* actor_{nullptr};
};

// ============================================================================
// vtkCustomInteractorStyle - standalone VTK interactor style
// (no longer inherits from pcl::visualization::PCLVisualizerInteractorStyle)
// ============================================================================
class QPCL_ENGINE_LIB_API vtkCustomInteractorStyle
    : public vtkInteractorStyleRubberBandPick {
public:
    static vtkCustomInteractorStyle* New();
    vtkTypeMacro(vtkCustomInteractorStyle, vtkInteractorStyleRubberBandPick);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /** \brief Empty constructor. */
    vtkCustomInteractorStyle();

    /** \brief Empty destructor */
    ~vtkCustomInteractorStyle() override;

    // ======== Initialization (ported from PCLVisualizerInteractorStyle) ========

    /** \brief Initialization routine. Must be called before anything else. */
    virtual void Initialize();

    // ======== Actor map management ========

    /** \brief Pass a pointer to the cloud actor map */
    inline void setCloudActorMap(const PclUtils::CloudActorMapPtr& actors) {
        cloud_actors_ = actors;
    }

    /** \brief Pass a pointer to the shape actor map */
    inline void setShapeActorMap(const PclUtils::ShapeActorMapPtr& actors) {
        shape_actors_ = actors;
    }

    /** \brief Get the cloud actor map pointer. */
    inline PclUtils::CloudActorMapPtr getCloudActorMap() {
        return cloud_actors_;
    }

    /** \brief Get the shape actor map pointer. */
    inline PclUtils::ShapeActorMapPtr getShapeActorMap() {
        return shape_actors_;
    }

    /** \brief Pass a set of renderers to the interactor style. */
    void setRendererCollection(
            vtkSmartPointer<vtkRendererCollection>& rens) {
        rens_ = rens;
    }

    // ======== Event callback registration ========

    /** \brief Register a callback function for mouse events */
    PclUtils::SignalConnection registerMouseCallback(
            std::function<void(const PclUtils::MouseEvent&)> cb);

    /** \brief Register a callback function for keyboard events */
    PclUtils::SignalConnection registerKeyboardCallback(
            std::function<void(const PclUtils::KeyboardEvent&)> cb);

    /** \brief Register a callback function for point picking events */
    PclUtils::SignalConnection registerPointPickingCallback(
            std::function<void(const PclUtils::PointPickingEvent&)> cb);

    /** \brief Register a callback function for area picking events */
    PclUtils::SignalConnection registerAreaPickingCallback(
            std::function<void(const PclUtils::AreaPickingEvent&)> cb);

    // ======== Screenshot & Camera ========

    /** \brief Save the current rendered image to disk, as a PNG screenshot. */
    void saveScreenshot(const std::string& file);

    /** \brief Save the camera parameters to disk, as a .cam file. */
    bool saveCameraParameters(const std::string& file);

    /** \brief Get camera parameters of a given viewport (0 means default). */
    void getCameraParameters(PclUtils::Camera& camera, int viewport = 0) const;

    /** \brief Load camera parameters from a camera parameter file. */
    bool loadCameraParameters(const std::string& file);

    /** \brief Set camera parameters from PclUtils::Camera struct. */
    void setCameraParameters(const PclUtils::Camera& camera, int viewport = 0);

    /** \brief Set camera file for camera parameter saving/restoring. */
    void setCameraFile(const std::string& file) { camera_file_ = file; }

    /** \brief Get camera file for camera parameter saving/restoring. */
    std::string getCameraFile() const { return camera_file_; }

    /** \brief Change the default keyboard modifier from ALT to something else. */
    inline void setKeyboardModifier(
            const PclUtils::InteractorKeyboardModifier& modifier) {
        modifier_ = modifier;
    }

    void toggleAreaPicking();

    /** \brief Set render window. */
    inline void setRenderWindow(const vtkSmartPointer<vtkRenderWindow>& win) {
        win_ = win;
    }

public:
    /**
     * Access to adding or removing manipulators.
     */
    void AddManipulator(vtkCameraManipulator* m);

    /**
     * Removes all manipulators.
     */
    void RemoveAllManipulators();

    //@{
    /**
     * Accessor for the collection of camera manipulators.
     */
    vtkGetObjectMacro(CameraManipulators, vtkCollection);
    //@}

    //@{
    /**
     * Propagates the center to the manipulators.
     */
    vtkSetVector3Macro(CenterOfRotation, double);
    vtkGetVector3Macro(CenterOfRotation, double);
    //@}

    //@{
    /**
     * Propagates the rotation factor to the manipulators.
     */
    vtkSetMacro(RotationFactor, double);
    vtkGetMacro(RotationFactor, double);
    //@}

    /**
     * Returns the chosen manipulator based on the modifiers.
     */
    virtual vtkCameraManipulator* FindManipulator(int button,
                                                  int shift,
                                                  int control);

    /**
     * Dolly the renderer's camera to a specific point
     */
    static void DollyToPosition(double fact,
                                int* position,
                                vtkRenderer* renderer);

    /**
     * Translate the renderer's camera
     */
    static void TranslateCamera(
            vtkRenderer* renderer, int toX, int toY, int fromX, int fromY);

    using vtkInteractorStyleTrackballCamera::Dolly;

protected:
    /** \brief Interactor style internal method. Zoom in. */
    void zoomIn();

    /** \brief Interactor style internal method. Zoom out. */
    void zoomOut();

    // Keyboard events
    void OnKeyDown() override;
    void OnKeyUp() override;

    /** \brief Gets called whenever a key is pressed. */
    void OnChar() override;

    // mouse button events
    void OnMouseMove() override;
    void OnLeftButtonDown() override;
    void OnLeftButtonUp() override;
    void OnMiddleButtonDown() override;
    void OnMiddleButtonUp() override;
    void OnRightButtonDown() override;
    void OnRightButtonUp() override;
    void OnMouseWheelForward() override;
    void OnMouseWheelBackward() override;

    /** \brief Gets called periodically if a timer is set. */
    void OnTimer() override;

    void Dolly(double factor) override;

    /** \brief LUT display helper */
    void updateLookUpTableDisplay(bool add_lut = false);

    /** \brief Get camera parameters from a string vector. */
    bool getCameraParameters(const std::vector<std::string>& camera);

    // ======== Data members (ported from PCLVisualizerInteractorStyle) ========
protected:
    /** \brief Set to true after initialization is complete. */
    bool init_{false};

    /** \brief Collection of vtkRenderers stored internally. */
    vtkSmartPointer<vtkRendererCollection> rens_;

    /** \brief Cloud actor map stored internally. */
    PclUtils::CloudActorMapPtr cloud_actors_{nullptr};

    /** \brief Shape map stored internally. */
    PclUtils::ShapeActorMapPtr shape_actors_{nullptr};

    /** \brief The current window width/height. */
    int win_height_{0}, win_width_{0};

    /** \brief The current window position x/y. */
    int win_pos_x_{0}, win_pos_y_{0};

    /** \brief The maximum resizeable window width/height. */
    int max_win_height_{0}, max_win_width_{0};

    /** \brief Set to true if the grid actor is enabled. */
    bool grid_enabled_{false};
    /** \brief Actor for 2D grid on screen. */
    vtkSmartPointer<vtkLegendScaleActor> grid_actor_;

    /** \brief Set to true if the LUT actor is enabled. */
    bool lut_enabled_{false};
    /** \brief Actor for 2D lookup table on screen. */
    vtkSmartPointer<vtkScalarBarActor> lut_actor_;

    /** \brief A PNG writer for screenshot captures. */
    vtkSmartPointer<vtkPNGWriter> snapshot_writer_;
    /** \brief Internal window to image filter. */
    vtkSmartPointer<vtkWindowToImageFilter> wif_;
    /** \brief Stores the point picker when switching to an area picker. */
    vtkSmartPointer<vtkPointPicker> point_picker_;

    PclUtils::Signal<void(const PclUtils::MouseEvent&)> mouse_signal_;
    PclUtils::Signal<void(const PclUtils::KeyboardEvent&)> keyboard_signal_;
    PclUtils::Signal<void(const PclUtils::PointPickingEvent&)>
            point_picking_signal_;
    PclUtils::Signal<void(const PclUtils::AreaPickingEvent&)>
            area_picking_signal_;

    /** \brief True if we're using red-blue colors for anaglyphic stereo. */
    bool stereo_anaglyph_mask_default_{false};

    /** \brief A VTK Mouse Callback object, used for point picking. */
    vtkSmartPointer<CVPointPickingCallback> mouse_callback_;

    /** \brief The keyboard modifier to use. Default: Alt. */
    PclUtils::InteractorKeyboardModifier modifier_{};

    /** \brief Camera file for saving/restoring. */
    std::string camera_file_;
    /** \brief Camera struct for saving/restoring. */
    PclUtils::Camera camera_;
    /** \brief Whether camera has been saved. */
    bool camera_saved_{false};
    /** \brief The render window (used when interactor not available). */
    vtkSmartPointer<vtkRenderWindow> win_;

    /** \brief LUT actor ID */
    std::string lut_actor_id_;

    // Camera manipulator members
    vtkCameraManipulator* CurrentManipulator;
    double CenterOfRotation[3];
    double RotationFactor;
    vtkCollection* CameraManipulators;

    void OnButtonDown(int button, int shift, int control);
    void OnButtonUp(int button);
    void ResetLights();

    friend class CVPointPickingCallback;

    vtkCustomInteractorStyle(const vtkCustomInteractorStyle&) = delete;
    void operator=(const vtkCustomInteractorStyle&) = delete;
};

}  // namespace VTKExtensions
