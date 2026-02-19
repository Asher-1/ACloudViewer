// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include <Utils/PCLCloud.h>

#include "base/CustomContextItem.h"
#include "qPCL.h"

// CV_DB_LIB
#include <ecvColorTypes.h>
#include <ecvDrawContext.h>
#include <ecvGenericVisualizer2D.h>

// PclUtils ImageViewer (replaces pcl::visualization::ImageViewer)
#include "base/CVImageViewer.h"

// VTK
#include <vtkSmartPointer.h>

// Forward declarations
class vtkRenderer;
class vtkRenderWindow;
class vtkRenderWindowInteractor;
#if VTK_MAJOR_VERSION >= 6
class vtkImageSlice;
class vtkImageSliceMapper;
#endif

namespace PclUtils {

/**
 * @brief 2D Image Visualizer for CloudViewer
 * 
 * This class provides 2D image visualization capabilities, supporting
 * multi-layer image compositing, text overlay, and interactive picking.
 * It extends ecvGenericVisualizer2D and PclUtils::ImageViewer.
 * 
 * Features include:
 * - Multi-layer image rendering with opacity control
 * - QImage support for Qt integration
 * - Text annotation and overlay
 * - Interactive picking
 * - Camera preservation for mode switching
 * - Window resize handling
 * 
 * @see ecvGenericVisualizer2D
 * @see PclUtils::ImageViewer
 */
class QPCL_ENGINE_LIB_API ImageVis : public ecvGenericVisualizer2D,
                                     public PclUtils::ImageViewer {
    // Q_OBJECT
public:
    /**
     * @brief Constructor for 2D image visualizer
     * @param viewerName Name identifier for this visualizer instance
     * @param autoInit Whether to automatically initialize VTK components (default: false)
     * 
     * Creates a new 2D image viewer with multi-layer support.
     * 
     * @note Constructor is public but may be used with care to ensure proper initialization.
     */
    ImageVis(const std::string& viewerName, bool autoInit = false);
    
    /**
     * @brief Set up interactor for the render window
     * @param interactor VTK render window interactor to configure
     * @param win VTK render window associated with the interactor
     * 
     * Configures the interactor style for 2D image interaction
     * (pan/zoom instead of 3D rotation).
     */
    void setupInteractor(vtkSmartPointer<vtkRenderWindowInteractor> interactor,
                         vtkSmartPointer<vtkRenderWindow> win);
    
    /**
     * @brief Set the render window
     * @param win VTK render window to use for display
     */
    void setRenderWindow(vtkSmartPointer<vtkRenderWindow> win);
    
    /**
     * @brief Get the render window
     * @return Smart pointer to the VTK render window
     */
    vtkSmartPointer<vtkRenderWindow> getRenderWindow();
    
    /**
     * @brief Get the render window interactor
     * @return Smart pointer to the VTK render window interactor
     */
    vtkSmartPointer<vtkRenderWindowInteractor> getRenderWindowInteractor();
    
    /**
     * @brief Set the render window interactor
     * @param interactor VTK render window interactor to use
     */
    void setRenderWindowInteractor(
            vtkSmartPointer<vtkRenderWindowInteractor> interactor);
    
    /**
     * @brief Get the VTK renderer
     * @return Smart pointer to the VTK renderer
     */
    vtkSmartPointer<vtkRenderer> getRender();
    
    /**
     * @brief Set the VTK renderer
     * @param render VTK renderer to use
     */
    void setRender(vtkSmartPointer<vtkRenderer> render);

    // =====================================================================
    // Layer Management
    // =====================================================================
    
    /**
     * @brief Check if a layer exists
     * @param id Unique identifier of the layer to check
     * @return true if a layer with the specified ID was found
     * 
     * Checks whether an image layer with the given ID has been added to the visualizer.
     */
    bool contains(const std::string& id) const;

    /**
     * @brief Get layer by ID
     * @param id Unique identifier of the layer
     * @return Pointer to Layer object, or nullptr if not found
     * 
     * Retrieves the layer object for direct manipulation.
     */
    Layer* getLayer(const std::string& id);
    
    /**
     * @brief Change layer opacity
     * @param opacity Opacity value (0.0=transparent, 1.0=opaque)
     * @param viewID Unique identifier of the layer
     * 
     * Adjusts the transparency of a specific layer.
     */
    void changeOpacity(double opacity, const std::string& viewID);
    
    /**
     * @brief Show or hide layer actors
     * @param visibility true to show, false to hide
     * @param viewID Unique identifier of the layer
     * 
     * Controls visibility of a layer without removing it.
     */
    void hideShowActors(bool visibility, const std::string& viewID);

    /**
     * @brief Remove a layer from the viewer
     * @param layer_id Unique identifier of the layer to remove
     * 
     * Permanently removes a layer and its associated data.
     */
    void removeLayer(const std::string& layer_id);

    /**
     * @brief Create a new 2D rendering layer
     * @param layer_id Unique identifier for the new layer
     * @param x X position of the layer in pixels
     * @param y Y position of the layer in pixels
     * @param width Width of the layer in pixels
     * @param height Height of the layer in pixels
     * @param opacity Layer opacity (0.0=transparent, 1.0=opaque, default: 0.5)
     * @param fill_box Whether to initialize with black background (default: true)
     * @return Iterator to the created layer in the layer map
     * 
     * Creates a new rendering layer at the specified position and size.
     * Layers can be stacked for compositing effects.
     */
    LayerMap::iterator createLayer(const std::string& layer_id,
                                   int x,
                                   int y,
                                   int width,
                                   int height,
                                   double opacity = 0.5,
                                   bool fill_box = true);

    // =====================================================================
    // Image Display Methods
    // =====================================================================
    
    /**
     * @brief Add RGB image to existing layer
     * @param qimage Qt image to display
     * @param x X position within the layer
     * @param y Y position within the layer
     * @param layer_id Target layer identifier
     * @param opacity Image opacity (default: 1.0)
     * 
     * Adds a QImage to an existing layer at the specified position.
     */
    void addRGBImage(const QImage& qimage,
                     unsigned x,
                     unsigned y,
                     const std::string& layer_id,
                     double opacity = 1.0);

    /**
     * @brief Add QImage as a new layer
     * @param qimage Qt image to display
     * @param layer_id Layer identifier (default: "image")
     * @param opacity Layer opacity (default: 1.0)
     * 
     * Creates a new layer and displays the QImage in it.
     * Uses ParaView-style vtkQImageToImageSource for efficient rendering.
     */
    void addQImage(const QImage& qimage,
                   const std::string& layer_id = "image",
                   double opacity = 1.0);

    /**
     * @brief Add text annotation to layer
     * @param x X position in pixels
     * @param y Y position in pixels
     * @param text_string Text to display
     * @param r Red color component (0.0-1.0)
     * @param g Green color component (0.0-1.0)
     * @param b Blue color component (0.0-1.0)
     * @param layer_id Target layer identifier (default: "line")
     * @param opacity Text opacity (default: 1.0)
     * @param fontSize Font size in points (default: 10)
     * @param bold Whether to use bold font (default: false)
     * @return true if text was added successfully
     * 
     * Adds text overlay at the specified position with customizable appearance.
     */
    bool addText(unsigned int x,
                 unsigned int y,
                 const std::string& text_string,
                 double r,
                 double g,
                 double b,
                 const std::string& layer_id = "line",
                 double opacity = 1.0,
                 int fontSize = 10,
                 bool bold = false);

public:
    // =====================================================================
    // 2D Viewer Control and Picking
    // =====================================================================
    
    /**
     * @brief Enable or disable 2D viewer mode
     * @param state true to enable 2D viewer mode
     * 
     * Switches between 2D image viewing mode and normal 3D mode.
     * In 2D mode, camera is set to orthographic projection and
     * interaction is limited to pan/zoom.
     */
    void enable2Dviewer(bool state);

    /**
     * @brief Pick item at screen coordinates
     * @param x Screen X coordinate in pixels
     * @param y Screen Y coordinate in pixels
     * @return View ID of picked item, or empty string if none
     * 
     * Performs picking at the specified screen location to identify
     * the layer or element under the cursor.
     */
    std::string pickItem(int x, int y);

private:
    // =====================================================================
    // Event Handlers
    // =====================================================================
    
    /// Internal mouse event handler
    void mouseEventProcess(const PclUtils::MouseEvent& event, void* args);
    
    /// Mouse event connection handle
    PclUtils::SignalConnection m_mouseConnection;
    
    /// Pick item from mouse event
    std::string pickItem(const PclUtils::MouseEvent& event);

    /**
     * @brief Static callback for window resize events
     * @param caller VTK object that triggered the event
     * @param eventId VTK event identifier
     * @param clientData User data (pointer to ImageVis instance)
     * @param callData Additional event data
     * @static
     */
    static void WindowResizeCallback(vtkObject* caller,
                                     unsigned long eventId,
                                     void* clientData,
                                     void* callData);
    
    /// Handle window resize event
    void onWindowResize();
    
    /// Update image scales after resize
    void updateImageScales();
    
#if VTK_MAJOR_VERSION >= 6
    /**
     * @brief Update image slice transformation
     * @param imageSlice VTK image slice to update
     * @param width Image width in pixels
     * @param height Image height in pixels
     * 
     * Adjusts the image slice transformation to maintain correct aspect ratio
     * and position after window resize.
     */
    void updateImageSliceTransform(vtkImageSlice* imageSlice,
                                   unsigned width,
                                   unsigned height);
#endif

    /**
     * @brief Set interactor style for image viewing
     * 
     * Switches to image-specific interactor style (pan/zoom instead of rotate).
     * Saves the original style for later restoration to prevent crashes
     * when switching back to 3D mode.
     */
    void setImageInteractorStyle();

    // =====================================================================
    // Member Variables
    // =====================================================================
    
    /// Main render window interactor
    vtkSmartPointer<vtkRenderWindowInteractor> m_mainInteractor;
    
    /// Callback command for window resize events
    vtkSmartPointer<vtkCallbackCommand> m_windowResizeCallback;

    /// Original interactor style (saved for restoration)
    /// Prevents crashes when getCamera is called on PCLVisualizerInteractorStyle
    vtkSmartPointer<vtkInteractorObserver> m_originalInteractorStyle;

    /**
     * @brief Camera parameters for mode switching
     * 
     * Stores original camera parameters to restore when exiting image mode.
     */
    struct CameraParams {
        bool parallelProjection;      ///< Orthographic vs perspective
        double focalPoint[3];          ///< Camera focal point
        double position[3];            ///< Camera position
        double viewUp[3];              ///< View up vector
        double viewAngle;              ///< Field of view angle
        double parallelScale;          ///< Scale for orthographic projection
        double clippingRange[2];       ///< Near/far clipping planes
    };
    
    /// Saved camera parameters
    CameraParams m_originalCameraParams;
    
    /// Whether camera parameters have been saved
    bool m_cameraParamsSaved;

    /**
     * @brief Image information per layer
     * 
     * Stores image slice info for each layer (ParaView-style rendering).
     */
    struct ImageInfo {
        unsigned originalWidth;        ///< Original image width
        unsigned originalHeight;       ///< Original image height
#if VTK_MAJOR_VERSION >= 6
        vtkSmartPointer<vtkImageSlice> imageSlice;        ///< VTK image slice
        vtkSmartPointer<vtkImageSliceMapper> imageMapper; ///< VTK image mapper
#endif
    };
    
    /// Map of image information (layer ID -> image info)
    std::map<std::string, ImageInfo> m_imageInfoMap;
};

typedef std::shared_ptr<ImageVis> ImageVisPtr;
}  // namespace PclUtils
