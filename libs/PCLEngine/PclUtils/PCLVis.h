// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "base/CVVisualizerTypes.h"
#include "base/WidgetMap.h"
#include "qPCL.h"

// Forward declaration
namespace PclUtils {
namespace renders {
class TextureRenderManager;
}
}  // namespace PclUtils

// CV_DB_LIB
#include <ecvColorTypes.h>
#include <ecvDisplayTools.h>  // For AxesGridProperties
#include <ecvDrawContext.h>
#include <ecvGenericVisualizer3D.h>
#include <ecvHObject.h>

// VTK
#include <vtkBoundingBox.h>  // needed for iVar
#include <vtkSmartPointer.h>

class vtkAlgorithmOutput;
class vtkDataArray;
class vtkLODActor;
class vtkCamera;
class vtkPolyData;
class vtkRender;
class vtkPointPicker;
class vtkAreaPicker;
class vtkPropPicker;
class vtkAbstractWidget;
class vtkRenderWindow;
class vtkRenderWindowInteractor;
class vtkRendererCollection;
class vtkRenderer;
class vtkMatrix4x4;
class vtkLightKit;
class vtkCubeAxesActor;
class vtkCameraOrientationWidget;
class vtkOrientationMarkerWidget;
class vtkFollower;
class vtkTextActor;
class ccGenericMesh;
class ccPointCloud;
class ccMesh;
class ccPolyline;
class ccHObject;
class ccBBox;
class ecvOrientedBBox;
class ccSensor;
class ecvPointpickingTools;
class ccMaterial;
class ccMaterialSet;

namespace cloudViewer {
namespace geometry {
class LineSet;
}
}  // namespace cloudViewer

namespace VTKExtensions {
class vtkPVCenterAxesActor;
class vtkCustomInteractorStyle;
}  // namespace VTKExtensions

namespace PclUtils {

/**
 * @brief PCL-based 3D Visualizer for CloudViewer
 *
 * This class provides a comprehensive 3D visualization framework built on VTK,
 * supporting point clouds, meshes, shapes, and advanced rendering features.
 * It extends ecvGenericVisualizer3D and provides PCL-compatible APIs.
 *
 * Features include:
 * - Point cloud and mesh rendering
 * - Texture mapping support
 * - Camera manipulation and control
 * - Interactive picking (point, area, actor)
 * - Multiple viewports
 * - Coordinate systems and axes
 * - Text and caption rendering
 * - Screenshot capture
 *
 * @see ecvGenericVisualizer3D
 */
class QPCL_ENGINE_LIB_API PCLVis : public ecvGenericVisualizer3D {
    Q_OBJECT
public:
    /**
     * @brief Default constructor (deprecated)
     * @deprecated Use the renderer/window constructor instead
     * @param interactor_style Custom VTK interactor style for user interaction
     * @param viewerName Name identifier for this visualizer instance
     * @param initIterator Whether to initialize the render window interactor
     * @param argc Command line argument count (for VTK initialization)
     * @param argv Command line arguments (for VTK initialization)
     */
    PCLVis(vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                   interactor_style,
           const std::string& viewerName = "",
           bool initIterator = false,
           int argc = 0,
           char** argv = nullptr);

    /**
     * @brief Constructor with existing renderer and window
     * @param ren VTK renderer to use for visualization
     * @param wind VTK render window for display output
     * @param interactor_style Custom VTK interactor style for user interaction
     * @param viewerName Name identifier for this visualizer instance
     * @param initIterator Whether to initialize the render window interactor
     * @param argc Command line argument count (for VTK initialization)
     * @param argv Command line arguments (for VTK initialization)
     */
    PCLVis(vtkSmartPointer<vtkRenderer> ren,
           vtkSmartPointer<vtkRenderWindow> wind,
           vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                   interactor_style,
           const std::string& viewerName = "",
           bool initIterator = false,
           int argc = 0,
           char** argv = nullptr);

    /**
     * @brief Virtual destructor
     *
     * Cleans up all VTK resources, actors, and internal state.
     */
    virtual ~PCLVis();

    /**
     * @brief Perform initialization tasks
     *
     * Sets up the visualization environment, including renderers,
     * lighting, and default settings. Should be called after construction.
     */
    void initialize();

    /**
     * @brief Configure the center axes display
     *
     * Sets up the center axes actor with appropriate scale and visibility
     * for scene orientation reference.
     */
    void configCenterAxes();

    /**
     * @brief Configure the interactor style
     * @param interactor_style Custom interactor style to apply
     *
     * Updates the current interactor style used for camera manipulation
     * and user interaction.
     */
    void configInteractorStyle(
            vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                    interactor_style);

public:
    // =====================================================================
    // Methods previously inherited from pcl::visualization::PCLVisualizer
    // Now reimplemented using direct VTK calls.
    // =====================================================================

    /**
     * @brief Check if an entity with the given ID exists
     * @param id Unique identifier to search for
     * @return true if a cloud, shape, or widget with the ID exists
     *
     * Searches across all actor maps (clouds, shapes, widgets) to
     * determine if the specified ID is currently in use.
     */
    bool contains(const std::string& id) const;

    /**
     * @brief Get the cloud actor map
     * @return Shared pointer to the cloud actor map
     *
     * Provides access to the internal map storing all point cloud actors.
     */
    inline PclUtils::CloudActorMapPtr getCloudActorMap() {
        return cloud_actor_map_;
    }

    /**
     * @brief Get the shape actor map
     * @return Shared pointer to the shape actor map
     *
     * Provides access to the internal map storing all shape actors
     * (lines, spheres, cubes, etc.).
     */
    inline PclUtils::ShapeActorMapPtr getShapeActorMap() {
        return shape_actor_map_;
    }

    /**
     * @brief Get the renderer collection
     * @return Smart pointer to the VTK renderer collection
     *
     * Returns the collection of all VTK renderers managed by this visualizer.
     */
    inline vtkSmartPointer<vtkRendererCollection> getRendererCollection() {
        return rens_;
    }

    /**
     * @brief Get the render window
     * @return Smart pointer to the VTK render window
     *
     * Returns the VTK render window used for display output.
     */
    inline vtkSmartPointer<vtkRenderWindow> getRenderWindow() { return win_; }

    /**
     * @brief Get the current interactor style
     * @return Smart pointer to the custom interactor style
     *
     * Returns the currently active interactor style used for
     * camera manipulation and user interaction.
     */
    inline vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
    getInteractorStyle() {
        return m_interactorStyle;
    }

    /**
     * @brief Remove a point cloud from the visualizer
     * @param id Unique identifier of the point cloud to remove
     * @param viewport Viewport ID (default: 0)
     * @return true if successfully removed, false if not found
     *
     * Removes the point cloud actor and associated data from the specified
     * viewport.
     */
    bool removePointCloud(const std::string& id, int viewport = 0);

    /**
     * @brief Remove a shape from the visualizer
     * @param id Unique identifier of the shape to remove
     * @param viewport Viewport ID (default: 0)
     * @return true if successfully removed, false if not found
     *
     * Removes shape actors (lines, spheres, cubes, etc.) from the specified
     * viewport.
     */
    bool removeShape(const std::string& id, int viewport = 0);

    /**
     * @brief Remove all point clouds from the visualizer
     * @param viewport Viewport ID (default: 0 removes from all viewports)
     * @return true if any clouds were removed
     *
     * Clears all point cloud actors from the specified viewport.
     */
    bool removeAllPointClouds(int viewport = 0);

    /**
     * @brief Remove all shapes from the visualizer
     * @param viewport Viewport ID (default: 0 removes from all viewports)
     * @return true if any shapes were removed
     *
     * Clears all shape actors from the specified viewport.
     */
    bool removeAllShapes(int viewport = 0);

    /**
     * @brief Remove a polygon mesh from the visualizer
     * @param id Unique identifier of the mesh to remove
     * @param viewport Viewport ID (default: 0)
     * @return true if successfully removed, false if not found
     *
     * Removes mesh actors from the specified viewport.
     */
    bool removePolygonMesh(const std::string& id, int viewport = 0);

    /**
     * @brief Set rendering properties for a point cloud (single value)
     * @param property Property type (CV_VISUALIZER_POINT_SIZE,
     * CV_VISUALIZER_OPACITY, etc.)
     * @param val1 Property value
     * @param id Unique identifier of the point cloud
     * @param viewport Viewport ID (default: 0)
     * @return true if property was set successfully
     *
     * Supported properties:
     * - CV_VISUALIZER_POINT_SIZE: Point size in pixels
     * - CV_VISUALIZER_OPACITY: Transparency (0.0-1.0)
     * - CV_VISUALIZER_REPRESENTATION: Point/wireframe/surface
     * - CV_VISUALIZER_SHADING: Flat/Gouraud/Phong shading
     */
    bool setPointCloudRenderingProperties(int property,
                                          double val1,
                                          const std::string& id,
                                          int viewport = 0);

    /**
     * @brief Set rendering properties for a point cloud (RGB color)
     * @param property Property type (typically CV_VISUALIZER_COLOR)
     * @param val1 Red component (0.0-1.0)
     * @param val2 Green component (0.0-1.0)
     * @param val3 Blue component (0.0-1.0)
     * @param id Unique identifier of the point cloud
     * @param viewport Viewport ID (default: 0)
     * @return true if property was set successfully
     */
    bool setPointCloudRenderingProperties(int property,
                                          double val1,
                                          double val2,
                                          double val3,
                                          const std::string& id,
                                          int viewport = 0);

    /**
     * @brief Get rendering properties for a point cloud
     * @param property Property type to query
     * @param value Output parameter for property value
     * @param id Unique identifier of the point cloud
     * @return true if property was retrieved successfully
     */
    bool getPointCloudRenderingProperties(int property,
                                          double& value,
                                          const std::string& id);

    /**
     * @brief Set rendering properties for a shape (single value)
     * @param property Property type (size, opacity, line width, etc.)
     * @param val1 Property value
     * @param id Unique identifier of the shape
     * @param viewport Viewport ID (default: 0)
     * @return true if property was set successfully
     */
    bool setShapeRenderingProperties(int property,
                                     double val1,
                                     const std::string& id,
                                     int viewport = 0);

    /**
     * @brief Set rendering properties for a shape (RGB color)
     * @param property Property type (typically CV_VISUALIZER_COLOR)
     * @param val1 Red component (0.0-1.0)
     * @param val2 Green component (0.0-1.0)
     * @param val3 Blue component (0.0-1.0)
     * @param id Unique identifier of the shape
     * @param viewport Viewport ID (default: 0)
     * @return true if property was set successfully
     */
    bool setShapeRenderingProperties(int property,
                                     double val1,
                                     double val2,
                                     double val3,
                                     const std::string& id,
                                     int viewport = 0);

    /**
     * @brief Add 2D text overlay to the visualizer
     * @param text Text string to display
     * @param xpos X position in screen pixels
     * @param ypos Y position in screen pixels
     * @param fontsize Font size in points
     * @param color Text color (RGB)
     * @param id Unique identifier for this text (default: "text")
     * @param viewport Viewport ID (default: 0)
     * @return true if text was added successfully
     *
     * Adds a 2D text overlay anchored at screen coordinates.
     */
    bool addText(const std::string& text,
                 int xpos,
                 int ypos,
                 int fontsize,
                 const ecvColor::Rgbf& color,
                 const std::string& id = "text",
                 int viewport = 0);

    /**
     * @brief Update existing 2D text overlay
     * @param text New text string
     * @param xpos New X position in screen pixels
     * @param ypos New Y position in screen pixels
     * @param id Unique identifier of the text to update (default: "text")
     * @return true if text was updated successfully
     */
    bool updateText(const std::string& text,
                    int xpos,
                    int ypos,
                    const std::string& id = "text");

    /**
     * @brief Add 3D text at a world-space position
     * @param text Text string to display
     * @param position 3D world coordinates for text placement
     * @param textScale Scale factor for text size (default: 1.0)
     * @param color Text color (RGB, default: white)
     * @param id Unique identifier for this text (default: "text3d")
     * @param viewport Viewport ID (default: 0)
     * @return true if text was added successfully
     *
     * Places 3D text in world space that transforms with the scene.
     */
    bool addText3D(const std::string& text,
                   const CCVector3d& position,
                   double textScale = 1.0,
                   const ecvColor::Rgbf& color = ecvColor::Rgbf(1.0f,
                                                                1.0f,
                                                                1.0f),
                   const std::string& id = "text3d",
                   int viewport = 0);

    /**
     * @brief Get camera parameters
     * @param camera Output parameter for camera state
     * @param viewport Viewport ID (default: 0)
     *
     * Retrieves current camera position, focal point, view up vector,
     * and projection parameters.
     */
    void getCameraParameters(PclUtils::Camera& camera, int viewport = 0) const;

    /**
     * @brief Set camera parameters
     * @param camera Camera state to apply
     * @param viewport Viewport ID (default: 0)
     *
     * Applies camera position, focal point, view up vector,
     * and projection parameters.
     */
    void setCameraParameters(const PclUtils::Camera& camera, int viewport = 0);

    /**
     * @brief Register mouse event callback
     * @param cb Callback function for mouse events
     * @return Connection handle for managing callback lifetime
     *
     * Registers a callback to receive mouse button, move, and wheel events.
     */
    SignalConnection registerMouseCallback(
            std::function<void(const PclUtils::MouseEvent&)> cb);

    /**
     * @brief Register keyboard event callback
     * @param cb Callback function for keyboard events
     * @return Connection handle for managing callback lifetime
     *
     * Registers a callback to receive key press and release events.
     */
    SignalConnection registerKeyboardCallback(
            std::function<void(const PclUtils::KeyboardEvent&)> cb);

    /**
     * @brief Register point picking callback
     * @param cb Callback function for point picking events
     * @return Connection handle for managing callback lifetime
     *
     * Registers a callback to receive events when a 3D point is picked.
     */
    SignalConnection registerPointPickingCallback(
            std::function<void(const PclUtils::PointPickingEvent&)> cb);

    /**
     * @brief Register area picking callback
     * @param cb Callback function for area picking events
     * @return Connection handle for managing callback lifetime
     *
     * Registers a callback to receive events when a rectangular area is
     * selected.
     */
    SignalConnection registerAreaPickingCallback(
            std::function<void(const PclUtils::AreaPickingEvent&)> cb);

    /**
     * @brief Reset camera to view a specific cloud
     * @param id Unique identifier of the cloud (default: "cloud")
     *
     * Adjusts camera to frame the specified cloud optimally.
     */
    void resetCameraViewpoint(const std::string& id = "cloud");

    /**
     * @brief Create a new viewport
     * @param xmin Normalized X minimum coordinate (0.0-1.0)
     * @param ymin Normalized Y minimum coordinate (0.0-1.0)
     * @param xmax Normalized X maximum coordinate (0.0-1.0)
     * @param ymax Normalized Y maximum coordinate (0.0-1.0)
     * @param viewport Output parameter for new viewport ID
     *
     * Creates a new viewport region for multi-view rendering.
     */
    void createViewPort(
            double xmin, double ymin, double xmax, double ymax, int& viewport);

    /**
     * @brief Add coordinate system axes
     * @param scale Scale factor for axes size (default: 1.0)
     * @param id Unique identifier (default: "reference")
     * @param viewport Viewport ID (default: 0)
     *
     * Adds RGB axes (X=red, Y=green, Z=blue) at the origin.
     */
    void addCoordinateSystem(double scale = 1.0,
                             const std::string& id = "reference",
                             int viewport = 0);

    /**
     * @brief Add transformed coordinate system axes
     * @param scale Scale factor for axes size
     * @param t Transformation matrix (position and orientation)
     * @param id Unique identifier (default: "reference")
     * @param viewport Viewport ID (default: 0)
     *
     * Adds RGB axes at a specified pose in world space.
     */
    void addCoordinateSystem(double scale,
                             const Eigen::Affine3f& t,
                             const std::string& id = "reference",
                             int viewport = 0);

    /**
     * @brief Remove coordinate system axes
     * @param id Unique identifier (default: "reference")
     * @param viewport Viewport ID (default: 0)
     * @return true if coordinate system was removed
     */
    bool removeCoordinateSystem(const std::string& id = "reference",
                                int viewport = 0);

    /**
     * @brief Set camera position and orientation
     * @param pos Camera position in world coordinates
     * @param focal Focal point (look-at target) in world coordinates
     * @param up View up vector (camera's up direction)
     * @param viewport Viewport ID (default: 0)
     *
     * Sets complete camera pose using position, focal point, and up vector.
     */
    void setCameraPosition(const CCVector3d& pos,
                           const CCVector3d& focal,
                           const CCVector3d& up,
                           int viewport = 0);

    /**
     * @brief Set camera position with default focal point
     * @param pos Camera position in world coordinates
     * @param up View up vector (camera's up direction)
     * @param viewport Viewport ID (default: 0)
     *
     * Sets camera position and up vector; focal point remains unchanged.
     */
    void setCameraPosition(const CCVector3d& pos,
                           const CCVector3d& up,
                           int viewport = 0);

    /**
     * @brief Save camera parameters to file
     * @param file Output file path
     *
     * Serializes camera state for later restoration.
     */
    void saveCameraParameters(const std::string& file);

    /**
     * @brief Load camera parameters from file
     * @param file Input file path
     *
     * Restores camera state from a previously saved file.
     */
    void loadCameraParameters(const std::string& file);

    /**
     * @brief Set full screen mode
     * @param state true to enable full screen, false for windowed
     */
    void setFullScreen(bool state);

    /**
     * @brief Set camera clipping planes
     * @param znear Near clipping distance (front plane)
     * @param zfar Far clipping distance (back plane)
     * @param viewport Viewport ID (default: 0)
     *
     * Controls depth range for rendering. Objects outside this range are
     * clipped.
     */
    void setCameraClipDistances(double znear, double zfar, int viewport = 0);

    /**
     * @brief Set camera field of view
     * @param fovy Vertical field of view in radians
     * @param viewport Viewport ID (default: 0)
     *
     * Sets the perspective camera's field of view angle.
     */
    void setCameraFieldOfView(double fovy, int viewport = 0);

    /**
     * @brief Save screenshot to file
     * @param file Output image file path (format determined by extension)
     *
     * Captures the current render window to an image file (PNG, JPEG, etc.).
     */
    void saveScreenshot(const std::string& file);

    /**
     * @brief Enable/disable VBO rendering
     * @param useVbos true to enable VBO (Vertex Buffer Objects)
     *
     * @note This is a no-op on modern VTK versions (VBOs are always used).
     */
    void setUseVbos(bool useVbos);

    /**
     * @brief Set the entity ID for lookup table display
     * @param viewID Unique identifier of the cloud/shape
     *
     * Specifies which entity's scalar field is displayed in the color bar.
     */
    void setLookUpTableID(const std::string& viewID);

    /**
     * @brief Add an axis-aligned bounding box
     * @param minPt Minimum corner point (x_min, y_min, z_min)
     * @param maxPt Maximum corner point (x_max, y_max, z_max)
     * @param color RGB color (default: white)
     * @param id Unique identifier (default: "cube")
     * @param viewport Viewport ID (default: 0)
     * @return true if cube was added successfully
     *
     * Adds a wireframe or solid cube aligned with coordinate axes.
     */
    bool addCube(const CCVector3d& minPt,
                 const CCVector3d& maxPt,
                 const ecvColor::Rgbf& color = ecvColor::Rgbf(1.0f, 1.0f, 1.0f),
                 const std::string& id = "cube",
                 int viewport = 0);

    /**
     * @brief Add a 3D line segment
     * @param p1 Start point in world coordinates
     * @param p2 End point in world coordinates
     * @param color RGB color for the line
     * @param id Unique identifier (default: "line")
     * @param viewport Viewport ID (default: 0)
     * @return true if line was added successfully
     *
     * Adds a straight line between two 3D points.
     */
    bool addLine(const CCVector3d& p1,
                 const CCVector3d& p2,
                 const ecvColor::Rgbf& color,
                 const std::string& id = "line",
                 int viewport = 0);

    /**
     * @brief Add a 3D sphere
     * @param center Center position in world coordinates
     * @param radius Sphere radius
     * @param color RGB color (default: white)
     * @param id Unique identifier (default: "sphere")
     * @param viewport Viewport ID (default: 0)
     * @return true if sphere was added successfully
     *
     * Adds a sphere primitive at the specified location.
     */
    bool addSphere(const CCVector3d& center,
                   double radius,
                   const ecvColor::Rgbf& color = ecvColor::Rgbf(1.0f,
                                                                1.0f,
                                                                1.0f),
                   const std::string& id = "sphere",
                   int viewport = 0);

    /**
     * @brief Add a point cloud from VTK polydata
     * @param polydata VTK polydata containing point geometry
     * @param colors Optional VTK color array (unsigned char RGB, 3 components)
     * @param id Unique identifier for the cloud
     * @param viewport Viewport ID (default: 0)
     * @return true if point cloud was added successfully
     *
     * Adds a point cloud directly from pre-built VTK data structures.
     * This is a low-level method for VTK pipeline integration.
     */
    bool addPointCloud(vtkSmartPointer<vtkPolyData> polydata,
                       vtkSmartPointer<vtkDataArray> colors,
                       const std::string& id,
                       int viewport = 0);

    /**
     * @brief Update an existing point cloud from VTK polydata
     * @param polydata VTK polydata containing updated geometry
     * @param colors Optional VTK color array (unsigned char RGB, 3 components)
     * @param id Unique identifier of the cloud to update
     * @return true if point cloud was updated successfully
     *
     * Updates an existing point cloud's geometry and colors without recreating
     * the actor.
     */
    bool updatePointCloud(vtkSmartPointer<vtkPolyData> polydata,
                          vtkSmartPointer<vtkDataArray> colors,
                          const std::string& id);

    // =====================================================================
    // End of formerly-inherited methods
    // =====================================================================

public:
    /**
     * @brief Hide PCL marker axes widget
     *
     * Hides the orientation marker axes display.
     */
    void hidePclMarkerAxes();

    /**
     * @brief Check if PCL marker axes are visible
     * @return true if marker axes are currently shown
     */
    bool pclMarkerAxesShown();

    /**
     * @brief Show PCL marker axes widget
     * @param interactor Render window interactor (optional, uses default if
     * nullptr)
     *
     * Displays the orientation marker axes in the corner of the viewport.
     */
    void showPclMarkerAxes(vtkRenderWindowInteractor* interactor = nullptr);

    /**
     * @brief Hide orientation marker widget axes
     *
     * Hides the VTK orientation marker widget.
     */
    void hideOrientationMarkerWidgetAxes();

    /**
     * @brief Show orientation marker widget axes
     * @param interactor Render window interactor to attach the widget
     *
     * Displays the VTK orientation marker widget for scene orientation
     * reference.
     */
    void showOrientationMarkerWidgetAxes(vtkRenderWindowInteractor* interactor);

    /**
     * @brief Toggle orientation marker widget visibility
     *
     * Switches between showing and hiding the orientation marker widget.
     */
    void toggleOrientationMarkerWidgetAxes();

    /**
     * @brief Remove a VTK actor from the renderer
     * @param actor Smart pointer to the VTK prop/actor to remove
     * @param viewport Viewport ID (default: 0 removes from all viewports)
     * @return true if actor was removed successfully
     *
     * Low-level method to remove actors from the rendering pipeline.
     */
    bool removeActorFromRenderer(const vtkSmartPointer<vtkProp>& actor,
                                 int viewport = 0);

    /**
     * @brief Add a VTK actor to the renderer
     * @param actor Smart pointer to the VTK prop/actor to add
     * @param viewport Viewport ID (default: 0 adds to all viewports)
     *
     * Low-level method to add actors directly to the rendering pipeline.
     */
    void addActorToRenderer(const vtkSmartPointer<vtkProp>& actor,
                            int viewport = 0);

    /**
     * @brief Force render window update
     *
     * Triggers an immediate refresh of the render window.
     * Call this after making changes to actors or properties that
     * don't automatically trigger a render.
     */
    void UpdateScreen();

    /**
     * @brief Configure the render window interactor
     * @param iren VTK render window interactor
     * @param win VTK render window
     *
     * Sets up the interactor with the render window and configures
     * event handling and interaction styles.
     */
    void setupInteractor(vtkRenderWindowInteractor* iren, vtkRenderWindow* win);

    /**
     * @brief Get the render window interactor
     * @return Smart pointer to the VTK render window interactor
     *
     * Provides access to the interactor for advanced control of
     * user interaction and events.
     */
    inline vtkSmartPointer<vtkRenderWindowInteractor>
    getRenderWindowInteractor() {
        return (interactor_);
    }

    // =====================================================================
    // Camera Tools
    // =====================================================================

    /**
     * @brief Get camera state as PclUtils::Camera
     * @param viewport Viewport ID (default: 0)
     * @return Camera object containing position, focal point, and view
     * parameters
     */
    PclUtils::Camera getCamera(int viewport = 0);

    /**
     * @brief Get VTK camera object
     * @param viewport Viewport ID (default: 0)
     * @return Smart pointer to the VTK camera
     *
     * Provides direct access to the underlying VTK camera for advanced control.
     */
    vtkSmartPointer<vtkCamera> getVtkCamera(int viewport = 0);

    /**
     * @brief Set model-view transformation matrix
     * @param viewMat 4x4 transformation matrix
     * @param viewport Viewport ID (default: 0)
     *
     * Applies a model-view matrix to the camera for custom transformations.
     */
    void setModelViewMatrix(const ccGLMatrixd& viewMat, int viewport = 0);

    /**
     * @brief Get parallel projection scale
     * @param viewport Viewport ID (default: 0)
     * @return Current parallel scale value
     *
     * Returns the scale factor for orthographic/parallel projection.
     */
    double getParallelScale(int viewport = 0);

    /**
     * @brief Set parallel projection scale
     * @param scale Scale factor for orthographic projection
     * @param viewport Viewport ID (default: 0)
     *
     * Controls zoom level in orthographic projection mode.
     */
    void setParallelScale(double scale, int viewport = 0);

    /**
     * @brief Enable orthographic projection
     * @param viewport Viewport ID (default: 0)
     *
     * Switches to parallel/orthographic projection (no perspective distortion).
     */
    void setOrthoProjection(int viewport = 0);

    /**
     * @brief Enable perspective projection
     * @param viewport Viewport ID (default: 0)
     *
     * Switches to perspective projection (realistic depth perception).
     */
    void setPerspectiveProjection(int viewport = 0);

    /**
     * @brief Check if perspective projection is active
     * @param viewport Viewport ID (default: 0)
     * @return true if using perspective projection, false if orthographic
     */
    bool getPerspectiveState(int viewport = 0);

    /**
     * @brief Get auto-update camera position flag
     * @return true if camera position updates automatically
     */
    inline bool getAutoUpateCameraPos() { return m_autoUpdateCameraPos; }

    /**
     * @brief Set auto-update camera position flag
     * @param state true to enable automatic camera position updates
     */
    inline void setAutoUpateCameraPos(bool state) {
        m_autoUpdateCameraPos = state;
    }

    /**
     * @brief Rotate camera around a custom axis
     * @param pos Screen position for rotation center (2D pixel coordinates)
     * @param axis 3D axis of rotation in world coordinates
     * @param angle Rotation angle in degrees
     * @param viewport Viewport ID (default: 0)
     *
     * Performs camera rotation around a specified 3D axis anchored at screen
     * position.
     */
    void rotateWithAxis(const CCVector2i& pos,
                        const CCVector3d& axis,
                        double angle,
                        int viewport = 0);

public:
    // =====================================================================
    // Rotation Center and Camera Manipulation
    // =====================================================================

    /**
     * @brief Get the current center of rotation
     * @param center Output array for rotation center coordinates [x, y, z]
     *
     * Retrieves the 3D point around which camera rotation occurs.
     */
    void getCenterOfRotation(double center[3]);

    /**
     * @brief Reset center of rotation to focal point
     * @param viewport Viewport ID (default: 0)
     *
     * Sets the rotation center back to the camera's focal point.
     */
    void resetCenterOfRotation(int viewport = 0);

    /**
     * @brief Expand bounding box by transformation matrix
     * @param bounds Bounding box to expand [xmin, xmax, ymin, ymax, zmin, zmax]
     * @param matrix 4x4 transformation matrix
     * @static
     *
     * Utility method to compute transformed bounding box extents.
     */
    static void ExpandBounds(double bounds[6], vtkMatrix4x4* matrix);

    /**
     * @brief Set the center of rotation
     * @param x X coordinate of rotation center
     * @param y Y coordinate of rotation center
     * @param z Z coordinate of rotation center
     *
     * Defines the 3D point around which camera rotation occurs.
     * Requires appropriate interaction style to take effect.
     */
    void setCenterOfRotation(double x, double y, double z);

    /**
     * @brief Set the center of rotation (array overload)
     * @param xyz Array containing rotation center coordinates [x, y, z]
     */
    inline void setCenterOfRotation(double xyz[3]) {
        setCenterOfRotation(xyz[0], xyz[1], xyz[2]);
    }

    /**
     * @brief Set rotation speed factor
     * @param factor Rotation speed multiplier (higher = faster rotation)
     */
    void setRotationFactor(double factor);

    /**
     * @brief Get rotation speed factor
     * @return Current rotation speed multiplier
     */
    double getRotationFactor();

    // =====================================================================
    // Center Axes and Interactor Style Configuration
    // =====================================================================

    /**
     * @brief Set center axes visibility
     * @param visible true to show center axes, false to hide
     *
     * Controls visibility of the scene center axes indicator.
     */
    void setCenterAxesVisibility(bool visible);

    /**
     * @brief Set 2D camera manipulators
     * @param manipulators Array of 9 integers defining mouse/key button
     * mappings
     * @virtual
     *
     * Configures camera control behavior for 2D visualization mode.
     */
    virtual void setCamera2DManipulators(const int manipulators[9]);

    /**
     * @brief Set 3D camera manipulators
     * @param manipulators Array of 9 integers defining mouse/key button
     * mappings
     * @virtual
     *
     * Configures camera control behavior for 3D visualization mode.
     */
    virtual void setCamera3DManipulators(const int manipulators[9]);

    /**
     * @brief Set camera manipulators for custom style
     * @param style Custom interactor style to configure
     * @param manipulators Array of 9 integers defining mouse/key button
     * mappings
     */
    void setCameraManipulators(VTKExtensions::vtkCustomInteractorStyle* style,
                               const int manipulators[9]);

    /**
     * @brief Set 2D mouse wheel motion factor
     * @param factor Zoom speed multiplier for 2D mode
     * @virtual
     */
    virtual void setCamera2DMouseWheelMotionFactor(double factor);

    /**
     * @brief Set 3D mouse wheel motion factor
     * @param factor Zoom speed multiplier for 3D mode
     * @virtual
     */
    virtual void setCamera3DMouseWheelMotionFactor(double factor);

    /**
     * @brief Update center axes display
     * @virtual
     *
     * Refreshes the center axes actor's scale and position
     * based on current scene bounds.
     */
    virtual void updateCenterAxes();

    /**
     * @brief Synchronize geometry bounds across nodes
     * @param viewport Viewport ID (default: 0)
     *
     * @note This method is designed for parallel/distributed rendering
     * contexts. In single-process mode, it simply updates local bounds.
     */
    void synchronizeGeometryBounds(int viewport = 0);

    // =====================================================================
    // Depth Query and Camera Distance Control
    // =====================================================================

    /**
     * @brief Get depth value from z-buffer at screen position
     * @param x Screen X coordinate in pixels
     * @param y Screen Y coordinate in pixels
     * @return Normalized depth value (0.0 = near plane, 1.0 = far plane)
     *
     * Queries the OpenGL depth buffer for depth testing and picking.
     */
    double getGLDepth(int x, int y);

    /**
     * @brief Get camera focal distance
     * @param viewport Viewport ID (default: 0)
     * @return Distance from camera to focal point
     */
    double getCameraFocalDistance(int viewport = 0);

    /**
     * @brief Set camera focal distance
     * @param focal_distance Distance from camera to focal point
     * @param viewport Viewport ID (default: 0)
     *
     * Adjusts how far the camera is from its focal point.
     */
    void setCameraFocalDistance(double focal_distance, int viewport = 0);

    /**
     * @brief Zoom camera by factor
     * @param zoomFactor Zoom multiplier (>1.0 = zoom in, <1.0 = zoom out)
     * @param viewport Viewport ID (default: 0)
     *
     * In perspective mode: decreases view angle by the factor.
     * In parallel mode: decreases parallel scale by the factor.
     *
     * @note Ignored when UseExplicitProjectionTransformMatrix is true.
     */
    void zoomCamera(double zoomFactor, int viewport = 0);

    /**
     * @brief Get projection transformation matrix
     * @param proj Output 4x4 projection matrix
     *
     * Retrieves the current camera projection matrix.
     */
    void getProjectionTransformMatrix(Eigen::Matrix4d& proj);

    /**
     * @brief Get model-view transformation matrix
     * @param view Output 4x4 model-view matrix
     *
     * Retrieves the current camera view transformation matrix.
     */
    void getModelViewTransformMatrix(Eigen::Matrix4d& view);

    // =====================================================================
    // Camera Reset and Clipping Range
    // =====================================================================

    /**
     * @brief Reset camera clipping range
     * @param viewport Viewport ID (default: 0)
     *
     * Automatically adjusts near/far clipping planes based on scene bounds.
     */
    void resetCameraClippingRange(int viewport = 0);

    /**
     * @brief Internal helper for resetting camera clipping range
     *
     * Calls resetCameraClippingRange(0) for viewport 0.
     */
    void internalResetCameraClippingRange() {
        this->resetCameraClippingRange(0);
    }

    /**
     * @brief Reset camera to view bounding box
     * @param bbox Bounding box to frame
     *
     * Positions camera to view the entire bounding box.
     */
    void resetCamera(const ccBBox* bbox);

    /**
     * @brief Reset camera to view axis-aligned box
     * @param xMin Minimum X coordinate
     * @param xMax Maximum X coordinate
     * @param yMin Minimum Y coordinate
     * @param yMax Maximum Y coordinate
     * @param zMin Minimum Z coordinate
     * @param zMax Maximum Z coordinate
     *
     * Centers camera on the bounding box center and adjusts distance
     * to fit all geometry while preserving view direction.
     */
    void resetCamera(double xMin,
                     double xMax,
                     double yMin,
                     double yMax,
                     double zMin,
                     double zMax);

    /**
     * @brief Reset camera to view entire scene
     *
     * Automatically frames all visible geometry.
     */
    void resetCamera();

    /**
     * @brief Reset camera to view bounds array
     * @param bounds Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
     */
    inline void resetCamera(double bounds[6]) {
        resetCamera(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4],
                    bounds[5]);
    }

    /**
     * @brief Compute reasonable clipping range for scene
     * @param range Output array [near, far] for clipping distances
     * @param viewport Viewport ID (default: 0)
     *
     * Calculates appropriate near/far plane distances based on scene bounds.
     */
    void getReasonableClippingRange(double range[2], int viewport = 0);

    /**
     * @brief Expand bounds by transformation matrix
     * @param bounds Bounding box to expand [xmin, xmax, ymin, ymax, zmin, zmax]
     * @param matrix 4x4 transformation matrix
     *
     * Utility to compute transformed bounding box.
     */
    void expandBounds(double bounds[6], vtkMatrix4x4* matrix);

    /**
     * @brief Set camera view angle
     * @param viewAngle Vertical field of view in degrees
     * @param viewport Viewport ID (default: 0)
     *
     * Sets the perspective camera's field of view.
     */
    void setCameraViewAngle(double viewAngle, int viewport = 0);

    // =====================================================================
    // Draw Methods for CV_db Entities
    // =====================================================================

    /**
     * @brief Draw sensor visualization
     * @param context Drawing context with rendering parameters
     * @param sensor Sensor object to visualize
     *
     * Renders sensor frustum or representation.
     */
    void draw(const CC_DRAW_CONTEXT& context, const ccSensor* sensor);

    /**
     * @brief Draw line set
     * @param context Drawing context with rendering parameters
     * @param lineset Line set geometry to draw
     *
     * Renders 3D line set from CloudViewer geometry.
     */
    void draw(const CC_DRAW_CONTEXT& context,
              const cloudViewer::geometry::LineSet* lineset);

    // ==================== Direct CV_db → VTK Draw Methods ====================
    // These methods bypass PCL data format conversion for maximum efficiency.
    // Data is converted directly from CV_db types to VTK polydata.

    /**
     * @brief Draw point cloud directly (bypassing PCL conversion)
     * @param context Draw context with rendering parameters
     * @param cloud CloudViewer point cloud to draw
     *
     * High-performance rendering path that converts ccPointCloud
     * directly to VTK polydata without intermediate PCL format.
     * Supports colors, normals, and scalar fields.
     */
    void drawDirect(const CC_DRAW_CONTEXT& context, ccPointCloud* cloud);

    /**
     * @brief Draw mesh directly (bypassing PCL conversion)
     * @param context Draw context with rendering parameters
     * @param mesh CloudViewer mesh to draw (non-textured)
     *
     * High-performance rendering path for meshes.
     * Converts ccGenericMesh directly to VTK polydata.
     * For textured meshes, use addTextureMeshFromCCMesh.
     */
    void drawMeshDirect(const CC_DRAW_CONTEXT& context, ccGenericMesh* mesh);

    /**
     * @brief Draw polyline directly (bypassing PCL conversion)
     * @param context Draw context with rendering parameters
     * @param polyline CloudViewer polyline to draw
     * @param closed Whether the polyline forms a closed loop
     *
     * Renders 2D/3D polylines with optional closure.
     */
    void drawPolylineDirect(const CC_DRAW_CONTEXT& context,
                            ccPolyline* polyline,
                            bool closed);

    /**
     * @brief Update shading mode directly from point cloud
     * @param context Draw context with rendering parameters
     * @param cloud Source point cloud (nullptr to use existing data)
     *
     * Synchronizes normals and RGB colors to VTK polydata for
     * selection extraction and updates shading mode (Flat/Phong).
     * Used for dynamic shading updates without recreating actors.
     */
    void updateShadingModeDirect(const CC_DRAW_CONTEXT& context,
                                 ccPointCloud* cloud);

    /**
     * @brief Update normal glyphs directly from point cloud
     * @param context Draw context with rendering parameters
     * @param cloud Source point cloud (nullptr to use existing data)
     *
     * Updates the visualization of point normals as arrow glyphs.
     */
    void updateNormalsDirect(const CC_DRAW_CONTEXT& context,
                             ccPointCloud* cloud);

    /**
     * @brief Apply transformation to entities
     * @param context Draw context containing transformation matrix
     *
     * Applies model transformation to actors specified in the context.
     */
    void transformEntities(const CC_DRAW_CONTEXT& context);

    /**
     * @brief Get VTK transformation from context
     * @param context Draw context with transformation parameters
     * @param origin Origin point for transformation
     * @return VTK transform object
     *
     * Creates a VTK transform from CloudViewer transformation context.
     */
    vtkSmartPointer<vtkTransform> getTransformation(
            const CC_DRAW_CONTEXT& context, const CCVector3d& origin);

    /**
     * @brief Remove entities specified in context
     * @param context Draw context identifying entities to remove
     * @return true if entities were removed
     */
    bool removeEntities(const CC_DRAW_CONTEXT& context);

    /**
     * @brief Show or hide actors by view ID
     * @param visibility true to show, false to hide
     * @param viewID Unique identifier of actor(s) to affect
     * @param viewport Viewport ID (default: 0)
     */
    void hideShowActors(bool visibility,
                        const std::string& viewID,
                        int viewport = 0);

    /**
     * @brief Show or hide widgets by view ID
     * @param visibility true to show, false to hide
     * @param viewID Unique identifier of widget(s) to affect
     * @param viewport Viewport ID (default: 0)
     */
    void hideShowWidgets(bool visibility,
                         const std::string& viewID,
                         int viewport = 0);

    // =====================================================================
    // Scalar Bar (Color Legend) and Caption Management
    // =====================================================================

    /**
     * @brief Add color scalar bar (color legend)
     * @param context Draw context containing scalar field information
     * @return true if scalar bar was added successfully
     *
     * Creates a color bar showing the mapping from scalar values to colors.
     */
    bool addScalarBar(const CC_DRAW_CONTEXT& context);

    /**
     * @brief Update existing scalar bar
     * @param context Draw context with updated scalar field information
     * @return true if scalar bar was updated successfully
     *
     * Updates the scalar bar's color map, range, and labels.
     */
    bool updateScalarBar(const CC_DRAW_CONTEXT& context);

    /**
     * @brief Add caption widget with 3D anchor
     * @param text Caption text to display
     * @param pos2D 2D screen position (normalized 0.0-1.0)
     * @param anchorPos 3D world position to anchor the caption
     * @param color Text color (RGBA)
     * @param fontSize Font size in points (default: 10)
     * @param viewID Unique identifier (default: "caption")
     * @param anchorDragable Whether anchor point can be dragged (default:
     * false)
     * @param viewport Viewport ID (default: 0)
     * @return true if caption was added successfully
     *
     * Creates a caption with a leader line connecting 2D text to a 3D point.
     */
    bool addCaption(const std::string& text,
                    const CCVector2& pos2D,
                    const CCVector3& anchorPos,
                    const ecvColor::Rgbaf& color,
                    int fontSize = 10,
                    const std::string& viewID = "caption",
                    bool anchorDragable = false,
                    int viewport = 0);

    /**
     * @brief Update existing caption widget
     * @param text Updated caption text
     * @param pos2D Updated 2D screen position (normalized 0.0-1.0)
     * @param anchorPos Updated 3D world anchor position
     * @param color Updated text color (RGBA)
     * @param fontSize Updated font size in points (default: 10)
     * @param viewID Unique identifier of caption to update (default: "caption")
     * @param viewport Viewport ID (default: 0)
     * @return true if caption was updated successfully
     */
    bool updateCaption(const std::string& text,
                       const CCVector2& pos2D,
                       const CCVector3& anchorPos,
                       const ecvColor::Rgbaf& color,
                       int fontSize = 10,
                       const std::string& viewID = "caption",
                       int viewport = 0);

    /**
     * @brief Get caption widget 2D position
     * @param viewID Unique identifier of caption widget
     * @param posX Output X position (normalized 0.0-1.0, left to right)
     * @param posY Output Y position (normalized 0.0-1.0, bottom to top in VTK)
     * @return true if position retrieved successfully, false if widget not
     * found
     *
     * Retrieves the current 2D screen position of a caption widget.
     */
    bool getCaptionPosition(const std::string& viewID,
                            float& posX,
                            float& posY);

    // =====================================================================
    // Texture Mapping and Management
    // =====================================================================

    /**
     * @brief Update texture materials
     * @param context Draw context identifying the mesh
     * @param materials Material set containing texture information
     * @return true if textures were updated successfully
     *
     * Updates texture coordinates and material properties for a mesh.
     */
    bool updateTexture(const CC_DRAW_CONTEXT& context,
                       const ccMaterialSet* materials);

    /**
     * @brief Add textured mesh from ccGenericMesh (recommended)
     * @param mesh ccGenericMesh object with geometry and ccMaterialSet
     * @param id Unique identifier for the mesh
     * @param viewport Viewport ID (default: 0)
     * @return true if textured mesh was added successfully
     *
     * @note This method directly uses ccMaterialSet, bypassing pcl::TexMaterial
     * encoding for better performance and quality.
     */
    bool addTextureMeshFromCCMesh(ccGenericMesh* mesh,
                                  const std::string& id,
                                  int viewport = 0);

    /**
     * @brief Load multi-texture mesh from OBJ file
     * @param obj_path Path to OBJ file
     * @param id Unique identifier for the mesh
     * @param viewport Viewport ID (default: 0)
     * @param quality Texture quality level 0-3 (default: 2)
     * @param enable_cache Whether to cache loaded textures (default: true)
     * @return true if mesh was loaded and added successfully
     *
     * Loads an OBJ file with MTL materials and textures.
     * Supports multiple textures per mesh.
     */
    bool addTextureMeshFromOBJ(const std::string& obj_path,
                               const std::string& id,
                               int viewport = 0,
                               int quality = 2,
                               bool enable_cache = true);

    /**
     * @brief Load multi-texture mesh from OBJ with advanced options
     * @param obj_path Path to OBJ geometry file
     * @param mtl_path Path to MTL material file
     * @param id Unique identifier for the mesh
     * @param viewport Viewport ID (default: 0)
     * @param max_texture_size Maximum texture dimension in pixels (default:
     * 4096)
     * @param use_mipmaps Whether to generate mipmaps for textures (default:
     * true)
     * @param enable_cache Whether to cache loaded textures (default: true)
     * @return true if mesh was loaded and added successfully
     *
     * Advanced texture loading with explicit material file and size control.
     */
    bool addTextureMeshFromOBJAdvanced(const std::string& obj_path,
                                       const std::string& mtl_path,
                                       const std::string& id,
                                       int viewport = 0,
                                       int max_texture_size = 4096,
                                       bool use_mipmaps = true,
                                       bool enable_cache = true);

    /**
     * @brief Clear texture cache
     *
     * Releases all cached textures to free memory.
     * Useful after loading many textured meshes.
     */
    void clearTextureCache();

    /**
     * @brief Get texture cache statistics
     * @param count Output parameter for number of cached textures
     * @param memory_bytes Output parameter for approximate memory usage in
     * bytes
     *
     * Provides information about texture cache utilization.
     */
    void getTextureCacheInfo(size_t& count, size_t& memory_bytes) const;

    // =====================================================================
    // Oriented Bounding Box Visualization
    // =====================================================================

    /**
     * @brief Add oriented bounding box with transformation matrix
     * @param trans 4x4 transformation matrix (position and orientation)
     * @param width Box width (X dimension)
     * @param height Box height (Y dimension)
     * @param depth Box depth (Z dimension)
     * @param r Red color component (0.0-1.0, default: 1.0)
     * @param g Green color component (0.0-1.0, default: 1.0)
     * @param b Blue color component (0.0-1.0, default: 1.0)
     * @param id Unique identifier (default: "cube")
     * @param viewport Viewport ID (default: 0)
     * @return true if oriented box was added successfully
     *
     * Adds a wireframe or solid box with arbitrary position and orientation.
     */
    bool addOrientedCube(const ccGLMatrixd& trans,
                         double width,
                         double height,
                         double depth,
                         double r = 1.0,
                         double g = 1.0,
                         double b = 1.0,
                         const std::string& id = "cube",
                         int viewport = 0);

    /**
     * @brief Add oriented bounding box with translation and rotation
     * @param translation Position in world coordinates
     * @param rotation Orientation as quaternion
     * @param width Box width (X dimension)
     * @param height Box height (Y dimension)
     * @param depth Box depth (Z dimension)
     * @param r Red color component (0.0-1.0, default: 1.0)
     * @param g Green color component (0.0-1.0, default: 1.0)
     * @param b Blue color component (0.0-1.0, default: 1.0)
     * @param id Unique identifier (default: "cube")
     * @param viewport Viewport ID (default: 0)
     * @return true if oriented box was added successfully
     */
    bool addOrientedCube(const Eigen::Vector3f& translation,
                         const Eigen::Quaternionf& rotation,
                         double width,
                         double height,
                         double depth,
                         double r = 1.0,
                         double g = 1.0,
                         double b = 1.0,
                         const std::string& id = "cube",
                         int viewport = 0);

    /**
     * @brief Add oriented bounding box from ecvOrientedBBox
     * @param obb Oriented bounding box object
     * @param id Unique identifier (default: "cube")
     * @param viewport Viewport ID (default: 0)
     * @return true if oriented box was added successfully
     *
     * Convenience method to visualize CloudViewer oriented bounding boxes.
     */
    bool addOrientedCube(const ecvOrientedBBox& obb,
                         const std::string& id = "cube",
                         int viewport = 0);

    /**
     * @brief Display text from draw context
     * @param context Draw context containing text elements to display
     *
     * Renders all text and labels specified in the draw context.
     */
    void displayText(const CC_DRAW_CONTEXT& context);

private:
    /** \brief Helper to create a shape actor from a VTK source output.
     *  Handles contains-check, mapper, actor, color, addToRenderer, and
     *  inserting into shape_actor_map_.
     */
    bool addShapeActor(vtkAlgorithmOutput* sourceOutput,
                       const ecvColor::Rgbf& color,
                       const std::string& id,
                       int viewport);

    /** \brief Overload: create a shape actor from pre-built polydata. */
    bool addShapeActor(vtkSmartPointer<vtkPolyData> polydata,
                       const ecvColor::Rgbf& color,
                       const std::string& id,
                       int viewport);

    // Texture rendering manager
    std::unique_ptr<renders::TextureRenderManager> texture_render_manager_;

    // Store transformation matrices to prevent memory leaks
    // Maps view ID to transformation matrix smart pointer
    std::map<std::string, vtkSmartPointer<vtkMatrix4x4>> transformation_map_;

public:
    // =====================================================================
    // Point Cloud and Mesh Rendering Properties
    // =====================================================================

    /**
     * @brief Set point cloud point size
     * @param pointSize Point size in pixels
     * @param viewID Unique identifier of the point cloud
     * @param viewport Viewport ID (default: 0)
     */
    void setPointSize(const unsigned char pointSize,
                      const std::string& viewID,
                      int viewport = 0);

    /**
     * @brief Set active scalar field name for coloring
     * @param viewID Unique identifier of the cloud/mesh
     * @param scalarName Name of the scalar field to use
     * @param viewport Viewport ID (default: 0)
     *
     * Switches which scalar field is used for color mapping.
     */
    void setScalarFieldName(const std::string& viewID,
                            const std::string& scalarName,
                            int viewport = 0);

    /**
     * @brief Add scalar field to VTK polydata
     * @param viewID Unique identifier of the cloud
     * @param cloud Source point cloud containing scalar fields
     * @param scalarFieldIndex Index of scalar field to add
     * @param viewport Viewport ID (default: 0)
     *
     * Transfers a single scalar field from ccPointCloud to VTK for
     * visualization.
     */
    void addScalarFieldToVTK(const std::string& viewID,
                             ccPointCloud* cloud,
                             int scalarFieldIndex,
                             int viewport = 0);

    /**
     * @brief Synchronize all scalar fields to VTK
     * @param viewID Unique identifier of the cloud
     * @param cloud Source point cloud containing scalar fields
     * @param viewport Viewport ID (default: 0)
     *
     * Transfers all scalar fields from ccPointCloud to VTK.
     * Useful for switching between multiple fields at runtime.
     */
    void syncAllScalarFieldsToVTK(const std::string& viewID,
                                  ccPointCloud* cloud,
                                  int viewport = 0);

    // =====================================================================
    // Source Object Management (for Selection/Picking)
    // =====================================================================

    /**
     * @brief Associate source object with view ID
     * @param obj CloudViewer object to associate
     * @param viewID Unique identifier for this visualization
     *
     * Stores reference to original ccHObject for selection extraction.
     */
    void setCurrentSourceObject(ccHObject* obj, const std::string& viewID);

    /**
     * @brief Remove source object association
     * @param viewID Unique identifier to disassociate
     */
    void removeSourceObject(const std::string& viewID);

    /**
     * @brief Get source object by view ID
     * @param viewID Unique identifier
     * @return Pointer to ccHObject, or nullptr if not found
     */
    ccHObject* getSourceObject(const std::string& viewID) const;

    /**
     * @brief Get source point cloud by view ID
     * @param viewID Unique identifier
     * @return Pointer to ccPointCloud, or nullptr if not found/not a cloud
     */
    ccPointCloud* getSourceCloud(const std::string& viewID) const;

    /**
     * @brief Get source mesh by view ID
     * @param viewID Unique identifier
     * @return Pointer to ccMesh, or nullptr if not found/not a mesh
     */
    ccMesh* getSourceMesh(const std::string& viewID) const;

    /**
     * @brief Check if source object exists for view ID
     * @param viewID Unique identifier
     * @return true if source object is associated
     */
    bool hasSourceObject(const std::string& viewID) const;

    // =====================================================================
    // Color and Appearance Control
    // =====================================================================

    /**
     * @brief Set point cloud to uniform color
     * @param r Red component (0.0-1.0)
     * @param g Green component (0.0-1.0)
     * @param b Blue component (0.0-1.0)
     * @param viewID Unique identifier of the point cloud
     * @param viewport Viewport ID (default: 0)
     *
     * Overrides per-point colors with a single uniform color.
     */
    void setPointCloudUniqueColor(double r,
                                  double g,
                                  double b,
                                  const std::string& viewID,
                                  int viewport = 0);

    /**
     * @brief Reset to scalar field coloring
     * @param viewID Unique identifier of the cloud/mesh
     * @param flag true to enable scalar coloring, false to disable
     * @param viewport Viewport ID (default: 0)
     *
     * Switches between scalar field colors and uniform/per-vertex colors.
     */
    void resetScalarColor(const std::string& viewID,
                          bool flag = true,
                          int viewport = 0);

    /**
     * @brief Set shape to uniform color
     * @param r Red component (0.0-1.0)
     * @param g Green component (0.0-1.0)
     * @param b Blue component (0.0-1.0)
     * @param viewID Unique identifier of the shape
     * @param viewport Viewport ID (default: 0)
     */
    void setShapeUniqueColor(float r,
                             float g,
                             float b,
                             const std::string& viewID,
                             int viewport = 0);

    /**
     * @brief Set line width for polylines/wireframes
     * @param lineWidth Line width in pixels
     * @param viewID Unique identifier of the shape/mesh
     * @param viewport Viewport ID (default: 0)
     */
    void setLineWidth(const unsigned char lineWidth,
                      const std::string& viewID,
                      int viewport = 0);

    /**
     * @brief Set mesh rendering mode
     * @param mode Rendering mode (points, wireframe, surface, etc.)
     * @param viewID Unique identifier of the mesh
     * @param viewport Viewport ID (default: 0)
     */
    void setMeshRenderingMode(MESH_RENDERING_MODE mode,
                              const std::string& viewID,
                              int viewport = 0);

    /**
     * @brief Set lighting mode for entity
     * @param viewID Unique identifier of the entity
     * @param viewport Viewport ID (default: 0)
     *
     * Updates lighting calculations for the specified entity.
     */
    void setLightMode(const std::string& viewID, int viewport = 0);

    /**
     * @brief Set point cloud opacity
     * @param opacity Opacity value (0.0=transparent, 1.0=opaque)
     * @param viewID Unique identifier of the point cloud
     * @param viewport Viewport ID (default: 0)
     */
    void setPointCloudOpacity(double opacity,
                              const std::string& viewID,
                              int viewport = 0);

    /**
     * @brief Set shape opacity
     * @param opacity Opacity value (0.0=transparent, 1.0=opaque)
     * @param viewID Unique identifier of the shape
     * @param viewport Viewport ID (default: 0)
     */
    void setShapeOpacity(double opacity,
                         const std::string& viewID,
                         int viewport = 0);

    /**
     * @brief Set mesh opacity
     * @param opacity Opacity value (0.0=transparent, 1.0=opaque)
     * @param viewID Unique identifier of the mesh
     * @param viewport Viewport ID (default: 0)
     */
    void setMeshOpacity(double opacity,
                        const std::string& viewID,
                        int viewport = 0);

    /**
     * @brief Set shape shading mode
     * @param mode Shading mode (flat, Gouraud, Phong)
     * @param viewID Unique identifier of the shape
     * @param viewport Viewport ID (default: 0)
     */
    void setShapeShadingMode(SHADING_MODE mode,
                             const std::string& viewID,
                             int viewport = 0);

    /**
     * @brief Set mesh shading mode
     * @param mode Shading mode (flat, Gouraud, Phong)
     * @param viewID Unique identifier of the mesh
     * @param viewport Viewport ID (default: 0)
     */
    void setMeshShadingMode(SHADING_MODE mode,
                            const std::string& viewID,
                            int viewport = 0);

    // =====================================================================
    // Actor and Widget Query Methods
    // =====================================================================

    /**
     * @brief Get PCL-compatible interactor style
     * @return Smart pointer to custom interactor style
     *
     * Returns the currently active interactor style.
     */
    vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
    getPCLInteractorStyle();

    /**
     * @brief Get VTK actor by view ID
     * @param viewId Unique identifier
     * @return Pointer to vtkActor, or nullptr if not found or not an actor
     */
    vtkActor* getActorById(const std::string& viewId);

    /**
     * @brief Get VTK prop by view ID
     * @param viewId Unique identifier
     * @return Pointer to vtkProp, or nullptr if not found
     *
     * Props are more general than actors and include actors, volumes, etc.
     */
    vtkProp* getPropById(const std::string& viewId);

    /**
     * @brief Get collection of props by view ID
     * @param viewId Unique identifier
     * @return Smart pointer to prop collection (may contain multiple props)
     */
    vtkSmartPointer<vtkPropCollection> getPropCollectionById(
            const std::string& viewId);

    /**
     * @brief Get view ID from VTK actor
     * @param actor Pointer to vtkProp to search for
     * @return View ID string, or empty string if not found
     *
     * Reverse lookup to find the ID associated with an actor.
     */
    std::string getIdByActor(vtkProp* actor);

    /**
     * @brief Get VTK widget by view ID
     * @param viewId Unique identifier
     * @return Pointer to vtkAbstractWidget, or nullptr if not found
     */
    vtkAbstractWidget* getWidgetById(const std::string& viewId);

    /**
     * @brief Get current renderer for viewport
     * @param viewport Viewport ID (default: 0)
     * @return Pointer to vtkRenderer, or nullptr if viewport not found
     *
     * Retrieves the renderer associated with a specific viewport.
     */
    vtkRenderer* getCurrentRenderer(int viewport = 0);

public:
    // =====================================================================
    // Widget and Prop Containment Checks
    // =====================================================================

    /**
     * @brief Check if widget or prop exists
     * @param id Unique identifier to search for
     * @return true if a widget or prop with this ID was found
     *
     * Searches both widget and prop maps for the given ID.
     */
    inline bool containWidget(const std::string& id) const {
        return (m_widget_map->find(id) != m_widget_map->end() ||
                m_prop_map->find(id) != m_prop_map->end());
    }

    /**
     * @brief Get widget actor map
     * @return Shared pointer to widget actor map
     *
     * Provides access to the internal map of all widgets.
     */
    inline WidgetActorMapPtr getWidgetActorMap() { return (m_widget_map); }

    /**
     * @brief Get prop actor map
     * @return Shared pointer to prop actor map
     *
     * Provides access to the internal map of all props.
     */
    inline PropActorMapPtr getPropActorMap() { return (m_prop_map); }

    /**
     * @brief Get visible geometry bounding box
     * @return VTK bounding box enclosing all visible geometry
     */
    inline vtkBoundingBox getVisibleGeometryBounds() { return GeometryBounds; }

    /**
     * @brief Get 2D interactor style
     * @return Smart pointer to 2D interactor style
     *
     * Returns the interactor style configured for 2D visualization.
     */
    inline vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
    get2DInteractorStyle() {
        return TwoDInteractorStyle;
    }

    /**
     * @brief Get 3D interactor style
     * @return Smart pointer to 3D interactor style
     *
     * Returns the interactor style configured for 3D visualization.
     */
    inline vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
    get3DInteractorStyle() {
        return ThreeDInteractorStyle;
    }

protected:
    // =====================================================================
    // Core visualization state (previously from PCLVisualizer base class)
    // =====================================================================

    /// VTK renderer collection (supports multiple viewports)
    vtkSmartPointer<vtkRendererCollection> rens_;

    /// VTK render window for display output
    vtkSmartPointer<vtkRenderWindow> win_;

    /// VTK render window interactor for user input
    vtkSmartPointer<vtkRenderWindowInteractor> interactor_;

    /// Map of point cloud actors (view ID -> actor data)
    PclUtils::CloudActorMapPtr cloud_actor_map_;

    /// Map of shape actors (view ID -> actor data)
    PclUtils::ShapeActorMapPtr shape_actor_map_;

    /// Map of widget actors (view ID -> widget pointer)
    WidgetActorMapPtr m_widget_map;

    /// Map of prop actors (view ID -> prop pointer)
    PropActorMapPtr m_prop_map;

    /// Bounding box of all visible geometry
    vtkBoundingBox GeometryBounds;

    /// Center axes actor for scene orientation reference
    vtkSmartPointer<VTKExtensions::vtkPVCenterAxesActor> m_centerAxes;

    /// Interactor style for 2D visualization mode
    vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
            TwoDInteractorStyle;

    /// Interactor style for 3D visualization mode
    vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
            ThreeDInteractorStyle;

    /// Current active interactor style
    vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle> m_interactorStyle;

private:
    /// Remove all widgets associated with a view ID
    bool removeWidgets(const std::string& viewId, int viewport);

    /// Remove point clouds associated with a view ID
    void removePointClouds(const std::string& viewId, int viewport = 0);

    /// Remove shapes associated with a view ID
    void removeShapes(const std::string& viewId, int viewport = 0);

    /// Remove mesh associated with a view ID
    void removeMesh(const std::string& viewId, int viewport = 0);

    /// Remove 2D text associated with a view ID
    void removeText2D(const std::string& viewId, int viewport = 0);

    /// Remove 3D text associated with a view ID
    void removeText3D(const std::string& viewId, int viewport = 0);

    /// Remove all actors and widgets from a viewport
    void removeALL(int viewport = 0);

private:
    /// Register mouse event callbacks with the interactor
    void registerMouse();

    /// Register keyboard event callbacks with the interactor
    void registerKeyboard();

    /// Register area picking callbacks with the interactor
    void registerAreaPicking();

    /// Register point picking callbacks with the interactor
    void registerPointPicking();

    /// Configure and register the interactor style
    void registerInteractorStyle(bool useDefault = false);

    /// Internal handler for point picking events
    void pointPickingProcess(const PclUtils::PointPickingEvent& event);

    /// Internal handler for area picking events
    void areaPickingEventProcess(const PclUtils::AreaPickingEvent& event);

    /// Internal handler for mouse events
    void mouseEventProcess(const PclUtils::MouseEvent& event);

    /// Internal handler for keyboard events
    void keyboardEventProcess(const PclUtils::KeyboardEvent& event);

public:
    // =====================================================================
    // Picking and Selection Tools
    // =====================================================================

    /**
     * @brief Check if point picking is enabled
     * @return true if point picking mode is active
     */
    inline bool isPointPickingEnabled() { return m_pointPickingEnabled; }

    /**
     * @brief Enable or disable point picking
     * @param state true to enable point picking
     */
    inline void setPointPickingEnabled(bool state) {
        m_pointPickingEnabled = state;
    }

    /**
     * @brief Toggle point picking mode
     *
     * Switches between enabled and disabled point picking.
     */
    inline void togglePointPicking() {
        setPointPickingEnabled(!isPointPickingEnabled());
    }

    /**
     * @brief Check if area picking is enabled
     * @return true if area picking mode is active
     */
    inline bool isAreaPickingEnabled() { return m_areaPickingEnabled; }

    /**
     * @brief Enable or disable area picking
     * @param state true to enable area picking (rubber band selection)
     */
    inline void setAreaPickingEnabled(bool state) {
        m_areaPickingEnabled = state;
    }

    /**
     * @brief Check if actor picking is enabled
     * @return true if actor picking mode is active
     */
    inline bool isActorPickingEnabled() { return m_actorPickingEnabled; }

    /**
     * @brief Enable or disable actor picking
     * @param state true to enable actor picking (click to select entities)
     */
    inline void setActorPickingEnabled(bool state) {
        m_actorPickingEnabled = state;
    }

    /**
     * @brief Toggle actor picking mode
     *
     * Switches between enabled and disabled actor picking.
     */
    inline void toggleActorPicking() {
        setActorPickingEnabled(!isActorPickingEnabled());
    }

    /**
     * @brief Toggle area picking mode
     *
     * Switches between enabled and disabled area picking (rubber band
     * selection).
     */
    void toggleAreaPicking();

    /**
     * @brief Process exit callback
     *
     * Handles cleanup when the visualizer is closing.
     */
    void exitCallbackProcess();

    /**
     * @brief Set area picking mode
     * @param state true to enable area picking mode
     *
     * Configures the interactor for area/rubber band selection.
     */
    void setAreaPickingMode(bool state);

    /**
     * @brief Pick actor at screen coordinates
     * @param x Screen X coordinate in pixels
     * @param y Screen Y coordinate in pixels
     * @return Pointer to picked vtkActor, or nullptr if no actor at position
     *
     * Performs actor picking at a specific screen location.
     */
    vtkActor* pickActor(double x, double y);

    /**
     * @brief Pick item in rectangular region
     * @param x0 Starting X coordinate (default: -1, uses current position)
     * @param y0 Starting Y coordinate (default: -1, uses current position)
     * @param x1 Ending X offset from start (default: 5.0)
     * @param y1 Ending Y offset from start (default: 5.0)
     * @return View ID of picked item, or empty string if none
     *
     * Picks items within a small rectangular region around the specified point.
     */
    std::string pickItem(double x0 = -1,
                         double y0 = -1,
                         double x1 = 5.0,
                         double y1 = 5.0);

    /**
     * @brief Render scene to QImage
     * @param zoomFactor Resolution multiplier (default: 1)
     * @param renderOverlayItems Whether to include 2D overlays (default: false)
     * @param silent Suppress status messages (default: false)
     * @param viewport Viewport ID to render (default: 0)
     * @return QImage containing rendered scene
     *
     * Captures the current viewport to an image for export or display.
     * Higher zoomFactor produces higher resolution output.
     */
    QImage renderToImage(int zoomFactor = 1,
                         bool renderOverlayItems = false,
                         bool silent = false,
                         int viewport = 0);

protected:
    // =====================================================================
    // Utility Variables
    // =====================================================================

    /// Current interaction mode
    int m_currentMode;

    /// Point picking mode enabled flag
    bool m_pointPickingEnabled;

    /// Area picking (rubber band) mode enabled flag
    bool m_areaPickingEnabled;

    /// Actor picking mode enabled flag
    bool m_actorPickingEnabled;

    /// Auto-update camera position flag
    bool m_autoUpdateCameraPos;

    /// Mutex for thread-safe cloud operations
    std::mutex m_cloud_mutex;

signals:
    /**
     * @brief Signal emitted when an actor is picked
     * @param actor Pointer to the picked VTK actor
     *
     * Qt signal emitted during actor picking events.
     */
    void interactorPickedEvent(vtkActor* actor);

public:
    // =====================================================================
    // View Properties (ParaView-compatible)
    // =====================================================================

    /**
     * @brief Set global default light intensity
     * @param intensity Light intensity (0.0-1.0)
     *
     * Sets the default light intensity for new objects and updates
     * the headlight intensity in the renderer.
     */
    void setLightIntensity(double intensity);

    /**
     * @brief Get global default light intensity
     * @return Current global light intensity (0.0-1.0)
     */
    double getLightIntensity() const;

    /**
     * @brief Set per-object light intensity
     * @param viewID Unique identifier of the object
     * @param intensity Light intensity for this object (0.0-1.0)
     * @param viewport Viewport ID (default: 0)
     *
     * Sets a custom light intensity for a specific object, overriding
     * the global default. Affects the object's material properties.
     */
    void setObjectLightIntensity(const std::string& viewID,
                                 double intensity,
                                 int viewport = 0);

    /**
     * @brief Get per-object light intensity
     * @param viewID Unique identifier of the object
     * @return Object-specific light intensity, or global default if not set
     */
    double getObjectLightIntensity(const std::string& viewID) const;

    /**
     * @brief Apply light properties to a VTK actor
     * @param actor Pointer to vtkActor to configure
     * @param viewID View ID for per-object settings (optional)
     *
     * Applies light intensity settings to an actor's material properties.
     * Uses per-object intensity if available, otherwise global default.
     * Called automatically when actors are created or updated.
     */
    void applyLightPropertiesToActor(vtkActor* actor,
                                     const std::string& viewID = "");

    // =====================================================================
    // Data Axes Grid (Unified Interface with AxesGridProperties)
    // =====================================================================

    /**
     * @brief Set data axes grid properties
     * @param viewID Unique identifier for this axes grid
     * @param props Axes grid configuration (visibility, labels, ticks, etc.)
     *
     * Configures or creates a data axes grid (cube axes) for visualizing
     * coordinate system and scale. Uses unified AxesGridProperties struct
     * for ParaView-compatible configuration.
     */
    void SetDataAxesGridProperties(const std::string& viewID,
                                   const AxesGridProperties& props);

    /**
     * @brief Get data axes grid properties
     * @param viewID Unique identifier of the axes grid
     * @param props Output parameter for current configuration
     *
     * Retrieves the current configuration of a data axes grid.
     */
    void GetDataAxesGridProperties(const std::string& viewID,
                                   AxesGridProperties& props) const;

    /**
     * @brief Remove data axes grid
     * @param viewID Unique identifier of the axes grid to remove
     *
     * Removes the specified data axes grid from the visualization.
     */
    void RemoveDataAxesGrid(const std::string& viewID);

    /**
     * @brief Toggle camera orientation widget
     * @param show true to show, false to hide
     *
     * Controls visibility of the camera orientation widget
     * (typically shown in a corner of the viewport).
     */
    void ToggleCameraOrientationWidget(bool show);

    /**
     * @brief Check if camera orientation widget is shown
     * @return true if camera orientation widget is currently visible
     */
    bool IsCameraOrientationWidgetShown() const;

protected:
    // =====================================================================
    // Widgets and Pickers
    // =====================================================================

    /// Orientation marker widget for axes display
    vtkSmartPointer<vtkOrientationMarkerWidget> m_axes_widget;

    /// Point picker for 3D point selection
    vtkSmartPointer<vtkPointPicker> m_point_picker;

    /// Area picker for rectangular region selection
    vtkSmartPointer<vtkAreaPicker> m_area_picker;

    /// Prop picker for actor/entity selection
    vtkSmartPointer<vtkPropPicker> m_propPicker;

    /// Selected slice indices (for volume/image data)
    std::vector<int> m_selected_slice;

    // =====================================================================
    // Source Object Management
    // =====================================================================

    /// Map of source objects for selection extraction (view ID -> ccHObject)
    std::map<std::string, ccHObject*> m_sourceObjectMap;

    // =====================================================================
    // View Properties (ParaView-compatible)
    // =====================================================================

    /// Global default light intensity (0.0-1.0)
    double m_lightIntensity;

    /// Per-object light intensity map (view ID -> intensity)
    std::unordered_map<std::string, double> m_objectLightIntensity;

    // =====================================================================
    // Data Axes and Camera Widgets
    // =====================================================================

    /// Data axes grid map (view ID -> cube axes actor)
    std::map<std::string, vtkSmartPointer<vtkCubeAxesActor>> m_dataAxesGridMap;

    /// Camera orientation widget (ParaView-style)
    vtkSmartPointer<vtkCameraOrientationWidget> m_cameraOrientationWidget;
};

typedef std::shared_ptr<PCLVis> PCLVisPtr;
}  // namespace PclUtils
