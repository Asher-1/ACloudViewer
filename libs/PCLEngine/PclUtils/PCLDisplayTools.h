// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ImageVis.h"
#include "PCLVis.h"
#include "Tools/Common/ecvTools.h"
#include "VTKExtensions/Widgets/QVTKWidgetCustom.h"
#include "qPCL.h"

// CV_CORE_LIB
#include <CVMath.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>

// system
#include <list>
#include <string>

class ccHObject;
class ImageVis;
class ccSensor;
class ccGenericMesh;
class ccImage;
class ecvOrientedBBox;
class ccPointCloud;
class QMainWindow;

namespace cloudViewer {
namespace geometry {
class LineSet;
}
}  // namespace cloudViewer

/**
 * @brief Display tools for CloudViewer using PCL/VTK backend
 *
 * This class provides a comprehensive display interface that bridges
 * CloudViewer's display abstractions with PCL/VTK visualization backend. It
 * manages both 2D and 3D visualization, camera control, entity rendering, and
 * user interaction.
 *
 * Key responsibilities:
 * - Entity rendering (point clouds, meshes, polylines, sensors, images)
 * - Camera manipulation and projection control
 * - Viewport management
 * - Widget and interaction handling
 * - Coordinate transformations between screen and world space
 * - ParaView-compatible view properties (light intensity, axes grid)
 *
 * @see ecvDisplayTools
 * @see PclUtils::PCLVis
 * @see PclUtils::ImageVis
 */
class QPCL_ENGINE_LIB_API PCLDisplayTools : public ecvDisplayTools {
public:
    /**
     * @brief Default constructor
     *
     * Creates a new PCL display tools instance.
     * Actual visualization components are initialized via registerVisualizer().
     */
    PCLDisplayTools() = default;

    /**
     * @brief Virtual destructor
     *
     * Cleans up VTK/Qt resources and visualization components.
     */
    virtual ~PCLDisplayTools() override;

public
    :  // =====================================================================
    // Visualizer Access and Coordinate Transformations
    // =====================================================================

    /**
     * @brief Get 3D visualizer interface
     * @return Pointer to generic 3D visualizer (PCLVis)
     */
    inline virtual ecvGenericVisualizer3D* getVisualizer3D() override {
        return get3DViewer();
    }

    /**
     * @brief Get 2D visualizer interface
     * @return Pointer to generic 2D visualizer (ImageVis)
     */
    inline virtual ecvGenericVisualizer2D* getVisualizer2D() override {
        return get2DViewer();
    }

    /**
     * @brief Get Qt/VTK widget
     * @return Pointer to custom QVTKWidget
     *
     * Provides access to the underlying Qt/VTK widget for advanced control.
     */
    inline QVTKWidgetCustom* getQVtkWidget() { return this->m_vtkWidget; }

    /**
     * @brief Convert 2D display coordinates to 3D world coordinates
     * @param input2D Display coordinates (screen space, with depth)
     * @param output3D Output world coordinates
     */
    inline virtual void toWorldPoint(const CCVector3d& input2D,
                                     CCVector3d& output3D) override {
        getQVtkWidget()->toWorldPoint(input2D, output3D);
    }

    /**
     * @brief Convert 2D display coordinates to 3D world coordinates
     * @param input2D Display coordinates (screen space, with depth)
     * @param output3D Output world coordinates
     */
    inline virtual void toWorldPoint(const CCVector3& input2D,
                                     CCVector3d& output3D) override {
        getQVtkWidget()->toWorldPoint(input2D, output3D);
    }

    /**
     * @brief Convert 3D world coordinates to 2D display coordinates
     * @param worldPos World coordinates
     * @param displayPos Output display coordinates (screen space)
     */
    virtual void toDisplayPoint(const CCVector3d& worldPos,
                                CCVector3d& displayPos) override {
        getQVtkWidget()->toDisplayPoint(worldPos, displayPos);
    }

    /**
     * @brief Convert 3D world coordinates to 2D display coordinates
     * @param worldPos World coordinates
     * @param displayPos Output display coordinates (screen space)
     */
    virtual void toDisplayPoint(const CCVector3& worldPos,
                                CCVector3d& displayPos) override {
        getQVtkWidget()->toDisplayPoint(worldPos, displayPos);
    }

    // =====================================================================
    // Display and Rendering Methods
    // =====================================================================

    /**
     * @brief Display text from draw context
     * @param context Draw context containing text elements
     */
    virtual void displayText(const CC_DRAW_CONTEXT& context) override;

    /**
     * @brief Toggle 2D viewer mode
     * @param state true to enable 2D mode, false for 3D mode
     *
     * Switches between 2D image viewing and 3D visualization modes.
     */
    virtual void toggle2Dviewer(bool state) override;

    /**
     * @brief Draw interactive widgets
     * @param param Widget parameters (type, position, properties)
     *
     * Renders interactive widgets for measurements, annotations, etc.
     */
    virtual void drawWidgets(const WIDGETS_PARAMETER& param) override;

    /**
     * @brief Change entity rendering properties
     * @param param Property parameters (entity ID, property type, values)
     *
     * Modifies visual properties like color, opacity, size, etc.
     */
    virtual void changeEntityProperties(PROPERTY_PARAM& param) override;

    /**
     * @brief Transform camera view matrix
     * @param viewMat View transformation matrix to apply
     *
     * Applies a custom view matrix to the camera.
     */
    virtual void transformCameraView(const ccGLMatrixd& viewMat) override;

    /**
     * @brief Transform camera projection matrix
     * @param projMat Projection transformation matrix to apply
     *
     * Applies a custom projection matrix to the camera.
     */
    virtual void transformCameraProjection(const ccGLMatrixd& projMat) override;

    /**
     * @brief Draw CloudViewer entity
     * @param context Draw context with rendering parameters
     * @param obj CloudViewer object to draw (cloud, mesh, polyline, etc.)
     *
     * Main entry point for rendering ccHObject entities.
     */
    virtual void draw(const CC_DRAW_CONTEXT& context,
                      const ccHObject* obj) override;

    /**
     * @brief Update mesh texture materials
     * @param context Draw context identifying the mesh
     * @param mesh Mesh with updated material/texture information
     *
     * Refreshes texture data for textured meshes.
     */
    virtual void updateMeshTextures(const CC_DRAW_CONTEXT& context,
                                    const ccGenericMesh* mesh) override;

    /**
     * @brief Check if entity needs update
     * @param viewID View identifier (output/input)
     * @param obj Entity to check
     * @return true if entity needs to be redrawn
     *
     * Determines whether an entity has changed and requires re-rendering.
     */
    bool checkEntityNeedUpdate(std::string& viewID, const ccHObject* obj);

    /**
     * @brief Draw axis-aligned bounding box
     * @param context Draw context with rendering parameters
     * @param bbox Bounding box to draw
     */
    virtual void drawBBox(const CC_DRAW_CONTEXT& context,
                          const ccBBox* bbox) override;

    /**
     * @brief Draw oriented bounding box
     * @param context Draw context with rendering parameters
     * @param obb Oriented bounding box to draw
     */
    virtual void drawOrientedBBox(const CC_DRAW_CONTEXT& context,
                                  const ecvOrientedBBox* obb) override;

    /**
     * @brief Check if orientation marker is shown
     * @return true if orientation marker is visible
     */
    virtual bool orientationMarkerShown() override;

    /**
     * @brief Toggle orientation marker visibility
     * @param state true to show, false to hide
     */
    virtual void toggleOrientationMarker(bool state) override;

    /**
     * @brief Remove entities from visualization
     * @param context Draw context identifying entities to remove
     */
    virtual void removeEntities(const CC_DRAW_CONTEXT& context) override;

    /**
     * @brief Show or hide entities
     * @param context Draw context identifying entities and visibility state
     * @return true if entities were found and affected
     */
    virtual bool hideShowEntities(const CC_DRAW_CONTEXT& context) override;

    /** \brief Create a new viewport from [xmin,ymin] -> [xmax,ymax].
     * \param[in] xmin the minimum X coordinate for the viewport (0.0 <= 1.0)
     * \param[in] ymin the minimum Y coordinate for the viewport (0.0 <= 1.0)
     * \param[in] xmax the maximum X coordinate for the viewport (0.0 <= 1.0)
     * \param[in] ymax the maximum Y coordinate for the viewport (0.0 <= 1.0)
     * \param[in] viewport the id of the new viewport
     *
     * \note If no renderer for the current window exists, one will be created,
     * and the viewport will be set to 0 ('all'). In case one or multiple
     * renderers do exist, the viewport ID will be set to the total number of
     * renderers - 1.
     */
    inline virtual void createViewPort(double xmin,
                                       double ymin,
                                       double xmax,
                                       double ymax,
                                       int& viewport) override {
        m_visualizer3D->createViewPort(xmin, ymin, xmax, ymax, viewport);
    }

    inline virtual void resetCameraViewpoint(
            const std::string& viewID) override {
        m_visualizer3D->resetCameraViewpoint(viewID);
    }

    inline virtual void setBackgroundColor(
            const CC_DRAW_CONTEXT& context) override {
        getQVtkWidget()->setBackgroundColor(
                ecvTools::TransFormRGB(context.backgroundCol),
                ecvTools::TransFormRGB(context.backgroundCol2),
                context.drawBackgroundGradient);
    }

    inline virtual void showOrientationMarker() override {
        m_visualizer3D->showPclMarkerAxes(
                m_visualizer3D->getRenderWindowInteractor());
    }

    inline virtual void drawCoordinates(double scale = 1.0,
                                        const std::string& id = "reference",
                                        int viewport = 0) override {
        m_visualizer3D->addCoordinateSystem(scale, id, viewport);
    }

    inline virtual void rotateWithAxis(const CCVector2i& pos,
                                       const CCVector3d& axis,
                                       double angle,
                                       int viewport = 0) override {
        m_visualizer3D->rotateWithAxis(pos, axis, angle, viewport);
    }

public:
    // set and get camera parameters
    inline virtual void resetCamera() override {
        m_visualizer3D->resetCamera();
    }
    inline virtual void resetCamera(const ccBBox* bbox) override {
        m_visualizer3D->resetCamera(bbox);
    }
    inline virtual void updateCamera() override {
        // PCL's updateCamera() is deprecated and will be removed in PCL 1.15
        // Only call it for PCL versions < 1.15
#if defined(PCL_VERSION_COMPARE)
#if PCL_VERSION_COMPARE(<, 1, 13, 0)
        m_visualizer3D->updateCamera();
#endif
#elif defined(PCL_MAJOR_VERSION) && defined(PCL_MINOR_VERSION)
#if (PCL_MAJOR_VERSION < 1) || \
        (PCL_MAJOR_VERSION == 1 && PCL_MINOR_VERSION < 13)
        m_visualizer3D->updateCamera();
#endif
#else
        // If version macros are not available, assume older version and call it
        // This maintains backward compatibility
        // m_visualizer3D->updateCamera();
#endif
    }

    inline virtual void updateScene() override {
        getQVtkWidget()->updateScene();
    }

    inline virtual void setAutoUpateCameraPos(bool state) override {
        if (this->m_visualizer3D) {
            this->m_visualizer3D->setAutoUpateCameraPos(state);
        }
    }

    /**
     * Get the current center of rotation
     */
    inline virtual void getCenterOfRotation(double center[3]) override {
        if (this->m_visualizer3D) {
            this->m_visualizer3D->getCenterOfRotation(center);
        }
    }

    /**
     * Resets the center of rotation to the focal point.
     */
    inline virtual void resetCenterOfRotation(int viewport = 0) override {
        if (this->m_visualizer3D) {
            this->m_visualizer3D->resetCenterOfRotation(viewport);
        }
    }

    /**
     * Set the center of rotation. For this to work,
     * one should have appropriate interaction style
     * and camera manipulators that use the center of rotation
     * They are setup correctly by default
     */
    inline virtual void setCenterOfRotation(double x,
                                            double y,
                                            double z) override {
        if (this->m_visualizer3D) {
            this->m_visualizer3D->setCenterOfRotation(x, y, z);
        }
    }
    inline void setCenterOfRotation(const double xyz[3]) {
        this->setCenterOfRotation(xyz[0], xyz[1], xyz[2]);
    }

    virtual void setPivotVisibility(bool state) override {
        if (this->m_visualizer3D) {
            this->m_visualizer3D->setCenterAxesVisibility(state);
        }
    }

    inline virtual void resetCameraClippingRange(int viewport = 0) override {
        if (m_visualizer3D) {
            m_visualizer3D->resetCameraClippingRange(viewport);
        }
    }

    inline virtual double getGLDepth(int x, int y) override {
        if (m_visualizer3D) {
            return m_visualizer3D->getGLDepth(x, y);
        } else {
            return 1.0;
        }
    }

    inline virtual void zoomCamera(double zoomFactor,
                                   int viewport = 0) override {
        m_visualizer3D->zoomCamera(zoomFactor, viewport);
    }

    inline virtual double getCameraFocalDistance(int viewport = 0) override {
        return m_visualizer3D->getCameraFocalDistance(viewport);
    }
    inline virtual void setCameraFocalDistance(double focal_distance,
                                               int viewport = 0) override {
        m_visualizer3D->setCameraFocalDistance(focal_distance, viewport);
    }

    inline virtual void getCameraPos(double* pos, int viewport = 0) override {
        const PclUtils::Camera& cam = m_visualizer3D->getCamera(viewport);
        pos[0] = cam.pos[0];
        pos[1] = cam.pos[1];
        pos[2] = cam.pos[2];
    }
    inline virtual void getCameraFocal(double* focal,
                                       int viewport = 0) override {
        const PclUtils::Camera& cam = m_visualizer3D->getCamera(viewport);
        focal[0] = cam.focal[0];
        focal[1] = cam.focal[1];
        focal[2] = cam.focal[2];
    }
    inline virtual void getCameraUp(double* up, int viewport = 0) override {
        const PclUtils::Camera& cam = m_visualizer3D->getCamera(viewport);
        up[0] = cam.view[0];
        up[1] = cam.view[1];
        up[2] = cam.view[2];
    }

    inline virtual void setCameraPosition(const CCVector3d& pos,
                                          int viewport = 0) override {
        getQVtkWidget()->setCameraPosition(pos);
    }

    inline virtual void setCameraPosition(const double* pos,
                                          const double* focal,
                                          const double* up,
                                          int viewport = 0) override {
        m_visualizer3D->setCameraPosition(
                CCVector3d(pos[0], pos[1], pos[2]),
                CCVector3d(focal[0], focal[1], focal[2]),
                CCVector3d(up[0], up[1], up[2]), viewport);
    }

    inline virtual void setCameraPosition(const double* pos,
                                          const double* up,
                                          int viewport = 0) override {
        m_visualizer3D->setCameraPosition(CCVector3d(pos[0], pos[1], pos[2]),
                                          CCVector3d(up[0], up[1], up[2]),
                                          viewport);
    }

    inline virtual void setCameraPosition(double pos_x,
                                          double pos_y,
                                          double pos_z,
                                          double view_x,
                                          double view_y,
                                          double view_z,
                                          double up_x,
                                          double up_y,
                                          double up_z,
                                          int viewport = 0) override {
        m_visualizer3D->setCameraPosition(CCVector3d(pos_x, pos_y, pos_z),
                                          CCVector3d(view_x, view_y, view_z),
                                          CCVector3d(up_x, up_y, up_z),
                                          viewport);
    }

    inline virtual void setRenderWindowSize(int xw, int yw) override {
        getQVtkWidget()->GetRenderWindow()->SetPosition(0, 0);
        getQVtkWidget()->GetRenderWindow()->SetSize(xw, yw);
    }

    inline virtual void fullScreen(bool state) override {
        m_visualizer3D->setFullScreen(state);
    }

    inline virtual void setOrthoProjection(int viewport = 0) override {
        if (m_visualizer3D) {
            m_visualizer3D->setOrthoProjection(viewport);
            m_visualizer3D->resetCameraClippingRange(viewport);
        }
    }
    inline virtual void setPerspectiveProjection(int viewport = 0) override {
        if (m_visualizer3D) {
            m_visualizer3D->setPerspectiveProjection(viewport);
            m_visualizer3D->resetCameraClippingRange(viewport);
        }
    }

    // set and get clip distances (near and far)
    inline virtual void getCameraClip(double* clipPlanes,
                                      int viewport = 0) override {
        const PclUtils::Camera& cam = m_visualizer3D->getCamera(viewport);
        clipPlanes[0] = cam.clip[0];
        clipPlanes[1] = cam.clip[1];
    }
    inline virtual void setCameraClip(double znear,
                                      double zfar,
                                      int viewport = 0) override {
        if (m_visualizer3D) {
            m_visualizer3D->setCameraClipDistances(znear, zfar, viewport);
        }
    }

    // set and get view angle in y direction or zoom factor in perspective mode
    inline virtual double getCameraFovy(int viewport = 0) override {
        return cloudViewer::RadiansToDegrees(
                m_visualizer3D->getCamera(viewport).fovy);
    }
    inline virtual void setCameraFovy(double fovy, int viewport = 0) override {
        m_visualizer3D->setCameraFieldOfView(fovy, viewport);
    }

    // get zoom factor in parallel mode
    virtual double getParallelScale(int viewport = 0) override;
    virtual void setParallelScale(double scale, int viewport = 0) override;

    /** \brief Save the current rendered image to disk, as a PNG screen shot.
     * \param[in] file the name of the PNG file
     */
    inline virtual void saveScreenshot(const std::string& file) override {
        m_visualizer3D->saveScreenshot(file);
    }

    /** \brief Save or Load the current rendered camera parameters to disk or
     * current camera. \param[in] file the name of the param file
     */
    inline virtual void saveCameraParameters(const std::string& file) override {
        m_visualizer3D->saveCameraParameters(file);
    }
    inline virtual void loadCameraParameters(const std::string& file) override {
        m_visualizer3D->loadCameraParameters(file);
    }

    /** \brief Use Vertex Buffer Objects renderers.
     * This is an optimization for the obsolete OpenGL backend. Modern OpenGL2
     * backend (VTK6.3) uses vertex buffer objects by default, transparently for
     * the user. \param[in] use_vbos set to true to use VBOs
     */
    inline virtual void setUseVbos(bool useVbos) override {
        m_visualizer3D->setUseVbos(useVbos);
    }

    /** \brief Set the ID of a cloud or shape to be used for LUT display
     * \param[in] id The id of the cloud/shape look up table to be displayed
     * The look up table is displayed by pressing 'u' in the PCLVisualizer */
    inline virtual void setLookUpTableID(const std::string& viewID) override {
        m_visualizer3D->setLookUpTableID(viewID);
    }

    virtual void getProjectionMatrix(double* projArray,
                                     int viewport = 0) override;
    virtual void getViewMatrix(double* ViewArray, int viewport = 0) override;
    virtual void setViewMatrix(const ccGLMatrixd& viewMat,
                               int viewport = 0) override;

    virtual void changeOpacity(double opacity,
                               const std::string& viewID,
                               int viewport = 0) override;

public:
    // =====================================================================
    // Viewer Access and Picking
    // =====================================================================

    /**
     * @brief Get 3D visualizer (PCLVis)
     * @return Pointer to PCLVis instance
     */
    inline PclUtils::PCLVis* get3DViewer() { return m_visualizer3D.get(); }

    /**
     * @brief Get 2D visualizer (ImageVis)
     * @return Pointer to ImageVis instance
     */
    inline PclUtils::ImageVis* get2DViewer() { return m_visualizer2D.get(); }

    /**
     * @brief Pick 2D label at screen coordinates
     * @param x Screen X coordinate in pixels
     * @param y Screen Y coordinate in pixels
     * @return Label text at the picked position, or empty if none
     *
     * Used for picking text labels in 2D overlay.
     */
    virtual QString pick2DLabel(int x, int y) override;

    /**
     * @brief Pick 3D item at screen coordinates
     * @param x Screen X coordinate (default: -1, uses cursor position)
     * @param y Screen Y coordinate (default: -1, uses cursor position)
     * @return View ID of picked item, or empty if none
     *
     * Performs 3D picking to identify entities at screen position.
     */
    virtual QString pick3DItem(int x = -1, int y = -1) override;

    /**
     * @brief Pick object at screen coordinates
     * @param x Screen X coordinate (default: -1, uses cursor position)
     * @param y Screen Y coordinate (default: -1, uses cursor position)
     * @return View ID of picked object, or empty if none
     *
     * Alternative picking method for objects.
     */
    virtual QString pickObject(double x = -1, double y = -1) override;

    /**
     * @brief Render viewport to image
     * @param zoomFactor Resolution multiplier (default: 1)
     * @param renderOverlayItems Whether to include 2D overlays (default: false)
     * @param silent Suppress status messages (default: false)
     * @param viewport Viewport ID to render (default: 0)
     * @return QImage containing rendered scene
     *
     * Captures the current viewport to an image for export or display.
     */
    virtual QImage renderToImage(int zoomFactor = 1,
                                 bool renderOverlayItems = false,
                                 bool silent = false,
                                 int viewport = 0) override;

    /**
     * @brief Set scale bar visibility
     * @param visible true to show scale bar, false to hide
     */
    virtual void setScaleBarVisible(bool visible) override {
        if (getQVtkWidget()) getQVtkWidget()->setScaleBarVisible(visible);
    }

    //  ========== View Properties (ParaView-compatible) ==========
    // Note: These are PCLDisplayTools-specific methods, not part of
    // ecvDisplayTools interface They delegate to PCLVis for VTK-level
    // implementation

    /**
     * @brief Set Data Axes Grid properties (ParaView-style)
     * Each ccHObject has its own Data Axes Grid bound to its viewID
     * @param viewID The view ID of the ccHObject
     * @param visible Whether the grid is visible
     * @param color RGB color of axes and grid lines
     * @param lineWidth Width of grid lines
     * @param spacing Spacing between grid lines
     * @param subdivisions Number of subdivisions
     * @param showLabels Whether to show axis labels
     * @param opacity Opacity of axes and grid lines
     */
    void SetDataAxesGridProperties(const std::string& viewID,
                                   bool visible,
                                   const std::array<double, 3>& color,
                                   double lineWidth,
                                   double spacing,
                                   int subdivisions,
                                   bool showLabels,
                                   double opacity);

    /**
     * @brief Get Data Axes Grid properties for a specific object
     * @param viewID The view ID of the ccHObject
     */
    void GetDataAxesGridProperties(const std::string& viewID,
                                   bool& visible,
                                   std::array<double, 3>& color,
                                   double& lineWidth,
                                   double& spacing,
                                   int& subdivisions,
                                   bool& showLabels,
                                   double& opacity) const;

    /**
     * @brief Toggle Camera Orientation Widget visibility (ParaView-style)
     * @param show true to show, false to hide
     */
    void ToggleCameraOrientationWidget(bool show);

    /**
     * @brief Check if Camera Orientation Widget is shown
     * @return true if visible, false otherwise
     */
    bool IsCameraOrientationWidgetShown() const;

    // Override base class virtual methods
    void toggleCameraOrientationWidget(bool show) override;
    bool isCameraOrientationWidgetShown() const override;

    /**
     * @brief Set global default light intensity (ParaView-style)
     * Modifies the renderer's headlight intensity for the entire scene.
     * @param intensity Light intensity (0.0-1.0, default 1.0)
     */
    void setLightIntensity(double intensity) override;

    /**
     * @brief Get current global default light intensity
     * @return Current light intensity (0.0-1.0)
     */
    double getLightIntensity() const override;

    /**
     * @brief Set light intensity for a specific object (per-object)
     * @param viewID The view ID of the target ccHObject
     * @param intensity Light intensity (0.0-1.0)
     */
    void setObjectLightIntensity(const QString& viewID,
                                 double intensity) override;

    /**
     * @brief Get light intensity for a specific object
     * @param viewID The view ID of the target ccHObject
     * @return Object's light intensity (falls back to global default)
     */
    double getObjectLightIntensity(const QString& viewID) const override;

    // ========================================================================
    // Data Axes Grid (Unified Interface with AxesGridProperties)
    // ========================================================================

    /**
     * @brief Set Data Axes Grid properties (Unified Interface)
     * @param viewID The view ID of the ccHObject
     * @param props All axes grid properties encapsulated in AxesGridProperties
     * struct
     * @param viewport Viewport ID (default: 0)
     */
    void setDataAxesGridProperties(const QString& viewID,
                                   const AxesGridProperties& props,
                                   int viewport = 0) override;

    /**
     * @brief Get Data Axes Grid properties (Unified Interface)
     * @param viewID The view ID of the ccHObject
     * @param props Output: current axes grid properties
     * @param viewport Viewport ID (default: 0)
     */
    void getDataAxesGridProperties(const QString& viewID,
                                   AxesGridProperties& props,
                                   int viewport = 0) const override;

private:
    // =====================================================================
    // Entity-specific Drawing Methods
    // =====================================================================

    /// Draw point cloud entity
    void drawPointCloud(const CC_DRAW_CONTEXT& context, ccPointCloud* ecvCloud);

    /// Draw mesh entity
    void drawMesh(CC_DRAW_CONTEXT& context, ccGenericMesh* mesh);

    /// Draw polyline entity
    void drawPolygon(const CC_DRAW_CONTEXT& context, ccPolyline* polyline);

    /// Draw line set entity
    void drawLines(const CC_DRAW_CONTEXT& context,
                   cloudViewer::geometry::LineSet* lineset);

    /// Draw image entity (2D image overlay)
    void drawImage(const CC_DRAW_CONTEXT& context, ccImage* image);

    /// Draw sensor entity (frustum or representation)
    void drawSensor(const CC_DRAW_CONTEXT& context, ccSensor* sensor);

    /**
     * @brief Update entity color properties
     * @param context Draw context with color information
     * @param ent Entity to update
     * @return true if color was updated
     *
     * Updates the color of an entity based on context settings.
     */
    bool updateEntityColor(const CC_DRAW_CONTEXT& context, ccHObject* ent);

protected:
    // =====================================================================
    // Protected Members
    // =====================================================================

    /// Qt/VTK widget for OpenGL rendering
    QVTKWidgetCustom* m_vtkWidget = nullptr;

    /// 2D image visualizer instance
    PclUtils::ImageVisPtr m_visualizer2D = nullptr;

    /// 3D point cloud/mesh visualizer instance
    PclUtils::PCLVisPtr m_visualizer3D = nullptr;

    /**
     * @brief Register visualizer with Qt widget
     * @param widget Main window to register with
     * @param stereoMode Whether to enable stereo rendering (default: false)
     *
     * Initializes VTK/Qt widget and creates visualizer instances.
     */
    virtual void registerVisualizer(QMainWindow* widget,
                                    bool stereoMode = false) override;
};
