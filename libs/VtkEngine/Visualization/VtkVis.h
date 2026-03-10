// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file VtkVis.h
/// @brief Main 3D visualizer for ACloudViewer (VTK-based rendering).

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

#include <map>
#include <mutex>
#include <thread>

#include "WidgetMap.h"
#include "qVTK.h"

// VtkRendering
#include <VtkRendering/Core/ActorMap.h>

// Forward declaration
namespace Visualization {
namespace renders {
class TextureRenderManager;
}
}  // namespace Visualization

// CV_DB_LIB
#include <ecvColorTypes.h>
#include <ecvDisplayTools.h>  // For AxesGridProperties
#include <ecvDrawContext.h>
#include <ecvGenericVisualizer3D.h>
#include <ecvHObject.h>

// VTK
#include <vtkBoundingBox.h>  // needed for iVar
#include <vtkCellArray.h>
#include <vtkDataSetMapper.h>
#include <vtkLODActor.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>

class vtkCamera;
class vtkRender;
class vtkRenderer;
class vtkRendererCollection;
class vtkPointPicker;
class vtkAreaPicker;
class vtkPropPicker;
class vtkAbstractWidget;
class vtkMatrix4x4;
class vtkLightKit;
class vtkCubeAxesActor;
class vtkCameraOrientationWidget;
class vtkOrientationMarkerWidget;
class vtkPolyData;
class vtkTransform;
class ccGenericMesh;
class ccPointCloud;
class ccMesh;
class ccHObject;
class ccBBox;
class ecvOrientedBBox;
class ccSensor;
class ecvPointpickingTools;
class ccMaterial;
class ccMaterialSet;
class ccPolyline;

namespace cloudViewer {
namespace geometry {
class LineSet;
}
}  // namespace cloudViewer

namespace VTKExtensions {
class vtkPVCenterAxesActor;
class vtkCustomInteractorStyle;
}  // namespace VTKExtensions

namespace Visualization {

/// @class VtkVis
/// @brief Main 3D visualizer for ACloudViewer. Uses VTK directly for rendering.
/// Replaces the former pcl::visualization::PCLVisualizer inheritance
/// with a clean, PCL-free VTK implementation.
class QVTK_ENGINE_LIB_API VtkVis : public ecvGenericVisualizer3D {
    Q_OBJECT
public:
    /** @param interactor_style Custom interactor style for mouse/keyboard
     *  @param viewerName Name for the viewer window
     *  @param initIterator Whether to initialize the iterator
     *  @param argc Command-line argument count
     *  @param argv Command-line arguments
     */
    VtkVis(vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                   interactor_style,
           const std::string& viewerName = "",
           bool initIterator = false,
           int argc = 0,
           char** argv = nullptr);
    /** @param ren VTK renderer
     *  @param wind VTK render window
     *  @param interactor_style Custom interactor style
     *  @param viewerName Name for the viewer window
     *  @param initIterator Whether to initialize the iterator
     *  @param argc Command-line argument count
     *  @param argv Command-line arguments
     */
    VtkVis(vtkSmartPointer<vtkRenderer> ren,
           vtkSmartPointer<vtkRenderWindow> wind,
           vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                   interactor_style,
           const std::string& viewerName = "",
           bool initIterator = false,
           int argc = 0,
           char** argv = nullptr);

    virtual ~VtkVis();

    /// Performs initialization (renderer, camera, etc.).
    void initialize();

    /// Configures the center axes actor.
    void configCenterAxes();

    /** @param interactor_style Custom interactor style to configure
     */
    void configInteractorStyle(
            vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                    interactor_style);

public:
    /** \brief Marker Axes.
     */
    void hidePclMarkerAxes();
    /** @return true if PCL marker axes are currently shown
     */
    bool pclMarkerAxesShown();
    /** @param interactor Optional interactor for the axes widget
     */
    void showPclMarkerAxes(vtkRenderWindowInteractor* interactor = nullptr);
    void hideOrientationMarkerWidgetAxes();
    /** @param interactor Interactor for the orientation marker widget
     */
    void showOrientationMarkerWidgetAxes(vtkRenderWindowInteractor* interactor);
    void toggleOrientationMarkerWidgetAxes();

    /** \brief Internal method. Adds a vtk actor to screen.
     * \param[in] actor a pointer to the vtk actor object
     * \param[in] viewport the view port where the actor should be added to
     * (default: all)
     */
    /** @param actor VTK prop to remove
     *  @param viewport Viewport ID (0 = all)
     *  @return true on success
     */
    bool removeActorFromRenderer(const vtkSmartPointer<vtkProp>& actor,
                                 int viewport = 0);

    /** @param actor VTK prop to add
     *  @param viewport Viewport ID (0 = all)
     */
    void addActorToRenderer(const vtkSmartPointer<vtkProp>& actor,
                            int viewport = 0);

    /**
     * @brief UpdateScreen - Updates/refreshes the render window
     * This method forces a render update after actor changes
     */
    void UpdateScreen();

    /**
     * @brief setupInteractor override to init interactor_
     * @param iren VTK render window interactor
     * @param win VTK render window
     */
    void setupInteractor(vtkRenderWindowInteractor* iren, vtkRenderWindow* win);

    /** \brief Get a pointer to the current interactor style used. */
    inline vtkSmartPointer<vtkRenderWindowInteractor>
    getRenderWindowInteractor() {
        return (interactor_);
    }

    /// Camera parameters (replaces pcl::visualization::Camera).
    struct CameraParams {
        double pos[3] = {0, 0, 0};
        double focal[3] = {0, 0, 0};
        double view[3] = {0, 0, 1};
        double clip[2] = {0.01, 1000};
        double fovy = 0.8575;
        double window_size[2] = {0, 0};
        double window_pos[2] = {0, 0};
    };

    /** @param viewport Viewport ID (default 0)
     *  @return Camera parameters (position, focal, view, etc.)
     */
    CameraParams getCamera(int viewport = 0);
    /** @param viewport Viewport ID (default 0)
     *  @return VTK camera smart pointer
     */
    vtkSmartPointer<vtkCamera> getVtkCamera(int viewport = 0);

    /** @param viewMat Model-view matrix to apply
     *  @param viewport Viewport ID (default 0)
     */
    void setModelViewMatrix(const ccGLMatrixd& viewMat, int viewport = 0);
    /** @param viewport Viewport ID (default 0)
     *  @return Parallel scale (zoom in parallel projection)
     */
    double getParallelScale(int viewport = 0);
    /** @param scale Parallel scale value
     *  @param viewport Viewport ID (default 0)
     */
    void setParallelScale(double scale, int viewport = 0);

    /** @param viewport Viewport ID (default 0)
     */
    void setOrthoProjection(int viewport = 0);
    /** @param viewport Viewport ID (default 0)
     */
    void setPerspectiveProjection(int viewport = 0);
    /** @param viewport Viewport ID (default 0)
     *  @return true if perspective projection is active
     */
    bool getPerspectiveState(int viewport = 0);

    inline bool getAutoUpateCameraPos() { return m_autoUpdateCameraPos; }
    inline void setAutoUpateCameraPos(bool state) {
        m_autoUpdateCameraPos = state;
    }

    /** @param pos Screen position for rotation
     *  @param axis Rotation axis in 3D
     *  @param angle Rotation angle (radians)
     *  @param viewport Viewport ID (default 0)
     */
    void rotateWithAxis(const CCVector2i& pos,
                        const CCVector3d& axis,
                        double angle,
                        int viewport = 0);

public:
    /**
     * Get the current center of rotation
     * @param center Output array [3] for center coordinates
     */
    void getCenterOfRotation(double center[3]);
    /**
     * Resets the center of rotation to the focal point.
     * @param viewport Viewport ID (default 0)
     */
    void resetCenterOfRotation(int viewport = 0);

    /** @param bounds Bounding box [6] to expand
     *  @param matrix Transformation matrix to apply
     */
    static void ExpandBounds(double bounds[6], vtkMatrix4x4* matrix);

    /**
     * Set the center of rotation. For this to work,
     * one should have appropriate interaction style
     * and camera manipulators that use the center of rotation
     * They are setup correctly by default
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     */
    void setCenterOfRotation(double x, double y, double z);
    inline void setCenterOfRotation(double xyz[3]) {
        setCenterOfRotation(xyz[0], xyz[1], xyz[2]);
    }

    /** @param factor Rotation sensitivity factor
     */
    void setRotationFactor(double factor);
    /** @return Current rotation factor
     */
    double getRotationFactor();

    /// Forwarded to center axes.
    /** @param visible true to show, false to hide
     */
    void setCenterAxesVisibility(bool);

    /// Forwarded to vtkPVInteractorStyle if present on local processes.
    /** @param manipulators Array of 9 manipulator IDs
     */
    virtual void setCamera2DManipulators(const int manipulators[9]);
    /** @param manipulators Array of 9 manipulator IDs
     */
    virtual void setCamera3DManipulators(const int manipulators[9]);
    /** @param style Interactor style to configure
     *  @param manipulators Array of 9 manipulator IDs
     */
    void setCameraManipulators(VTKExtensions::vtkCustomInteractorStyle* style,
                               const int manipulators[9]);
    /** @param factor Mouse wheel zoom factor for 2D
     */
    virtual void setCamera2DMouseWheelMotionFactor(double factor);
    /** @param factor Mouse wheel zoom factor for 3D
     */
    virtual void setCamera3DMouseWheelMotionFactor(double factor);
    /**
     * updateCenterAxes().
     * updates CenterAxes's scale and position.
     */
    virtual void updateCenterAxes();

    /**
     * Synchronizes bounds information on all nodes.
     * \note CallOnAllProcesses
     * @param viewport Viewport ID (default 0)
     */
    void synchronizeGeometryBounds(int viewport = 0);

    /** @param x Screen X coordinate
     *  @param y Screen Y coordinate
     *  @return Depth value for z-buffer at (x,y)
     */
    double getGLDepth(int x, int y);

    /** @param viewport Viewport ID (default 0)
     *  @return Camera focal distance
     */
    double getCameraFocalDistance(int viewport = 0);
    /** @param focal_distance Focal distance to set
     *  @param viewport Viewport ID (default 0)
     */
    void setCameraFocalDistance(double focal_distance, int viewport = 0);

    /**
     * In perspective mode, decrease the view angle by the specified factor.
     * In parallel mode, decrease the parallel scale by the specified factor.
     * A value greater than 1 is a zoom-in, a value less than 1 is a zoom-out.
     * @note This setting is ignored when UseExplicitProjectionTransformMatrix
     * is true.
     */
    /** @param zoomFactor Zoom factor (>1 zoom in, <1 zoom out)
     *  @param viewport Viewport ID (default 0)
     */
    void zoomCamera(double zoomFactor, int viewport = 0);

    /** @param proj Output projection matrix
     *  @param viewport Viewport ID (default 0)
     */
    void getProjectionTransformMatrix(Eigen::Matrix4d& proj, int viewport = 0);

    /** @param view Output model-view matrix
     *  @param viewport Viewport ID (default 0)
     */
    void getModelViewTransformMatrix(Eigen::Matrix4d& view, int viewport = 0);

    /**
     * Automatically set up the camera based on a specified bounding box
     * (xmin, xmax, ymin, ymax, zmin, zmax). Camera will reposition itself so
     * that its focal point is the center of the bounding box, and adjust its
     * distance and position to preserve its initial view plane normal
     * (i.e., vector defined from camera position to focal point). Note: is
     * the view plane is parallel to the view up axis, the view up axis will
     * be reset to one of the three coordinate axes.
     */
    /** @param viewport Viewport ID (default 0)
     */
    void resetCameraClippingRange(int viewport = 0);
    void internalResetCameraClippingRange() {
        this->resetCameraClippingRange(0);
    }
    /** @param bbox Bounding box to fit camera to
     */
    void resetCamera(const ccBBox* bbox);
    /** @param xMin,xMax,yMin,yMax,zMin,zMax Bounding box extents
     */
    void resetCamera(double xMin,
                     double xMax,
                     double yMin,
                     double yMax,
                     double zMin,
                     double zMax);
    void resetCamera();
    /** @param bounds Bounding box array [6]
     */
    inline void resetCamera(double bounds[6]) {
        resetCamera(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4],
                    bounds[5]);
    }
    /** @param range Output [2] for near/far clip distances
     *  @param viewport Viewport ID (default 0)
     */
    void getReasonableClippingRange(double range[2], int viewport = 0);
    /** @param bounds Bounding box [6] to expand
     *  @param matrix Transformation matrix
     */
    void expandBounds(double bounds[6], vtkMatrix4x4* matrix);
    /** @param viewAngle Camera field-of-view angle (degrees)
     *  @param viewport Viewport ID (default 0)
     */
    void setCameraViewAngle(double viewAngle, int viewport = 0);

    /// Draw a point cloud (direct CV_db to VTK, no PCL intermediate).
    void drawPointCloud(const CC_DRAW_CONTEXT& context, ccPointCloud* cloud);
    /// Draw a mesh (direct CV_db to VTK, no PCL intermediate).
    void drawMesh(const CC_DRAW_CONTEXT& context, ccGenericMesh* mesh);
    /// Draw a polyline (direct CV_db to VTK, no PCL intermediate).
    void drawPolyline(const CC_DRAW_CONTEXT& context,
                      ccPolyline* polyline,
                      bool closed);
    /// Draw a sensor visualization.
    void drawSensor(const CC_DRAW_CONTEXT& context, const ccSensor* sensor);
    /// Draw a line set visualization.
    void drawLineSet(const CC_DRAW_CONTEXT& context,
                     const cloudViewer::geometry::LineSet* lineset);

    void transformEntities(const CC_DRAW_CONTEXT& context);
    vtkSmartPointer<vtkTransform> getTransformation(
            const CC_DRAW_CONTEXT& context, const CCVector3d& origin);
    void updateNormals(const CC_DRAW_CONTEXT& context, ccPointCloud* cloud);
    void updateShadingMode(const CC_DRAW_CONTEXT& context, ccPointCloud* cloud);
    bool removeEntities(const CC_DRAW_CONTEXT& context);
    void hideShowActors(bool visibility,
                        const std::string& viewID,
                        int viewport = 0);
    void hideShowWidgets(bool visibility,
                         const std::string& viewID,
                         int viewport = 0);

    /** @param context Draw context
     *  @return true on success
     */
    bool addScalarBar(const CC_DRAW_CONTEXT& context);
    /** @param context Draw context
     *  @return true on success
     */
    bool updateScalarBar(const CC_DRAW_CONTEXT& context);
    bool addCaption(const std::string& text,
                    const CCVector2& pos2D,
                    const CCVector3& anchorPos,
                    double r,
                    double g,
                    double b,
                    double a,
                    int fontSize = 10,
                    const std::string& viewID = "caption",
                    bool anchorDragable = false,
                    int viewport = 0);

    bool updateCaption(const std::string& text,
                       const CCVector2& pos2D,
                       const CCVector3& anchorPos,
                       double r,
                       double g,
                       double b,
                       double a,
                       int fontSize = 10,
                       const std::string& viewID = "caption",
                       int viewport = 0);

    //! Get caption widget 2D position (normalized coordinates 0.0-1.0)
    /** Returns false if widget not found or invalid
     *  \param viewID widget view ID
     *  \param posX output X position (0.0-1.0, left to right)
     *  \param posY output Y position (0.0-1.0, bottom to top in VTK coordinate
     *system) \return true if position retrieved successfully
     **/
    bool getCaptionPosition(const std::string& viewID,
                            float& posX,
                            float& posY);

    bool addPolyline(vtkSmartPointer<vtkPolyData> polydata,
                     double r,
                     double g,
                     double b,
                     float width = 1.0f,
                     const std::string& id = "multiline",
                     int viewport = 0);
    /** @param context Draw context
     *  @param materials Material set for texture update
     *  @return true on success
     */
    bool updateTexture(const CC_DRAW_CONTEXT& context,
                       const ccMaterialSet* materials);
    /**
     * @brief Add texture mesh directly from ccGenericMesh (preferred)
     * @param mesh ccGenericMesh object containing geometry and materials
     * @param id Unique identifier for the mesh
     * @param viewport Viewport ID (default: 0)
     * @return true on success
     * @note This method directly uses ccMaterialSet, avoiding pcl::TexMaterial
     * encoding
     */
    bool addTextureMeshFromCCMesh(ccGenericMesh* mesh,
                                  const std::string& id,
                                  int viewport = 0);

    /**
     * @brief Load multi-texture mesh from OBJ file (enhanced version)
     * @param obj_path OBJ file path
     * @param id Unique identifier
     * @param viewport Viewport ID
     * @param quality Texture quality (0=low, 1=medium, 2=high, 3=original)
     * @param enable_cache Whether to enable texture cache
     * @return Returns true on success
     */
    bool addTextureMeshFromOBJ(const std::string& obj_path,
                               const std::string& id,
                               int viewport = 0,
                               int quality = 2,
                               bool enable_cache = true);

    /**
     * @brief Load multi-texture mesh from OBJ file (advanced options)
     * @param obj_path OBJ file path
     * @param mtl_path MTL file path (optional)
     * @param id Unique identifier
     * @param viewport Viewport ID
     * @param max_texture_size Maximum texture size
     * @param use_mipmaps Whether to use mipmaps
     * @param enable_cache Whether to enable texture cache
     * @return Returns true on success
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
     */
    void clearTextureCache();

    /**
     * @brief Get texture cache information
     * @param count Output: cached texture count
     * @param memory_bytes Output: cache memory usage (bytes)
     */
    void getTextureCacheInfo(size_t& count, size_t& memory_bytes) const;

    bool addOrientedCube(const ccGLMatrixd& trans,
                         double width,
                         double height,
                         double depth,
                         double r = 1.0,
                         double g = 1.0,
                         double b = 1.0,
                         const std::string& id = "cube",
                         int viewport = 0);
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
    bool addOrientedCube(const ecvOrientedBBox& obb,
                         const std::string& id = "cube",
                         int viewport = 0);

    /// Add axis-aligned cube (bounding box wireframe/surface).
    bool addCube(double xMin,
                 double xMax,
                 double yMin,
                 double yMax,
                 double zMin,
                 double zMax,
                 double r,
                 double g,
                 double b,
                 const std::string& id = "cube",
                 int viewport = 0);

    /// Add line segment between two points.
    bool addLine(double x1,
                 double y1,
                 double z1,
                 double x2,
                 double y2,
                 double z2,
                 double r,
                 double g,
                 double b,
                 const std::string& id = "line",
                 int viewport = 0);

    /// Add sphere at center with given radius.
    bool addSphere(double cx,
                   double cy,
                   double cz,
                   double radius,
                   double r,
                   double g,
                   double b,
                   const std::string& id = "sphere",
                   int viewport = 0);

    void displayText(const CC_DRAW_CONTEXT& context);

private:
    // Texture rendering manager
    std::unique_ptr<renders::TextureRenderManager> texture_render_manager_;

    // Store transformation matrices to prevent memory leaks
    // Maps view ID to transformation matrix smart pointer
    std::map<std::string, vtkSmartPointer<vtkMatrix4x4>> transformation_map_;

public:
    /** @param pointSize Point size in pixels
     *  @param viewID Entity view ID
     *  @param viewport Viewport ID (default 0)
     */
    void setPointSize(const unsigned char pointSize,
                      const std::string& viewID,
                      int viewport = 0);
    /** @param viewID Entity view ID
     *  @param scalarName Name of scalar field to display
     *  @param viewport Viewport ID (default 0)
     */
    void setScalarFieldName(const std::string& viewID,
                            const std::string& scalarName,
                            int viewport = 0);

    /**
     * @brief Add scalar field data from ccPointCloud to VTK polydata
     * @param viewID The cloud ID
     * @param cloud The ccPointCloud containing scalar field
     * @param scalarFieldIndex Index of the scalar field to extract
     * @param viewport Viewport ID
     */
    void addScalarFieldToVTK(const std::string& viewID,
                             ccPointCloud* cloud,
                             int scalarFieldIndex,
                             int viewport = 0);

    /**
     * @brief Sync ALL scalar fields from ccPointCloud to VTK polydata
     *
     * This method ensures all scalar fields in the ccPointCloud are available
     * in the VTK polydata for selection/extraction operations.
     *
     * @param viewID The cloud ID
     * @param cloud The ccPointCloud containing scalar fields
     * @param viewport Viewport ID
     */
    void syncAllScalarFieldsToVTK(const std::string& viewID,
                                  ccPointCloud* cloud,
                                  int viewport = 0);

    /**
     * @brief Set the source object for selection operations
     *
     * This stores a reference to the original ccHObject (ccPointCloud or
     * ccMesh) that is being visualized. Used for direct extraction during
     * selection operations to bypass VTK→ccHObject conversion.
     * Supports multiple objects in the scene via viewID mapping.
     *
     * @param obj The source object (ccPointCloud or ccMesh)
     * @param viewID The view ID for the object
     */
    void setCurrentSourceObject(ccHObject* obj, const std::string& viewID);

    /**
     * @brief Remove a source object from the map
     * @param viewID The view ID of the object to remove
     */
    void removeSourceObject(const std::string& viewID);

    /**
     * @brief Get the source object for a given viewID
     * @param viewID The view ID to look up
     * @return The source ccHObject or nullptr if not found
     */
    ccHObject* getSourceObject(const std::string& viewID) const;

    /**
     * @brief Get the source object as ccPointCloud
     * @param viewID The view ID to look up
     * @return ccPointCloud pointer or nullptr if not a point cloud
     */
    ccPointCloud* getSourceCloud(const std::string& viewID) const;

    /**
     * @brief Get the source object as ccMesh
     * @param viewID The view ID to look up
     * @return ccMesh pointer or nullptr if not a mesh
     */
    ccMesh* getSourceMesh(const std::string& viewID) const;

    /**
     * @brief Check if a source object exists for the given viewID
     * @param viewID The view ID to check
     * @return true if source object exists
     */
    bool hasSourceObject(const std::string& viewID) const;

    void setPointCloudUniqueColor(double r,
                                  double g,
                                  double b,
                                  const std::string& viewID,
                                  int viewport = 0);
    void resetScalarColor(const std::string& viewID,
                          bool flag = true,
                          int viewport = 0);
    void setShapeUniqueColor(float r,
                             float g,
                             float b,
                             const std::string& viewID,
                             int viewport = 0);
    void setLineWidth(const unsigned char lineWidth,
                      const std::string& viewID,
                      int viewport = 0);
    void setMeshRenderingMode(MESH_RENDERING_MODE mode,
                              const std::string& viewID,
                              int viewport = 0);
    void setLightMode(const std::string& viewID, int viewport = 0);
    void setPointCloudOpacity(double opacity,
                              const std::string& viewID,
                              int viewport = 0);
    void setShapeOpacity(double opacity,
                         const std::string& viewID,
                         int viewport = 0);

    /**
     * @brief Set opacity for mesh (textured or non-textured)
     *
     * This method properly handles transparency for meshes, including:
     * - Meshes with textures: enables depth peeling and alpha blending
     * - Meshes without textures: simple opacity setting
     * - Automatic transparency rendering configuration when opacity < 1.0
     *
     * @param opacity Opacity value [0.0, 1.0] where 0.0 = fully
     * transparent, 1.0 = opaque
     * @param viewID The unique identifier of the mesh
     * @param viewport The viewport ID (default: 0)
     */
    void setMeshOpacity(double opacity,
                        const std::string& viewID,
                        int viewport = 0);

    /*
     * value = 0, PCL_VISUALIZER_SHADING_FLAT
     * value = 1, PCL_VISUALIZER_SHADING_GOURAUD
     * value = 2, PCL_VISUALIZER_SHADING_PHONG
     */
    void setShapeShadingMode(SHADING_MODE mode,
                             const std::string& viewID,
                             int viewport = 0);
    void setMeshShadingMode(SHADING_MODE mode,
                            const std::string& viewID,
                            int viewport = 0);

    /** @param viewId Entity or shape view ID
     *  @return VTK actor or nullptr
     */
    vtkActor* getActorById(const std::string& viewId);
    /** @param viewId Prop view ID
     *  @return VTK prop or nullptr
     */
    vtkProp* getPropById(const std::string& viewId);
    /** @param viewId View ID
     *  @return Prop collection for the view
     */
    vtkSmartPointer<vtkPropCollection> getPropCollectionById(
            const std::string& viewId);
    /** @param actor VTK prop to look up
     *  @return View ID string or empty if not found
     */
    std::string getIdByActor(vtkProp* actor);
    /** @param viewId Widget view ID
     *  @return VTK abstract widget or nullptr
     */
    vtkAbstractWidget* getWidgetById(const std::string& viewId);

    /**
     * Get the Current Renderer in the list.
     * @param viewport Viewport ID (default 0)
     * @return Renderer pointer, or NULL when at the end of the list
     */
    vtkRenderer* getCurrentRenderer(int viewport = 0);

public:
    /** \brief Check if the widgets or props with the given id was already added
     * to this visualizer. \param[in] id the id of the widgets or props to check
     * \return true if a widgets or props with the specified id was found
     */
    inline bool containWidget(const std::string& id) const {
        return (m_widget_map->find(id) != m_widget_map->end() ||
                m_prop_map->find(id) != m_prop_map->end());
    }

    /** \brief Return a pointer to the WidgetActorMap this visualizer uses. */
    inline WidgetActorMapPtr getWidgetActorMap() { return (m_widget_map); }

    /** \brief Return a pointer to the PropActorMap this visualizer uses. */
    inline PropActorMapPtr getPropActorMap() { return (m_prop_map); }

    inline vtkBoundingBox getVisibleGeometryBounds() { return GeometryBounds; }

    inline vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
    get2DInteractorStyle() {
        return TwoDInteractorStyle;
    }

    inline vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
    get3DInteractorStyle() {
        return ThreeDInteractorStyle;
    }

protected:
    /** \brief Internal list with actor pointers and name IDs for widgets. */
    WidgetActorMapPtr m_widget_map;

    /** \brief Internal list with actor pointers and name IDs for props. */
    PropActorMapPtr m_prop_map;

    vtkBoundingBox GeometryBounds;
    vtkSmartPointer<VTKExtensions::vtkPVCenterAxesActor> m_centerAxes;
    vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
            TwoDInteractorStyle;
    vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
            ThreeDInteractorStyle;

private:
    bool removeWidgets(const std::string& viewId, int viewport);
    void removePointClouds(const std::string& viewId, int viewport = 0);
    void removeShapes(const std::string& viewId, int viewport = 0);
    void removeMesh(const std::string& viewId, int viewport = 0);
    void removeText2D(const std::string& viewId, int viewport = 0);
    void removeText3D(const std::string& viewId, int viewport = 0);
    void removeALL(int viewport = 0);

private:
    void registerMouse();
    void registerKeyboard();
    void registerAreaPicking();
    void registerPointPicking();

    void registerInteractorStyle(bool useDefault = false);

    static void OnPointPicking(vtkObject* caller,
                               unsigned long event_id,
                               void* client_data,
                               void* call_data);
    static void OnAreaPicking(vtkObject* caller,
                              unsigned long event_id,
                              void* client_data,
                              void* call_data);
    static void OnKeyPress(vtkObject* caller,
                           unsigned long event_id,
                           void* client_data,
                           void* call_data);
    static void OnRightButtonPress(vtkObject* caller,
                                   unsigned long event_id,
                                   void* client_data,
                                   void* call_data);

public:
    // Util Tools
    inline bool isPointPickingEnabled() { return m_pointPickingEnabled; }
    inline void setPointPickingEnabled(bool state) {
        m_pointPickingEnabled = state;
    }
    inline void togglePointPicking() {
        setPointPickingEnabled(!isPointPickingEnabled());
    }

    inline bool isAreaPickingEnabled() { return m_areaPickingEnabled; }
    inline void setAreaPickingEnabled(bool state) {
        m_areaPickingEnabled = state;
    }

    inline bool isActorPickingEnabled() { return m_actorPickingEnabled; }
    inline void setActorPickingEnabled(bool state) {
        m_actorPickingEnabled = state;
    }
    inline void toggleActorPicking() {
        setActorPickingEnabled(!isActorPickingEnabled());
    }

    void toggleAreaPicking();
    void exitCallbackProcess();
    /** @param state true to enable area picking
     */
    void setAreaPickingMode(bool state);
    /** @param x Screen X coordinate
     *  @param y Screen Y coordinate
     *  @return Picked actor or nullptr
     */
    vtkActor* pickActor(double x, double y);
    /** @param x0,y0,x1,y1 Pick rectangle (default: single point)
     *  @return View ID of picked item or empty string
     */
    std::string pickItem(double x0 = -1,
                         double y0 = -1,
                         double x1 = 5.0,
                         double y1 = 5.0);

    /** @param zoomFactor Image zoom factor (default 1)
     *  @param renderOverlayItems Whether to include overlay items
     *  @param silent Suppress render window updates
     *  @param viewport Viewport ID (default 0)
     *  @return Rendered image
     */
    QImage renderToImage(int zoomFactor = 1,
                         bool renderOverlayItems = false,
                         bool silent = false,
                         int viewport = 0);

protected:
    // Util Variables
    int m_currentMode;
    bool m_pointPickingEnabled;
    bool m_areaPickingEnabled;
    bool m_actorPickingEnabled;

    bool m_autoUpdateCameraPos;

    std::mutex m_cloud_mutex;

signals:
    void interactorPickedEvent(vtkActor* actor);

public:
    // ========== View Properties (ParaView-compatible) ==========

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

    // ========================================================================
    // Data Axes Grid (Unified Interface with DataAxesGridProperties struct)
    // ========================================================================

    /**
     * @brief Set Data Axes Grid properties (Unified Interface)
     *
     * Data Axes Grid shows axes and grid lines around the data bounds.
     * Uses vtkCubeAxesActor with FlyModeToOuterEdges.
     * Each ccHObject has its own Data Axes Grid bound to its viewID.
     *
     * @param viewID The view ID of the ccHObject to bind the axes grid to
     * @param props All axes grid properties encapsulated in AxesGridProperties
     * struct
     */
    void SetDataAxesGridProperties(const std::string& viewID,
                                   const AxesGridProperties& props);

    /**
     * @brief Get Data Axes Grid properties (Unified Interface)
     * @param viewID The view ID of the ccHObject
     * @param props Output: current axes grid properties
     */
    void GetDataAxesGridProperties(const std::string& viewID,
                                   AxesGridProperties& props) const;

    /**
     * @brief Remove Data Axes Grid for a specific object
     * @param viewID The view ID of the ccHObject
     */
    void RemoveDataAxesGrid(const std::string& viewID);

    /**
     * @brief Toggle Camera Orientation Widget visibility (ParaView-style)
     *
     * The Camera Orientation Widget provides an interactive 3D gizmo
     * for controlling camera orientation. Uses vtkCameraOrientationWidget.
     *
     * @param show true to show, false to hide
     */
    void ToggleCameraOrientationWidget(bool show);

    /**
     * @brief Check if Camera Orientation Widget is shown
     * @return true if visible, false otherwise
     */
    bool IsCameraOrientationWidgetShown() const;

public:
    // ---------- Methods formerly inherited from PCLVisualizer ----------

    /// Check if a cloud, shape, or coordinate with the given id exists.
    bool contains(const std::string& id) const;

    /// Get the renderer collection.
    vtkSmartPointer<vtkRendererCollection> getRendererCollection();

    /// Get the render window.
    vtkSmartPointer<vtkRenderWindow> getRenderWindow();

    /// Get the cloud actor map.
    VtkRendering::CloudActorMapPtr getCloudActorMap() {
        return cloud_actor_map_;
    }

    /// Get the shape actor map.
    VtkRendering::ShapeActorMapPtr getShapeActorMap() {
        return shape_actor_map_;
    }

    /// Add a coordinate system.
    void addCoordinateSystem(double scale,
                             const std::string& id = "reference",
                             int viewport = 0);

    /// Add 2D text to the viewport.
    bool addText(const std::string& text,
                 int xpos,
                 int ypos,
                 int fontsize,
                 double r,
                 double g,
                 double b,
                 const std::string& id,
                 int viewport = 0);

    /// Update existing 2D text content and position.
    bool updateText(const std::string& text,
                    int xpos,
                    int ypos,
                    const std::string& id);

    /// Add 3D text that follows the camera.
    bool addText3D(const std::string& text,
                   double px,
                   double py,
                   double pz,
                   double text_scale = 1.0,
                   double r = 1.0,
                   double g = 1.0,
                   double b = 1.0,
                   const std::string& id = "",
                   int viewport = 0);

    /// Remove a point cloud by ID.
    bool removePointCloud(const std::string& id, int viewport = 0);

    /// Remove a shape by ID.
    bool removeShape(const std::string& id, int viewport = 0);

    /// Remove a polygon mesh by ID.
    bool removePolygonMesh(const std::string& id, int viewport = 0);

    /// Add a pre-built vtkPolyData as a point cloud actor.
    bool addPointCloud(vtkSmartPointer<vtkPolyData> polydata,
                       const std::string& id = "cloud",
                       int viewport = 0);

    /// Update an existing point cloud actor with new polydata.
    bool updatePointCloud(vtkSmartPointer<vtkPolyData> polydata,
                          const std::string& id = "cloud");

    /// Set point cloud rendering properties.
    bool setPointCloudRenderingProperties(int property,
                                          double val,
                                          const std::string& id,
                                          int viewport = 0);
    /// Get point cloud rendering properties.
    bool getPointCloudRenderingProperties(int property,
                                          double& val,
                                          const std::string& id,
                                          int viewport = 0);
    bool setPointCloudRenderingProperties(int property,
                                          double v1,
                                          double v2,
                                          double v3,
                                          const std::string& id,
                                          int viewport = 0);

    /// Set shape rendering properties.
    bool setShapeRenderingProperties(int property,
                                     double val,
                                     const std::string& id,
                                     int viewport = 0);
    bool setShapeRenderingProperties(int property,
                                     double v1,
                                     double v2,
                                     double v3,
                                     const std::string& id,
                                     int viewport = 0);

    /// Create a new viewport.
    void createViewPort(
            double xmin, double ymin, double xmax, double ymax, int& viewport);

    /// Reset camera viewpoint for a given entity.
    void resetCameraViewpoint(const std::string& id);

    /// Set camera position (various overloads).
    void setCameraPosition(double pos_x,
                           double pos_y,
                           double pos_z,
                           double view_x,
                           double view_y,
                           double view_z,
                           double up_x,
                           double up_y,
                           double up_z,
                           int viewport = 0);
    void setCameraPosition(double pos_x,
                           double pos_y,
                           double pos_z,
                           double up_x,
                           double up_y,
                           double up_z,
                           int viewport = 0);

    /// Set camera clip distances.
    void setCameraClipDistances(double znear, double zfar, int viewport = 0);

    /// Set camera field of view.
    void setCameraFieldOfView(double fovy, int viewport = 0);

    /// Set full screen mode.
    void setFullScreen(bool mode);

    /// Save screenshot to file.
    void saveScreenshot(const std::string& file);

    /// Save/load camera parameters.
    void saveCameraParameters(const std::string& file);
    void loadCameraParameters(const std::string& file);

    /// Use vertex buffer objects.
    void setUseVbos(bool use_vbos);

    /// Set lookup table ID.
    void setLookUpTableID(const std::string& id);

    /// Set background color.
    void setBackgroundColor(double r, double g, double b, int viewport = 0);

    /// Rendering property constants (replaces pcl::visualization:: enums)
    enum RenderingProperties {
        PCL_VISUALIZER_POINT_SIZE = 0,
        PCL_VISUALIZER_OPACITY,
        PCL_VISUALIZER_LINE_WIDTH,
        PCL_VISUALIZER_FONT_SIZE,
        PCL_VISUALIZER_COLOR,
        PCL_VISUALIZER_REPRESENTATION,
        PCL_VISUALIZER_IMMEDIATE_RENDERING,
        PCL_VISUALIZER_SHADING,
        PCL_VISUALIZER_LUT,
        PCL_VISUALIZER_LUT_RANGE
    };

    enum ShadingRepresentationProperties {
        PCL_VISUALIZER_SHADING_FLAT = 0,
        PCL_VISUALIZER_SHADING_GOURAUD,
        PCL_VISUALIZER_SHADING_PHONG
    };

    enum RepresentationProperties {
        PCL_VISUALIZER_REPRESENTATION_POINTS = 0,
        PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
        PCL_VISUALIZER_REPRESENTATION_SURFACE
    };

protected:
    // VTK management (replaces PCLVisualizer members)
    vtkSmartPointer<vtkRendererCollection> rens_;
    vtkSmartPointer<vtkRenderWindow> win_;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor_;

    VtkRendering::CloudActorMapPtr cloud_actor_map_;
    VtkRendering::ShapeActorMapPtr shape_actor_map_;
    VtkRendering::CoordinateActorMapPtr coordinate_actor_map_;

    vtkSmartPointer<vtkOrientationMarkerWidget> m_axes_widget;
    vtkSmartPointer<vtkPointPicker> m_point_picker;
    vtkSmartPointer<vtkAreaPicker> m_area_picker;
    vtkSmartPointer<vtkPropPicker> m_propPicker;

    std::vector<int> m_selected_slice;

    std::map<std::string, ccHObject*> m_sourceObjectMap;

    // View Properties (ParaView-compatible)
    double m_lightIntensity;  // Current light intensity (0.0-1.0)
    std::map<std::string, double>
            m_objectLightIntensity;  // Per-object light intensity

    // Axes Grid actors (ParaView-style)
    // Data Axes Grid: one per object (viewID -> actor mapping)
    // Data Axes Grid: per-object, bound to viewID
    std::map<std::string, vtkSmartPointer<vtkCubeAxesActor>> m_dataAxesGridMap;

    // Camera Orientation Widget (ParaView-style)
    vtkSmartPointer<vtkCameraOrientationWidget> m_cameraOrientationWidget;
};

typedef std::shared_ptr<VtkVis> VtkVisPtr;

}  // namespace Visualization
