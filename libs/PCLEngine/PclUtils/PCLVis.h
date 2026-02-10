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

// LOCAL
#include <Utils/PCLCloud.h>

#include <map>
#include <mutex>
#include <thread>

#include <boost/signals2/connection.hpp>

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

class vtkLODActor;
class vtkCamera;
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

class QPCL_ENGINE_LIB_API PCLVis : public ecvGenericVisualizer3D {
    Q_OBJECT
public:
    //! Default constructor (deprecated - use the renderer/window constructor)
    PCLVis(vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                   interactor_style,
           const std::string& viewerName = "",
           bool initIterator = false,
           int argc = 0,
           char** argv = nullptr);
    PCLVis(vtkSmartPointer<vtkRenderer> ren,
           vtkSmartPointer<vtkRenderWindow> wind,
           vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                   interactor_style,
           const std::string& viewerName = "",
           bool initIterator = false,
           int argc = 0,
           char** argv = nullptr);

    virtual ~PCLVis();

    // do some initialization jobs
    void initialize();

    // center axes configuration
    void configCenterAxes();

    void configInteractorStyle(
            vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                    interactor_style);

public:
    // =====================================================================
    // Methods previously inherited from pcl::visualization::PCLVisualizer
    // Now reimplemented using direct VTK calls.
    // =====================================================================

    /** \brief Check if a cloud, shape, or widget with the given ID exists. */
    bool contains(const std::string& id) const;

    /** \brief Return a pointer to the CloudActorMap. */
    inline PclUtils::CloudActorMapPtr getCloudActorMap() {
        return cloud_actor_map_;
    }

    /** \brief Return a pointer to the ShapeActorMap. */
    inline PclUtils::ShapeActorMapPtr getShapeActorMap() {
        return shape_actor_map_;
    }

    /** \brief Return a pointer to the RendererCollection. */
    inline vtkSmartPointer<vtkRendererCollection> getRendererCollection() {
        return rens_;
    }

    /** \brief Return a pointer to the RenderWindow. */
    inline vtkSmartPointer<vtkRenderWindow> getRenderWindow() { return win_; }

    /** \brief Get the current interactor style. */
    inline vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
    getInteractorStyle() {
        return m_interactorStyle;
    }

    /** \brief Remove a point cloud from the visualizer. */
    bool removePointCloud(const std::string& id, int viewport = 0);

    /** \brief Remove a shape from the visualizer. */
    bool removeShape(const std::string& id, int viewport = 0);

    /** \brief Remove all point clouds from the visualizer. */
    bool removeAllPointClouds(int viewport = 0);

    /** \brief Remove all shapes from the visualizer. */
    bool removeAllShapes(int viewport = 0);

    /** \brief Remove a polygon mesh from the visualizer. */
    bool removePolygonMesh(const std::string& id, int viewport = 0);

    /** \brief Set rendering properties for a point cloud.
     *  Supports: CV_VISUALIZER_POINT_SIZE, CV_VISUALIZER_OPACITY,
     *            CV_VISUALIZER_COLOR, CV_VISUALIZER_REPRESENTATION,
     *            CV_VISUALIZER_SHADING
     */
    bool setPointCloudRenderingProperties(int property,
                                          double val1,
                                          const std::string& id,
                                          int viewport = 0);
    bool setPointCloudRenderingProperties(int property,
                                          double val1,
                                          double val2,
                                          double val3,
                                          const std::string& id,
                                          int viewport = 0);

    /** \brief Get rendering properties for a point cloud. */
    bool getPointCloudRenderingProperties(int property,
                                          double& value,
                                          const std::string& id);

    /** \brief Set rendering properties for a shape. */
    bool setShapeRenderingProperties(int property,
                                     double val1,
                                     const std::string& id,
                                     int viewport = 0);
    bool setShapeRenderingProperties(int property,
                                     double val1,
                                     double val2,
                                     double val3,
                                     const std::string& id,
                                     int viewport = 0);

    /** \brief Add 2D text to the visualizer. */
    bool addText(const std::string& text,
                 int xpos,
                 int ypos,
                 int fontsize,
                 double r,
                 double g,
                 double b,
                 const std::string& id = "text",
                 int viewport = 0);

    /** \brief Update existing 2D text. */
    bool updateText(const std::string& text,
                    int xpos,
                    int ypos,
                    const std::string& id = "text");

    /** \brief Add 3D text at a position. */
    template <typename PointT>
    bool addText3D(const std::string& text,
                   const PointT& position,
                   double textScale = 1.0,
                   double r = 1.0,
                   double g = 1.0,
                   double b = 1.0,
                   const std::string& id = "text3d",
                   int viewport = 0);

    /** \brief Add a polygon from a point cloud. */
    template <typename PointT>
    bool addPolygon(const pcl::PointCloud<PointT>& cloud,
                    double r,
                    double g,
                    double b,
                    const std::string& id = "polygon",
                    int viewport = 0);

    /** \brief Add a polygon from a PlanarPolygon. */
    template <typename PointT>
    bool addPolygon(const pcl::PlanarPolygon<PointT>& polygon,
                    double r,
                    double g,
                    double b,
                    const std::string& id = "polygon",
                    int viewport = 0);

    /** \brief Get camera parameters. */
    void getCameraParameters(PclUtils::Camera& camera, int viewport = 0) const;

    /** \brief Set camera parameters. */
    void setCameraParameters(const PclUtils::Camera& camera, int viewport = 0);

    /** \brief Register mouse callback. */
    boost::signals2::connection registerMouseCallback(
            std::function<void(const PclUtils::MouseEvent&)> cb);

    /** \brief Register keyboard callback. */
    boost::signals2::connection registerKeyboardCallback(
            std::function<void(const PclUtils::KeyboardEvent&)> cb);

    /** \brief Register point picking callback. */
    boost::signals2::connection registerPointPickingCallback(
            std::function<void(const PclUtils::PointPickingEvent&)> cb);

    /** \brief Register area picking callback. */
    boost::signals2::connection registerAreaPickingCallback(
            std::function<void(const PclUtils::AreaPickingEvent&)> cb);

    /** \brief Reset camera viewpoint for a specific cloud. */
    void resetCameraViewpoint(const std::string& id = "cloud");

    /** \brief Create a new viewport. */
    void createViewPort(double xmin, double ymin, double xmax, double ymax,
                        int& viewport);

    /** \brief Add coordinate system. */
    void addCoordinateSystem(double scale = 1.0,
                             const std::string& id = "reference",
                             int viewport = 0);
    void addCoordinateSystem(double scale,
                             const Eigen::Affine3f& t,
                             const std::string& id = "reference",
                             int viewport = 0);

    /** \brief Remove coordinate system. */
    bool removeCoordinateSystem(const std::string& id = "reference",
                                int viewport = 0);

    /** \brief Set camera position using pos, focal, up. */
    void setCameraPosition(double pos_x, double pos_y, double pos_z,
                           double view_x, double view_y, double view_z,
                           double up_x, double up_y, double up_z,
                           int viewport = 0);

    /** \brief Set camera position using pos and up. */
    void setCameraPosition(double pos_x, double pos_y, double pos_z,
                           double up_x, double up_y, double up_z,
                           int viewport = 0);

    /** \brief Save camera parameters to file. */
    void saveCameraParameters(const std::string& file);

    /** \brief Load camera parameters from file. */
    void loadCameraParameters(const std::string& file);

    /** \brief Add a point cloud to the visualizer. */
    template <typename PointT>
    bool addPointCloud(
            const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
            const std::string& id = "cloud",
            int viewport = 0);

    /** \brief Add a point cloud with a color handler. */
    template <typename PointT>
    bool addPointCloud(
            const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
            const PclUtils::PointCloudColorHandler<PointT>& handler,
            const std::string& id = "cloud",
            int viewport = 0);

    /** \brief Update a point cloud. */
    template <typename PointT>
    bool updatePointCloud(
            const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
            const std::string& id = "cloud");

    /** \brief Update a point cloud with a color handler. */
    template <typename PointT>
    bool updatePointCloud(
            const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
            const PclUtils::PointCloudColorHandler<PointT>& handler,
            const std::string& id = "cloud");

    /** \brief Add point cloud normals as lines. */
    template <typename PointNT>
    bool addPointCloudNormals(
            const typename pcl::PointCloud<PointNT>::ConstPtr& cloud,
            int level = 100,
            float scale = 0.02f,
            const std::string& id = "cloud",
            int viewport = 0);

    /** \brief Add a polygon mesh. */
    bool addPolygonMesh(const pcl::PolygonMesh& mesh,
                        const std::string& id = "polygon",
                        int viewport = 0);

    /** \brief Update a polygon mesh. */
    bool updatePolygonMesh(const pcl::PolygonMesh& mesh,
                           const std::string& id = "polygon");

    /** \brief Set full screen mode. */
    void setFullScreen(bool state);

    /** \brief Set camera clip distances. */
    void setCameraClipDistances(double znear, double zfar, int viewport = 0);

    /** \brief Set camera field of view (radians). */
    void setCameraFieldOfView(double fovy, int viewport = 0);

    /** \brief Save a screenshot to file. */
    void saveScreenshot(const std::string& file);

    /** \brief Enable/disable VBO rendering (no-op on modern VTK). */
    void setUseVbos(bool useVbos);

    /** \brief Set the cloud/shape id used for LUT display. */
    void setLookUpTableID(const std::string& viewID);

    /** \brief Add a cube (axis-aligned box). */
    bool addCube(double xmin, double xmax,
                 double ymin, double ymax,
                 double zmin, double zmax,
                 double r = 1.0, double g = 1.0, double b = 1.0,
                 const std::string& id = "cube",
                 int viewport = 0);

    /** \brief Add a line between two 3D points. */
    template <typename PointT>
    bool addLine(const PointT& pt1, const PointT& pt2,
                 double r, double g, double b,
                 const std::string& id = "line",
                 int viewport = 0);

    /** \brief Add a sphere at a 3D position. */
    template <typename PointT>
    bool addSphere(const PointT& center, double radius,
                   double r = 1.0, double g = 1.0, double b = 1.0,
                   const std::string& id = "sphere",
                   int viewport = 0);

    // =====================================================================
    // End of formerly-inherited methods
    // =====================================================================

public:
    /** \brief Marker Axes.
     */
    void hidePclMarkerAxes();
    bool pclMarkerAxesShown();
    void showPclMarkerAxes(vtkRenderWindowInteractor* interactor = nullptr);
    void hideOrientationMarkerWidgetAxes();
    void showOrientationMarkerWidgetAxes(vtkRenderWindowInteractor* interactor);
    void toggleOrientationMarkerWidgetAxes();

    /** \brief Internal method. Adds a vtk actor to screen.
     * \param[in] actor a pointer to the vtk actor object
     * \param[in] viewport the view port where the actor should be added to
     * (default: all)
     */
    bool removeActorFromRenderer(const vtkSmartPointer<vtkProp>& actor,
                                 int viewport = 0);

    void addActorToRenderer(const vtkSmartPointer<vtkProp>& actor,
                            int viewport = 0);

    /**
     * @brief UpdateScreen - Updates/refreshes the render window
     * This method forces a render update after actor changes
     */
    void UpdateScreen();

    /**
     * @brief setupInteractor - sets up the interactor with the render window
     * @param iren
     * @param win
     */
    void setupInteractor(vtkRenderWindowInteractor* iren, vtkRenderWindow* win);

    /** \brief Get a pointer to the current interactor style used. */
    inline vtkSmartPointer<vtkRenderWindowInteractor>
    getRenderWindowInteractor() {
        return (interactor_);
    }

    // Camera Tools
    PclUtils::Camera getCamera(int viewport = 0);
    vtkSmartPointer<vtkCamera> getVtkCamera(int viewport = 0);

    void setModelViewMatrix(const ccGLMatrixd& viewMat, int viewport = 0);
    double getParallelScale(int viewport = 0);
    void setParallelScale(double scale, int viewport = 0);

    void setOrthoProjection(int viewport = 0);
    void setPerspectiveProjection(int viewport = 0);
    bool getPerspectiveState(int viewport = 0);

    inline bool getAutoUpateCameraPos() { return m_autoUpdateCameraPos; }
    inline void setAutoUpateCameraPos(bool state) {
        m_autoUpdateCameraPos = state;
    }

    void rotateWithAxis(const CCVector2i& pos,
                        const CCVector3d& axis,
                        double angle,
                        int viewport = 0);

public:
    /**
     * Get the current center of rotation
     */
    void getCenterOfRotation(double center[3]);
    /**
     * Resets the center of rotation to the focal point.
     */
    void resetCenterOfRotation(int viewport = 0);

    static void ExpandBounds(double bounds[6], vtkMatrix4x4* matrix);

    /**
     * Set the center of rotation. For this to work,
     * one should have appropriate interaction style
     * and camera manipulators that use the center of rotation
     * They are setup correctly by default
     */
    void setCenterOfRotation(double x, double y, double z);
    inline void setCenterOfRotation(double xyz[3]) {
        setCenterOfRotation(xyz[0], xyz[1], xyz[2]);
    }

    void setRotationFactor(double factor);
    double getRotationFactor();

    //*****************************************************************
    // Forwarded to center axes.
    void setCenterAxesVisibility(bool);

    //*****************************************************************
    // Forwarded to vtkPVInteractorStyle if present on local processes.
    virtual void setCamera2DManipulators(const int manipulators[9]);
    virtual void setCamera3DManipulators(const int manipulators[9]);
    void setCameraManipulators(VTKExtensions::vtkCustomInteractorStyle* style,
                               const int manipulators[9]);
    virtual void setCamera2DMouseWheelMotionFactor(double factor);
    virtual void setCamera3DMouseWheelMotionFactor(double factor);
    /**
     * updateCenterAxes().
     * updates CenterAxes's scale and position.
     */
    virtual void updateCenterAxes();

    /**
     * Synchronizes bounds information on all nodes.
     * \note CallOnAllProcesses
     */
    void synchronizeGeometryBounds(int viewport = 0);

    // Return a depth value for z-buffer
    double getGLDepth(int x, int y);

    double getCameraFocalDistance(int viewport = 0);
    void setCameraFocalDistance(double focal_distance, int viewport = 0);

    /**
     * In perspective mode, decrease the view angle by the specified factor.
     * In parallel mode, decrease the parallel scale by the specified factor.
     * A value greater than 1 is a zoom-in, a value less than 1 is a zoom-out.
     * @note This setting is ignored when UseExplicitProjectionTransformMatrix
     * is true.
     */
    void zoomCamera(double zoomFactor, int viewport = 0);

    void getProjectionTransformMatrix(Eigen::Matrix4d& proj);

    void getModelViewTransformMatrix(Eigen::Matrix4d& view);

    /**
     * Automatically set up the camera based on a specified bounding box
     * (xmin, xmax, ymin, ymax, zmin, zmax). Camera will reposition itself so
     * that its focal point is the center of the bounding box, and adjust its
     * distance and position to preserve its initial view plane normal
     * (i.e., vector defined from camera position to focal point). Note: is
     * the view plane is parallel to the view up axis, the view up axis will
     * be reset to one of the three coordinate axes.
     */
    void resetCameraClippingRange(int viewport = 0);
    void internalResetCameraClippingRange() {
        this->resetCameraClippingRange(0);
    }
    void resetCamera(const ccBBox* bbox);
    void resetCamera(double xMin,
                     double xMax,
                     double yMin,
                     double yMax,
                     double zMin,
                     double zMax);
    void resetCamera();
    inline void resetCamera(double bounds[6]) {
        resetCamera(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4],
                    bounds[5]);
    }
    void getReasonableClippingRange(double range[2], int viewport = 0);
    void expandBounds(double bounds[6], vtkMatrix4x4* matrix);
    void setCameraViewAngle(double viewAngle, int viewport = 0);

    // Draw methods (PCL format paths - legacy)
    void draw(const CC_DRAW_CONTEXT& context, const PCLCloud::Ptr& smCloud);
    void draw(const CC_DRAW_CONTEXT& context, const PCLMesh::Ptr& pclMesh);
    void draw(const CC_DRAW_CONTEXT& context,
              const PCLTextureMesh::Ptr& textureMesh);
    void draw(const CC_DRAW_CONTEXT& context,
              const PCLPolygon::Ptr& pclPolygon,
              bool closed = false);
    void draw(const CC_DRAW_CONTEXT& context, const ccSensor* sensor);
    void draw(const CC_DRAW_CONTEXT& context,
              const cloudViewer::geometry::LineSet* lineset);

    // ==================== Direct CV_db → VTK Draw Methods ====================
    // These methods bypass PCL data format conversion for maximum efficiency.
    // Data is converted directly from CV_db types to VTK polydata.

    /**
     * @brief Draw point cloud directly from ccPointCloud (no PCL conversion)
     * @param context Draw context with rendering parameters
     * @param cloud The ccPointCloud to draw
     */
    void drawDirect(const CC_DRAW_CONTEXT& context, ccPointCloud* cloud);

    /**
     * @brief Draw mesh directly from ccGenericMesh (no PCL conversion)
     * @param context Draw context with rendering parameters
     * @param mesh The ccGenericMesh to draw (non-textured path)
     */
    void drawMeshDirect(const CC_DRAW_CONTEXT& context, ccGenericMesh* mesh);

    /**
     * @brief Draw polyline directly from ccPolyline (no PCL conversion)
     * @param context Draw context with rendering parameters
     * @param polyline The ccPolyline to draw
     * @param closed Whether the polyline is closed (forms a polygon)
     */
    void drawPolylineDirect(const CC_DRAW_CONTEXT& context,
                            ccPolyline* polyline,
                            bool closed);

    /**
     * @brief Update shading mode directly from ccPointCloud (no PCL
     * conversion) Syncs normals and RGB colors to VTK polydata for Find Data /
     * selection extraction, and controls shading mode (Flat/Phong).
     * @param context Draw context with rendering parameters
     * @param cloud The ccPointCloud source data (can be nullptr)
     */
    void updateShadingModeDirect(const CC_DRAW_CONTEXT& context,
                                 ccPointCloud* cloud);

    /**
     * @brief Update normal glyphs directly from ccPointCloud (no PCL
     * conversion)
     * @param context Draw context with rendering parameters
     * @param cloud The ccPointCloud source data (can be nullptr)
     */
    void updateNormalsDirect(const CC_DRAW_CONTEXT& context,
                             ccPointCloud* cloud);

    void transformEntities(const CC_DRAW_CONTEXT& context);
    vtkSmartPointer<vtkTransform> getTransformation(
            const CC_DRAW_CONTEXT& context, const CCVector3d& origin);
    void updateNormals(const CC_DRAW_CONTEXT& context,
                       const PCLCloud::Ptr& smCloud);
    void updateShadingMode(const CC_DRAW_CONTEXT& context, PCLCloud& smCloud);
    bool removeEntities(const CC_DRAW_CONTEXT& context);
    void hideShowActors(bool visibility,
                        const std::string& viewID,
                        int viewport = 0);
    void hideShowWidgets(bool visibility,
                         const std::string& viewID,
                         int viewport = 0);

    bool addScalarBar(const CC_DRAW_CONTEXT& context);
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

    bool addPolyline(const PCLPolygon::ConstPtr pclPolygon,
                     double r,
                     double g,
                     double b,
                     float width = 1.0f,
                     const std::string& id = "multiline",
                     int viewport = 0);
    bool updateTexture(const CC_DRAW_CONTEXT& context,
                       const ccMaterialSet* materials);
    /**
     * @brief Add texture mesh from PCLTextureMesh
     * @deprecated Use addTextureMeshFromCCMesh instead
     */
    bool addTextureMesh(const PCLTextureMesh& mesh,
                        const std::string& id,
                        int viewport);
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
     */
    bool addTextureMeshFromOBJ(const std::string& obj_path,
                               const std::string& id,
                               int viewport = 0,
                               int quality = 2,
                               bool enable_cache = true);

    /**
     * @brief Load multi-texture mesh from OBJ file (advanced options)
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
    void displayText(const CC_DRAW_CONTEXT& context);

private:
    // Texture rendering manager
    std::unique_ptr<renders::TextureRenderManager> texture_render_manager_;

    // Store transformation matrices to prevent memory leaks
    // Maps view ID to transformation matrix smart pointer
    std::map<std::string, vtkSmartPointer<vtkMatrix4x4>> transformation_map_;

public:
    void setPointSize(const unsigned char pointSize,
                      const std::string& viewID,
                      int viewport = 0);
    void setScalarFieldName(const std::string& viewID,
                            const std::string& scalarName,
                            int viewport = 0);

    void addScalarFieldToVTK(const std::string& viewID,
                             ccPointCloud* cloud,
                             int scalarFieldIndex,
                             int viewport = 0);

    void syncAllScalarFieldsToVTK(const std::string& viewID,
                                  ccPointCloud* cloud,
                                  int viewport = 0);

    void setCurrentSourceObject(ccHObject* obj, const std::string& viewID);
    void removeSourceObject(const std::string& viewID);
    ccHObject* getSourceObject(const std::string& viewID) const;
    ccPointCloud* getSourceCloud(const std::string& viewID) const;
    ccMesh* getSourceMesh(const std::string& viewID) const;
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
    void setMeshOpacity(double opacity,
                        const std::string& viewID,
                        int viewport = 0);
    void setShapeShadingMode(SHADING_MODE mode,
                             const std::string& viewID,
                             int viewport = 0);
    void setMeshShadingMode(SHADING_MODE mode,
                            const std::string& viewID,
                            int viewport = 0);

    vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
    getPCLInteractorStyle();
    vtkActor* getActorById(const std::string& viewId);
    vtkProp* getPropById(const std::string& viewId);
    vtkSmartPointer<vtkPropCollection> getPropCollectionById(
            const std::string& viewId);
    std::string getIdByActor(vtkProp* actor);
    vtkAbstractWidget* getWidgetById(const std::string& viewId);

    /**
     * Get the Current Renderer in the list.
     * Return NULL when at the end of the list.
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
    // =====================================================================
    // Core visualization state (previously from PCLVisualizer base class)
    // =====================================================================

    /** \brief The renderer collection. */
    vtkSmartPointer<vtkRendererCollection> rens_;

    /** \brief The render window. */
    vtkSmartPointer<vtkRenderWindow> win_;

    /** \brief The render window interactor. */
    vtkSmartPointer<vtkRenderWindowInteractor> interactor_;

    /** \brief The cloud actor map. */
    PclUtils::CloudActorMapPtr cloud_actor_map_;

    /** \brief The shape actor map. */
    PclUtils::ShapeActorMapPtr shape_actor_map_;

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
    vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
            m_interactorStyle;

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

    void pointPickingProcess(const PclUtils::PointPickingEvent& event);
    void areaPickingEventProcess(const PclUtils::AreaPickingEvent& event);
    void mouseEventProcess(const PclUtils::MouseEvent& event);
    void keyboardEventProcess(const PclUtils::KeyboardEvent& event);

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
    void setAreaPickingMode(bool state);
    vtkActor* pickActor(double x, double y);
    std::string pickItem(double x0 = -1,
                         double y0 = -1,
                         double x1 = 5.0,
                         double y1 = 5.0);

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

    void setLightIntensity(double intensity);
    double getLightIntensity() const;

    // ========================================================================
    // Data Axes Grid (Unified Interface with DataAxesGridProperties struct)
    // ========================================================================

    void SetDataAxesGridProperties(const std::string& viewID,
                                   const AxesGridProperties& props);
    void GetDataAxesGridProperties(const std::string& viewID,
                                   AxesGridProperties& props) const;
    void RemoveDataAxesGrid(const std::string& viewID);
    void ToggleCameraOrientationWidget(bool show);
    bool IsCameraOrientationWidgetShown() const;

protected:
    vtkSmartPointer<vtkOrientationMarkerWidget> m_axes_widget;
    vtkSmartPointer<vtkPointPicker> m_point_picker;
    vtkSmartPointer<vtkAreaPicker> m_area_picker;
    vtkSmartPointer<vtkPropPicker> m_propPicker;

    std::vector<int> m_selected_slice;

    // Source objects for selection operations (allows direct extraction)
    std::map<std::string, ccHObject*> m_sourceObjectMap;

    // View Properties (ParaView-compatible)
    double m_lightIntensity;  // Current light intensity (0.0-1.0)

    // Data Axes Grid: per-object, bound to viewID
    std::map<std::string, vtkSmartPointer<vtkCubeAxesActor>> m_dataAxesGridMap;

    // Camera Orientation Widget (ParaView-style)
    vtkSmartPointer<vtkCameraOrientationWidget> m_cameraOrientationWidget;
};

typedef std::shared_ptr<PCLVis> PCLVisPtr;
}  // namespace PclUtils

// Additional VTK includes for template impls
#include <vtkFollower.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSphereSource.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkVectorText.h>

// Template implementations
#include <vtkCellArray.h>
#include <vtkDataSetMapper.h>
#include <vtkLODActor.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>

namespace PclUtils {

template <typename PointT>
bool PCLVis::addPointCloud(
        const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
        const std::string& id,
        int viewport) {
    PclUtils::PointCloudColorHandlerCustom<PointT> handler(cloud, 255, 255, 255);
    return addPointCloud<PointT>(cloud, handler, id, viewport);
}

template <typename PointT>
bool PCLVis::addPointCloud(
        const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
        const PclUtils::PointCloudColorHandler<PointT>& handler,
        const std::string& id,
        int viewport) {
    if (contains(id)) return false;

    // Use PclUtils geometry handler to get VTK points
    PclUtils::PointCloudGeometryHandlerXYZ<PointT> geometry(cloud);
    vtkSmartPointer<vtkPoints> points;
    geometry.getGeometry(points);

    // Create polydata from points
    vtkSmartPointer<vtkPolyData> polydata =
            vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    // Create vertex cells for point rendering
    vtkIdType npts = points->GetNumberOfPoints();
    vtkSmartPointer<vtkCellArray> vertices =
            vtkSmartPointer<vtkCellArray>::New();
    for (vtkIdType i = 0; i < npts; i++) {
        vertices->InsertNextCell(1, &i);
    }
    polydata->SetVerts(vertices);

    // Get colors from handler
    vtkSmartPointer<vtkDataArray> scalars = handler.getColor();
    if (scalars) {
        polydata->GetPointData()->SetScalars(scalars);
    }

    // Create mapper
    vtkSmartPointer<vtkDataSetMapper> mapper =
            vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(polydata);
    mapper->SetScalarModeToUsePointData();
    mapper->InterpolateScalarsBeforeMappingOff();
    mapper->ScalarVisibilityOn();
    mapper->SetColorModeToDirectScalars();

    // Create actor
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetPointSize(1.0);

    addActorToRenderer(actor, viewport);
    PclUtils::CloudActorEntry entry;
    entry.actor = actor;
    (*cloud_actor_map_)[id] = entry;
    return true;
}

template <typename PointT>
bool PCLVis::updatePointCloud(
        const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
        const std::string& id) {
    PclUtils::PointCloudColorHandlerCustom<PointT> handler(cloud, 255, 255, 255);
    return updatePointCloud<PointT>(cloud, handler, id);
}

template <typename PointT>
bool PCLVis::updatePointCloud(
        const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
        const PclUtils::PointCloudColorHandler<PointT>& handler,
        const std::string& id) {
    auto it = cloud_actor_map_->find(id);
    if (it == cloud_actor_map_->end()) return false;

    vtkActor* actor = it->second.actor;
    if (!actor) return false;

    // Use PclUtils geometry handler to get VTK points
    PclUtils::PointCloudGeometryHandlerXYZ<PointT> geometry(cloud);
    vtkSmartPointer<vtkPoints> points;
    geometry.getGeometry(points);

    // Create polydata from points
    vtkSmartPointer<vtkPolyData> polydata =
            vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    // Create vertex cells
    vtkIdType npts = points->GetNumberOfPoints();
    vtkSmartPointer<vtkCellArray> vertices =
            vtkSmartPointer<vtkCellArray>::New();
    for (vtkIdType i = 0; i < npts; i++) {
        vertices->InsertNextCell(1, &i);
    }
    polydata->SetVerts(vertices);

    // Get colors from handler
    vtkSmartPointer<vtkDataArray> scalars = handler.getColor();
    if (scalars) {
        polydata->GetPointData()->SetScalars(scalars);
    }

    // Update the mapper's input data
    vtkDataSetMapper* mapper =
            vtkDataSetMapper::SafeDownCast(actor->GetMapper());
    if (mapper) {
        mapper->SetInputData(polydata);
        mapper->Update();
    }
    actor->Modified();
    return true;
}

template <typename PointNT>
bool PCLVis::addPointCloudNormals(
        const typename pcl::PointCloud<PointNT>::ConstPtr& cloud,
        int level,
        float scale,
        const std::string& id,
        int viewport) {
    if (contains(id)) return false;
    if (!cloud || cloud->empty()) return false;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();

    vtkIdType ptIdx = 0;
    for (std::size_t i = 0; i < cloud->size(); i += level) {
        const PointNT& p = cloud->at(i);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
            !std::isfinite(p.z))
            continue;
        if (!std::isfinite(p.normal_x) || !std::isfinite(p.normal_y) ||
            !std::isfinite(p.normal_z))
            continue;

        points->InsertNextPoint(p.x, p.y, p.z);
        points->InsertNextPoint(p.x + p.normal_x * scale,
                                p.y + p.normal_y * scale,
                                p.z + p.normal_z * scale);

        vtkIdType lineIds[2] = {ptIdx, ptIdx + 1};
        lines->InsertNextCell(2, lineIds);
        ptIdx += 2;
    }

    vtkSmartPointer<vtkPolyData> polydata =
            vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetLines(lines);

    vtkSmartPointer<vtkDataSetMapper> mapper =
            vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(polydata);

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(1.0, 0.0, 0.0);

    addActorToRenderer(actor, viewport);
    PclUtils::CloudActorEntry entry;
    entry.actor = actor;
    (*cloud_actor_map_)[id] = entry;
    return true;
}

template <typename PointT>
bool PCLVis::addPolygon(const pcl::PointCloud<PointT>& cloud,
                        double r, double g, double b,
                        const std::string& id, int viewport) {
    if (contains(id)) return false;
    if (cloud.empty()) return false;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> polygon = vtkSmartPointer<vtkCellArray>::New();

    vtkIdType npts = static_cast<vtkIdType>(cloud.size());
    std::vector<vtkIdType> ids(npts);
    for (vtkIdType i = 0; i < npts; i++) {
        points->InsertNextPoint(cloud[i].x, cloud[i].y, cloud[i].z);
        ids[i] = i;
    }
    polygon->InsertNextCell(npts, ids.data());

    vtkSmartPointer<vtkPolyData> polydata =
            vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetPolys(polygon);

    vtkSmartPointer<vtkDataSetMapper> mapper =
            vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(polydata);

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(r, g, b);
    actor->GetProperty()->SetRepresentationToWireframe();

    addActorToRenderer(actor, viewport);
    (*shape_actor_map_)[id] = actor;
    return true;
}

template <typename PointT>
bool PCLVis::addPolygon(const pcl::PlanarPolygon<PointT>& polygon,
                        double r, double g, double b,
                        const std::string& id, int viewport) {
    pcl::PointCloud<PointT> cloud;
    cloud.points = polygon.getContour();
    cloud.width = static_cast<std::uint32_t>(cloud.points.size());
    cloud.height = 1;
    return addPolygon<PointT>(cloud, r, g, b, id, viewport);
}

template <typename PointT>
bool PCLVis::addText3D(const std::string& text,
                       const PointT& position,
                       double textScale,
                       double r, double g, double b,
                       const std::string& id, int viewport) {
    if (contains(id)) return false;

    vtkSmartPointer<vtkVectorText> textSource =
            vtkSmartPointer<vtkVectorText>::New();
    textSource->SetText(text.c_str());
    textSource->Update();

    vtkSmartPointer<vtkTransform> transform =
            vtkSmartPointer<vtkTransform>::New();
    transform->Translate(position.x, position.y, position.z);
    transform->Scale(textScale, textScale, textScale);

    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter =
            vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformFilter->SetTransform(transform);
    transformFilter->SetInputConnection(textSource->GetOutputPort());
    transformFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> textMapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    textMapper->SetInputConnection(transformFilter->GetOutputPort());

    vtkSmartPointer<vtkFollower> textActor =
            vtkSmartPointer<vtkFollower>::New();
    textActor->SetMapper(textMapper);
    textActor->GetProperty()->SetColor(r, g, b);

    // Make the text face the camera
    vtkRenderer* renderer = nullptr;
    if (rens_) {
        rens_->InitTraversal();
        renderer = rens_->GetNextItem();
    }
    if (renderer) {
        textActor->SetCamera(renderer->GetActiveCamera());
    }

    addActorToRenderer(textActor, viewport);
    (*shape_actor_map_)[id] = textActor;
    return true;
}

template <typename PointT>
bool PCLVis::addLine(const PointT& pt1, const PointT& pt2,
                     double r, double g, double b,
                     const std::string& id, int viewport) {
    if (contains(id)) return false;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->InsertNextPoint(pt1.x, pt1.y, pt1.z);
    points->InsertNextPoint(pt2.x, pt2.y, pt2.z);

    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
    vtkIdType lineIds[2] = {0, 1};
    lines->InsertNextCell(2, lineIds);

    vtkSmartPointer<vtkPolyData> polydata =
            vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetLines(lines);

    vtkSmartPointer<vtkDataSetMapper> mapper =
            vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(polydata);

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(r, g, b);

    addActorToRenderer(actor, viewport);
    (*shape_actor_map_)[id] = actor;
    return true;
}

template <typename PointT>
bool PCLVis::addSphere(const PointT& center, double radius,
                       double r, double g, double b,
                       const std::string& id, int viewport) {
    if (contains(id)) return false;

    vtkSmartPointer<vtkSphereSource> sphere =
            vtkSmartPointer<vtkSphereSource>::New();
    sphere->SetCenter(center.x, center.y, center.z);
    sphere->SetRadius(radius);
    sphere->SetPhiResolution(16);
    sphere->SetThetaResolution(16);
    sphere->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(sphere->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(r, g, b);

    addActorToRenderer(actor, viewport);
    (*shape_actor_map_)[id] = actor;
    return true;
}

}  // namespace PclUtils
