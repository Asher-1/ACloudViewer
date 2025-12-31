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

#include "WidgetMap.h"
#include "qPCL.h"

// Forward declaration
namespace PclUtils {
namespace renders {
class TextureRenderManager;
}
}  // namespace PclUtils

// ECV_DB_LIB
#include <ecvColorTypes.h>
#include <ecvDrawContext.h>
#include <ecvGenericVisualizer3D.h>

// VTK
#include <vtkBoundingBox.h>  // needed for iVar

// PCL
#include <pcl/visualization/pcl_visualizer.h>

class vtkLODActor;
class vtkCamera;
class vtkRender;
class vtkPointPicker;
class vtkAreaPicker;
class vtkPropPicker;
class vtkAbstractWidget;
class vtkRenderWindow;
class vtkMatrix4x4;
class ccGenericMesh;
class ccPointCloud;
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
class QPCL_ENGINE_LIB_API PCLVis : public ecvGenericVisualizer3D,
                                   public pcl::visualization::PCLVisualizer {
    Q_OBJECT
public:
    //! Default constructor
    PCLVis(vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>
                   interactor_style,
           const std::string& viewerName = "",
           bool initIterator = false,
           int argc = 0,
           char** argv = nullptr);  // deprecated!
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
     * @brief setupInteractor override to init interactor_
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
    pcl::visualization::Camera getCamera(int viewport = 0);
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
    inline void resetCamera() {
        pcl::visualization::PCLVisualizer::resetCamera();
    }
    inline void resetCamera(double bounds[6]) {
        resetCamera(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4],
                    bounds[5]);
    }
    void getReasonableClippingRange(double range[2], int viewport = 0);
    void expandBounds(double bounds[6], vtkMatrix4x4* matrix);
    void setCameraViewAngle(double viewAngle, int viewport = 0);

    void draw(const CC_DRAW_CONTEXT& context, const PCLCloud::Ptr& smCloud);
    void draw(const CC_DRAW_CONTEXT& context, const PCLMesh::Ptr& pclMesh);
    void draw(const CC_DRAW_CONTEXT& context,
              const PCLTextureMesh::Ptr& textureMesh);
    void draw(const CC_DRAW_CONTEXT& context,
              const PCLPolygon::Ptr& pclPolygon,
              bool closed);
    void draw(const CC_DRAW_CONTEXT& context, const ccSensor* sensor);
    void draw(const CC_DRAW_CONTEXT& context,
              const cloudViewer::geometry::LineSet* lineset);

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
     * @brief Add texture mesh from PCLTextureMesh (deprecated)
     * @deprecated Use addTextureMeshFromCCMesh instead to avoid
     * pcl::TexMaterial encoding
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
     * @param count Output cached texture count
     * @param memory_bytes Output cache memory usage (bytes)
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

    vtkSmartPointer<pcl::visualization::PCLVisualizerInteractorStyle>
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
    vtkSmartPointer<pcl::visualization::PCLVisualizerInteractorStyle>
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

    void pointPickingProcess(const pcl::visualization::PointPickingEvent& event,
                             void* args);
    void areaPickingEventProcess(
            const pcl::visualization::AreaPickingEvent& event, void* args);
    void mouseEventProcess(const pcl::visualization::MouseEvent& event,
                           void* args);
    void keyboardEventProcess(const pcl::visualization::KeyboardEvent& event,
                              void* args);

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

protected:
    vtkSmartPointer<vtkOrientationMarkerWidget> m_axes_widget;
    vtkSmartPointer<vtkPointPicker> m_point_picker;
    vtkSmartPointer<vtkAreaPicker> m_area_picker;
    vtkSmartPointer<vtkPropPicker> m_propPicker;

    std::vector<int> m_selected_slice;
};

typedef std::shared_ptr<PCLVis> PCLVisPtr;
}  // namespace PclUtils
