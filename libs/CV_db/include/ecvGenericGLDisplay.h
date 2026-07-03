// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVGeom.h>

#include <QFlags>
#include <QFont>
#include <QJsonObject>
#include <QRect>
#include <QString>
#include <list>
#include <string>
#include <vector>

#include "CV_db.h"
#include "ecvGLMatrix.h"
#include "ecvGuiParameters.h"

class ccBBox;
class ccHObject;
class ccDrawableObject;
class ccGenericMesh;
class ccInteractor;
class ecvOrientedBBox;
class QWidget;
#include "ecvViewportParameters.h"
struct AxesGridProperties;
struct ccGLCameraParameters;
struct ccGLDrawContext;
struct ecvViewContext;
struct PROPERTY_PARAM;
struct WIDGETS_PARAMETER;
struct ecvHotZone;
struct ecvClickableItem;

/// Per-window display interface for multi-window rendering.
///
/// Each 3D view window implements this interface, holding its own
/// viewport parameters, camera state, and scene/window DB roots.
/// The global ecvDisplayTools (owned by ecvViewManager) delegates to the
/// "active" ecvGenericGLDisplay instance so that existing static call sites
/// keep working while new code can address a specific window.
///
/// Design references:
///   CloudCompare  ccGenericGLDisplay / ccGLWindowInterface
///   ParaView      pqView / pqRenderView
///   MeshLab       GLArea
class CV_DB_LIB_API ecvGenericGLDisplay {
public:
    virtual ~ecvGenericGLDisplay() = default;

    // ================================================================
    // Enums — shared across all display implementations.
    // Previously defined in ecvDisplayTools; moved here so the
    // per-view interface can reference them without a circular
    // dependency.  Existing code using ecvDisplayTools::PICKING_MODE
    // etc. still compiles because C++ resolves inherited names.
    // ================================================================

    enum PICKING_MODE {
        NO_PICKING,
        ENTITY_PICKING,
        ENTITY_RECT_PICKING,
        FAST_PICKING,
        POINT_PICKING,
        TRIANGLE_PICKING,
        POINT_OR_TRIANGLE_PICKING,
        POINT_OR_TRIANGLE_OR_LABEL_PICKING,
        LABEL_PICKING,
        DEFAULT_PICKING,
    };

    enum INTERACTION_FLAG {
        INTERACT_NONE = 0,
        INTERACT_ROTATE = 1,
        INTERACT_PAN = 2,
        INTERACT_CTRL_PAN = 4,
        INTERACT_ZOOM_CAMERA = 8,
        INTERACT_2D_ITEMS = 16,
        INTERACT_CLICKABLE_ITEMS = 32,
        INTERACT_TRANSFORM_ENTITIES = 64,
        INTERACT_SIG_RB_CLICKED = 128,
        INTERACT_SIG_LB_CLICKED = 256,
        INTERACT_SIG_MOUSE_MOVED = 512,
        INTERACT_SIG_BUTTON_RELEASED = 1024,
        INTERACT_SIG_MB_CLICKED = 2048,
        INTERACT_SEND_ALL_SIGNALS =
                INTERACT_SIG_RB_CLICKED | INTERACT_SIG_LB_CLICKED |
                INTERACT_SIG_MB_CLICKED | INTERACT_SIG_MOUSE_MOVED |
                INTERACT_SIG_BUTTON_RELEASED,
        MODE_PAN_ONLY = INTERACT_PAN | INTERACT_ZOOM_CAMERA |
                        INTERACT_2D_ITEMS | INTERACT_CLICKABLE_ITEMS,
        MODE_TRANSFORM_CAMERA = INTERACT_ROTATE | MODE_PAN_ONLY,
        MODE_TRANSFORM_ENTITIES =
                INTERACT_ROTATE | INTERACT_PAN | INTERACT_ZOOM_CAMERA |
                INTERACT_TRANSFORM_ENTITIES | INTERACT_CLICKABLE_ITEMS,
    };
    Q_DECLARE_FLAGS(INTERACTION_FLAGS, INTERACTION_FLAG)

    enum MessagePosition {
        LOWER_LEFT_MESSAGE,
        UPPER_CENTER_MESSAGE,
        SCREEN_CENTER_MESSAGE,
    };

    enum MessageType {
        CUSTOM_MESSAGE,
        SCREEN_SIZE_MESSAGE,
        PERSPECTIVE_STATE_MESSAGE,
        SUN_LIGHT_STATE_MESSAGE,
        CUSTOM_LIGHT_STATE_MESSAGE,
        MANUAL_TRANSFORMATION_MESSAGE,
        MANUAL_SEGMENTATION_MESSAGE,
        ROTAION_LOCK_MESSAGE,
        FULL_SCREEN_MESSAGE,
    };

    enum PivotVisibility {
        PIVOT_HIDE,
        PIVOT_SHOW_ON_MOVE,
        PIVOT_ALWAYS_SHOW,
    };

    // ================================================================
    // Identification
    // ================================================================

    virtual int getUniqueID() const = 0;
    virtual QString getTitle() const = 0;

    // ================================================================
    // Refresh / redraw
    // ================================================================

    virtual void redraw(bool only2D = false, bool forceRedraw = true) = 0;
    virtual void refresh(bool only2D = false) = 0;
    virtual void toBeRefreshed() = 0;

    // ================================================================
    // Viewport / camera
    // ================================================================

    virtual const ecvViewportParameters& getViewportParameters() const = 0;
    virtual void setViewportParameters(const ecvViewportParameters& params) = 0;
    virtual void setPerspectiveState(bool state,
                                     bool objectCenteredView,
                                     bool persistDefault = true) = 0;
    virtual bool perspectiveView() const = 0;
    virtual bool objectCenteredView() const = 0;

    virtual void getGLCameraParameters(ccGLCameraParameters& params) const;
    virtual void getVisibleObjectsBB(ccBBox& box) const;
    virtual void updateConstellationCenterAndZoom(const ccBBox* box = nullptr);

    virtual QRect getGLViewport() const;
    virtual int glWidth() const;
    virtual int glHeight() const;
    virtual int getDevicePixelRatio() const;

    /// VTK-backed views override to persist camera in layout/session JSON.
    virtual QJsonObject saveLayoutCameraState() const;
    virtual void loadLayoutCameraState(const QJsonObject& cameraJson);

    // ================================================================
    // Scene / own DB
    // ================================================================

    virtual void setSceneDB(ccHObject* root) = 0;
    virtual ccHObject* getSceneDB() = 0;
    virtual ccHObject* getOwnDB() = 0;
    virtual void addToOwnDB(ccHObject* obj, bool noDependency = true) = 0;
    virtual void removeFromOwnDB(ccHObject* obj) = 0;

    // ================================================================
    // Qt widget bridge
    // ================================================================

    virtual QWidget* asWidget() = 0;
    virtual const QWidget* asWidget() const = 0;

    // ================================================================
    // Interaction / picking (per-view)
    // ================================================================

    virtual void setInteractionMode(INTERACTION_FLAGS flags);
    virtual INTERACTION_FLAGS getInteractionMode() const;
    virtual void setPickingMode(PICKING_MODE mode);
    virtual PICKING_MODE getPickingMode() const;

    // ================================================================
    // Display context / parameters (per-view)
    // ================================================================

    virtual void getContext(ccGLDrawContext& context) const;
    virtual const ecvGui::ParamStruct& getDisplayParameters() const;
    virtual void setDisplayParameters(const ecvGui::ParamStruct& params,
                                      bool thisWindowOnly = false);
    virtual bool hasOverriddenDisplayParameters() const = 0;

    // ================================================================
    // HotZone / clickable items
    // ================================================================

    virtual void drawClickableItems(int xStart, int& yStart);

    // ================================================================
    // ================================================================

    /// Access the per-view state container.
    /// Returns nullptr for generic stubs; ecvDisplayTools (VTK engine) returns
    /// nullptr; vtkGLView overrides to return &m_ctx.
    virtual ecvViewContext* viewContext() { return nullptr; }
    virtual const ecvViewContext* viewContext() const { return nullptr; }

    /// Per-view active interactor items (labels, etc. being
    /// dragged/interacted). Default returns a process-wide static list;
    /// subclasses override with their own per-window storage.
    virtual std::list<ccInteractor*>& activeItemsRef();

    /// Hot zone pointer for +/- point size and overlay controls (per view).
    virtual ecvHotZone*& hotZonePtrRef();

    /// Clickable overlay regions built during DrawClickableItems (per view).
    virtual std::vector<ecvClickableItem>& clickableItemsRef();

    // ================================================================
    // Per-view entity operations (Phase 3: replaces static dispatch)
    // ================================================================

    virtual void invalidateViewport() {}
    virtual void deprecate3DLayer() {}
    virtual void displayNewMessage(const QString& message,
                                   MessagePosition pos,
                                   bool append = false,
                                   int displayMaxDelay_sec = 2,
                                   MessageType type = CUSTOM_MESSAGE) {
        Q_UNUSED(message);
        Q_UNUSED(pos);
        Q_UNUSED(append);
        Q_UNUSED(displayMaxDelay_sec);
        Q_UNUSED(type);
    }

    virtual QFont textDisplayFont() const;

    virtual void display2DText(const QString& text,
                               int x,
                               int y,
                               unsigned char align,
                               float bkgAlpha,
                               const unsigned char* rgbColor,
                               const QFont* font,
                               const QString& id);

    virtual void moveCamera(float dx, float dy, float dz);
    virtual void rotateBaseViewMat(const ccGLMatrixd& rotMat);

    virtual void loadCameraParameters(const std::string& file);
    virtual void saveCameraParameters(const std::string& file);

    // ================================================================
    // Phase 7a: Per-view VTK operations
    //
    // Virtual methods for entity rendering, widget management, and
    // camera control.  Default implementations are empty stubs.
    // ecvDisplayTools (via VtkDisplayTools) provides shared implementations;
    // vtkGLView overrides them for per-view behavior.
    // Call sites migrate from:
    //     ecvDisplayTools::DrawWidgets(param);
    // to:
    //     context.display->drawWidgets(param);
    // ================================================================

    virtual void draw(const ccGLDrawContext& context, const ccHObject* obj) {
        Q_UNUSED(context);
        Q_UNUSED(obj);
    }
    virtual void drawBBox(const ccGLDrawContext& context, const ccBBox* bbox) {
        Q_UNUSED(context);
        Q_UNUSED(bbox);
    }
    virtual void drawBBoxBatch(const ccGLDrawContext& context,
                               const std::vector<ccBBox>& boxes) {
        Q_UNUSED(context);
        Q_UNUSED(boxes);
    }
    virtual void drawOrientedBBox(const ccGLDrawContext& context,
                                  const ecvOrientedBBox* obb) {
        Q_UNUSED(context);
        Q_UNUSED(obb);
    }
    virtual void updateMeshTextures(const ccGLDrawContext& context,
                                    const ccGenericMesh* mesh) {
        Q_UNUSED(context);
        Q_UNUSED(mesh);
    }

    virtual void drawWidgets(const WIDGETS_PARAMETER& param) {
        Q_UNUSED(param);
    }
    virtual void removeWidgets(const WIDGETS_PARAMETER& param) {
        Q_UNUSED(param);
    }

    virtual bool hideShowEntities(const ccGLDrawContext& context) {
        Q_UNUSED(context);
        return true;
    }
    virtual void removeEntities(const ccGLDrawContext& context) {
        Q_UNUSED(context);
    }
    virtual void changeEntityProperties(PROPERTY_PARAM& param) {
        Q_UNUSED(param);
    }

    virtual void updateCamera() {}
    virtual void updateScene() {}
    virtual void renderScene() { updateScene(); }
    virtual void resetCamera(const ccBBox* bbox) { Q_UNUSED(bbox); }
    virtual void resetCamera() {}
    virtual void setBackgroundColor(const ccGLDrawContext& context) {
        Q_UNUSED(context);
    }
    virtual void toggle2Dviewer(bool state) { Q_UNUSED(state); }
    virtual void toggleExclusiveFullScreen(bool state) { Q_UNUSED(state); }

    virtual QString pick2DLabel(int x, int y) {
        Q_UNUSED(x);
        Q_UNUSED(y);
        return {};
    }
    virtual QString pick3DItem(int x = -1, int y = -1) {
        Q_UNUSED(x);
        Q_UNUSED(y);
        return {};
    }
    virtual QString pickObject(double x = -1, double y = -1) {
        Q_UNUSED(x);
        Q_UNUSED(y);
        return {};
    }

    // ================================================================
    // Phase 7a wave 2: Additional per-view VTK operations
    // ================================================================

    virtual CCVector3d toVtkCoordinates(int x, int y, int z = 0) {
        Q_UNUSED(x);
        Q_UNUSED(y);
        Q_UNUSED(z);
        return CCVector3d(0, 0, 0);
    }
    virtual bool getClick3DPos(int x, int y, CCVector3d& pos) {
        Q_UNUSED(x);
        Q_UNUSED(y);
        Q_UNUSED(pos);
        return false;
    }

    virtual void setView(CC_VIEW_ORIENTATION orientation) {
        Q_UNUSED(orientation);
    }
    virtual CCVector3d getCurrentViewDir() const {
        return CCVector3d(0, 0, -1);
    }
    virtual void setPivotPoint(const CCVector3d& P,
                               bool autoRedraw = true,
                               bool verbose = false) {
        Q_UNUSED(P);
        Q_UNUSED(autoRedraw);
        Q_UNUSED(verbose);
    }
    virtual void setPivotVisibility(PivotVisibility vis) { Q_UNUSED(vis); }
    virtual void setAutoPickPivotAtCenter(bool state) { Q_UNUSED(state); }
    virtual void resetCenterOfRotation(int viewport = 0) { Q_UNUSED(viewport); }
    virtual bool isRotationAxisLocked() const { return false; }
    virtual void lockRotationAxis(bool state, const CCVector3d& axis) {
        Q_UNUSED(state);
        Q_UNUSED(axis);
    }

    virtual void toggleCameraOrientationWidget(bool state) { Q_UNUSED(state); }
    virtual void toggleOrientationMarker(bool state) { Q_UNUSED(state); }
    virtual void toggleDebugTrace() {}

    /// Refresh 2D screen-space labels anchored to the 3D scene (per-view hook).
    virtual void update2DLabels(bool immediateUpdate = false) {
        Q_UNUSED(immediateUpdate);
    }

    virtual bool renderToFile(const QString& filename,
                              float zoomFactor = 1.0f,
                              bool dontScale = false) {
        Q_UNUSED(filename);
        Q_UNUSED(zoomFactor);
        Q_UNUSED(dontScale);
        return false;
    }
    virtual void removeBB(const QString& viewId) { Q_UNUSED(viewId); }
    virtual void removeBB(const ccGLDrawContext& context) { Q_UNUSED(context); }
    virtual void setExclusiveFullScreenFlag(bool state) { Q_UNUSED(state); }

    virtual double getObjectLightIntensity(const QString& viewID) const {
        Q_UNUSED(viewID);
        return 1.0;
    }
    virtual void setObjectLightIntensity(const QString& viewID,
                                         double intensity) {
        Q_UNUSED(viewID);
        Q_UNUSED(intensity);
    }
    virtual double getLightIntensity() const { return 1.0; }
    virtual void setLightIntensity(double intensity) { Q_UNUSED(intensity); }

    virtual void getDataAxesGridProperties(const QString& viewID,
                                           AxesGridProperties& props,
                                           int viewport = 0) const {
        Q_UNUSED(viewID);
        Q_UNUSED(props);
        Q_UNUSED(viewport);
    }
    virtual void setDataAxesGridProperties(const QString& viewID,
                                           const AxesGridProperties& props,
                                           int viewport = 0) {
        Q_UNUSED(viewID);
        Q_UNUSED(props);
        Q_UNUSED(viewport);
    }

    // ================================================================
    // Phase 2.6: Additional per-view operations for QVTKWidgetCustom
    // ================================================================

    virtual void filterByEntityType(std::vector<ccHObject*>& entities,
                                    CV_CLASS_ENUM type) {
        Q_UNUSED(entities);
        Q_UNUSED(type);
    }
    virtual void updateActiveItemsList(int x, int y, bool centerItems) {
        Q_UNUSED(x);
        Q_UNUSED(y);
        Q_UNUSED(centerItems);
    }
    virtual double computeActualPixelSize() const { return 1.0; }
    virtual void updateNamePoseRecursive() {}
    virtual void showPivotSymbol(bool state) { Q_UNUSED(state); }
    virtual bool exclusiveFullScreen() const { return false; }
    virtual CCVector3d convertMousePositionToOrientation(int x, int y) {
        Q_UNUSED(x);
        Q_UNUSED(y);
        return CCVector3d(0, 0, 1);
    }
    virtual bool processClickableItems(int x, int y) {
        Q_UNUSED(x);
        Q_UNUSED(y);
        return false;
    }
    virtual void updateZoom(float zoomFactor) { Q_UNUSED(zoomFactor); }
    virtual void resizeGL(int w, int h) {
        Q_UNUSED(w);
        Q_UNUSED(h);
    }
    virtual void setViewportDefaultPointSize(float size) { Q_UNUSED(size); }
    virtual void setViewportDefaultLineWidth(float width) { Q_UNUSED(width); }
    virtual void setZNearCoef(double coef) { Q_UNUSED(coef); }
    virtual void setFov(float fov_deg) { Q_UNUSED(fov_deg); }
    virtual void setPointSizeOnView(float size) { Q_UNUSED(size); }
    virtual void rotateWithAxis(const CCVector2i& mousePos,
                                const CCVector3d& axis,
                                double angle_deg) {
        Q_UNUSED(mousePos);
        Q_UNUSED(axis);
        Q_UNUSED(angle_deg);
    }
    virtual void startPicking(
            PICKING_MODE mode, int x, int y, int w = 0, int h = 0) {
        Q_UNUSED(mode);
        Q_UNUSED(x);
        Q_UNUSED(y);
        Q_UNUSED(w);
        Q_UNUSED(h);
    }
    virtual void redraw2DLabel() {}

    virtual void scheduleFullRedraw(int delayMs) { Q_UNUSED(delayMs); }
    virtual void stopDeferredPicking() {}
    virtual void startDeferredPickingFor(ecvGenericGLDisplay* targetView) {
        Q_UNUSED(targetView);
    }
    virtual qint64 elapsedMs() const { return 0; }

    // ================================================================
    // Lifecycle notification (ref: CloudCompare aboutToBeRemoved)
    // ================================================================

    virtual void aboutToBeRemoved(ccDrawableObject* /*obj*/) {}

    virtual bool acceptsBoundEntitiesFrom(
            const ecvGenericGLDisplay* /*primaryView*/) const {
        return false;
    }

    // ================================================================
    // Static registry: QWidget* -> ecvGenericGLDisplay*
    // ================================================================

    static ecvGenericGLDisplay* FromWidget(QWidget* widget);
    static void RegisterGLDisplay(QWidget* widget,
                                  ecvGenericGLDisplay* display);
    static void UnregisterGLDisplay(QWidget* widget);
};

Q_DECLARE_OPERATORS_FOR_FLAGS(ecvGenericGLDisplay::INTERACTION_FLAGS)
