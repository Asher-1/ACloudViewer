// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// CVVisualizerTypes.h - PclUtils types replacing pcl::visualization equivalents
// These types allow PCLVis to work without inheriting from PCLVisualizer.

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Eigen
#include <Eigen/Core>

// PCL point cloud (data containers only, no visualization)
#include <pcl/PCLPointField.h>
#include <pcl/point_cloud.h>

// VTK
#include <vtkIdTypeArray.h>
#include <vtkLODActor.h>
#include <vtkMatrix4x4.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

class vtkProp;

/**
 * @namespace PclUtils
 * @brief Utilities and types for PCL-based visualization
 * 
 * This namespace contains custom types and utilities that replace or extend
 * PCL visualization components, providing a more flexible and maintainable
 * interface for 3D visualization in CloudViewer.
 */
namespace PclUtils {

// ============================================================================
// Signal / Connection - lightweight replacement for boost::signals2
// Thread-safe multicast signal with connection management.
// ============================================================================

// Forward-declare Signal so SignalConnection's friend declaration is valid.
template <typename Signature>
class Signal;

/**
 * @brief Represents a connection to a Signal
 * 
 * Manages the lifetime of a callback connection to a Signal.
 * Can be used to disconnect the callback when no longer needed.
 * 
 * @see Signal
 */
class SignalConnection {
public:
    /**
     * @brief Default constructor (unconnected state)
     */
    SignalConnection() = default;

    /**
     * @brief Disconnect this slot from its signal
     * 
     * After calling this, the callback will no longer be invoked.
     * Safe to call multiple times.
     */
    void disconnect() {
        if (auto d = disconnect_fn_.lock()) {
            (*d)();
        }
    }

    /**
     * @brief Check if this connection is still active
     * @return true if connected to a signal, false if disconnected
     */
    bool connected() const { return !disconnect_fn_.expired(); }

private:
    template <typename>
    friend class Signal;

    explicit SignalConnection(std::shared_ptr<std::function<void()>> fn)
        : disconnect_fn_(fn) {}

    std::weak_ptr<std::function<void()>> disconnect_fn_;
};

/**
 * @brief Lightweight multicast signal (replaces boost::signals2::signal)
 * 
 * Thread-safe signal/slot mechanism for event handling.
 * Multiple callbacks can be connected to a single signal, and all will
 * be invoked when the signal is triggered.
 * 
 * Usage example:
 * @code
 *   Signal<void(const MouseEvent&)> mouse_signal;
 *   auto conn = mouse_signal.connect([](const MouseEvent& e) {
 *       std::cout << "Mouse at " << e.getX() << ", " << e.getY() << std::endl;
 *   });
 *   mouse_signal(event);  // fires all connected slots
 *   conn.disconnect();    // removes the slot
 * @endcode
 * 
 * @tparam Args... Callback function signature
 */
template <typename... Args>
class Signal<void(Args...)> {
    struct Slot {
        std::size_t id;
        std::function<void(Args...)> fn;
        std::shared_ptr<std::function<void()>> disconnect_handle;
    };

public:
    Signal() = default;

    // Non-copyable, movable
    Signal(const Signal&) = delete;
    Signal& operator=(const Signal&) = delete;
    Signal(Signal&&) = default;
    Signal& operator=(Signal&&) = default;

    /** \brief Connect a callback. Returns a SignalConnection. */
    SignalConnection connect(std::function<void(Args...)> fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto slot_id = next_id_++;
        auto disconnect_handle =
                std::make_shared<std::function<void()>>([this, slot_id]() {
                    std::lock_guard<std::mutex> lock2(mutex_);
                    slots_.erase(
                            std::remove_if(slots_.begin(), slots_.end(),
                                           [slot_id](const Slot& s) {
                                               return s.id == slot_id;
                                           }),
                            slots_.end());
                });
        slots_.push_back({slot_id, std::move(fn), disconnect_handle});
        return SignalConnection(disconnect_handle);
    }

    /** \brief Fire the signal — invoke all connected slots. */
    void operator()(Args... args) const {
        std::vector<std::function<void(Args...)>> snapshot;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot.reserve(slots_.size());
            for (auto& s : slots_) snapshot.push_back(s.fn);
        }
        for (auto& fn : snapshot) fn(args...);
    }

    /** \brief Returns true if no slots are connected. */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_.empty();
    }

    /** \brief Number of connected slots. */
    std::size_t num_slots() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_.size();
    }

    /** \brief Disconnect all slots. */
    void disconnect_all_slots() {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.clear();
    }

private:
    mutable std::mutex mutex_;
    std::vector<Slot> slots_;
    std::size_t next_id_{0};
};

// ============================================================================
// Vector3ub - replaces pcl::visualization::Vector3ub
// ============================================================================
using Vector3ub = Eigen::Array<unsigned char, 3, 1>;

// ============================================================================
// CloudActorEntry - replaces pcl::visualization::CloudActor
// ============================================================================
struct CloudActorEntry {
    CloudActorEntry() = default;
    virtual ~CloudActorEntry() = default;

    /** \brief The actor holding the data to render. */
    vtkSmartPointer<vtkLODActor> actor;

    /** \brief The viewpoint transformation matrix. */
    vtkSmartPointer<vtkMatrix4x4> viewpoint_transformation_;

    /** \brief Internal cell array. Used for optimizing updatePointCloud. */
    vtkSmartPointer<vtkIdTypeArray> cells;
};

// ============================================================================
// Actor Maps - replaces pcl::visualization::CloudActorMap/ShapeActorMap
// ============================================================================
using CloudActorMap = std::unordered_map<std::string, CloudActorEntry>;
using CloudActorMapPtr = std::shared_ptr<CloudActorMap>;

using ShapeActorMap =
        std::unordered_map<std::string, vtkSmartPointer<vtkProp>>;
using ShapeActorMapPtr = std::shared_ptr<ShapeActorMap>;

using CoordinateActorMap =
        std::unordered_map<std::string, vtkSmartPointer<vtkProp>>;
using CoordinateActorMapPtr = std::shared_ptr<CoordinateActorMap>;

// ============================================================================
// Camera - replaces pcl::visualization::Camera
// ============================================================================
struct Camera {
    /** \brief Focal point or lookAt. */
    double focal[3]{0.0, 0.0, 0.0};
    /** \brief Position of the camera. */
    double pos[3]{0.0, 0.0, 0.0};
    /** \brief Up vector of the camera. */
    double view[3]{0.0, 0.0, 0.0};
    /** \brief Clipping planes of the camera. */
    double clip[2]{0.0, 0.0};
    /** \brief Field of view angle in y-direction (radians). */
    double fovy{0.0};
    /** \brief The window size. */
    int window_size[2]{0, 0};
    /** \brief The window position. */
    int window_pos[2]{0, 0};

    /** \brief Compute the View matrix from camera parameters.
     *  \param[out] view the view matrix (column-major Eigen 4x4)
     */
    void computeViewMatrix(Eigen::Matrix4d& view) const {
        // Camera Z-axis (back)  = normalize(pos - focal)
        Eigen::Vector3d z(pos[0] - focal[0], pos[1] - focal[1],
                          pos[2] - focal[2]);
        z.normalize();
        // Camera X-axis (right) = normalize(up × Z)
        Eigen::Vector3d up(this->view[0], this->view[1], this->view[2]);
        Eigen::Vector3d x = up.cross(z);
        x.normalize();
        // Camera Y-axis = Z × X
        Eigen::Vector3d y = z.cross(x);

        view = Eigen::Matrix4d::Identity();
        view(0, 0) = x[0];
        view(0, 1) = x[1];
        view(0, 2) = x[2];
        view(0, 3) = -x.dot(Eigen::Vector3d(pos[0], pos[1], pos[2]));
        view(1, 0) = y[0];
        view(1, 1) = y[1];
        view(1, 2) = y[2];
        view(1, 3) = -y.dot(Eigen::Vector3d(pos[0], pos[1], pos[2]));
        view(2, 0) = z[0];
        view(2, 1) = z[1];
        view(2, 2) = z[2];
        view(2, 3) = -z.dot(Eigen::Vector3d(pos[0], pos[1], pos[2]));
    }

    /** \brief Compute the Projection matrix from camera parameters.
     *  \param[out] proj the projection matrix (column-major Eigen 4x4)
     */
    void computeProjectionMatrix(Eigen::Matrix4d& proj) const {
        double aspect = (window_size[1] != 0)
                                ? static_cast<double>(window_size[0]) /
                                          window_size[1]
                                : 1.0;
        double nearVal = clip[0];
        double farVal = clip[1];
        double f = 1.0 / std::tan(fovy / 2.0);

        proj = Eigen::Matrix4d::Zero();
        proj(0, 0) = f / aspect;
        proj(1, 1) = f;
        proj(2, 2) = (farVal + nearVal) / (nearVal - farVal);
        proj(2, 3) = (2.0 * farVal * nearVal) / (nearVal - farVal);
        proj(3, 2) = -1.0;
    }
};

// ============================================================================
// Rendering Properties - replaces pcl::visualization::RenderingProperties
// ============================================================================
enum RenderingProperties {
    CV_VISUALIZER_POINT_SIZE,
    CV_VISUALIZER_OPACITY,
    CV_VISUALIZER_LINE_WIDTH,
    CV_VISUALIZER_FONT_SIZE,
    CV_VISUALIZER_COLOR,
    CV_VISUALIZER_REPRESENTATION,
    CV_VISUALIZER_IMMEDIATE_RENDERING,
    CV_VISUALIZER_SHADING,
    CV_VISUALIZER_LUT,
    CV_VISUALIZER_LUT_RANGE
};

enum RenderingRepresentationProperties {
    CV_VISUALIZER_REPRESENTATION_POINTS,
    CV_VISUALIZER_REPRESENTATION_WIREFRAME,
    CV_VISUALIZER_REPRESENTATION_SURFACE
};

enum ShadingRepresentationProperties {
    CV_VISUALIZER_SHADING_FLAT,
    CV_VISUALIZER_SHADING_GOURAUD,
    CV_VISUALIZER_SHADING_PHONG
};

// ============================================================================
// Keyboard modifier - replaces pcl::visualization::InteractorKeyboardModifier
// ============================================================================
enum InteractorKeyboardModifier {
    INTERACTOR_KB_MOD_ALT,
    INTERACTOR_KB_MOD_CTRL,
    INTERACTOR_KB_MOD_SHIFT
};

// ============================================================================
// MouseEvent - replaces pcl::visualization::MouseEvent
// ============================================================================
class MouseEvent {
public:
    enum Type {
        MouseMove = 1,
        MouseButtonPress,
        MouseButtonRelease,
        MouseScrollDown,
        MouseScrollUp,
        MouseDblClick
    };

    enum MouseButton {
        NoButton = 0,
        LeftButton,
        MiddleButton,
        RightButton,
        VScroll
    };

    MouseEvent(const Type& type,
               const MouseButton& button,
               unsigned int x,
               unsigned int y,
               bool alt,
               bool ctrl,
               bool shift,
               bool selection_mode = false)
        : type_(type),
          button_(button),
          pointer_x_(x),
          pointer_y_(y),
          key_state_(0),
          selection_mode_(selection_mode) {
        if (alt) key_state_ = 1;
        if (ctrl) key_state_ |= 2;
        if (shift) key_state_ |= 4;
    }

    const Type& getType() const { return type_; }
    void setType(const Type& type) { type_ = type; }
    const MouseButton& getButton() const { return button_; }
    void setButton(const MouseButton& button) { button_ = button; }
    unsigned int getX() const { return pointer_x_; }
    unsigned int getY() const { return pointer_y_; }
    unsigned int getKeyboardModifiers() const { return key_state_; }
    bool isAltPressed() const { return (key_state_ & 1) != 0; }
    bool isCtrlPressed() const { return (key_state_ & 2) != 0; }
    bool isShiftPressed() const { return (key_state_ & 4) != 0; }
    bool getSelectionMode() const { return selection_mode_; }

private:
    Type type_;
    MouseButton button_;
    unsigned int pointer_x_;
    unsigned int pointer_y_;
    unsigned int key_state_;
    bool selection_mode_;
};

// ============================================================================
// KeyboardEvent - replaces pcl::visualization::KeyboardEvent
// ============================================================================
class KeyboardEvent {
public:
    static const unsigned int Alt = 1;
    static const unsigned int Ctrl = 2;
    static const unsigned int Shift = 4;

    KeyboardEvent(bool action,
                  const std::string& key_sym,
                  unsigned char key,
                  bool alt,
                  bool ctrl,
                  bool shift)
        : action_(action), key_sym_(key_sym), key_(key) {
        if (alt) key_state_ = Alt;
        if (ctrl) key_state_ |= Ctrl;
        if (shift) key_state_ |= Shift;
    }

    bool isAltPressed() const { return (key_state_ & Alt) != 0; }
    bool isCtrlPressed() const { return (key_state_ & Ctrl) != 0; }
    bool isShiftPressed() const { return (key_state_ & Shift) != 0; }
    unsigned char getKeyCode() const { return key_; }
    const std::string& getKeySym() const { return key_sym_; }
    bool keyDown() const { return action_; }
    bool keyUp() const { return !action_; }

private:
    bool action_;
    unsigned int key_state_{0};
    unsigned char key_;
    std::string key_sym_;
};

// ============================================================================
// PointPickingEvent - replaces pcl::visualization::PointPickingEvent
// ============================================================================
class PointPickingEvent {
public:
    PointPickingEvent(int idx)
        : PointPickingEvent(idx, -1, -1, -1) {}
    PointPickingEvent(int idx,
                      float x,
                      float y,
                      float z,
                      const std::string& name = "")
        : idx_(idx),
          idx2_(-1),
          x_(x),
          y_(y),
          z_(z),
          x2_(),
          y2_(),
          z2_(),
          name_(name) {}
    PointPickingEvent(int idx1,
                      int idx2,
                      float x1,
                      float y1,
                      float z1,
                      float x2,
                      float y2,
                      float z2)
        : idx_(idx1),
          idx2_(idx2),
          x_(x1),
          y_(y1),
          z_(z1),
          x2_(x2),
          y2_(y2),
          z2_(z2) {}

    int getPointIndex() const { return idx_; }

    void getPoint(float& x, float& y, float& z) const {
        x = x_;
        y = y_;
        z = z_;
    }

    int getPointIndex(int idx1, int idx2) const {
        idx1 = idx_;
        idx2 = idx2_;
        return idx_;
    }

    void getPoints(float& x1,
                   float& y1,
                   float& z1,
                   float& x2,
                   float& y2,
                   float& z2) const {
        x1 = x_;
        y1 = y_;
        z1 = z_;
        x2 = x2_;
        y2 = y2_;
        z2 = z2_;
    }

    const std::string& getCloudName() const { return name_; }

private:
    int idx_, idx2_;
    float x_, y_, z_;
    float x2_, y2_, z2_;
    std::string name_;
};

// ============================================================================
// AreaPickingEvent - replaces pcl::visualization::AreaPickingEvent
// ============================================================================
class AreaPickingEvent {
public:
    using Indices = std::vector<int>;

    AreaPickingEvent(std::map<std::string, Indices> cloud_indices)
        : cloud_indices_(std::move(cloud_indices)) {}

    bool getPointsIndices(Indices& indices) const {
        if (cloud_indices_.empty()) return false;
        for (const auto& i : cloud_indices_)
            indices.insert(indices.cend(), i.second.cbegin(), i.second.cend());
        return true;
    }

    std::vector<std::string> getCloudNames() const {
        std::vector<std::string> names;
        names.reserve(cloud_indices_.size());
        for (const auto& i : cloud_indices_) names.push_back(i.first);
        return names;
    }

    Indices getPointsIndices(const std::string& name) const {
        const auto cloud = cloud_indices_.find(name);
        if (cloud == cloud_indices_.cend()) return {};
        return cloud->second;
    }

private:
    std::map<std::string, Indices> cloud_indices_;
};

// ============================================================================
// PointCloudColorHandler - replaces pcl::visualization::PointCloudColorHandler
// Base class for color handlers that produce VTK data arrays from point clouds.
// ============================================================================
template <typename PointT>
class PointCloudColorHandler {
public:
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;

    PointCloudColorHandler() = default;
    explicit PointCloudColorHandler(const PointCloudConstPtr& cloud)
        : cloud_(cloud) {}
    virtual ~PointCloudColorHandler() = default;

    /** \brief Check if this handler is capable of handling the input data. */
    virtual bool isCapable() const { return capable_; }

    /** \brief Obtain the actual color for the input dataset as a VTK data
     * array.
     *  \return smart pointer to VTK array, or null on failure */
    virtual vtkSmartPointer<vtkDataArray> getColor() const = 0;

    /** \brief Abstract getName method. */
    virtual std::string getName() const = 0;

    /** \brief Abstract getFieldName method. */
    virtual std::string getFieldName() const = 0;

    /** \brief Set the input cloud. */
    virtual void setInputCloud(const PointCloudConstPtr& cloud) {
        cloud_ = cloud;
    }

    /** \brief Get the input cloud. */
    const PointCloudConstPtr& getInputCloud() const { return cloud_; }

protected:
    PointCloudConstPtr cloud_;
    bool capable_{false};
    int field_idx_{-1};
    std::vector<pcl::PCLPointField> fields_;
};

// ============================================================================
// PointCloudColorHandlerCustom - replaces
// pcl::visualization::PointCloudColorHandlerCustom
// Constant-color handler: all points get the same RGB value.
// ============================================================================
template <typename PointT>
class PointCloudColorHandlerCustom : public PointCloudColorHandler<PointT> {
public:
    using typename PointCloudColorHandler<PointT>::PointCloud;
    using typename PointCloudColorHandler<PointT>::PointCloudConstPtr;
    using PointCloudColorHandler<PointT>::cloud_;
    using PointCloudColorHandler<PointT>::capable_;

    PointCloudColorHandlerCustom(const PointCloudConstPtr& cloud,
                                 double r,
                                 double g,
                                 double b)
        : PointCloudColorHandler<PointT>(cloud),
          r_(static_cast<unsigned char>(r)),
          g_(static_cast<unsigned char>(g)),
          b_(static_cast<unsigned char>(b)) {
        capable_ = true;
    }

    std::string getName() const override { return "PointCloudColorHandlerCustom"; }
    std::string getFieldName() const override { return ""; }

    vtkSmartPointer<vtkDataArray> getColor() const override {
        if (!capable_ || !cloud_) return nullptr;

        auto scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);
        scalars->SetName("Colors");
        const vtkIdType npts =
                static_cast<vtkIdType>(cloud_->size());
        scalars->SetNumberOfTuples(npts);
        unsigned char* colors = scalars->GetPointer(0);
        for (vtkIdType i = 0; i < npts; ++i) {
            colors[i * 3 + 0] = r_;
            colors[i * 3 + 1] = g_;
            colors[i * 3 + 2] = b_;
        }
        return scalars;
    }

private:
    unsigned char r_, g_, b_;
};

// ============================================================================
// PointCloudColorHandlerRGBField - replaces
// pcl::visualization::PointCloudColorHandlerRGBField
// Extracts per-point RGB from the cloud's rgb/rgba field.
// ============================================================================
template <typename PointT>
class PointCloudColorHandlerRGBField : public PointCloudColorHandler<PointT> {
public:
    using typename PointCloudColorHandler<PointT>::PointCloudConstPtr;
    using PointCloudColorHandler<PointT>::cloud_;
    using PointCloudColorHandler<PointT>::capable_;

    PointCloudColorHandlerRGBField() = default;
    explicit PointCloudColorHandlerRGBField(const PointCloudConstPtr& cloud)
        : PointCloudColorHandler<PointT>(cloud) {
        capable_ = true;  // Assume capable for RGB-bearing point types
    }

    std::string getName() const override {
        return "PointCloudColorHandlerRGBField";
    }
    std::string getFieldName() const override { return "rgb"; }

    vtkSmartPointer<vtkDataArray> getColor() const override;
    // Implementation needs pcl point trait access – provided in
    // CVVisualizerTypesImpl.h
};

// ============================================================================
// PointCloudGeometryHandlerXYZ - replaces
// pcl::visualization::PointCloudGeometryHandlerXYZ
// Extracts XYZ coordinates from a point cloud as vtkPoints.
// ============================================================================
template <typename PointT>
class PointCloudGeometryHandlerXYZ {
public:
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;

    explicit PointCloudGeometryHandlerXYZ(const PointCloudConstPtr& cloud)
        : cloud_(cloud) {}

    /** \brief Obtain the geometry (XYZ) as vtkPoints. */
    void getGeometry(vtkSmartPointer<vtkPoints>& points) const {
        if (!cloud_ || cloud_->empty()) return;
        points = vtkSmartPointer<vtkPoints>::New();
        points->SetDataTypeToFloat();
        points->SetNumberOfPoints(
                static_cast<vtkIdType>(cloud_->size()));
        for (std::size_t i = 0; i < cloud_->size(); ++i) {
            const auto& p = (*cloud_)[i];
            points->SetPoint(static_cast<vtkIdType>(i), p.x, p.y, p.z);
        }
    }

private:
    PointCloudConstPtr cloud_;
};

}  // namespace PclUtils

