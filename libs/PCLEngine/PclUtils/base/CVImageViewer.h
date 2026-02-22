// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <pcl/correspondence.h>
#include <pcl/geometry/planar_polygon.h>
#include <pcl/memory.h>
#include <pcl/point_types.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "base/CVContextItem.h"
#include "base/CVFloatImageUtils.h"
#include "base/CVVisualizerTypes.h"  // PclUtils::MouseEvent, KeyboardEvent, Vector3ub

class vtkImageSlice;
class vtkContextActor;
class vtkImageViewer;
class vtkImageFlip;
class vtkRenderer;
class vtkImageData;

namespace PclUtils {

// ============================================================================
// ImageViewerInteractorStyle – replaces
// pcl::visualization::ImageViewerInteractorStyle
// ============================================================================
class ImageViewerInteractorStyle : public vtkInteractorStyleImage {
public:
    static ImageViewerInteractorStyle* New();
    ImageViewerInteractorStyle();

    void OnMouseWheelForward() override {}
    void OnMouseWheelBackward() override {}
    void OnMiddleButtonDown() override {}
    void OnRightButtonDown() override {}
    void OnLeftButtonDown() override;
    void OnChar() override;

    void adjustCamera(vtkImageData* image, vtkRenderer* ren);
    void adjustCamera(vtkRenderer* ren);
};

// ============================================================================
// Color constants
// ============================================================================
static const Vector3ub green_color(0, 255, 0);
static const Vector3ub red_color(255, 0, 0);
static const Vector3ub blue_color(0, 0, 255);

// ============================================================================
// ImageViewer – replaces pcl::visualization::ImageViewer
// ============================================================================
class ImageViewer {
public:
    using Ptr = std::shared_ptr<ImageViewer>;
    using ConstPtr = std::shared_ptr<const ImageViewer>;

    explicit ImageViewer(const std::string& window_title = "");
    virtual ~ImageViewer();

    // ------------------------------------------------------------------
    // Interactor style
    // ------------------------------------------------------------------
    void setInteractorStyle(vtkInteractorObserver* style) {
        interactor_->SetInteractorStyle(style);
    }

    // ------------------------------------------------------------------
    // Mono image
    // ------------------------------------------------------------------
    void showMonoImage(const unsigned char* data,
                       unsigned width,
                       unsigned height,
                       const std::string& layer_id = "mono_image",
                       double opacity = 1.0);
    void addMonoImage(const unsigned char* data,
                      unsigned width,
                      unsigned height,
                      const std::string& layer_id = "mono_image",
                      double opacity = 1.0);

    inline void showMonoImage(
            const pcl::PointCloud<pcl::Intensity>::ConstPtr& cloud,
            const std::string& layer_id = "mono_image",
            double opacity = 1.0) {
        return showMonoImage(*cloud, layer_id, opacity);
    }
    inline void addMonoImage(
            const pcl::PointCloud<pcl::Intensity>::ConstPtr& cloud,
            const std::string& layer_id = "mono_image",
            double opacity = 1.0) {
        return addMonoImage(*cloud, layer_id, opacity);
    }
    void showMonoImage(const pcl::PointCloud<pcl::Intensity>& cloud,
                       const std::string& layer_id = "mono_image",
                       double opacity = 1.0);
    void addMonoImage(const pcl::PointCloud<pcl::Intensity>& cloud,
                      const std::string& layer_id = "mono_image",
                      double opacity = 1.0);

    inline void showMonoImage(
            const pcl::PointCloud<pcl::Intensity8u>::ConstPtr& cloud,
            const std::string& layer_id = "mono_image",
            double opacity = 1.0) {
        return showMonoImage(*cloud, layer_id, opacity);
    }
    inline void addMonoImage(
            const pcl::PointCloud<pcl::Intensity8u>::ConstPtr& cloud,
            const std::string& layer_id = "mono_image",
            double opacity = 1.0) {
        return addMonoImage(*cloud, layer_id, opacity);
    }
    void showMonoImage(const pcl::PointCloud<pcl::Intensity8u>& cloud,
                       const std::string& layer_id = "mono_image",
                       double opacity = 1.0);
    void addMonoImage(const pcl::PointCloud<pcl::Intensity8u>& cloud,
                      const std::string& layer_id = "mono_image",
                      double opacity = 1.0);

    // ------------------------------------------------------------------
    // RGB image
    // ------------------------------------------------------------------
    void showRGBImage(const unsigned char* data,
                      unsigned width,
                      unsigned height,
                      const std::string& layer_id = "rgb_image",
                      double opacity = 1.0);
    void addRGBImage(const unsigned char* data,
                     unsigned width,
                     unsigned height,
                     const std::string& layer_id = "rgb_image",
                     double opacity = 1.0,
                     bool autoresize = true);

    template <typename T>
    inline void showRGBImage(const typename pcl::PointCloud<T>::ConstPtr& cloud,
                             const std::string& layer_id = "rgb_image",
                             double opacity = 1.0) {
        return showRGBImage<T>(*cloud, layer_id, opacity);
    }
    template <typename T>
    inline void addRGBImage(const typename pcl::PointCloud<T>::ConstPtr& cloud,
                            const std::string& layer_id = "rgb_image",
                            double opacity = 1.0) {
        return addRGBImage<T>(*cloud, layer_id, opacity);
    }
    template <typename T>
    void showRGBImage(const pcl::PointCloud<T>& cloud,
                      const std::string& layer_id = "rgb_image",
                      double opacity = 1.0);
    template <typename T>
    void addRGBImage(const pcl::PointCloud<T>& cloud,
                     const std::string& layer_id = "rgb_image",
                     double opacity = 1.0);

    // ------------------------------------------------------------------
    // Float / short / angle images
    // ------------------------------------------------------------------
    void showFloatImage(const float* data,
                        unsigned int width,
                        unsigned int height,
                        float min_value = std::numeric_limits<float>::min(),
                        float max_value = std::numeric_limits<float>::max(),
                        bool grayscale = false,
                        const std::string& layer_id = "float_image",
                        double opacity = 1.0);
    void addFloatImage(const float* data,
                       unsigned int width,
                       unsigned int height,
                       float min_value = std::numeric_limits<float>::min(),
                       float max_value = std::numeric_limits<float>::max(),
                       bool grayscale = false,
                       const std::string& layer_id = "float_image",
                       double opacity = 1.0);
    void showShortImage(const unsigned short* data,
                        unsigned int width,
                        unsigned int height,
                        unsigned short min_value =
                                std::numeric_limits<unsigned short>::min(),
                        unsigned short max_value =
                                std::numeric_limits<unsigned short>::max(),
                        bool grayscale = false,
                        const std::string& layer_id = "short_image",
                        double opacity = 1.0);
    void addShortImage(const unsigned short* data,
                       unsigned int width,
                       unsigned int height,
                       unsigned short min_value =
                               std::numeric_limits<unsigned short>::min(),
                       unsigned short max_value =
                               std::numeric_limits<unsigned short>::max(),
                       bool grayscale = false,
                       const std::string& layer_id = "short_image",
                       double opacity = 1.0);
    void showAngleImage(const float* data,
                        unsigned width,
                        unsigned height,
                        const std::string& layer_id = "angle_image",
                        double opacity = 1.0);
    void addAngleImage(const float* data,
                       unsigned width,
                       unsigned height,
                       const std::string& layer_id = "angle_image",
                       double opacity = 1.0);
    void showHalfAngleImage(const float* data,
                            unsigned width,
                            unsigned height,
                            const std::string& layer_id = "half_angle_image",
                            double opacity = 1.0);
    void addHalfAngleImage(const float* data,
                           unsigned width,
                           unsigned height,
                           const std::string& layer_id = "half_angle_image",
                           double opacity = 1.0);

    // ------------------------------------------------------------------
    // Markers / points
    // ------------------------------------------------------------------
    void markPoint(std::size_t u,
                   std::size_t v,
                   Vector3ub fg_color,
                   Vector3ub bg_color = red_color,
                   double radius = 3.0,
                   const std::string& layer_id = "points",
                   double opacity = 1.0);
    void markPoints(const std::vector<int>& uv,
                    Vector3ub fg_color,
                    Vector3ub bg_color = red_color,
                    double size = 3.0,
                    const std::string& layer_id = "markers",
                    double opacity = 1.0);
    void markPoints(const std::vector<float>& uv,
                    Vector3ub fg_color,
                    Vector3ub bg_color = red_color,
                    double size = 3.0,
                    const std::string& layer_id = "markers",
                    double opacity = 1.0);

    // ------------------------------------------------------------------
    // Window / interaction
    // ------------------------------------------------------------------
    void setWindowTitle(const std::string& name);
    void spin();
    void spinOnce(int time = 1, bool force_redraw = true);

    // ------------------------------------------------------------------
    // Keyboard / mouse callbacks
    // ------------------------------------------------------------------
    SignalConnection registerKeyboardCallback(
            void (*callback)(const KeyboardEvent&, void*),
            void* cookie = nullptr) {
        return registerKeyboardCallback(
                [=](const KeyboardEvent& e) { (*callback)(e, cookie); });
    }
    template <typename T>
    SignalConnection registerKeyboardCallback(
            void (T::*callback)(const KeyboardEvent&, void*),
            T& instance,
            void* cookie = nullptr) {
        return registerKeyboardCallback([=, &instance](const KeyboardEvent& e) {
            (instance.*callback)(e, cookie);
        });
    }
    SignalConnection registerKeyboardCallback(
            std::function<void(const KeyboardEvent&)> cb);

    SignalConnection registerMouseCallback(void (*callback)(const MouseEvent&,
                                                            void*),
                                           void* cookie = nullptr) {
        return registerMouseCallback(
                [=](const MouseEvent& e) { (*callback)(e, cookie); });
    }
    template <typename T>
    SignalConnection registerMouseCallback(
            void (T::*callback)(const MouseEvent&, void*),
            T& instance,
            void* cookie = nullptr) {
        return registerMouseCallback([=, &instance](const MouseEvent& e) {
            (instance.*callback)(e, cookie);
        });
    }
    SignalConnection registerMouseCallback(
            std::function<void(const MouseEvent&)> cb);

    // ------------------------------------------------------------------
    // Position / size
    // ------------------------------------------------------------------
    void setPosition(int x, int y);
    void setSize(int xw, int yw);
    int* getSize();

    bool wasStopped() const { return stopped_; }
    void close() {
        stopped_ = true;
        interactor_->TerminateApp();
    }

    // ------------------------------------------------------------------
    // 2D primitives
    // ------------------------------------------------------------------
    bool addCircle(unsigned int x,
                   unsigned int y,
                   double radius,
                   const std::string& layer_id = "circles",
                   double opacity = 1.0);
    bool addCircle(unsigned int x,
                   unsigned int y,
                   double radius,
                   double r,
                   double g,
                   double b,
                   const std::string& layer_id = "circles",
                   double opacity = 1.0);
    bool addRectangle(const pcl::PointXY& min_pt,
                      const pcl::PointXY& max_pt,
                      const std::string& layer_id = "rectangles",
                      double opacity = 1.0);
    bool addRectangle(const pcl::PointXY& min_pt,
                      const pcl::PointXY& max_pt,
                      double r,
                      double g,
                      double b,
                      const std::string& layer_id = "rectangles",
                      double opacity = 1.0);
    bool addRectangle(unsigned int x_min,
                      unsigned int x_max,
                      unsigned int y_min,
                      unsigned int y_max,
                      const std::string& layer_id = "rectangles",
                      double opacity = 1.0);
    bool addRectangle(unsigned int x_min,
                      unsigned int x_max,
                      unsigned int y_min,
                      unsigned int y_max,
                      double r,
                      double g,
                      double b,
                      const std::string& layer_id = "rectangles",
                      double opacity = 1.0);
    template <typename T>
    bool addRectangle(const typename pcl::PointCloud<T>::ConstPtr& image,
                      const T& min_pt,
                      const T& max_pt,
                      const std::string& layer_id = "rectangles",
                      double opacity = 1.0);
    template <typename T>
    bool addRectangle(const typename pcl::PointCloud<T>::ConstPtr& image,
                      const T& min_pt,
                      const T& max_pt,
                      double r,
                      double g,
                      double b,
                      const std::string& layer_id = "rectangles",
                      double opacity = 1.0);
    template <typename T>
    bool addRectangle(const typename pcl::PointCloud<T>::ConstPtr& image,
                      const pcl::PointCloud<T>& mask,
                      double r,
                      double g,
                      double b,
                      const std::string& layer_id = "rectangles",
                      double opacity = 1.0);
    template <typename T>
    bool addRectangle(const typename pcl::PointCloud<T>::ConstPtr& image,
                      const pcl::PointCloud<T>& mask,
                      const std::string& layer_id = "image_mask",
                      double opacity = 1.0);

    bool addFilledRectangle(unsigned int x_min,
                            unsigned int x_max,
                            unsigned int y_min,
                            unsigned int y_max,
                            const std::string& layer_id = "boxes",
                            double opacity = 0.5);
    bool addFilledRectangle(unsigned int x_min,
                            unsigned int x_max,
                            unsigned int y_min,
                            unsigned int y_max,
                            double r,
                            double g,
                            double b,
                            const std::string& layer_id = "boxes",
                            double opacity = 0.5);
    bool addLine(unsigned int x_min,
                 unsigned int y_min,
                 unsigned int x_max,
                 unsigned int y_max,
                 double r,
                 double g,
                 double b,
                 const std::string& layer_id = "line",
                 double opacity = 1.0);
    bool addLine(unsigned int x_min,
                 unsigned int y_min,
                 unsigned int x_max,
                 unsigned int y_max,
                 const std::string& layer_id = "line",
                 double opacity = 1.0);
    bool addText(unsigned int x,
                 unsigned int y,
                 const std::string& text,
                 double r,
                 double g,
                 double b,
                 const std::string& layer_id = "line",
                 double opacity = 1.0);
    bool addText(unsigned int x,
                 unsigned int y,
                 const std::string& text,
                 const std::string& layer_id = "line",
                 double opacity = 1.0);

    // ------------------------------------------------------------------
    // Mask / polygon / correspondences (template)
    // ------------------------------------------------------------------
    template <typename T>
    bool addMask(const typename pcl::PointCloud<T>::ConstPtr& image,
                 const pcl::PointCloud<T>& mask,
                 double r,
                 double g,
                 double b,
                 const std::string& layer_id = "image_mask",
                 double opacity = 0.5);
    template <typename T>
    bool addMask(const typename pcl::PointCloud<T>::ConstPtr& image,
                 const pcl::PointCloud<T>& mask,
                 const std::string& layer_id = "image_mask",
                 double opacity = 0.5);
    template <typename T>
    bool addPlanarPolygon(const typename pcl::PointCloud<T>::ConstPtr& image,
                          const pcl::PlanarPolygon<T>& polygon,
                          double r,
                          double g,
                          double b,
                          const std::string& layer_id = "planar_polygon",
                          double opacity = 1.0);
    template <typename T>
    bool addPlanarPolygon(const typename pcl::PointCloud<T>::ConstPtr& image,
                          const pcl::PlanarPolygon<T>& polygon,
                          const std::string& layer_id = "planar_polygon",
                          double opacity = 1.0);

    // ------------------------------------------------------------------
    // Layer management
    // ------------------------------------------------------------------
    bool addLayer(const std::string& layer_id,
                  int width,
                  int height,
                  double opacity = 0.5);
    void removeLayer(const std::string& layer_id);

    template <typename PointT>
    bool showCorrespondences(const pcl::PointCloud<PointT>& source_img,
                             const pcl::PointCloud<PointT>& target_img,
                             const pcl::Correspondences& correspondences,
                             int nth = 1,
                             const std::string& layer_id = "correspondences");

protected:
    void render();

    void convertIntensityCloudToUChar(
            const pcl::PointCloud<pcl::Intensity>& cloud,
            std::vector<unsigned char>& data);
    void convertIntensityCloud8uToUChar(
            const pcl::PointCloud<pcl::Intensity8u>& cloud,
            std::vector<unsigned char>& data);
    template <typename T>
    void convertRGBCloudToUChar(const pcl::PointCloud<T>& cloud,
                                std::vector<unsigned char>& data);

    void resetStoppedFlag() { stopped_ = false; }

    void emitMouseEvent(unsigned long event_id);
    void emitKeyboardEvent(unsigned long event_id);

    static void MouseCallback(vtkObject*,
                              unsigned long eid,
                              void* clientdata,
                              void* calldata);
    static void KeyboardCallback(vtkObject*,
                                 unsigned long eid,
                                 void* clientdata,
                                 void* calldata);

protected:  // types
    struct ExitMainLoopTimerCallback : public vtkCommand {
        ExitMainLoopTimerCallback() : right_timer_id(-1), window(nullptr) {}
        static ExitMainLoopTimerCallback* New() {
            return new ExitMainLoopTimerCallback;
        }
        void Execute(vtkObject*,
                     unsigned long event_id,
                     void* call_data) override {
            if (event_id != vtkCommand::TimerEvent) return;
            int timer_id = *static_cast<int*>(call_data);
            if (timer_id != right_timer_id) return;
            window->interactor_->TerminateApp();
        }
        int right_timer_id;
        ImageViewer* window;
    };
    struct ExitCallback : public vtkCommand {
        ExitCallback() : window(nullptr) {}
        static ExitCallback* New() { return new ExitCallback; }
        void Execute(vtkObject*, unsigned long event_id, void*) override {
            if (event_id != vtkCommand::ExitEvent) return;
            window->stopped_ = true;
            window->interactor_->TerminateApp();
        }
        ImageViewer* window;
    };

protected:
    /** \brief Internal structure describing a layer. */
    struct Layer {
        Layer() {}
        vtkSmartPointer<vtkContextActor> actor;
        std::string layer_name;
    };

    using LayerMap = std::vector<Layer>;

    LayerMap::iterator createLayer(const std::string& layer_id,
                                   int width,
                                   int height,
                                   double opacity = 0.5,
                                   bool fill_box = true);

    Signal<void(const MouseEvent&)> mouse_signal_;
    Signal<void(const KeyboardEvent&)> keyboard_signal_;

    vtkSmartPointer<vtkRenderWindowInteractor> interactor_;
    vtkSmartPointer<vtkCallbackCommand> mouse_command_;
    vtkSmartPointer<vtkCallbackCommand> keyboard_command_;

    vtkSmartPointer<ExitMainLoopTimerCallback> exit_main_loop_timer_callback_;
    vtkSmartPointer<ExitCallback> exit_callback_;

    vtkSmartPointer<vtkRenderWindow> win_;
    vtkSmartPointer<vtkRenderer> ren_;
    vtkSmartPointer<vtkImageSlice> slice_;
    vtkSmartPointer<ImageViewerInteractorStyle> interactor_style_;

    std::vector<unsigned char> data_;
    std::size_t data_size_;
    bool stopped_;
    int timer_id_;

    LayerMap layer_map_;
    vtkSmartPointer<vtkImageFlip> algo_;
    std::vector<unsigned char*> image_data_;

    struct LayerComparator {
        LayerComparator(const std::string& str) : str_(str) {}
        const std::string& str_;
        bool operator()(const Layer& layer) const {
            return layer.layer_name == str_;
        }
    };
};

}  // namespace PclUtils

// ============================================================================
// Template implementations
// ============================================================================
#include "CVImageViewer.hpp"
