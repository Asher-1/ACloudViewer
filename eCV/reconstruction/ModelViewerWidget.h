// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <LineSet.h>
#include <ecvBBox.h>

#include <QtCore>
#include <QtOpenGL>

#include "ColorMaps.h"
#include "ImageViewerWidget.h"
#include "MovieGrabberWidget.h"
#include "OptionManager.h"
#include "PointViewerWidget.h"
#include "RenderOptions.h"
#include "base/database.h"
#include "base/reconstruction.h"

class ccPointCloud;
class ccCameraSensor;
class MainWindow;

namespace cloudViewer {

class ModelViewerWidget : public QWidget {
public:
    const float kInitNearPlane = 1.0f;
    const float kMinNearPlane = 1e-3f;
    const float kMaxNearPlane = 1e5f;
    const float kNearPlaneScaleSpeed = 0.02f;
    const float kFarPlane = 1e5f;
    const float kInitFocusDistance = 100.0f;
    const float kMinFocusDistance = 1e-5f;
    const float kMaxFocusDistance = 1e8f;
    const float kFieldOfView = 25.0f;
    const float kFocusSpeed = 2.0f;
    const float kInitPointSize = 1.0f;
    const float kMinPointSize = 0.5f;
    const float kMaxPointSize = 100.0f;
    const float kPointScaleSpeed = 1.0f;
    const float kInitImageSize = 0.2f;
    const float kMinImageSize = 1e-6f;
    const float kMaxImageSize = 1e3f;
    const float kImageScaleSpeed = 0.1f;
    const int kDoubleClickInterval = 250;

    ModelViewerWidget(QWidget* parent, OptionManager* options, MainWindow* app);

    void Release();

    QWidget* getMainWindow();

    void ReloadReconstruction();
    void ClearReconstruction();

    int GetProjectionType() const;

    // Takes ownwership of the colormap objects.
    void SetPointColormap(PointColormapBase* colormap);
    void SetImageColormap(ImageColormapBase* colormap);

    void UpdateMovieGrabber();

    float ZoomScale();
    float AspectRatio() const;
    void ChangeFocusDistance(const float delta);
    void ChangePointSize(const float delta);
    void ChangeCameraSize(const float delta);

    void ResetView();

    ccGLMatrixd ModelViewMatrix() const;

    void SelectObject(ccHObject* entity,
                      unsigned subEntityID,
                      int x,
                      int y,
                      const CCVector3& P);
    //  void SelectObject(const CCVector3& p, int index, const std::string& id);
    void SelectMoviewGrabberView(const size_t view_idx);

    QImage GrabImage();
    void GrabMovie();

    void update();
    void StartRender();
    void EndRender(bool autoZoom = true);

    void ShowPointInfo(const colmap::point3D_t point3D_id);
    void ShowImageInfo(const colmap::image_t image_id);

    void SetPerspectiveProjection();
    void SetOrthogonalProjection();

    float PointSize() const;
    float ImageSize() const;
    void SetPointSize(const float point_size, bool autoUpdate = true);
    void SetImageSize(const float image_size, bool autoUpdate = true);

    void SetBackgroundColor(const float r, const float g, const float b);

    // Copy of current scene data that is displayed
    colmap::Reconstruction* reconstruction = nullptr;
    std::unordered_map<colmap::camera_t, colmap::Camera> cameras;
    std::unordered_map<colmap::image_t, colmap::Image> images;
    std::unordered_map<colmap::point3D_t, colmap::Point3D> points3D;
    std::vector<colmap::image_t> reg_image_ids;

    QLabel* statusbar_status_label;

private:
    void SetupView();

    void Upload();
    void UploadPointData(const bool selection_mode = false);
    void UploadPointConnectionData();
    void UploadImageData(const bool selection_mode = false);
    void UploadImageConnectionData();
    void UploadMovieGrabberData();

    void drawPointCloud(ccPointCloud* cloud);
    void resetPointCloud(ccPointCloud* cloud);
    void drawLines(geometry::LineSet& lineset);
    void resetLines(geometry::LineSet& lineset);
    void drawCameraSensors(std::vector<ccCameraSensor*>& sensors);
    void resetCameraSensors(std::vector<ccCameraSensor*>& sensors);
    void clearSensors(std::vector<ccCameraSensor*>& sensors);
    void updateSensors(std::vector<ccCameraSensor*>& sensors,
                       const std::vector<colmap::image_t>& image_ids);
    ccHObject* getSelectedCamera(colmap::image_t selected_id);

    OptionManager* options_;

    PointViewerWidget* point_viewer_widget_;
    DatabaseImageViewerWidget* image_viewer_widget_;
    MovieGrabberWidget* movie_grabber_widget_;

    std::unique_ptr<PointColormapBase> point_colormap_;
    std::unique_ptr<ImageColormapBase> image_colormap_;

    float focus_distance_;

    std::vector<std::pair<std::size_t, char>> selection_buffer_;
    colmap::image_t selected_image_id_;
    colmap::point3D_t selected_point3D_id_;
    std::size_t selected_movie_grabber_view_;

    MainWindow* app_;

    // for visualization
    ccPointCloud* cloud_sparse_;
    ccBBox bbox_;

    geometry::LineSet point_line_data_;
    geometry::LineSet image_line_data_;
    std::vector<ccCameraSensor*> sensors_;

    ccHObject* main_sensors_;

    geometry::LineSet movie_grabber_path_;
    std::vector<ccCameraSensor*> movie_grabber_sensors_;

    // Size of points (dynamic): does not require re-uploading of points.
    float point_size_;
    // Size of image models (not dynamic): requires re-uploading of image
    // models.
    float image_size_;
};

}  // namespace cloudViewer
