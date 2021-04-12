// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include <QtCore>
#include <QtOpenGL>

#include "ColorMaps.h"
#include "base/database.h"
#include "base/reconstruction.h"
#include "OptionManager.h"
#include "ImageViewerWidget.h"
#include "MovieGrabberWidget.h"
#include "PointViewerWidget.h"
#include "RenderOptions.h"

#include <LineSet.h>

class ccPointCloud;
class ccCameraSensor;

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
  const float kPointScaleSpeed = 0.1f;
  const float kInitImageSize = 0.2f;
  const float kMinImageSize = 1e-6f;
  const float kMaxImageSize = 1e3f;
  const float kImageScaleSpeed = 0.1f;
  const int kDoubleClickInterval = 250;

  ModelViewerWidget(QWidget* parent, OptionManager* options);
  virtual ~ModelViewerWidget() override;

  void ReloadReconstruction();
  void ClearReconstruction();

  int GetProjectionType() const;

  // Takes ownwership of the colormap objects.
  void SetPointColormap(PointColormapBase* colormap);
  void SetImageColormap(ImageColormapBase* colormap);

  void UpdateMovieGrabber();

  void ChangeFocusDistance(const float delta);
  void ChangeNearPlane(const float delta);
  void ChangePointSize(const float delta);
  void ChangeCameraSize(const float delta);

  QMatrix4x4 ModelViewMatrix() const;
  void SetModelViewMatrix(const QMatrix4x4& matrix);

  void SelectObject(const int x, const int y);
  void SelectMoviewGrabberView(const size_t view_idx);

  QImage GrabImage();
  void GrabMovie();

  void ShowPointInfo(const colmap::point3D_t point3D_id);
  void ShowImageInfo(const colmap::image_t image_id);

  float PointSize() const;
  float ImageSize() const;
  void SetPointSize(const float point_size);
  void SetImageSize(const float image_size);

  // Copy of current scene data that is displayed
  colmap::Reconstruction* reconstruction = nullptr;
  EIGEN_STL_UMAP(colmap::camera_t, colmap::Camera) cameras;
  EIGEN_STL_UMAP(colmap::image_t, colmap::Image) images;
  EIGEN_STL_UMAP(colmap::point3D_t, colmap::Point3D) points3D;
  std::vector<colmap::image_t> reg_image_ids;

  QLabel* statusbar_status_label;

 private:
  void mousePressEvent(QMouseEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;

  void SetupView();

  void Upload();
  void UploadPointData(const bool selection_mode = false);
  void UploadPointConnectionData();
  void UploadImageData(const bool selection_mode = false);
  void UploadImageConnectionData();
  void UploadMovieGrabberData();

  void update();
  void drawPointCloud(ccPointCloud* cloud);
  void resetPointCloud(ccPointCloud* cloud);
  void drawLines(geometry::LineSet& lineset);
  void resetLines(geometry::LineSet& lineset);
  void drawCameraSensors(std::vector<std::shared_ptr<ccCameraSensor>>& sensors);
  void resetCameraSensors(std::vector<std::shared_ptr<ccCameraSensor>>& sensors);

  void ComposeProjectionMatrix();

  float ZoomScale() const;
  float AspectRatio() const;
  float OrthographicWindowExtent() const;

  Eigen::Vector3f PositionToArcballVector(const float x, const float y) const;

  OptionManager* options_;

  QMatrix4x4 model_view_matrix_;
  QMatrix4x4 projection_matrix_;

  PointViewerWidget* point_viewer_widget_;
  DatabaseImageViewerWidget* image_viewer_widget_;
  MovieGrabberWidget* movie_grabber_widget_;

  std::unique_ptr<PointColormapBase> point_colormap_;
  std::unique_ptr<ImageColormapBase> image_colormap_;

  bool mouse_is_pressed_;
  QTimer mouse_press_timer_;
  QPoint prev_mouse_pos_;

  float focus_distance_;

  std::vector<std::pair<std::size_t, char>> selection_buffer_;
  colmap::image_t selected_image_id_;
  colmap::point3D_t selected_point3D_id_;
  std::size_t selected_movie_grabber_view_;

  // for visualization
  ccPointCloud* cloud_sparse_;
  ccPointCloud* cloud_dense_;

  geometry::LineSet point_line_data_;
  geometry::LineSet image_line_data_;
  std::vector<std::shared_ptr<ccCameraSensor>> sensors_;

  geometry::LineSet movie_grabber_path_;
  std::vector<std::shared_ptr<ccCameraSensor>> movie_grabber_sensors_;

  bool coordinate_grid_enabled_;

  // Size of points (dynamic): does not require re-uploading of points.
  float point_size_;
  // Size of image models (not dynamic): requires re-uploading of image models.
  float image_size_;
  // Near clipping plane.
  float near_plane_;

};

}  // namespace cloudViewer
