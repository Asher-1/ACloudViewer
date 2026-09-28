// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "image/undistortion.h"
#include "scene/reconstruction.h"
#include "util/base_controller.h"

namespace colmap {

// Upstream parity (d3ccaf35 controllers/undistorters.h): the undistortion
// controllers with per-class Options structs and BaseController semantics.
// Each controller synchronously drives the fork's *Impl Thread implementation
// from image/undistortion.{h,cc} (zero logic duplication); jpeg quality is
// accepted for interface parity but not applied yet (the fork Bitmap has no
// metadata hook - adjudicated below).

class COLMAPUndistorter : public BaseController {
 public:
  struct Options {
    CopyType copy_type = CopyType::COPY;
    int num_patch_match_src_images = 20;
    std::vector<image_t> image_ids;
    int jpeg_quality = -1;
    int num_threads = -1;
  };

  COLMAPUndistorter(Options options,
                    const UndistortCameraOptions& camera_options,
                    const Reconstruction& reconstruction,
                    const std::filesystem::path& image_path,
                    const std::filesystem::path& output_path);

  // Upstream parity: an external cancellation probe checked by the
  // implementation between undistortion tasks.
  void SetCheckIfStoppedFunc(std::function<bool()> func);

  void Run();

 private:
  Options options_;
  UndistortCameraOptions camera_options_;
  const Reconstruction& reconstruction_;
  std::filesystem::path image_path_;
  std::filesystem::path output_path_;
  std::function<bool()> check_if_stopped_func_;
};

class PMVSUndistorter : public BaseController {
 public:
  struct Options {
    int jpeg_quality = -1;
    int num_threads = -1;
  };

  PMVSUndistorter(const Options& options,
                  const UndistortCameraOptions& camera_options,
                  const Reconstruction& reconstruction,
                  const std::filesystem::path& image_path,
                  const std::filesystem::path& output_path);

  void Run();

 private:
  Options options_;
  UndistortCameraOptions camera_options_;
  const Reconstruction& reconstruction_;
  std::filesystem::path image_path_;
  std::filesystem::path output_path_;
};

class CMPMVSUndistorter : public BaseController {
 public:
  struct Options {
    int jpeg_quality = -1;
    int num_threads = -1;
  };

  CMPMVSUndistorter(const Options& options,
                    const UndistortCameraOptions& camera_options,
                    const Reconstruction& reconstruction,
                    const std::filesystem::path& image_path,
                    const std::filesystem::path& output_path);

  void Run();

 private:
  Options options_;
  UndistortCameraOptions camera_options_;
  const Reconstruction& reconstruction_;
  std::filesystem::path image_path_;
  std::filesystem::path output_path_;
};

class StandaloneImageUndistorter : public BaseController {
 public:
  struct Options {
    std::vector<std::pair<std::string, Camera>> image_names_and_cameras;
    CopyType copy_type = CopyType::COPY;
    int jpeg_quality = -1;
    int num_threads = -1;
  };

  StandaloneImageUndistorter(Options options,
                             const UndistortCameraOptions& camera_options,
                             const std::filesystem::path& image_path,
                             const std::filesystem::path& output_path);

  void Run();

 private:
  Options options_;
  UndistortCameraOptions camera_options_;
  std::filesystem::path image_path_;
  std::filesystem::path output_path_;
};

class StereoImageRectifier : public BaseController {
 public:
  struct Options {
    std::vector<std::pair<image_t, image_t>> stereo_pairs;
    int jpeg_quality = -1;
    int num_threads = -1;
  };

  StereoImageRectifier(Options options,
                       const UndistortCameraOptions& camera_options,
                       const Reconstruction& reconstruction,
                       const std::filesystem::path& image_path,
                       const std::filesystem::path& output_path);

  void Run();

 private:
  Options options_;
  UndistortCameraOptions camera_options_;
  const Reconstruction& reconstruction_;
  std::filesystem::path image_path_;
  std::filesystem::path output_path_;
};

}  // namespace colmap
