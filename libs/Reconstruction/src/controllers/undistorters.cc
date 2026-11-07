// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Upstream parity (d3ccaf35 controllers/undistorters.cc) implemented as
// thin synchronous controllers over the fork's *Impl Thread implementations
// in image/undistortion.cc. jpeg_quality is accepted for interface parity;
// applying it requires a Bitmap metadata hook the fork does not have yet.

#include "controllers/undistorters.h"

namespace colmap {

COLMAPUndistorter::COLMAPUndistorter(Options options,
                                     const UndistortCameraOptions& camera_options,
                                     const Reconstruction& reconstruction,
                                     const std::filesystem::path& image_path,
                                     const std::filesystem::path& output_path)
    : options_(std::move(options)),
      camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {}

void COLMAPUndistorter::SetCheckIfStoppedFunc(std::function<bool()> func) {
  check_if_stopped_func_ = std::move(func);
}

void COLMAPUndistorter::Run() {
  COLMAPUndistorterImpl impl(camera_options_,
                             const_cast<Reconstruction*>(&reconstruction_),
                             image_path_,
                             output_path_,
                             options_.num_patch_match_src_images,
                             options_.copy_type,
                             options_.image_ids);
  if (check_if_stopped_func_) {
    impl.SetCheckIfStoppedFunc(check_if_stopped_func_);
  }
  impl.Start();
  impl.Wait();
}

PMVSUndistorter::PMVSUndistorter(const Options& options,
                                 const UndistortCameraOptions& camera_options,
                                 const Reconstruction& reconstruction,
                                 const std::filesystem::path& image_path,
                                 const std::filesystem::path& output_path)
    : options_(options),
      camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {}

void PMVSUndistorter::Run() {
  PMVSUndistorterImpl impl(camera_options_,
                           const_cast<Reconstruction*>(&reconstruction_),
                           image_path_,
                           output_path_);
  impl.Start();
  impl.Wait();
}

CMPMVSUndistorter::CMPMVSUndistorter(
        const Options& options,
        const UndistortCameraOptions& camera_options,
        const Reconstruction& reconstruction,
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_path)
    : options_(options),
      camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {}

void CMPMVSUndistorter::Run() {
  CMPMVSUndistorterImpl impl(camera_options_,
                             const_cast<Reconstruction*>(&reconstruction_),
                             image_path_,
                             output_path_);
  impl.Start();
  impl.Wait();
}

StandaloneImageUndistorter::StandaloneImageUndistorter(
        Options options,
        const UndistortCameraOptions& camera_options,
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_path)
    : options_(std::move(options)),
      camera_options_(camera_options),
      image_path_(image_path),
      output_path_(output_path) {}

void StandaloneImageUndistorter::Run() {
  PureImageUndistorterImpl impl(camera_options_,
                                image_path_,
                                output_path_,
                                options_.image_names_and_cameras);
  impl.Start();
  impl.Wait();
}

StereoImageRectifier::StereoImageRectifier(
        Options options,
        const UndistortCameraOptions& camera_options,
        const Reconstruction& reconstruction,
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_path)
    : options_(std::move(options)),
      camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {}

void StereoImageRectifier::Run() {
  StereoImageRectifierImpl impl(camera_options_,
                                const_cast<Reconstruction*>(&reconstruction_),
                                image_path_,
                                output_path_,
                                options_.stereo_pairs);
  impl.Start();
  impl.Wait();
}

}  // namespace colmap
