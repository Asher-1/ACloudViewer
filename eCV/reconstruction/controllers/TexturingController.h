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

#include "base/reconstruction.h"
#include "util/misc.h"
#include "util/alignment.h"
#include "util/threading.h"

#include "camera/PinholeCameraTrajectory.h"

namespace colmap {
class Reconstruction;
}

namespace cloudViewer {

struct TexturingOptions {
  // show verbose information
  bool verbose = true;

  // Display a 3D representation showing the a cloud and a list of camera with their 6DOf poses
  bool show_cameras = false;

  // obj textured mesh save precision
  int save_precision = 5;

  // textured mesh file path
  std::string meshed_file_path = "";

  std::string textured_file_path = "";

  bool Check() const;
};

// Undistort images and export undistorted cameras, as required by the
// mvs::PatchMatchController class.
class TexturingReconstruction : public colmap::Thread {
 public:
  TexturingReconstruction(
      const TexturingOptions& options,
      const colmap::Reconstruction& reconstruction,
      const std::string& image_path,
      const std::string& output_path,
      const std::vector<colmap::image_t>& image_ids = std::vector<colmap::image_t>());

 private:
  void Run();

  bool Texturing(const colmap::image_t image_id, std::size_t index);

  TexturingOptions options_;
  const std::string image_path_;
  const std::string output_path_;
  const std::vector<colmap::image_t> image_ids_;
  const colmap::Reconstruction& reconstruction_;
  std::vector<std::string> image_names_;
  camera::PinholeCameraTrajectory camera_trajectory_;
};
}  // namespace colmap
