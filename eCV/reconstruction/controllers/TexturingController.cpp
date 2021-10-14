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

#include "TexturingController.h"

#include <fstream>

#include "base/reconstruction.h"

// CV_CORE_LIB
#include <FileSystem.h>

// QPCL_ENGINE_LIB
#ifdef USE_PCL_BACKEND
#include <Tools/PclTools.h>
#endif

namespace cloudViewer {

using namespace colmap;

TexturingReconstruction::TexturingReconstruction(const TexturingOptions& options,
                                                 const Reconstruction& reconstruction,
                                                 const std::string& image_path,
                                                 const std::string& output_path,
                                                 const std::vector<image_t>& image_ids)
    : options_(options),
      image_path_(image_path),
      output_path_(output_path),
      image_ids_(image_ids),
      reconstruction_(reconstruction) {}

void TexturingReconstruction::Run() {
  PrintHeading1("Mesh Texturing");

  CreateDirIfNotExists(JoinPaths(output_path_, "images"));
  CreateDirIfNotExists(JoinPaths(output_path_, "sparse"));
  CreateDirIfNotExists(JoinPaths(output_path_, "stereo"));
  CreateDirIfNotExists(JoinPaths(output_path_, "stereo/depth_maps"));
  std::string parent_path = utility::filesystem::GetFileParentDirectory(
                            options_.textured_file_path);
  CreateDirIfNotExists(parent_path);
  CreateDirIfNotExists(JoinPaths(parent_path, "images"));

  ThreadPool thread_pool;
  std::vector<std::future<bool>> futures;
  futures.reserve(reconstruction_.NumRegImages());
  camera_trajectory_.parameters_.clear();
  if (image_ids_.empty()) {
    camera_trajectory_.parameters_.resize(reconstruction_.NumRegImages());
    for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
      const image_t image_id = reconstruction_.RegImageIds().at(i);
      futures.push_back(
          thread_pool.AddTask(&TexturingReconstruction::Texturing, this, image_id, i));
    }
  } else {
    std::size_t index = 0;
    camera_trajectory_.parameters_.resize(image_ids_.size());
    for (const image_t image_id : image_ids_) {
      futures.push_back(
          thread_pool.AddTask(&TexturingReconstruction::Texturing, this, image_id, index));
      index += 1;
    }
  }

  // Only use the image names for the successfully textured mesh
  // when writing the MVS config files
  image_names_.clear();
  for (size_t i = 0; i < futures.size(); ++i) {
    if (IsStopped()) {
      break;
    }

    if (options_.verbose)
    {
        std::cout << StringPrintf("texture image [%d/%d]", i + 1, futures.size()) << std::endl;
        CVLog::Print("texture image [%d/%d]", i + 1, futures.size());
    }

    if (futures[i].get()) {
      if (image_ids_.empty()) {
        const image_t image_id = reconstruction_.RegImageIds().at(i);
        image_names_.push_back(reconstruction_.Image(image_id).Name());
      } else {
        image_names_.push_back(reconstruction_.Image(image_ids_[i]).Name());
      }
    }
  }

  // check camera trajectory validation
  for (int i = 0; i < camera_trajectory_.parameters_.size(); ++i) {
      auto& cameraParams = camera_trajectory_.parameters_[i];
      if (!cameraParams.intrinsic_.IsValid()) {
          CVLog::Error("Invalid camera intrinsic parameters found and ignore texturing!");
          return;
      }
  }

#ifdef USE_PCL_BACKEND
  PCLTextureMesh::Ptr texturedMesh = PclTools::CreateTexturingMesh(options_.meshed_file_path,
                                                                   camera_trajectory_,
                                                                   options_.show_cameras,
                                                                   options_.verbose);
  if (texturedMesh) {
      if(!PclTools::SaveOBJFile(options_.textured_file_path, *texturedMesh,
                                static_cast<unsigned>(options_.save_precision)))
      {
          CVLog::Warning("Saving textured mesh to %s failed!", options_.textured_file_path.c_str());
      }
      else {
          CVLog::Print("Save textured mesh to %s successfully!", options_.textured_file_path.c_str());
      }
      std::cout << "Texturing reconstruction successful..." << std::endl;
      CVLog::Print("Texturing reconstruction successful...");
  } else {
      std::cout << "Texturing reconstruction failed..." << std::endl;
      CVLog::Print("Texturing reconstruction failed...");
  }
  GetTimer().PrintMinutes();
#else
  QMessageBox::critical(this, "",
                        tr("Dense texturing reconstruction requires pcl, which "
                           "is not available on your system."));
#endif
}

bool TexturingReconstruction::Texturing(const image_t image_id, std::size_t index) {
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = reconstruction_.Camera(image.CameraId());

  const std::string input_image_path = JoinPaths(image_path_, image.Name());
  const std::string texture_file = JoinPaths("images", image.Name());

  std::string target_file_path = JoinPaths(utility::filesystem::GetFileParentDirectory(
                                               options_.textured_file_path), texture_file);
  if (!ExistsFile(target_file_path) && ExistsFile(JoinPaths(output_path_, texture_file)))
  {
      FileCopy(JoinPaths(output_path_, texture_file), target_file_path);
  }

  auto& cameraParams = camera_trajectory_.parameters_[index];
  cameraParams.texture_file_ = texture_file;
  const Eigen::Matrix3x4d proj_matrix = image.InverseProjectionMatrix();
  Eigen::Matrix3d rotation = proj_matrix.leftCols<3>();
  Eigen::Vector3d translation = proj_matrix.rightCols<1>();
  ccGLMatrixd extrinsic = ccGLMatrixd::FromEigenMatrix3(rotation);
  extrinsic.setTranslation(translation.data());
  cameraParams.extrinsic_ = ccGLMatrixd::ToEigenMatrix4(extrinsic);
  std::string model_name = camera.ModelName();
  // https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
  if (model_name == "SIMPLE_PINHOLE" || model_name == "SIMPLE_RADIAL" ||
      model_name == "SIMPLE_RADIAL_FISHEYE" || model_name == "RADIAL" ||
      model_name == "RADIAL_FISHEYE")
  {
      // Simple pinhole: f, cx, cy
      cameraParams.intrinsic_.SetIntrinsics(static_cast<int>(camera.Width()),
                                            static_cast<int>(camera.Height()),
                                            camera.FocalLength(),
                                            camera.FocalLength(),
                                            camera.PrincipalPointX(),
                                            camera.PrincipalPointY());
  }
  else if (model_name == "PINHOLE" || model_name == "OPENCV" ||
           model_name == "OPENCV_FISHEYE" || model_name == "FULL_OPENCV" ||
           model_name == "FOV" || model_name == "THIN_PRISM_FISHEYE")
  {
      // Pinhole: fx, fy, cx, cy
      cameraParams.intrinsic_.SetIntrinsics(static_cast<int>(camera.Width()),
                                            static_cast<int>(camera.Height()),
                                            camera.FocalLengthX(),
                                            camera.FocalLengthY(),
                                            camera.PrincipalPointX(),
                                            camera.PrincipalPointY());
//      float fx = params[0];
//      float fy = params[1];
//      float dim_aspect = static_cast<float>(width) / height;
//      float pixel_aspect = fy / fx;
//      float img_aspect = dim_aspect * pixel_aspect;
//      if (img_aspect < 1.0f) {
//          camera_info.flen = fy / height;
//      } else {
//          camera_info.flen = fx / width;
//      }
//      camera_info.ppoint[0] = params[2] / width;
//      camera_info.ppoint[1] = params[3] / height;
  }
  else
  {
      std::string msg = "Unsupported camera model with texturing "
          "detected! If possible, re-run the SfM reconstruction with the "
          "SIMPLE_PINHOLE or the PINHOLE camera model (recommended). "
          "Otherwise, use the undistortion step in Colmap to obtain "
          "undistorted images and corresponding camera models without radial "
          "distortion.";
      CVLog::Error(msg.c_str());
      return false;
  }
  return true;
}

bool TexturingOptions::Check() const
{
    CHECK_OPTION_GT(save_precision, 0);
    return true;
}

}  // namespace colmap
