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

#include "exe/image.h"

#include "scene/reconstruction.h"
#include "image/undistortion.h"
#include "controllers/texturing_controller.h"
#include "controllers/incremental_pipeline.h"
#include "sfm/incremental_mapper.h"
#include "sfm/observation_manager.h"
#include "util/misc.h"
#include "controllers/option_manager.h"

namespace colmap {
namespace {

// Read stereo image pair names from a text file. The text file is expected to
// have one image pair per line, e.g.:
//
//      image_name1.jpg image_name2.jpg
//      image_name3.jpg image_name4.jpg
//      image_name5.jpg image_name6.jpg
//      ...
//
std::vector<std::pair<image_t, image_t>> ReadStereoImagePairs(
    const std::string& path, const Reconstruction& reconstruction) {
  const std::vector<std::string> stereo_pair_lines = ReadTextFileLines(path);

  std::vector<std::pair<image_t, image_t>> stereo_pairs;
  stereo_pairs.reserve(stereo_pair_lines.size());

  for (const auto& line : stereo_pair_lines) {
    const std::vector<std::string> names = StringSplit(line, " ");
    CHECK_EQ(names.size(), 2);

    const Image* image1 = reconstruction.FindImageWithName(names[0]);
    const Image* image2 = reconstruction.FindImageWithName(names[1]);

    CHECK_NOTNULL(image1);
    CHECK_NOTNULL(image2);

    stereo_pairs.emplace_back(image1->ImageId(), image2->ImageId());
  }

  return stereo_pairs;
}

}  // namespace

int RunImageDeleter(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_ids_path;
  std::string image_names_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "image_ids_path", &image_ids_path,
      "Path to text file containing one image_id to delete per line");
  options.AddDefaultOption(
      "image_names_path", &image_names_path,
      "Path to text file containing one image name to delete per line");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  if (!image_ids_path.empty()) {
    const auto image_ids = ReadTextFileLines(image_ids_path);

    for (const auto& image_id_str : image_ids) {
      if (image_id_str.empty()) {
        continue;
      }

      const image_t image_id = std::stoi(image_id_str);
      if (reconstruction.ExistsImage(image_id)) {
        const auto& image = reconstruction.Image(image_id);
        std::cout
            << StringPrintf(
                   "Deleting image_id=%d, image_name=%s from reconstruction",
                   image.ImageId(), image.Name().c_str())
            << std::endl;
        reconstruction.DeRegisterImage(image_id);
      } else {
        std::cout << StringPrintf(
                         "WARNING: Skipping image_id=%s, because it does not "
                         "exist in the reconstruction",
                         image_id_str.c_str())
                  << std::endl;
      }
    }
  }

  if (!image_names_path.empty()) {
    const auto image_names = ReadTextFileLines(image_names_path);

    for (const auto& image_name : image_names) {
      if (image_name.empty()) {
        continue;
      }

      const Image* image = reconstruction.FindImageWithName(image_name);
      if (image != nullptr) {
        std::cout
            << StringPrintf(
                   "Deleting image_id=%d, image_name=%s from reconstruction",
                   image->ImageId(), image->Name().c_str())
            << std::endl;
        reconstruction.DeRegisterImage(image->ImageId());
      } else {
        std::cout << StringPrintf(
                         "WARNING: Skipping image_name=%s, because it does not "
                         "exist in the reconstruction",
                         image_name.c_str())
                  << std::endl;
      }
    }
  }

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunImageFilterer(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  double max_extra_param = 100.0;
  int min_num_observations = 10;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_focal_length_ratio", &min_focal_length_ratio);
  options.AddDefaultOption("max_focal_length_ratio", &max_focal_length_ratio);
  options.AddDefaultOption("max_extra_param", &max_extra_param);
  options.AddDefaultOption("min_num_observations", &min_num_observations);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  const size_t num_reg_images = reconstruction.NumRegImages();

  // Registration belongs to frames. In particular, models read from disk can
  // have valid frame poses while the fork's legacy per-image flag is false.
  const std::vector<frame_t> filtered_frame_ids =
      ObservationManager(reconstruction)
          .FindFramesToFilter(min_focal_length_ratio,
                              max_focal_length_ratio,
                              max_extra_param,
                              min_num_observations);
  for (const frame_t frame_id : filtered_frame_ids) {
    reconstruction.DeRegisterFrame(frame_id);
  }

  const size_t num_filtered_images =
      num_reg_images - reconstruction.NumRegImages();

  std::cout << StringPrintf("Filtered %d images from a total of %d images",
                            num_filtered_images, num_reg_images)
            << std::endl;

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunImageRectifier(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string stereo_pairs_list;

  UndistortCameraOptions undistort_camera_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("stereo_pairs_list", &stereo_pairs_list);
  options.AddDefaultOption("blank_pixels",
                           &undistort_camera_options.blank_pixels);
  options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
  options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
  options.AddDefaultOption("max_image_size",
                           &undistort_camera_options.max_image_size);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  const auto stereo_pairs =
      ReadStereoImagePairs(stereo_pairs_list, reconstruction);

  StereoImageRectifierImpl rectifier(undistort_camera_options, &reconstruction,
                                  *options.image_path, output_path,
                                  stereo_pairs);
  rectifier.Start();
  rectifier.Wait();

  return EXIT_SUCCESS;
}

int RunImageRegistrator(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading1("Loading database");

  DatabaseCache database_cache;

  {
    auto database = Database::Open(*options.database_path);
    Timer timer;
    timer.Start();
    DatabaseCache::Options cache_options;
    cache_options.min_num_matches =
        static_cast<size_t>(options.mapper->min_num_matches);
    cache_options.ignore_watermarks = options.mapper->ignore_watermarks;
    cache_options.image_names = {options.mapper->image_names.begin(),
                                 options.mapper->image_names.end()};
    // An explicit registration list selects images to add, not the complete
    // existing model. The mapper also needs every already posed image because
    // BeginReconstruction reloads the model against this cache.
    if (!cache_options.image_names.empty()) {
      for (const image_t image_id : reconstruction.RegImageIds()) {
        cache_options.image_names.insert(
            reconstruction.Image(image_id).Name());
      }
    }
    // Registration may need an explicitly selected image with no remaining
    // correspondence edge to an already posed image. Keep its keypoints in
    // the cache, matching the upstream registrator's load contract.
    cache_options.load_all_images = true;
    database_cache.Load(*database, cache_options);
    std::cout << std::endl;
    timer.PrintMinutes();
  }

  std::cout << std::endl;

  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(&reconstruction);

  const auto mapper_options = options.mapper->Mapper();

  for (const auto& image : reconstruction.Images()) {
    // Registration is a frame-pose fact. The legacy image flag can be stale
    // after loading a fork model, so it must not schedule an already posed
    // image for registration again.
    if (image.second.HasPose()) {
      continue;
    }

    PrintHeading1("Registering image #" + std::to_string(image.first) + " (" +
                  std::to_string(reconstruction.NumRegImages() + 1) + ")");

    // Upstream parity (d3ccaf35 exe/image.cc): the visibility count lives in
    // the mapper's observation manager, not on the image entity.
    std::cout << "  => Image sees "
              << mapper.ObservationManager().NumVisiblePoints3D(image.first)
              << " / " << image.second.NumObservations() << " points"
              << std::endl;

    mapper.RegisterNextImage(mapper_options, image.first);
  }

  const bool kDiscardReconstruction = false;
  mapper.EndReconstruction(kDiscardReconstruction);

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunImageUndistorter(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string output_type = "COLMAP";
  std::string image_list_path;
  std::string copy_policy = "copy";
  int num_patch_match_src_images = 20;
  CopyType copy_type;

  UndistortCameraOptions undistort_camera_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("output_type", &output_type,
                           "{COLMAP, PMVS, CMP-MVS}");
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption("copy_policy", &copy_policy,
                           "{copy, soft-link, hard-link}");
  options.AddDefaultOption("num_patch_match_src_images",
                           &num_patch_match_src_images);
  options.AddDefaultOption("blank_pixels",
                           &undistort_camera_options.blank_pixels);
  options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
  options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
  options.AddDefaultOption("max_image_size",
                           &undistort_camera_options.max_image_size);
  options.AddDefaultOption("roi_min_x", &undistort_camera_options.roi_min_x);
  options.AddDefaultOption("roi_min_y", &undistort_camera_options.roi_min_y);
  options.AddDefaultOption("roi_max_x", &undistort_camera_options.roi_max_x);
  options.AddDefaultOption("roi_max_y", &undistort_camera_options.roi_max_y);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  CreateDirIfNotExists(output_path);

  PrintHeading1("Reading reconstruction");
  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  std::cout << StringPrintf(" => Reconstruction with %d images and %d points",
                            reconstruction.NumImages(),
                            reconstruction.NumPoints3D())
            << std::endl;

  std::vector<image_t> image_ids;
  if (!image_list_path.empty()) {
    const auto& image_names = ReadTextFileLines(image_list_path);
    for (const auto& image_name : image_names) {
      const Image* image = reconstruction.FindImageWithName(image_name);
      if (image != nullptr) {
        image_ids.push_back(image->ImageId());
      } else {
        std::cout << "WARN: Cannot find image " << image_name << std::endl;
      }
    }
  }

  StringToLower(&copy_policy);
  if (copy_policy == "copy") {
    copy_type = CopyType::COPY;
  } else if (copy_policy == "soft-link") {
    copy_type = CopyType::SOFT_LINK;
  } else if (copy_policy == "hard-link") {
    copy_type = CopyType::HARD_LINK;
  } else {
    std::cerr << "ERROR: Invalid `copy_policy` - supported values are "
                 "{'copy', 'soft-link', 'hard-link'}."
              << std::endl;
    return EXIT_FAILURE;
  }

  std::unique_ptr<Thread> undistorter;
  if (output_type == "COLMAP") {
    undistorter.reset(new COLMAPUndistorterImpl(
        undistort_camera_options, &reconstruction, *options.image_path,
        output_path, num_patch_match_src_images, copy_type, image_ids));
  } else if (output_type == "PMVS") {
    undistorter.reset(new PMVSUndistorterImpl(undistort_camera_options,
                                          &reconstruction, *options.image_path,
                                          output_path));
  } else if (output_type == "CMP-MVS") {
    undistorter.reset(new CMPMVSUndistorterImpl(undistort_camera_options,
                                            &reconstruction, *options.image_path,
                                            output_path));
  } else {
    std::cerr << "ERROR: Invalid `output_type` - supported values are "
                 "{'COLMAP', 'PMVS', 'CMP-MVS'}."
              << std::endl;
    return EXIT_FAILURE;
  }

  undistorter->Start();
  undistorter->Wait();

  return EXIT_SUCCESS;
}

int RunImageUndistorterStandalone(int argc, char** argv) {
  std::string input_file;
  std::string output_path;

  UndistortCameraOptions undistort_camera_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_file", &input_file);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("blank_pixels",
                           &undistort_camera_options.blank_pixels);
  options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
  options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
  options.AddDefaultOption("max_image_size",
                           &undistort_camera_options.max_image_size);
  options.AddDefaultOption("roi_min_x", &undistort_camera_options.roi_min_x);
  options.AddDefaultOption("roi_min_y", &undistort_camera_options.roi_min_y);
  options.AddDefaultOption("roi_max_x", &undistort_camera_options.roi_max_x);
  options.AddDefaultOption("roi_max_y", &undistort_camera_options.roi_max_y);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  CreateDirIfNotExists(output_path);

  // Loads a text file containing the image names and camera information.
  // The format of the text file is
  // image_name CAMERA_MODEL camera_params
  std::vector<std::pair<std::string, Camera>> image_names_and_cameras;

  {
    std::ifstream file(input_file);
    CHECK(file.is_open()) << input_file;

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
      StringTrim(&line);

      if (line.empty()) {
        continue;
      }

      std::string item;
      std::stringstream line_stream(line);

      // Loads the image name.
      std::string image_name;
      std::getline(line_stream, image_name, ' ');

      // Loads the camera and its parameters
      class Camera camera;

      std::getline(line_stream, item, ' ');
      if (!ExistsCameraModelWithName(item)) {
        std::cerr << "ERROR: Camera model " << item << " does not exist"
                  << std::endl;
        return EXIT_FAILURE;
      }
      camera.SetModelIdFromName(item);

      std::getline(line_stream, item, ' ');
      camera.SetWidth(std::stoll(item));

      std::getline(line_stream, item, ' ');
      camera.SetHeight(std::stoll(item));

      camera.Params().clear();
      while (!line_stream.eof()) {
        std::getline(line_stream, item, ' ');
        camera.Params().push_back(std::stold(item));
      }

      CHECK(camera.VerifyParams());

      image_names_and_cameras.emplace_back(image_name, camera);
    }
  }

  std::unique_ptr<Thread> undistorter;
  undistorter.reset(new PureImageUndistorterImpl(undistort_camera_options,
                                             *options.image_path, output_path,
                                             image_names_and_cameras));

  undistorter->Start();
  undistorter->Wait();

  return EXIT_SUCCESS;
}

int RunImageTexturer(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string mesh_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("mesh_path", &mesh_path);
  options.AddTexturingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsFile(mesh_path)) {
    std::cerr << "ERROR: `mesh_path` does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  PrintHeading1("Reading reconstruction");
  Reconstruction reconstruction;
  reconstruction.Read(JoinPaths(input_path, "sparse"));
  std::cout << StringPrintf(" => Reconstruction with %d images and %d points",
                            reconstruction.NumImages(),
                            reconstruction.NumPoints3D())
            << std::endl;

  options.texturing->meshed_file_path = mesh_path;
  std::string output_prefix;
  std::string output_extension;
  SplitFileExtension(output_path, &output_prefix, &output_extension);
  if (output_extension == ".obj" || output_extension == ".OBJ") {
    const auto output_dir = GetParentDir(output_path);
    if (!output_dir.empty()) {
      CreateDirIfNotExists(output_dir);
    }
    options.texturing->textured_file_path = output_path;
  } else {
    CreateDirIfNotExists(output_path);
    options.texturing->textured_file_path =
        JoinPaths(output_path, "textured_mesh.obj");
  }

  TexturingReconstruction texturer(*options.texturing, input_path);
  texturer.Start();
  texturer.Wait();

  if (!texturer.IsSuccess()) {
    std::cerr << "ERROR: mesh texturing failed" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
}  // namespace colmap
