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

#include "base/database.h"

#include <fstream>
#include <stdexcept>

#include "util/sqlite3_utils.h"
#include "util/string.h"
#include "util/version.h"

// W17.2 (fork parity, upstream dbb41680 scene/database.cc): only the
// database-layer logic (Merge, transactions, static state) remains here;
// the SQL implementation lives in scene/database_sqlite.cc.

namespace colmap {

const size_t Database::kMaxNumImages =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());

std::mutex Database::update_schema_mutex_;
void Database::Merge(const Database& database1, const Database& database2,
                     Database* merged_database) {
  // Merge the cameras.

  std::unordered_map<camera_t, camera_t> new_camera_ids1;
  for (const auto& camera : database1.ReadAllCameras()) {
    const camera_t new_camera_id = merged_database->WriteCamera(camera);
    new_camera_ids1.emplace(camera.CameraId(), new_camera_id);
  }

  std::unordered_map<camera_t, camera_t> new_camera_ids2;
  for (const auto& camera : database2.ReadAllCameras()) {
    const camera_t new_camera_id = merged_database->WriteCamera(camera);
    new_camera_ids2.emplace(camera.CameraId(), new_camera_id);
  }

  // Merge the images.

  std::unordered_map<image_t, image_t> new_image_ids1;
  for (auto& image : database1.ReadAllImages()) {
    image.SetCameraId(new_camera_ids1.at(image.CameraId()));
    CHECK(!merged_database->ExistsImageWithName(image.Name()))
        << "The two databases must not contain images with the same name, but "
           "the there are images with name "
        << image.Name() << " in both databases";
    const image_t new_image_id = merged_database->WriteImage(image);
    new_image_ids1.emplace(image.ImageId(), new_image_id);
    const auto keypoints = database1.ReadKeypoints(image.ImageId());
    const auto descriptors = database1.ReadDescriptors(image.ImageId());
    merged_database->WriteKeypoints(new_image_id, keypoints);
    if (database1.ExistsFloatDescriptors(image.ImageId())) {
      merged_database->WriteFloatDescriptors(
          new_image_id, database1.ReadFloatDescriptors(image.ImageId()),
          database1.ReadDescriptorType(image.ImageId()));
    } else {
      merged_database->WriteDescriptors(new_image_id, descriptors);
    }
  }

  std::unordered_map<image_t, image_t> new_image_ids2;
  for (auto& image : database2.ReadAllImages()) {
    image.SetCameraId(new_camera_ids2.at(image.CameraId()));
    CHECK(!merged_database->ExistsImageWithName(image.Name()))
        << "The two databases must not contain images with the same name, but "
           "the there are images with name "
        << image.Name() << " in both databases";
    const image_t new_image_id = merged_database->WriteImage(image);
    new_image_ids2.emplace(image.ImageId(), new_image_id);
    const auto keypoints = database2.ReadKeypoints(image.ImageId());
    const auto descriptors = database2.ReadDescriptors(image.ImageId());
    merged_database->WriteKeypoints(new_image_id, keypoints);
    if (database2.ExistsFloatDescriptors(image.ImageId())) {
      merged_database->WriteFloatDescriptors(
          new_image_id, database2.ReadFloatDescriptors(image.ImageId()),
          database2.ReadDescriptorType(image.ImageId()));
    } else {
      merged_database->WriteDescriptors(new_image_id, descriptors);
    }
  }

  // Merge the matches.

  for (const auto& matches : database1.ReadAllMatches()) {
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(matches.first, &image_id1, &image_id2);

    const image_t new_image_id1 = new_image_ids1.at(image_id1);
    const image_t new_image_id2 = new_image_ids1.at(image_id2);

    merged_database->WriteMatches(new_image_id1, new_image_id2, matches.second);
  }

  for (const auto& matches : database2.ReadAllMatches()) {
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(matches.first, &image_id1, &image_id2);

    const image_t new_image_id1 = new_image_ids2.at(image_id1);
    const image_t new_image_id2 = new_image_ids2.at(image_id2);

    merged_database->WriteMatches(new_image_id1, new_image_id2, matches.second);
  }

  // Merge the two-view geometries.

  {
    std::vector<image_pair_t> image_pair_ids;
    std::vector<TwoViewGeometry> two_view_geometries;
    database1.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

    for (size_t i = 0; i < two_view_geometries.size(); ++i) {
      image_t image_id1, image_id2;
      Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);

      const image_t new_image_id1 = new_image_ids1.at(image_id1);
      const image_t new_image_id2 = new_image_ids1.at(image_id2);

      merged_database->WriteTwoViewGeometry(new_image_id1, new_image_id2,
                                            two_view_geometries[i]);
    }
  }

  {
    std::vector<image_pair_t> image_pair_ids;
    std::vector<TwoViewGeometry> two_view_geometries;
    database2.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

    for (size_t i = 0; i < two_view_geometries.size(); ++i) {
      image_t image_id1, image_id2;
      Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);

      const image_t new_image_id1 = new_image_ids2.at(image_id1);
      const image_t new_image_id2 = new_image_ids2.at(image_id2);

      merged_database->WriteTwoViewGeometry(new_image_id1, new_image_id2,
                                            two_view_geometries[i]);
    }
  }
}
DatabaseTransaction::DatabaseTransaction(Database* database)
    : database_(database), database_lock_(database->transaction_mutex_) {
  CHECK_NOTNULL(database_);
  database_->BeginTransaction();
}

DatabaseTransaction::~DatabaseTransaction() { database_->EndTransaction(); }

}  // namespace colmap