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

namespace colmap {
namespace {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureKeypointsBlob;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptorsBlob;
typedef Eigen::Matrix<point2D_t, Eigen::Dynamic, 2, Eigen::RowMajor>
    FeatureMatchesBlob;

void SwapFeatureMatchesBlob(FeatureMatchesBlob* matches) {
  matches->col(0).swap(matches->col(1));
}

FeatureKeypointsBlob FeatureKeypointsToBlob(const FeatureKeypoints& keypoints) {
  const FeatureKeypointsBlob::Index kNumCols = 6;
  FeatureKeypointsBlob blob(keypoints.size(), kNumCols);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    blob(i, 0) = keypoints[i].x;
    blob(i, 1) = keypoints[i].y;
    blob(i, 2) = keypoints[i].a11;
    blob(i, 3) = keypoints[i].a12;
    blob(i, 4) = keypoints[i].a21;
    blob(i, 5) = keypoints[i].a22;
  }
  return blob;
}

FeatureKeypoints FeatureKeypointsFromBlob(const FeatureKeypointsBlob& blob) {
  FeatureKeypoints keypoints(static_cast<size_t>(blob.rows()));
  if (blob.cols() == 2) {
    for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
      keypoints[i] = FeatureKeypoint(blob(i, 0), blob(i, 1));
    }
  } else if (blob.cols() == 4) {
    for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
      keypoints[i] =
          FeatureKeypoint(blob(i, 0), blob(i, 1), blob(i, 2), blob(i, 3));
    }
  } else if (blob.cols() == 6) {
    for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
      keypoints[i] = FeatureKeypoint(blob(i, 0), blob(i, 1), blob(i, 2),
                                     blob(i, 3), blob(i, 4), blob(i, 5));
    }
  } else {
    LOG(FATAL) << "Keypoint format not supported";
  }
  return keypoints;
}

FeatureMatchesBlob FeatureMatchesToBlob(const FeatureMatches& matches) {
  const FeatureMatchesBlob::Index kNumCols = 2;
  FeatureMatchesBlob blob(matches.size(), kNumCols);
  for (size_t i = 0; i < matches.size(); ++i) {
    blob(i, 0) = matches[i].point2D_idx1;
    blob(i, 1) = matches[i].point2D_idx2;
  }
  return blob;
}

FeatureMatches FeatureMatchesFromBlob(const FeatureMatchesBlob& blob) {
  CHECK_EQ(blob.cols(), 2);
  FeatureMatches matches(static_cast<size_t>(blob.rows()));
  for (FeatureMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
    matches[i].point2D_idx1 = blob(i, 0);
    matches[i].point2D_idx2 = blob(i, 1);
  }
  return matches;
}

template <typename MatrixType>
MatrixType ReadStaticMatrixBlob(sqlite3_stmt* sql_stmt, const int rc,
                                const int col) {
  CHECK_GE(col, 0);

  MatrixType matrix;

  if (rc == SQLITE_ROW) {
    const size_t num_bytes =
        static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col));
    if (num_bytes > 0) {
      CHECK_EQ(num_bytes, matrix.size() * sizeof(typename MatrixType::Scalar));
      memcpy(reinterpret_cast<char*>(matrix.data()),
             sqlite3_column_blob(sql_stmt, col), num_bytes);
    } else {
      matrix = MatrixType::Zero();
    }
  } else {
    matrix = MatrixType::Zero();
  }

  return matrix;
}

template <typename MatrixType>
MatrixType ReadDynamicMatrixBlob(sqlite3_stmt* sql_stmt, const int rc,
                                 const int col) {
  CHECK_GE(col, 0);

  MatrixType matrix;

  if (rc == SQLITE_ROW) {
    const size_t rows =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 0));
    const size_t cols =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 1));

    CHECK_GE(rows, 0);
    CHECK_GE(cols, 0);
    matrix = MatrixType(rows, cols);

    const size_t num_bytes =
        static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col + 2));
    CHECK_EQ(matrix.size() * sizeof(typename MatrixType::Scalar), num_bytes);

    memcpy(reinterpret_cast<char*>(matrix.data()),
           sqlite3_column_blob(sql_stmt, col + 2), num_bytes);
  } else {
    const typename MatrixType::Index rows =
        (MatrixType::RowsAtCompileTime == Eigen::Dynamic)
            ? 0
            : MatrixType::RowsAtCompileTime;
    const typename MatrixType::Index cols =
        (MatrixType::ColsAtCompileTime == Eigen::Dynamic)
            ? 0
            : MatrixType::ColsAtCompileTime;
    matrix = MatrixType(rows, cols);
  }

  return matrix;
}

void BindPoseBlob(sqlite3_stmt* sql_stmt,
                 const int qvec_column,
                 const int tvec_column,
                 const Eigen::Vector4d& qvec,
                 const Eigen::Vector3d& tvec) {
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt, qvec_column, qvec.data(),
                                 sizeof(double) * qvec.size(), SQLITE_STATIC));
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt, tvec_column, tvec.data(),
                                 sizeof(double) * tvec.size(), SQLITE_STATIC));
}

bool ReadPoseBlob(sqlite3_stmt* sql_stmt,
                  const int qvec_column,
                  const int tvec_column,
                  Eigen::Vector4d* qvec,
                  Eigen::Vector3d* tvec) {
  if (sqlite3_column_bytes(sql_stmt, qvec_column) !=
          static_cast<int>(sizeof(double) * qvec->size()) ||
      sqlite3_column_bytes(sql_stmt, tvec_column) !=
          static_cast<int>(sizeof(double) * tvec->size())) {
    return false;
  }
  memcpy(qvec->data(), sqlite3_column_blob(sql_stmt, qvec_column),
         sizeof(double) * qvec->size());
  memcpy(tvec->data(), sqlite3_column_blob(sql_stmt, tvec_column),
         sizeof(double) * tvec->size());
  return qvec->allFinite() && tvec->allFinite() && qvec->squaredNorm() > 0.0;
}

template <typename MatrixType>
void WriteStaticMatrixBlob(sqlite3_stmt* sql_stmt, const MatrixType& matrix,
                           const int col) {
  SQLITE3_CALL(sqlite3_bind_blob(
      sql_stmt, col, reinterpret_cast<const char*>(matrix.data()),
      static_cast<int>(matrix.size() * sizeof(typename MatrixType::Scalar)),
      SQLITE_STATIC));
}

template <typename MatrixType>
void WriteDynamicMatrixBlob(sqlite3_stmt* sql_stmt, const MatrixType& matrix,
                            const int col) {
  CHECK_GE(matrix.rows(), 0);
  CHECK_GE(matrix.cols(), 0);
  CHECK_GE(col, 0);

  const size_t num_bytes = matrix.size() * sizeof(typename MatrixType::Scalar);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 0, matrix.rows()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 1, matrix.cols()));
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt, col + 2,
                                 reinterpret_cast<const char*>(matrix.data()),
                                 static_cast<int>(num_bytes), SQLITE_STATIC));
}

Camera ReadCameraRow(sqlite3_stmt* sql_stmt) {
  Camera camera;

  camera.SetCameraId(static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 0)));
  camera.SetModelId(sqlite3_column_int64(sql_stmt, 1));
  camera.SetWidth(static_cast<size_t>(sqlite3_column_int64(sql_stmt, 2)));
  camera.SetHeight(static_cast<size_t>(sqlite3_column_int64(sql_stmt, 3)));

  const size_t num_params_bytes =
      static_cast<size_t>(sqlite3_column_bytes(sql_stmt, 4));
  const size_t num_params = num_params_bytes / sizeof(double);
  CHECK_EQ(num_params, camera.NumParams());
  memcpy(camera.ParamsData(), sqlite3_column_blob(sql_stmt, 4),
         num_params_bytes);

  camera.SetPriorFocalLength(sqlite3_column_int64(sql_stmt, 5) != 0);

  return camera;
}

Image ReadImageRow(sqlite3_stmt* sql_stmt) {
  Image image;

  image.SetImageId(static_cast<image_t>(sqlite3_column_int64(sql_stmt, 0)));
  image.SetName(std::string(
      reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1))));
  image.SetCameraId(static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 2)));

  // NaNs are automatically converted to NULLs in SQLite.
  for (size_t i = 0; i < 4; ++i) {
    if (sqlite3_column_type(sql_stmt, i + 3) != SQLITE_NULL) {
      image.QvecPrior(i) = sqlite3_column_double(sql_stmt, i + 3);
    }
  }

  // NaNs are automatically converted to NULLs in SQLite.
  for (size_t i = 0; i < 3; ++i) {
    if (sqlite3_column_type(sql_stmt, i + 7) != SQLITE_NULL) {
      image.TvecPrior(i) = sqlite3_column_double(sql_stmt, i + 7);
    }
  }

  return image;
}

}  // namespace

const size_t Database::kMaxNumImages =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());

std::mutex Database::update_schema_mutex_;

Database::Database() : database_(nullptr) {}

Database::Database(const std::string& path) : Database() { Open(path); }

Database::~Database() { Close(); }

void Database::Open(const std::string& path) {
  Close();

  // SQLITE_OPEN_NOMUTEX specifies that the connection should not have a
  // mutex (so that we don't serialize the connection's operations).
  // Modifications to the database will still be serialized, but multiple
  // connections can read concurrently.
  SQLITE3_CALL(sqlite3_open_v2(
      path.c_str(), &database_,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
      nullptr));

  // Don't wait for the operating system to write the changes to disk
  SQLITE3_EXEC(database_, "PRAGMA synchronous=OFF", nullptr);

  // Use faster journaling mode
  SQLITE3_EXEC(database_, "PRAGMA journal_mode=WAL", nullptr);

  // Store temporary tables and indices in memory
  SQLITE3_EXEC(database_, "PRAGMA temp_store=MEMORY", nullptr);

  // Disabled by default
  SQLITE3_EXEC(database_, "PRAGMA foreign_keys=ON", nullptr);

  // Enable auto vacuum to reduce DB file size
  SQLITE3_EXEC(database_, "PRAGMA auto_vacuum=1", nullptr);

  PreMigrateTables();
  CreateTables();
  PostMigrateTables();
  PrepareSQLStatements();
}

void Database::Close() {
  if (database_ != nullptr) {
    FinalizeSQLStatements();
    SQLITE3_EXEC(database_, "VACUUM", nullptr);
    sqlite3_close_v2(database_);
    database_ = nullptr;
  }
}

bool Database::ExistsCamera(const camera_t camera_id) const {
  return ExistsRowId(sql_stmt_exists_camera_, camera_id);
}

bool Database::ExistsRig(const rig_t rig_id) const {
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT 1 FROM rigs WHERE rig_id=?;",
                                  -1, &stmt, nullptr));
  const bool exists = ExistsRowId(stmt, rig_id);
  SQLITE3_CALL(sqlite3_finalize(stmt));
  return exists;
}

bool Database::ExistsFrame(const frame_t frame_id) const {
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT 1 FROM frames WHERE frame_id=?;",
                                  -1, &stmt, nullptr));
  const bool exists = ExistsRowId(stmt, frame_id);
  SQLITE3_CALL(sqlite3_finalize(stmt));
  return exists;
}

bool Database::ExistsImage(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_image_id_, image_id);
}

bool Database::ExistsImageWithName(std::string name) const {
  return ExistsRowString(sql_stmt_exists_image_name_, name);
}

bool Database::ExistsKeypoints(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_keypoints_, image_id);
}

bool Database::ExistsDescriptors(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_descriptors_, image_id);
}

bool Database::ExistsFloatDescriptors(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_float_descriptors_, image_id);
}

bool Database::ExistsMatches(const image_t image_id1,
                             const image_t image_id2) const {
  return ExistsRowId(sql_stmt_exists_matches_,
                     ImagePairToPairId(image_id1, image_id2));
}

bool Database::ExistsInlierMatches(const image_t image_id1,
                                   const image_t image_id2) const {
  return ExistsRowId(sql_stmt_exists_two_view_geometry_,
                     ImagePairToPairId(image_id1, image_id2));
}

bool Database::ExistsTwoViewGeometry(const image_t image_id1,
                                     const image_t image_id2) const {
  return ExistsRowId(sql_stmt_exists_two_view_geometry_,
                     ImagePairToPairId(image_id1, image_id2));
}

size_t Database::NumCameras() const { return CountRows("cameras"); }
size_t Database::NumRigs() const { return CountRows("rigs"); }
size_t Database::NumFrames() const { return CountRows("frames"); }

size_t Database::NumImages() const { return CountRows("images"); }

size_t Database::NumKeypoints() const { return SumColumn("rows", "keypoints"); }

size_t Database::MaxNumKeypoints() const {
  return MaxColumn("rows", "keypoints");
}

size_t Database::NumKeypointsForImage(const image_t image_id) const {
  return CountRowsForEntry(sql_stmt_num_keypoints_, image_id);
}

size_t Database::NumDescriptors() const {
  return SumColumn("rows", "descriptors");
}

size_t Database::MaxNumDescriptors() const {
  return MaxColumn("rows", "descriptors");
}

size_t Database::NumDescriptorsForImage(const image_t image_id) const {
  return CountRowsForEntry(sql_stmt_num_descriptors_, image_id);
}

size_t Database::NumMatches() const { return SumColumn("rows", "matches"); }

size_t Database::NumInlierMatches() const {
  return SumColumn("rows", "two_view_geometries");
}

size_t Database::NumMatchedImagePairs() const { return CountRows("matches"); }

size_t Database::NumVerifiedImagePairs() const {
  return CountRows("two_view_geometries");
}

Camera Database::ReadCamera(const camera_t camera_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_camera_, 1, camera_id));

  Camera camera;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_camera_));
  if (rc == SQLITE_ROW) {
    camera = ReadCameraRow(sql_stmt_read_camera_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_camera_));

  return camera;
}

std::vector<Camera> Database::ReadAllCameras() const {
  std::vector<Camera> cameras;

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_cameras_)) == SQLITE_ROW) {
    cameras.push_back(ReadCameraRow(sql_stmt_read_cameras_));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_cameras_));

  return cameras;
}

Rig Database::ReadRig(const rig_t rig_id) const {
  sqlite3_stmt* ref_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(
      database_, "SELECT ref_sensor_id, ref_sensor_type, ref_camera_id FROM rigs "
      "WHERE rig_id=?;", -1, &ref_stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(ref_stmt, 1, rig_id));
  Rig rig;
  if (SQLITE3_CALL(sqlite3_step(ref_stmt)) != SQLITE_ROW) {
    SQLITE3_CALL(sqlite3_finalize(ref_stmt));
    return rig;
  }
  const bool has_generic_ref = sqlite3_column_type(ref_stmt, 0) != SQLITE_NULL &&
                               sqlite3_column_type(ref_stmt, 1) != SQLITE_NULL;
  const sensor_t ref_sensor_id(
      has_generic_ref ? static_cast<SensorType>(sqlite3_column_int(ref_stmt, 1))
                      : SensorType::CAMERA,
      has_generic_ref ? static_cast<uint32_t>(sqlite3_column_int64(ref_stmt, 0))
                      : static_cast<uint32_t>(sqlite3_column_int64(ref_stmt, 2)));
  SQLITE3_CALL(sqlite3_finalize(ref_stmt));
  rig.SetRigId(rig_id);
  rig.AddRefSensor(ref_sensor_id);
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(
      database_, "SELECT camera_id, qvec, tvec FROM rig_cameras WHERE rig_id=? "
      "ORDER BY camera_id;", -1, &stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, rig_id));
  struct CameraPose { camera_t camera_id; Eigen::Vector4d qvec; Eigen::Vector3d tvec; };
  std::vector<CameraPose> cameras;
  while (SQLITE3_CALL(sqlite3_step(stmt)) == SQLITE_ROW) {
    const camera_t camera_id = static_cast<camera_t>(sqlite3_column_int64(stmt, 0));
    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    CHECK(ReadPoseBlob(stmt, 1, 2, &qvec, &tvec));
    cameras.push_back({camera_id, qvec, tvec});
  }
  SQLITE3_CALL(sqlite3_finalize(stmt));
  for (const CameraPose& camera : cameras) {
    if (!rig.HasCamera(camera.camera_id)) {
      rig.AddCamera(camera.camera_id, camera.qvec, camera.tvec);
    }
  }
  sqlite3_stmt* sensor_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(
      database_, "SELECT sensor_id, sensor_type, qvec, tvec FROM rig_sensors "
      "WHERE rig_id=? ORDER BY sensor_type, sensor_id;", -1, &sensor_stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(sensor_stmt, 1, rig_id));
  while (SQLITE3_CALL(sqlite3_step(sensor_stmt)) == SQLITE_ROW) {
    const sensor_t sensor_id(
        static_cast<SensorType>(sqlite3_column_int(sensor_stmt, 1)),
        static_cast<uint32_t>(sqlite3_column_int64(sensor_stmt, 0)));
    if (rig.HasSensor(sensor_id)) continue;
    if (sqlite3_column_type(sensor_stmt, 2) == SQLITE_NULL ||
        sqlite3_column_type(sensor_stmt, 3) == SQLITE_NULL) {
      rig.AddSensor(sensor_id, std::nullopt, std::nullopt);
    } else {
      Eigen::Vector4d qvec;
      Eigen::Vector3d tvec;
      CHECK(ReadPoseBlob(sensor_stmt, 2, 3, &qvec, &tvec));
      rig.AddSensor(sensor_id, qvec, tvec);
    }
  }
  SQLITE3_CALL(sqlite3_finalize(sensor_stmt));
  return rig;
}

std::vector<Rig> Database::ReadAllRigs() const {
  std::vector<Rig> rigs;
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT rig_id FROM rigs ORDER BY rig_id;",
                                  -1, &stmt, nullptr));
  while (SQLITE3_CALL(sqlite3_step(stmt)) == SQLITE_ROW) {
    rigs.push_back(ReadRig(static_cast<rig_t>(sqlite3_column_int64(stmt, 0))));
  }
  SQLITE3_CALL(sqlite3_finalize(stmt));
  return rigs;
}

Frame Database::ReadFrame(const frame_t frame_id) const {
  sqlite3_stmt* frame_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT rig_id, has_pose, qvec, tvec "
                                  "FROM frames WHERE frame_id=?;", -1, &frame_stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(frame_stmt, 1, frame_id));
  Frame frame;
  if (SQLITE3_CALL(sqlite3_step(frame_stmt)) == SQLITE_ROW) {
    frame.SetFrameId(frame_id);
    frame.SetRigId(static_cast<rig_t>(sqlite3_column_int64(frame_stmt, 0)));
    if (sqlite3_column_int(frame_stmt, 1) != 0) {
      Eigen::Vector4d qvec;
      Eigen::Vector3d tvec;
      CHECK(ReadPoseBlob(frame_stmt, 2, 3, &qvec, &tvec));
      frame.SetRigFromWorld(qvec, tvec);
    }
  }
  SQLITE3_CALL(sqlite3_finalize(frame_stmt));
  if (frame.FrameId() == kInvalidFrameId) return frame;
  sqlite3_stmt* data_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(
      database_, "SELECT data_id, sensor_id, sensor_type FROM frame_data "
      "WHERE frame_id=? ORDER BY sensor_type, sensor_id, data_id;", -1, &data_stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(data_stmt, 1, frame_id));
  while (SQLITE3_CALL(sqlite3_step(data_stmt)) == SQLITE_ROW) {
    frame.AddDataId(data_t(
        sensor_t(static_cast<SensorType>(sqlite3_column_int(data_stmt, 2)),
                 static_cast<uint32_t>(sqlite3_column_int64(data_stmt, 1))),
        static_cast<uint64_t>(sqlite3_column_int64(data_stmt, 0))));
  }
  SQLITE3_CALL(sqlite3_finalize(data_stmt));
  if (frame.DataIds().empty()) {
    // Fork-legacy fallback: rows written before the W3-2a dual-write have no
    // frame_data entries, so derive them from frame_images. AddImageId's
    // image_id == camera_id assumption would otherwise duplicate the data
    // ids that the frame_data read above already restored.
    sqlite3_stmt* image_stmt = nullptr;
    SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT image_id FROM frame_images "
                                    "WHERE frame_id=? ORDER BY image_id;", -1, &image_stmt, nullptr));
    SQLITE3_CALL(sqlite3_bind_int64(image_stmt, 1, frame_id));
    while (SQLITE3_CALL(sqlite3_step(image_stmt)) == SQLITE_ROW) {
      frame.AddImageId(static_cast<image_t>(sqlite3_column_int64(image_stmt, 0)));
    }
    SQLITE3_CALL(sqlite3_finalize(image_stmt));
  }
  return frame;
}

std::vector<Frame> Database::ReadAllFrames() const {
  std::vector<Frame> frames;
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT frame_id FROM frames ORDER BY frame_id;",
                                  -1, &stmt, nullptr));
  while (SQLITE3_CALL(sqlite3_step(stmt)) == SQLITE_ROW) {
    frames.push_back(ReadFrame(static_cast<frame_t>(sqlite3_column_int64(stmt, 0))));
  }
  SQLITE3_CALL(sqlite3_finalize(stmt));
  return frames;
}

Image Database::ReadImage(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_image_id_, 1, image_id));

  Image image;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_id_));
  if (rc == SQLITE_ROW) {
    image = ReadImageRow(sql_stmt_read_image_id_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_id_));

  return image;
}

Image Database::ReadImageWithName(const std::string& name) const {
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt_read_image_name_, 1, name.c_str(),
                                 static_cast<int>(name.size()), SQLITE_STATIC));

  Image image;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_name_));
  if (rc == SQLITE_ROW) {
    image = ReadImageRow(sql_stmt_read_image_name_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_name_));

  return image;
}

std::vector<Image> Database::ReadAllImages() const {
  std::vector<Image> images;
  images.reserve(NumImages());

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_images_)) == SQLITE_ROW) {
    images.push_back(ReadImageRow(sql_stmt_read_images_));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_images_));

  return images;
}

FeatureKeypoints Database::ReadKeypoints(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_));
  const FeatureKeypointsBlob blob = ReadDynamicMatrixBlob<FeatureKeypointsBlob>(
      sql_stmt_read_keypoints_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_));

  return FeatureKeypointsFromBlob(blob);
}

FeatureDescriptors Database::ReadDescriptors(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_descriptors_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_descriptors_));
  const FeatureDescriptors descriptors =
      ReadDynamicMatrixBlob<FeatureDescriptors>(sql_stmt_read_descriptors_, rc,
                                                0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_descriptors_));

  return descriptors;
}

FeatureDescriptorsFloat Database::ReadFloatDescriptors(
    const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_float_descriptors_, 1,
                                  image_id));
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_float_descriptors_));
  const FeatureDescriptorsFloat descriptors =
      ReadDynamicMatrixBlob<FeatureDescriptorsFloat>(
          sql_stmt_read_float_descriptors_, rc, 0);
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_float_descriptors_));
  return descriptors;
}

FeatureDescriptorType Database::ReadDescriptorType(
    const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_descriptor_type_, 1,
                                  image_id));
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_descriptor_type_));
  const auto type = rc == SQLITE_ROW
                        ? sqlite3_column_int(sql_stmt_read_descriptor_type_, 0)
                        : static_cast<int>(FeatureDescriptorType::kSift);
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_descriptor_type_));
  return static_cast<FeatureDescriptorType>(type);
}

FeatureMatches Database::ReadMatches(image_t image_id1,
                                     image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_matches_, 1, pair_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_));
  FeatureMatchesBlob blob =
      ReadDynamicMatrixBlob<FeatureMatchesBlob>(sql_stmt_read_matches_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_));

  if (SwapImagePair(image_id1, image_id2)) {
    SwapFeatureMatchesBlob(&blob);
  }

  return FeatureMatchesFromBlob(blob);
}

std::vector<std::pair<image_pair_t, FeatureMatches>> Database::ReadAllMatches()
    const {
  std::vector<std::pair<image_pair_t, FeatureMatches>> all_matches;

  int rc;
  while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_all_))) ==
         SQLITE_ROW) {
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_matches_all_, 0));
    const FeatureMatchesBlob blob = ReadDynamicMatrixBlob<FeatureMatchesBlob>(
        sql_stmt_read_matches_all_, rc, 1);
    all_matches.emplace_back(pair_id, FeatureMatchesFromBlob(blob));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_all_));

  return all_matches;
}

TwoViewGeometry Database::ReadTwoViewGeometry(const image_t image_id1,
                                              const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_read_two_view_geometry_, 1, pair_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_two_view_geometry_));

  TwoViewGeometry two_view_geometry;

  FeatureMatchesBlob blob = ReadDynamicMatrixBlob<FeatureMatchesBlob>(
      sql_stmt_read_two_view_geometry_, rc, 0);

  two_view_geometry.config = static_cast<int>(
      sqlite3_column_int64(sql_stmt_read_two_view_geometry_, 3));

  // Bridge the fork's legacy two-view blob layout (plain F/E/H + qvec/tvec)
  // to the upstream struct with optional members.
  const Eigen::Matrix3d F_legacy = ReadStaticMatrixBlob<Eigen::Matrix3d>(
      sql_stmt_read_two_view_geometry_, rc, 4);
  const Eigen::Matrix3d E_legacy = ReadStaticMatrixBlob<Eigen::Matrix3d>(
      sql_stmt_read_two_view_geometry_, rc, 5);
  const Eigen::Matrix3d H_legacy = ReadStaticMatrixBlob<Eigen::Matrix3d>(
      sql_stmt_read_two_view_geometry_, rc, 6);
  const Eigen::Vector4d qvec_legacy = ReadStaticMatrixBlob<Eigen::Vector4d>(
      sql_stmt_read_two_view_geometry_, rc, 7);
  const Eigen::Vector3d tvec_legacy = ReadStaticMatrixBlob<Eigen::Vector3d>(
      sql_stmt_read_two_view_geometry_, rc, 8);
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond(qvec_legacy(0), qvec_legacy(1),
                                 qvec_legacy(2), qvec_legacy(3)),
              tvec_legacy);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_two_view_geometry_));

  two_view_geometry.inlier_matches = FeatureMatchesFromBlob(blob);
  // The write path stores the transposed matrix so that its column-major
  // memory equals the row-major original; undo that transpose here.
  two_view_geometry.F = F_legacy.transpose();
  two_view_geometry.E = E_legacy.transpose();
  two_view_geometry.H = H_legacy.transpose();

  if (SwapImagePair(image_id1, image_id2)) {
    two_view_geometry.Invert();
  }

  return two_view_geometry;
}

void Database::ReadTwoViewGeometries(
    std::vector<image_pair_t>* image_pair_ids,
    std::vector<TwoViewGeometry>* two_view_geometries) const {
  int rc;
  while ((rc = SQLITE3_CALL(sqlite3_step(
              sql_stmt_read_two_view_geometries_))) == SQLITE_ROW) {
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_two_view_geometries_, 0));
    image_pair_ids->push_back(pair_id);

    TwoViewGeometry two_view_geometry;

    const FeatureMatchesBlob blob = ReadDynamicMatrixBlob<FeatureMatchesBlob>(
        sql_stmt_read_two_view_geometries_, rc, 1);
    two_view_geometry.inlier_matches = FeatureMatchesFromBlob(blob);

    two_view_geometry.config = static_cast<int>(
        sqlite3_column_int64(sql_stmt_read_two_view_geometries_, 4));

    const Eigen::Matrix3d F_legacy = ReadStaticMatrixBlob<Eigen::Matrix3d>(
        sql_stmt_read_two_view_geometries_, rc, 5);
    const Eigen::Matrix3d E_legacy = ReadStaticMatrixBlob<Eigen::Matrix3d>(
        sql_stmt_read_two_view_geometries_, rc, 6);
    const Eigen::Matrix3d H_legacy = ReadStaticMatrixBlob<Eigen::Matrix3d>(
        sql_stmt_read_two_view_geometries_, rc, 7);
    const Eigen::Vector4d qvec_legacy = ReadStaticMatrixBlob<Eigen::Vector4d>(
        sql_stmt_read_two_view_geometries_, rc, 8);
    const Eigen::Vector3d tvec_legacy = ReadStaticMatrixBlob<Eigen::Vector3d>(
        sql_stmt_read_two_view_geometries_, rc, 9);
    two_view_geometry.F = F_legacy.transpose();
    two_view_geometry.E = E_legacy.transpose();
    two_view_geometry.H = H_legacy.transpose();
    two_view_geometry.cam2_from_cam1 =
        Rigid3d(Eigen::Quaterniond(qvec_legacy(0), qvec_legacy(1),
                                   qvec_legacy(2), qvec_legacy(3)),
                tvec_legacy);

    two_view_geometries->push_back(two_view_geometry);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_two_view_geometries_));
}

std::map<image_pair_t, TwoViewGeometry> Database::ReadTwoViewGeometries()
        const {
  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);
  std::map<image_pair_t, TwoViewGeometry> result;
  for (size_t i = 0; i < image_pair_ids.size(); ++i) {
    result.emplace(image_pair_ids[i], two_view_geometries[i]);
  }
  return result;
}

void Database::ReadTwoViewGeometryNumInliers(
    std::vector<std::pair<image_t, image_t>>* image_pairs,
    std::vector<int>* num_inliers) const {
  const auto num_inlier_matches = NumInlierMatches();
  image_pairs->reserve(num_inlier_matches);
  num_inliers->reserve(num_inlier_matches);

  while (SQLITE3_CALL(sqlite3_step(
             sql_stmt_read_two_view_geometry_num_inliers_)) == SQLITE_ROW) {
    image_t image_id1;
    image_t image_id2;
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_two_view_geometry_num_inliers_, 0));
    PairIdToImagePair(pair_id, &image_id1, &image_id2);
    image_pairs->emplace_back(image_id1, image_id2);

    const int rows = static_cast<int>(
        sqlite3_column_int64(sql_stmt_read_two_view_geometry_num_inliers_, 1));
    num_inliers->push_back(rows);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_two_view_geometry_num_inliers_));
}

camera_t Database::WriteCamera(const Camera& camera,
                               const bool use_camera_id) const {
  if (use_camera_id) {
    CHECK(!ExistsCamera(camera.CameraId())) << "camera_id must be unique";
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_add_camera_, 1, camera.CameraId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_camera_, 1));
  }

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 2, camera.ModelId()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 3,
                                  static_cast<sqlite3_int64>(camera.Width())));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 4,
                                  static_cast<sqlite3_int64>(camera.Height())));

  const size_t num_params_bytes = sizeof(double) * camera.NumParams();
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt_add_camera_, 5, camera.ParamsData(),
                                 static_cast<int>(num_params_bytes),
                                 SQLITE_STATIC));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 6,
                                  camera.HasPriorFocalLength()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_camera_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_camera_));

  return static_cast<camera_t>(sqlite3_last_insert_rowid(database_));
}

rig_t Database::WriteRig(const Rig& rig, const bool use_rig_id) const {
  CHECK_GT(rig.NumSensors(), 0);
  const std::vector<camera_t> legacy_camera_ids = rig.CameraIds();
  const camera_t legacy_ref_camera_id =
      rig.RefCameraId() != kInvalidCameraId
          ? rig.RefCameraId()
          : (legacy_camera_ids.empty() ? rig.RefSensorId().id
                                       : legacy_camera_ids.front());
  sqlite3_stmt* rig_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO rigs(rig_id, ref_camera_id, ref_sensor_id, ref_sensor_type) "
                                  "VALUES(?, ?, ?, ?);", -1, &rig_stmt, nullptr));
  if (use_rig_id) {
    CHECK(!ExistsRig(rig.RigId()));
    SQLITE3_CALL(sqlite3_bind_int64(rig_stmt, 1, rig.RigId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(rig_stmt, 1));
  }
  SQLITE3_CALL(sqlite3_bind_int64(rig_stmt, 2, legacy_ref_camera_id));
  SQLITE3_CALL(sqlite3_bind_int64(rig_stmt, 3, rig.RefSensorId().id));
  SQLITE3_CALL(sqlite3_bind_int(rig_stmt, 4, static_cast<int>(rig.RefSensorId().type)));
  SQLITE3_CALL(sqlite3_step(rig_stmt));
  SQLITE3_CALL(sqlite3_finalize(rig_stmt));
  const rig_t rig_id = static_cast<rig_t>(sqlite3_last_insert_rowid(database_));
  sqlite3_stmt* camera_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO rig_cameras(rig_id, camera_id, qvec, tvec) "
                                  "VALUES(?, ?, ?, ?);", -1, &camera_stmt, nullptr));
  for (const camera_t camera_id : rig.CameraIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(camera_stmt, 1, rig_id));
    SQLITE3_CALL(sqlite3_bind_int64(camera_stmt, 2, camera_id));
    BindPoseBlob(camera_stmt, 3, 4, rig.CamFromRigQvec(camera_id),
                 rig.CamFromRigTvec(camera_id));
    SQLITE3_CALL(sqlite3_step(camera_stmt));
    SQLITE3_CALL(sqlite3_reset(camera_stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(camera_stmt));
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type, qvec, tvec) VALUES(?, ?, ?, ?, ?);", -1, &camera_stmt, nullptr));
  for (const sensor_t& sensor_id : rig.SensorIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(camera_stmt, 1, rig_id));
    SQLITE3_CALL(sqlite3_bind_int64(camera_stmt, 2, sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int(camera_stmt, 3, static_cast<int>(sensor_id.type)));
    if (rig.HasSensorFromRig(sensor_id)) {
      BindPoseBlob(camera_stmt, 4, 5, rig.SensorFromRigQvec(sensor_id), rig.SensorFromRigTvec(sensor_id));
    } else {
      SQLITE3_CALL(sqlite3_bind_null(camera_stmt, 4));
      SQLITE3_CALL(sqlite3_bind_null(camera_stmt, 5));
    }
    SQLITE3_CALL(sqlite3_step(camera_stmt));
    SQLITE3_CALL(sqlite3_reset(camera_stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(camera_stmt));
  return rig_id;
}

frame_t Database::WriteFrame(const Frame& frame, const bool use_frame_id) const {
  CHECK_GT(frame.DataIds().size(), 0);
  CHECK(ExistsRig(frame.RigId()));
  sqlite3_stmt* frame_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO frames(frame_id, rig_id, has_pose, qvec, tvec) "
                                  "VALUES(?, ?, ?, ?, ?);", -1, &frame_stmt, nullptr));
  if (use_frame_id) {
    CHECK(!ExistsFrame(frame.FrameId()));
    SQLITE3_CALL(sqlite3_bind_int64(frame_stmt, 1, frame.FrameId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(frame_stmt, 1));
  }
  SQLITE3_CALL(sqlite3_bind_int64(frame_stmt, 2, frame.RigId()));
  SQLITE3_CALL(sqlite3_bind_int(frame_stmt, 3, frame.HasPose() ? 1 : 0));
  if (frame.HasPose()) BindPoseBlob(frame_stmt, 4, 5, frame.RigFromWorldQvec(), frame.RigFromWorldTvec());
  else {
    SQLITE3_CALL(sqlite3_bind_null(frame_stmt, 4));
    SQLITE3_CALL(sqlite3_bind_null(frame_stmt, 5));
  }
  SQLITE3_CALL(sqlite3_step(frame_stmt));
  SQLITE3_CALL(sqlite3_finalize(frame_stmt));
  const frame_t frame_id = static_cast<frame_t>(sqlite3_last_insert_rowid(database_));
  sqlite3_stmt* image_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO frame_images(frame_id, image_id) VALUES(?, ?);",
                                  -1, &image_stmt, nullptr));
  for (const image_t image_id : frame.ImageIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(image_stmt, 1, frame_id));
    SQLITE3_CALL(sqlite3_bind_int64(image_stmt, 2, image_id));
    SQLITE3_CALL(sqlite3_step(image_stmt));
    SQLITE3_CALL(sqlite3_reset(image_stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(image_stmt));
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type) VALUES(?, ?, ?, ?);", -1, &image_stmt, nullptr));
  for (const data_t& data_id : frame.DataIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(image_stmt, 1, frame_id));
    SQLITE3_CALL(sqlite3_bind_int64(image_stmt, 2, data_id.id));
    SQLITE3_CALL(sqlite3_bind_int64(image_stmt, 3, data_id.sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int(image_stmt, 4, static_cast<int>(data_id.sensor_id.type)));
    SQLITE3_CALL(sqlite3_step(image_stmt));
    SQLITE3_CALL(sqlite3_reset(image_stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(image_stmt));
  return frame_id;
}

image_t Database::WriteImage(const Image& image,
                             const bool use_image_id) const {
  if (use_image_id) {
    CHECK(!ExistsImage(image.ImageId())) << "image_id must be unique";
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_image_, 1, image.ImageId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_image_, 1));
  }

  SQLITE3_CALL(sqlite3_bind_text(sql_stmt_add_image_, 2, image.Name().c_str(),
                                 static_cast<int>(image.Name().size()),
                                 SQLITE_STATIC));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_image_, 3, image.CameraId()));

  // NaNs are automatically converted to NULLs in SQLite.
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 4, image.QvecPrior(0)));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 5, image.QvecPrior(1)));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 6, image.QvecPrior(2)));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 7, image.QvecPrior(3)));

  // NaNs are automatically converted to NULLs in SQLite.
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 8, image.TvecPrior(0)));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 9, image.TvecPrior(1)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_add_image_, 10, image.TvecPrior(2)));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_image_));

  return static_cast<image_t>(sqlite3_last_insert_rowid(database_));
}

void Database::WriteKeypoints(const image_t image_id,
                              const FeatureKeypoints& keypoints) const {
  const FeatureKeypointsBlob blob = FeatureKeypointsToBlob(keypoints);

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_keypoints_, 1, image_id));
  WriteDynamicMatrixBlob(sql_stmt_write_keypoints_, blob, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_keypoints_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_keypoints_));
}

void Database::WriteDescriptors(const image_t image_id,
                                const FeatureDescriptors& descriptors) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_descriptors_, 1, image_id));
  WriteDynamicMatrixBlob(sql_stmt_write_descriptors_, descriptors, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_descriptors_));
}

void Database::WriteFloatDescriptors(const image_t image_id,
                                     const FeatureDescriptorsFloat& descriptors,
                                     const FeatureDescriptorType type) const {
  CHECK(descriptors.rows() >= 0 && descriptors.cols() >= 0);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_float_descriptors_, 1,
                                  image_id));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_float_descriptors_, 2,
                                  descriptors.rows()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_float_descriptors_, 3,
                                  descriptors.cols()));
  SQLITE3_CALL(sqlite3_bind_int(sql_stmt_write_float_descriptors_, 4,
                                static_cast<int>(type)));
  SQLITE3_CALL(sqlite3_bind_blob(
      sql_stmt_write_float_descriptors_, 5,
      reinterpret_cast<const char*>(descriptors.data()),
      static_cast<int>(descriptors.size() * sizeof(float)), SQLITE_STATIC));
  SQLITE3_CALL(sqlite3_step(sql_stmt_write_float_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_float_descriptors_));
}

void Database::WriteMatches(const image_t image_id1, const image_t image_id2,
                            const FeatureMatches& matches) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_matches_, 1, pair_id));

  // Important: the swapped data must live until the query is executed.
  FeatureMatchesBlob blob = FeatureMatchesToBlob(matches);
  if (SwapImagePair(image_id1, image_id2)) {
    SwapFeatureMatchesBlob(&blob);
    WriteDynamicMatrixBlob(sql_stmt_write_matches_, blob, 2);
  } else {
    WriteDynamicMatrixBlob(sql_stmt_write_matches_, blob, 2);
  }

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_matches_));
}

void Database::WriteTwoViewGeometry(
    const image_t image_id1, const image_t image_id2,
    const TwoViewGeometry& two_view_geometry) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_write_two_view_geometry_, 1, pair_id));

  const TwoViewGeometry* two_view_geometry_ptr = &two_view_geometry;

  // Invert the two-view geometry if the image pair has to be swapped.
  std::unique_ptr<TwoViewGeometry> swapped_two_view_geometry;
  if (SwapImagePair(image_id1, image_id2)) {
    swapped_two_view_geometry.reset(new TwoViewGeometry());
    *swapped_two_view_geometry = two_view_geometry;
    swapped_two_view_geometry->Invert();
    two_view_geometry_ptr = swapped_two_view_geometry.get();
  }

  const FeatureMatchesBlob inlier_matches =
      FeatureMatchesToBlob(two_view_geometry_ptr->inlier_matches);
  WriteDynamicMatrixBlob(sql_stmt_write_two_view_geometry_, inlier_matches, 2);

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_two_view_geometry_, 5,
                                  two_view_geometry_ptr->config));

  // Transpose the matrices to obtain row-major data layout.
  // Important: Do not move these objects inside the if-statement, because
  // the objects must live until `sqlite3_step` is called on the statement.
  // Bridge the upstream optional members to the fork's legacy blob layout:
  // absent matrices serialize as empty blobs; the pose as zero qvec/tvec.
  static const Eigen::Matrix3d kZero3d = Eigen::Matrix3d::Zero();
  const Eigen::Matrix3d Ft = two_view_geometry_ptr->F
                                 ? two_view_geometry_ptr->F->transpose()
                                 : kZero3d;
  const Eigen::Matrix3d Et = two_view_geometry_ptr->E
                                 ? two_view_geometry_ptr->E->transpose()
                                 : kZero3d;
  const Eigen::Matrix3d Ht = two_view_geometry_ptr->H
                                 ? two_view_geometry_ptr->H->transpose()
                                 : kZero3d;
  Eigen::Vector4d qvec = Eigen::Vector4d::Zero();
  Eigen::Vector3d tvec = Eigen::Vector3d::Zero();
  if (two_view_geometry_ptr->cam2_from_cam1) {
    const Eigen::Quaterniond& q =
        two_view_geometry_ptr->cam2_from_cam1->rotation();
    qvec << q.w(), q.x(), q.y(), q.z();
    tvec = two_view_geometry_ptr->cam2_from_cam1->translation();
  }

  if (two_view_geometry_ptr->inlier_matches.size() > 0) {
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Ft, 6);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Et, 7);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Ht, 8);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, qvec, 9);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, tvec, 10);
  } else {
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_,
                          Eigen::MatrixXd(0, 0), 6);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_,
                          Eigen::MatrixXd(0, 0), 7);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_,
                          Eigen::MatrixXd(0, 0), 8);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_,
                          Eigen::MatrixXd(0, 0), 9);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_,
                          Eigen::MatrixXd(0, 0), 10);
  }

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_two_view_geometry_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_two_view_geometry_));
}

void Database::UpdateCamera(const Camera& camera) const {
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 1, camera.ModelId()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_camera_, 2,
                                  static_cast<sqlite3_int64>(camera.Width())));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_camera_, 3,
                                  static_cast<sqlite3_int64>(camera.Height())));

  const size_t num_params_bytes = sizeof(double) * camera.NumParams();
  SQLITE3_CALL(
      sqlite3_bind_blob(sql_stmt_update_camera_, 4, camera.ParamsData(),
                        static_cast<int>(num_params_bytes), SQLITE_STATIC));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_camera_, 5,
                                  camera.HasPriorFocalLength()));

  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 6, camera.CameraId()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_camera_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_camera_));
}

void Database::UpdateRig(const Rig& rig) const {
  CHECK(ExistsRig(rig.RigId()));
  SQLITE3_EXEC(database_, StringPrintf("DELETE FROM rig_cameras WHERE rig_id=%u;", rig.RigId()).c_str(), nullptr);
  SQLITE3_EXEC(database_, StringPrintf("DELETE FROM rig_sensors WHERE rig_id=%u;", rig.RigId()).c_str(), nullptr);
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "UPDATE rigs SET ref_camera_id=?, ref_sensor_id=?, ref_sensor_type=? WHERE rig_id=?;", -1, &stmt, nullptr));
  const std::vector<camera_t> legacy_camera_ids = rig.CameraIds();
  const camera_t legacy_ref_camera_id =
      rig.RefCameraId() != kInvalidCameraId
          ? rig.RefCameraId()
          : (legacy_camera_ids.empty() ? rig.RefSensorId().id
                                       : legacy_camera_ids.front());
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, legacy_ref_camera_id));
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 2, rig.RefSensorId().id));
  SQLITE3_CALL(sqlite3_bind_int(stmt, 3, static_cast<int>(rig.RefSensorId().type)));
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 4, rig.RigId()));
  SQLITE3_CALL(sqlite3_step(stmt));
  SQLITE3_CALL(sqlite3_finalize(stmt));
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO rig_cameras(rig_id, camera_id, qvec, tvec) VALUES(?, ?, ?, ?);", -1, &stmt, nullptr));
  for (const camera_t camera_id : rig.CameraIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, rig.RigId()));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 2, camera_id));
    BindPoseBlob(stmt, 3, 4, rig.CamFromRigQvec(camera_id), rig.CamFromRigTvec(camera_id));
    SQLITE3_CALL(sqlite3_step(stmt));
    SQLITE3_CALL(sqlite3_reset(stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(stmt));
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type, qvec, tvec) VALUES(?, ?, ?, ?, ?);", -1, &stmt, nullptr));
  for (const sensor_t& sensor_id : rig.SensorIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, rig.RigId()));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 2, sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int(stmt, 3, static_cast<int>(sensor_id.type)));
    if (rig.HasSensorFromRig(sensor_id)) {
      BindPoseBlob(stmt, 4, 5, rig.SensorFromRigQvec(sensor_id), rig.SensorFromRigTvec(sensor_id));
    } else {
      SQLITE3_CALL(sqlite3_bind_null(stmt, 4));
      SQLITE3_CALL(sqlite3_bind_null(stmt, 5));
    }
    SQLITE3_CALL(sqlite3_step(stmt));
    SQLITE3_CALL(sqlite3_reset(stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(stmt));
}

void Database::UpdateFrame(const Frame& frame) const {
  CHECK(ExistsFrame(frame.FrameId()));
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "UPDATE frames SET rig_id=?, has_pose=?, qvec=?, tvec=? WHERE frame_id=?;", -1, &stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, frame.RigId()));
  SQLITE3_CALL(sqlite3_bind_int(stmt, 2, frame.HasPose() ? 1 : 0));
  if (frame.HasPose()) BindPoseBlob(stmt, 3, 4, frame.RigFromWorldQvec(), frame.RigFromWorldTvec());
  else { SQLITE3_CALL(sqlite3_bind_null(stmt, 3)); SQLITE3_CALL(sqlite3_bind_null(stmt, 4)); }
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 5, frame.FrameId()));
  SQLITE3_CALL(sqlite3_step(stmt));
  SQLITE3_CALL(sqlite3_finalize(stmt));
  SQLITE3_EXEC(database_, StringPrintf("DELETE FROM frame_images WHERE frame_id=%u;", frame.FrameId()).c_str(), nullptr);
  SQLITE3_EXEC(database_, StringPrintf("DELETE FROM frame_data WHERE frame_id=%u;", frame.FrameId()).c_str(), nullptr);
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO frame_images(frame_id, image_id) VALUES(?, ?);", -1, &stmt, nullptr));
  for (const image_t image_id : frame.ImageIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, frame.FrameId()));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 2, image_id));
    SQLITE3_CALL(sqlite3_step(stmt));
    SQLITE3_CALL(sqlite3_reset(stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(stmt));
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type) VALUES(?, ?, ?, ?);", -1, &stmt, nullptr));
  for (const data_t& data_id : frame.DataIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, frame.FrameId()));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 2, data_id.id));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 3, data_id.sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int(stmt, 4, static_cast<int>(data_id.sensor_id.type)));
    SQLITE3_CALL(sqlite3_step(stmt));
    SQLITE3_CALL(sqlite3_reset(stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(stmt));
}

void Database::UpdateImage(const Image& image) const {
  SQLITE3_CALL(
      sqlite3_bind_text(sql_stmt_update_image_, 1, image.Name().c_str(),
                        static_cast<int>(image.Name().size()), SQLITE_STATIC));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_image_, 2, image.CameraId()));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 3, image.QvecPrior(0)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 4, image.QvecPrior(1)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 5, image.QvecPrior(2)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 6, image.QvecPrior(3)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 7, image.TvecPrior(0)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 8, image.TvecPrior(1)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 9, image.TvecPrior(2)));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_image_, 10, image.ImageId()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_image_));
}

void Database::UpdateKeypoints(const image_t image_id,
                               const FeatureKeypoints& keypoints) const {
  const FeatureKeypointsBlob blob = FeatureKeypointsToBlob(keypoints);

  // UPDATE keypoints has four independent parameters: rows, cols, data and
  // the WHERE image_id (upstream dbb41680 UpdateKeypoints layout).
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_keypoints_, 1,
                                  static_cast<sqlite3_int64>(blob.rows())));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_keypoints_, 2,
                                  static_cast<sqlite3_int64>(blob.cols())));
  SQLITE3_CALL(
      sqlite3_bind_blob(sql_stmt_update_keypoints_, 3, blob.data(),
                        static_cast<int>(blob.size() * sizeof(float)),
                        SQLITE_STATIC));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_keypoints_, 4, image_id));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_keypoints_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_keypoints_));
}

void Database::DeleteMatches(const image_t image_id1,
                             const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_matches_, 1,
                                  static_cast<sqlite3_int64>(pair_id)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_matches_));
}

void Database::DeleteInlierMatches(const image_t image_id1,
                                   const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_two_view_geometry_, 1,
                                  static_cast<sqlite3_int64>(pair_id)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_two_view_geometry_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_two_view_geometry_));
}

void Database::ClearAllTables() const {
  ClearMatches();
  ClearTwoViewGeometries();
  ClearDescriptors();
  ClearKeypoints();
  ClearFrames();
  ClearRigs();
  ClearImages();
  ClearCameras();
}

void Database::ClearCameras() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_cameras_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_cameras_));
}

void Database::ClearRigs() const {
  SQLITE3_EXEC(database_, "DELETE FROM rig_sensors; DELETE FROM rig_cameras; DELETE FROM rigs;", nullptr);
}

void Database::ClearFrames() const {
  SQLITE3_EXEC(database_, "DELETE FROM frame_data; DELETE FROM frame_images; DELETE FROM frames;", nullptr);
}

void Database::ClearImages() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_images_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_images_));
}

void Database::ClearDescriptors() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_descriptors_));
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_float_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_float_descriptors_));
}

void Database::ClearKeypoints() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_keypoints_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_keypoints_));
}

void Database::ClearMatches() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_matches_));
}

void Database::ClearTwoViewGeometries() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_two_view_geometries_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_two_view_geometries_));
}

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

void Database::BeginTransaction() const {
  SQLITE3_EXEC(database_, "BEGIN TRANSACTION", nullptr);
}

void Database::EndTransaction() const {
  SQLITE3_EXEC(database_, "END TRANSACTION", nullptr);
}

void Database::PrepareSQLStatements() {
  sql_stmts_.clear();

  std::string sql;

  //////////////////////////////////////////////////////////////////////////////
  // num_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "SELECT rows FROM keypoints WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_num_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_num_keypoints_);

  sql = "SELECT rows FROM descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_num_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_num_descriptors_);

  //////////////////////////////////////////////////////////////////////////////
  // exists_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "SELECT 1 FROM cameras WHERE camera_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_camera_, 0));
  sql_stmts_.push_back(sql_stmt_exists_camera_);

  sql = "SELECT 1 FROM images WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_image_id_, 0));
  sql_stmts_.push_back(sql_stmt_exists_image_id_);

  sql = "SELECT 1 FROM images WHERE name = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_image_name_, 0));
  sql_stmts_.push_back(sql_stmt_exists_image_name_);

  sql = "SELECT 1 FROM keypoints WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_exists_keypoints_);

  sql = "SELECT 1 FROM descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_exists_descriptors_);

  sql = "SELECT 1 FROM float_descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_float_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_exists_float_descriptors_);

  sql = "SELECT 1 FROM matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_matches_, 0));
  sql_stmts_.push_back(sql_stmt_exists_matches_);

  sql = "SELECT 1 FROM two_view_geometries WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_two_view_geometry_, 0));
  sql_stmts_.push_back(sql_stmt_exists_two_view_geometry_);

  sql = "SELECT 1 FROM pose_priors WHERE pose_prior_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_pose_prior_, 0));
  sql_stmts_.push_back(sql_stmt_exists_pose_prior_);

  //////////////////////////////////////////////////////////////////////////////
  // add_*
  //////////////////////////////////////////////////////////////////////////////
  sql =
      "INSERT INTO cameras(camera_id, model, width, height, params, "
      "prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);";
  SQLITE3_CALL(
      sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_camera_, 0));
  sql_stmts_.push_back(sql_stmt_add_camera_);

  sql =
      "INSERT INTO images(image_id, name, camera_id, prior_qw, prior_qx, "
      "prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) VALUES(?, ?, ?, ?, ?, "
      "?, ?, ?, ?, ?);";
  SQLITE3_CALL(
      sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_image_, 0));
  sql_stmts_.push_back(sql_stmt_add_image_);

  //////////////////////////////////////////////////////////////////////////////
  // update_*
  //////////////////////////////////////////////////////////////////////////////
  sql =
      "UPDATE cameras SET model=?, width=?, height=?, params=?, "
      "prior_focal_length=? WHERE camera_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_camera_, 0));
  sql_stmts_.push_back(sql_stmt_update_camera_);

  sql =
      "UPDATE images SET name=?, camera_id=?, prior_qw=?, prior_qx=?, "
      "prior_qy=?, prior_qz=?, prior_tx=?, prior_ty=?, prior_tz=? WHERE "
      "image_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_image_, 0));
  sql_stmts_.push_back(sql_stmt_update_image_);

  //////////////////////////////////////////////////////////////////////////////
  // read_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "SELECT * FROM cameras WHERE camera_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_camera_, 0));
  sql_stmts_.push_back(sql_stmt_read_camera_);

  sql = "SELECT * FROM cameras;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_cameras_, 0));
  sql_stmts_.push_back(sql_stmt_read_cameras_);

  sql = "SELECT * FROM images WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_image_id_, 0));
  sql_stmts_.push_back(sql_stmt_read_image_id_);

  sql = "SELECT * FROM images WHERE name = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_image_name_, 0));
  sql_stmts_.push_back(sql_stmt_read_image_name_);

  sql = "SELECT * FROM images;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_images_, 0));
  sql_stmts_.push_back(sql_stmt_read_images_);

  sql = "SELECT rows, cols, data FROM keypoints WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_read_keypoints_);

  sql = "SELECT rows, cols, data FROM descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_read_descriptors_);

  sql = "SELECT rows, cols, data, type FROM float_descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_float_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_read_float_descriptors_);

  sql = "SELECT type FROM float_descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_descriptor_type_, 0));
  sql_stmts_.push_back(sql_stmt_read_descriptor_type_);

  sql = "SELECT rows, cols, data FROM matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_matches_, 0));
  sql_stmts_.push_back(sql_stmt_read_matches_);

  sql = "SELECT * FROM matches WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_matches_all_, 0));
  sql_stmts_.push_back(sql_stmt_read_matches_all_);

  sql =
      "SELECT rows, cols, data, config, F, E, H, qvec, tvec FROM "
      "two_view_geometries WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_two_view_geometry_, 0));
  sql_stmts_.push_back(sql_stmt_read_two_view_geometry_);

  sql = "SELECT * FROM two_view_geometries WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_two_view_geometries_, 0));
  sql_stmts_.push_back(sql_stmt_read_two_view_geometries_);

  sql = "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_two_view_geometry_num_inliers_,
                                  0));
  sql_stmts_.push_back(sql_stmt_read_two_view_geometry_num_inliers_);

  // Column order matches the pose_priors table definition and
  // ReadPosePriorRow below.
  sql = "SELECT pose_prior_id, data_id, sensor_id, sensor_type, position, "
        "position_covariance, coordinate_system, gravity FROM pose_priors "
        "WHERE pose_prior_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_pose_prior_, 0));
  sql_stmts_.push_back(sql_stmt_read_pose_prior_);

  sql = "SELECT pose_prior_id, data_id, sensor_id, sensor_type, position, "
        "position_covariance, coordinate_system, gravity FROM pose_priors;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_pose_priors_, 0));
  sql_stmts_.push_back(sql_stmt_read_pose_priors_);

  //////////////////////////////////////////////////////////////////////////////
  // write_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_write_keypoints_);

  sql = "UPDATE keypoints SET rows=?, cols=?, data=? WHERE image_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_update_keypoints_);

  sql = "UPDATE pose_priors SET data_id=?, sensor_id=?, sensor_type=?, "
        "position=?, position_covariance=?, coordinate_system=?, gravity=? "
        "WHERE pose_prior_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_pose_prior_, 0));
  sql_stmts_.push_back(sql_stmt_update_pose_prior_);

  sql =
      "INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_write_descriptors_);

  sql = "INSERT OR REPLACE INTO float_descriptors(image_id, rows, cols, type, data) VALUES(?, ?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_float_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_write_float_descriptors_);

  sql = "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_matches_, 0));
  sql_stmts_.push_back(sql_stmt_write_matches_);

  sql = "INSERT INTO two_view_geometries(pair_id, rows, cols, data, config, F, "
      "E, H, qvec, tvec) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_two_view_geometry_, 0));
  sql_stmts_.push_back(sql_stmt_write_two_view_geometry_);

  sql = "INSERT INTO pose_priors(pose_prior_id, data_id, sensor_id, "
        "sensor_type, position, position_covariance, coordinate_system, "
        "gravity) VALUES(?, ?, ?, ?, ?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_pose_prior_, 0));
  sql_stmts_.push_back(sql_stmt_write_pose_prior_);

  //////////////////////////////////////////////////////////////////////////////
  // delete_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "DELETE FROM matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_delete_matches_, 0));
  sql_stmts_.push_back(sql_stmt_delete_matches_);

  sql = "DELETE FROM two_view_geometries WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_delete_two_view_geometry_, 0));
  sql_stmts_.push_back(sql_stmt_delete_two_view_geometry_);

  //////////////////////////////////////////////////////////////////////////////
  // clear_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "DELETE FROM cameras;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_cameras_, 0));
  sql_stmts_.push_back(sql_stmt_clear_cameras_);

  sql = "DELETE FROM images;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_images_, 0));
  sql_stmts_.push_back(sql_stmt_clear_images_);

  sql = "DELETE FROM descriptors;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_clear_descriptors_);

  sql = "DELETE FROM float_descriptors;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_float_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_clear_float_descriptors_);

  sql = "DELETE FROM keypoints;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_clear_keypoints_);

  sql = "DELETE FROM matches;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_matches_, 0));
  sql_stmts_.push_back(sql_stmt_clear_matches_);

  sql = "DELETE FROM two_view_geometries;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_two_view_geometries_, 0));
  sql_stmts_.push_back(sql_stmt_clear_two_view_geometries_);

  sql = "DELETE FROM pose_priors;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_pose_priors_, 0));
  sql_stmts_.push_back(sql_stmt_clear_pose_priors_);
}

void Database::FinalizeSQLStatements() {
  for (const auto& sql_stmt : sql_stmts_) {
    SQLITE3_CALL(sqlite3_finalize(sql_stmt));
  }
}

void Database::CreateTables() const {
  CreateCameraTable();
  CreateRigTable();
  CreateRigSensorsTable();
  CreateRigCamerasTable();
  CreateFrameTable();
  CreateFrameDataTable();
  CreateFrameImagesTable();
  CreateImageTable();
  CreateKeypointsTable();
  CreateDescriptorsTable();
  CreateFloatDescriptorsTable();
  CreateMatchesTable();
  CreateTwoViewGeometriesTable();
  CreatePosePriorsTable();
}

void Database::CreateRigTable() const {
  SQLITE3_EXEC(database_, "CREATE TABLE IF NOT EXISTS rigs"
             " (rig_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
             "  ref_camera_id INTEGER, ref_sensor_id INTEGER NOT NULL,"
             "  ref_sensor_type INTEGER NOT NULL);", nullptr);
}

void Database::CreateRigSensorsTable() const {
  SQLITE3_EXEC(database_, "CREATE TABLE IF NOT EXISTS rig_sensors"
             " (rig_id INTEGER NOT NULL, sensor_id INTEGER NOT NULL, sensor_type INTEGER NOT NULL,"
             "  qvec BLOB, tvec BLOB, PRIMARY KEY(rig_id, sensor_id, sensor_type),"
             "  UNIQUE(sensor_id, sensor_type),"
             "  FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);", nullptr);
}

void Database::CreateRigCamerasTable() const {
  SQLITE3_EXEC(database_, "CREATE TABLE IF NOT EXISTS rig_cameras"
             " (rig_id INTEGER NOT NULL, camera_id INTEGER NOT NULL,"
             "  qvec BLOB NOT NULL, tvec BLOB NOT NULL,"
             "  PRIMARY KEY(rig_id, camera_id), UNIQUE(camera_id),"
             "  FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE,"
             "  FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));", nullptr);
}

void Database::CreateFrameTable() const {
  SQLITE3_EXEC(database_, "CREATE TABLE IF NOT EXISTS frames"
             " (frame_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
             "  rig_id INTEGER NOT NULL, has_pose INTEGER NOT NULL, qvec BLOB, tvec BLOB,"
             "  FOREIGN KEY(rig_id) REFERENCES rigs(rig_id));", nullptr);
}

void Database::CreateFrameDataTable() const {
  SQLITE3_EXEC(database_, "CREATE TABLE IF NOT EXISTS frame_data"
             " (frame_id INTEGER NOT NULL, data_id INTEGER NOT NULL, sensor_id INTEGER NOT NULL,"
             "  sensor_type INTEGER NOT NULL, PRIMARY KEY(frame_id, data_id, sensor_id, sensor_type),"
             "  UNIQUE(data_id, sensor_type),"
             "  FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE);", nullptr);
}

void Database::CreateFrameImagesTable() const {
  SQLITE3_EXEC(database_, "CREATE TABLE IF NOT EXISTS frame_images"
             " (frame_id INTEGER NOT NULL, image_id INTEGER NOT NULL,"
             "  PRIMARY KEY(frame_id, image_id), UNIQUE(image_id),"
             "  FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE,"
             "  FOREIGN KEY(image_id) REFERENCES images(image_id));", nullptr);
}

void Database::CreateCameraTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS cameras"
      "   (camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    model                INTEGER                             NOT NULL,"
      "    width                INTEGER                             NOT NULL,"
      "    height               INTEGER                             NOT NULL,"
      "    params               BLOB,"
      "    prior_focal_length   INTEGER                             NOT NULL);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateImageTable() const {
  const std::string sql = StringPrintf(
      "CREATE TABLE IF NOT EXISTS images"
      "   (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    name       TEXT                                NOT NULL UNIQUE,"
      "    camera_id  INTEGER                             NOT NULL,"
      "    prior_qw   REAL,"
      "    prior_qx   REAL,"
      "    prior_qy   REAL,"
      "    prior_qz   REAL,"
      "    prior_tx   REAL,"
      "    prior_ty   REAL,"
      "    prior_tz   REAL,"
      "CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < %d),"
      "FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));"
      "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);",
      kMaxNumImages);

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateKeypointsTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS keypoints"
      "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows      INTEGER               NOT NULL,"
      "    cols      INTEGER               NOT NULL,"
      "    data      BLOB,"
      "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateDescriptorsTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS descriptors"
      "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows      INTEGER               NOT NULL,"
      "    cols      INTEGER               NOT NULL,"
      "    data      BLOB,"
      "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateFloatDescriptorsTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS float_descriptors"
      "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows      INTEGER               NOT NULL,"
      "    cols      INTEGER               NOT NULL,"
      "    type      INTEGER               NOT NULL,"
      "    data      BLOB,"
      "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);";
  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateMatchesTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS matches"
      "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows     INTEGER               NOT NULL,"
      "    cols     INTEGER               NOT NULL,"
      "    data     BLOB);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateTwoViewGeometriesTable() const {
  if (ExistsTable("inlier_matches")) {
    SQLITE3_EXEC(database_,
                 "ALTER TABLE inlier_matches RENAME TO two_view_geometries;",
                 nullptr);
  } else {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS two_view_geometries"
        "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
        "    rows     INTEGER               NOT NULL,"
        "    cols     INTEGER               NOT NULL,"
        "    data     BLOB,"
        "    config   INTEGER               NOT NULL,"
        "    F        BLOB,"
        "    E        BLOB,"
        "    H        BLOB,"
        "    qvec     BLOB,"
        "    tvec     BLOB);";
    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }
}

void Database::CreatePosePriorsTable() const {
  // Upstream dbb41680 column order without the data-table foreign key (this
  // fork has no unified `data` table; the fork-legacy tables carry the
  // sensor references instead).
  SQLITE3_EXEC(database_,
               "CREATE TABLE IF NOT EXISTS pose_priors"
               "   (pose_prior_id  INTEGER  PRIMARY KEY  AUTOINCREMENT,"
               "    data_id        INTEGER               NOT NULL,"
               "    sensor_id      INTEGER               NOT NULL,"
               "    sensor_type    INTEGER               NOT NULL,"
               "    position       BLOB,"
               "    position_covariance BLOB,"
               "    coordinate_system   INTEGER          NOT NULL,"
               "    gravity        BLOB);",
               nullptr);
}


namespace {

// Returns whether the given table has a column of the given name.
bool ExistsColumnImpl(sqlite3* db, const std::string& table_name,
                      const std::string& column_name) {
    sqlite3_stmt* stmt;
    const std::string sql =
            "SELECT name FROM pragma_table_info('" + table_name +
            "') WHERE name = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr));
    SQLITE3_CALL(
            sqlite3_bind_text(stmt, 1, column_name.c_str(), -1, SQLITE_STATIC));
    const bool exists =
            SQLITE3_CALL(sqlite3_step(stmt)) == SQLITE_ROW;
    SQLITE3_CALL(sqlite3_finalize(stmt));
    return exists;
}

}  // namespace

void Database::PreMigrateTables() const {
    // Legacy upstream databases keyed pose priors by image IDs; the table is
    // renamed so that CreateTables() can recreate it in the new schema.
    if (ExistsTable("pose_priors") &&
        ExistsColumnImpl(database_, "pose_priors", "image_id")) {
        SQLITE3_EXEC(database_,
                     "ALTER TABLE pose_priors RENAME TO pose_priors_old;",
                     nullptr);
    }
}

int Database::ReadUserVersion() const {
    sqlite3_stmt* version_stmt;
    SQLITE3_CALL(sqlite3_prepare_v2(database_, "PRAGMA user_version;", -1,
                                    &version_stmt, nullptr));
    int user_version = 0;
    if (SQLITE3_CALL(sqlite3_step(version_stmt)) == SQLITE_ROW) {
        user_version = sqlite3_column_int(version_stmt, 0);
    }
    SQLITE3_CALL(sqlite3_finalize(version_stmt));
    return user_version;
}

void Database::PostMigrateTables() const {
    // Refuse databases written by a newer schema than this build supports.
    const int user_version = ReadUserVersion();
    if (user_version > GetDatabaseVersionNumber()) {
        throw std::runtime_error(
                "Database schema version " + std::to_string(user_version) +
                " is newer than the supported version " +
                std::to_string(GetDatabaseVersionNumber()) + ".");
    }

    // Legacy fork databases (pre user_version migration machinery, version
    // 395) stored sentinel poses/matrices for unknown values; migrate them
    // to NULL. Sentinels: identity qvec (w=1, little-endian) + zero tvec,
    // zero F/E/H matrices.
    if (user_version == kLegacyForkDatabaseVersionNumber) {
        const std::string zero48(48, '0');
        const std::string zero144(144, '0');
        SQLITE3_EXEC(
                database_,
                ("UPDATE two_view_geometries SET qvec = NULL WHERE qvec ="
                 " X'000000000000F03F" + std::string(48, '0') + "';")
                        .c_str(),
                nullptr);
        SQLITE3_EXEC(database_,
                     ("UPDATE two_view_geometries SET tvec = NULL WHERE tvec"
                      " = X'" + zero48 + "';")
                             .c_str(),
                     nullptr);
        SQLITE3_EXEC(database_,
                     ("UPDATE two_view_geometries SET F = NULL WHERE F ="
                      " X'" + zero144 + "';")
                             .c_str(),
                     nullptr);
        SQLITE3_EXEC(database_,
                     ("UPDATE two_view_geometries SET E = NULL WHERE E ="
                      " X'" + zero144 + "';")
                             .c_str(),
                     nullptr);
        SQLITE3_EXEC(database_,
                     ("UPDATE two_view_geometries SET H = NULL WHERE H ="
                      " X'" + zero144 + "';")
                             .c_str(),
                     nullptr);
    }

    // Stamp the schema version (moved from the legacy UpdateSchema tail).
    std::unique_lock<std::mutex> lock(update_schema_mutex_);
    const std::string update_user_version_sql =
            StringPrintf("PRAGMA user_version = %d;", GetDatabaseVersionNumber());
    SQLITE3_EXEC(database_, update_user_version_sql.c_str(), nullptr);
}
void Database::UpdateSchema() const {
  if (!ExistsColumn("rigs", "ref_sensor_id")) {
    SQLITE3_EXEC(database_, "ALTER TABLE rigs ADD COLUMN ref_sensor_id INTEGER;", nullptr);
  }
  if (!ExistsColumn("rigs", "ref_sensor_type")) {
    SQLITE3_EXEC(database_, "ALTER TABLE rigs ADD COLUMN ref_sensor_type INTEGER;", nullptr);
  }
  SQLITE3_EXEC(database_, "UPDATE rigs SET ref_sensor_id=ref_camera_id, ref_sensor_type=0 "
               "WHERE ref_sensor_id IS NULL OR ref_sensor_type IS NULL;", nullptr);
  if (!ExistsColumn("two_view_geometries", "F")) {
    SQLITE3_EXEC(database_,
                 "ALTER TABLE two_view_geometries ADD COLUMN F BLOB;", nullptr);
  }

  if (!ExistsColumn("two_view_geometries", "E")) {
    SQLITE3_EXEC(database_,
                 "ALTER TABLE two_view_geometries ADD COLUMN E BLOB;", nullptr);
  }

  if (!ExistsColumn("two_view_geometries", "H")) {
    SQLITE3_EXEC(database_,
                 "ALTER TABLE two_view_geometries ADD COLUMN H BLOB;", nullptr);
  }

  if (!ExistsColumn("two_view_geometries", "qvec")) {
    SQLITE3_EXEC(database_,
                 "ALTER TABLE two_view_geometries ADD COLUMN qvec BLOB;",
                 nullptr);
  }

  if (!ExistsColumn("two_view_geometries", "tvec")) {
    SQLITE3_EXEC(database_,
                 "ALTER TABLE two_view_geometries ADD COLUMN tvec BLOB;",
                 nullptr);
  }

  // Update user version number.
  std::unique_lock<std::mutex> lock(update_schema_mutex_);
  const std::string update_user_version_sql =
      StringPrintf("PRAGMA user_version = %d;", COLMAP_VERSION_NUMBER);
  SQLITE3_EXEC(database_, update_user_version_sql.c_str(), nullptr);
}

bool Database::ExistsTable(const std::string& table_name) const {
  const std::string sql =
      "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;";

  sqlite3_stmt* sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  SQLITE3_CALL(sqlite3_bind_text(sql_stmt, 1, table_name.c_str(),
                                 static_cast<int>(table_name.size()),
                                 SQLITE_STATIC));

  const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return exists;
}

bool Database::ExistsColumn(const std::string& table_name,
                            const std::string& column_name) const {
  const std::string sql =
      StringPrintf("PRAGMA table_info(%s);", table_name.c_str());

  sqlite3_stmt* sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  bool exists_column = false;
  while (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
    const std::string result =
        reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1));
    if (column_name == result) {
      exists_column = true;
      break;
    }
  }

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return exists_column;
}

bool Database::ExistsRowId(sqlite3_stmt* sql_stmt,
                           const sqlite3_int64 row_id) const {
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt, 1, static_cast<sqlite3_int64>(row_id)));

  const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

bool Database::ExistsRowString(sqlite3_stmt* sql_stmt,
                               const std::string& row_entry) const {
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt, 1, row_entry.c_str(),
                                 static_cast<int>(row_entry.size()),
                                 SQLITE_STATIC));

  const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

size_t Database::CountRows(const std::string& table) const {
  const std::string sql =
      StringPrintf("SELECT COUNT(*) FROM %s;", table.c_str());

  sqlite3_stmt* sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  size_t count = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return count;
}

size_t Database::CountRowsForEntry(sqlite3_stmt* sql_stmt,
                                   const sqlite3_int64 row_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, 1, row_id));

  size_t count = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return count;
}

size_t Database::SumColumn(const std::string& column,
                           const std::string& table) const {
  const std::string sql =
      StringPrintf("SELECT SUM(%s) FROM %s;", column.c_str(), table.c_str());

  sqlite3_stmt* sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  size_t sum = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    sum = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return sum;
}

size_t Database::MaxColumn(const std::string& column,
                           const std::string& table) const {
  const std::string sql =
      StringPrintf("SELECT MAX(%s) FROM %s;", column.c_str(), table.c_str());

  sqlite3_stmt* sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  size_t max = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    max = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return max;
}

DatabaseTransaction::DatabaseTransaction(Database* database)
    : database_(database), database_lock_(database->transaction_mutex_) {
  CHECK_NOTNULL(database_);
  database_->BeginTransaction();
}

DatabaseTransaction::~DatabaseTransaction() { database_->EndTransaction(); }


namespace {

// Row reader shared by ReadPosePrior and ReadAllPosePriors. The SELECT
// column order matches the pose_priors table definition.
PosePrior ReadPosePriorRow(sqlite3_stmt* sql_stmt) {
  PosePrior pose_prior;
  pose_prior.pose_prior_id = static_cast<pose_prior_t>(
      sqlite3_column_int64(sql_stmt, 0));
  const data_t corr_data(
      sensor_t(static_cast<SensorType>(sqlite3_column_int(sql_stmt, 3)),
               static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 2))),
      static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 1)));
  pose_prior.corr_data_id = corr_data;
  pose_prior.position =
      ReadStaticMatrixBlob<Eigen::Vector3d>(sql_stmt, SQLITE_ROW, 4);
  pose_prior.position_covariance =
      ReadStaticMatrixBlob<Eigen::Matrix3d>(sql_stmt, SQLITE_ROW, 5);
  pose_prior.gravity =
      ReadStaticMatrixBlob<Eigen::Vector3d>(sql_stmt, SQLITE_ROW, 7);
  pose_prior.coordinate_system =
      static_cast<PosePrior::CoordinateSystem>(
          sqlite3_column_int(sql_stmt, 6));
  return pose_prior;
}

}  // namespace

bool Database::ExistsPosePrior(pose_prior_t pose_prior_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_exists_pose_prior_, 1,
                                  static_cast<sqlite3_int64>(pose_prior_id)));
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_exists_pose_prior_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_exists_pose_prior_));
  return rc == SQLITE_ROW;
}

size_t Database::NumPosePriors() const {
  sqlite3_stmt* sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT COUNT(*) FROM"
                                 " pose_priors;", -1, &sql_stmt, nullptr));
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  const size_t count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  SQLITE3_CALL(sqlite3_finalize(sql_stmt));
  THROW_CHECK_EQ(rc, SQLITE_ROW);
  return count;
}

PosePrior Database::ReadPosePrior(pose_prior_t pose_prior_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_pose_prior_, 1,
                                  static_cast<sqlite3_int64>(pose_prior_id)));
  PosePrior pose_prior;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_pose_prior_));
  if (rc == SQLITE_ROW) {
    pose_prior = ReadPosePriorRow(sql_stmt_read_pose_prior_);
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_pose_prior_));
  return pose_prior;
}

std::vector<PosePrior> Database::ReadAllPosePriors() const {
  std::vector<PosePrior> pose_priors;
  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_pose_priors_)) ==
         SQLITE_ROW) {
    pose_priors.push_back(ReadPosePriorRow(sql_stmt_read_pose_priors_));
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_pose_priors_));
  return pose_priors;
}

pose_prior_t Database::WritePosePrior(const PosePrior& pose_prior,
                                      bool use_pose_prior_id) {
  if (use_pose_prior_id) {
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_write_pose_prior_, 1,
        static_cast<sqlite3_int64>(pose_prior.pose_prior_id)));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_pose_prior_, 1));
  }
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_write_pose_prior_, 2,
      static_cast<sqlite3_int64>(pose_prior.corr_data_id.id)));
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_write_pose_prior_, 3,
      static_cast<sqlite3_int64>(pose_prior.corr_data_id.sensor_id.id)));
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_write_pose_prior_, 4,
      static_cast<sqlite3_int64>(pose_prior.corr_data_id.sensor_id.type)));
  WriteStaticMatrixBlob(sql_stmt_write_pose_prior_, pose_prior.position, 5);
  WriteStaticMatrixBlob(sql_stmt_write_pose_prior_,
                        pose_prior.position_covariance, 6);
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_write_pose_prior_, 7,
      static_cast<sqlite3_int64>(pose_prior.coordinate_system)));
  WriteStaticMatrixBlob(sql_stmt_write_pose_prior_, pose_prior.gravity, 8);
  SQLITE3_CALL(sqlite3_step(sql_stmt_write_pose_prior_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_pose_prior_));
  return static_cast<pose_prior_t>(
      sqlite3_last_insert_rowid(database_));
}

void Database::UpdatePosePrior(const PosePrior& pose_prior) {
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_update_pose_prior_, 1,
      static_cast<sqlite3_int64>(pose_prior.corr_data_id.id)));
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_update_pose_prior_, 2,
      static_cast<sqlite3_int64>(pose_prior.corr_data_id.sensor_id.id)));
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_update_pose_prior_, 3,
      static_cast<sqlite3_int64>(pose_prior.corr_data_id.sensor_id.type)));
  WriteStaticMatrixBlob(sql_stmt_update_pose_prior_, pose_prior.position, 4);
  WriteStaticMatrixBlob(sql_stmt_update_pose_prior_,
                        pose_prior.position_covariance, 5);
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_update_pose_prior_, 6,
      static_cast<sqlite3_int64>(pose_prior.coordinate_system)));
  WriteStaticMatrixBlob(sql_stmt_update_pose_prior_, pose_prior.gravity, 7);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_pose_prior_, 8,
                                  static_cast<sqlite3_int64>(
                                      pose_prior.pose_prior_id)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_update_pose_prior_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_pose_prior_));
}

void Database::ClearPosePriors() {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_pose_priors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_pose_priors_));
}

}  // namespace colmap
