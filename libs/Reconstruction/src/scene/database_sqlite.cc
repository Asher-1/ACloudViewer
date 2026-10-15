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

#include "scene/database.h"
#include "scene/database_sqlite.h"

#include <fstream>
#include <stdexcept>

#include "util/sqlite3_utils.h"
#include "util/string.h"
#include "util/version.h"

// W17.2 (fork parity, upstream dbb41680 scene/database_sqlite.cc): the SQL
// implementation of the Database class lives here under the upstream file
// name. The fork keeps the concrete Database class in base/database.h (the
// abstract-interface split is tracked as W17.2b); this file mirrors the
// upstream database_sqlite.cc function-for-function so upstream diffs in the
// SQLite layer can be located mechanically.


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
  if (blob.rows() == 0) {
    // Upstream parity: images without keypoints (e.g. match-only test
    // fixtures) load as empty rather than failing the column check.
    return keypoints;
  } else if (blob.cols() == 2) {
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

// Reads a fork-legacy (qvec, tvec) blob pair; only used to migrate the
// fork-legacy rig tables to the upstream sensor_from_rig encoding.
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

// Upstream rig_sensors.sensor_from_rig encoding (WriteRigid3dToStringStream):
// 7 little-endian doubles [qw, qx, qy, qz, tx, ty, tz]. The fork's qvec is
// stored as (w, x, y, z), so it maps onto the blob element by element.
std::string SensorFromRigToBlob(const Eigen::Vector4d& qvec,
                                const Eigen::Vector3d& tvec) {
  std::ostringstream stream(std::ios::binary);
  for (int i = 0; i < 4; ++i) {
    WriteBinaryLittleEndian<double>(&stream, qvec(i));
  }
  for (int i = 0; i < 3; ++i) {
    WriteBinaryLittleEndian<double>(&stream, tvec(i));
  }
  return stream.str();
}

// Returns false for NULL (unknown sensor_from_rig).
bool SensorFromRigFromBlob(sqlite3_stmt* sql_stmt,
                           const int col,
                           Eigen::Vector4d* qvec,
                           Eigen::Vector3d* tvec) {
  if (sqlite3_column_type(sql_stmt, col) == SQLITE_NULL) {
    return false;
  }
  const size_t num_bytes =
      static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col));
  CHECK_EQ(num_bytes, 7 * sizeof(double));
  const std::string bytes(
      static_cast<const char*>(sqlite3_column_blob(sql_stmt, col)), num_bytes);
  std::istringstream stream(bytes, std::ios::binary);
  for (int i = 0; i < 4; ++i) {
    (*qvec)(i) = ReadBinaryLittleEndian<double>(&stream);
  }
  for (int i = 0; i < 3; ++i) {
    (*tvec)(i) = ReadBinaryLittleEndian<double>(&stream);
  }
  return true;
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
  camera.SetModelId(static_cast<CameraModelId>(
        sqlite3_column_int64(sql_stmt, 1)));
  camera.SetWidth(static_cast<size_t>(sqlite3_column_int64(sql_stmt, 2)));
  camera.SetHeight(static_cast<size_t>(sqlite3_column_int64(sql_stmt, 3)));

  const size_t num_params_bytes =
      static_cast<size_t>(sqlite3_column_bytes(sql_stmt, 4));
  const size_t num_params = num_params_bytes / sizeof(double);
  CHECK_EQ(num_params, camera.NumParams());
  memcpy(camera.ParamsData(), sqlite3_column_blob(sql_stmt, 4),
         num_params_bytes);

  // Upstream parity (f1ea91d0): column 5 carries the persisted
  // CameraSource; tolerate legacy readers with fewer columns and NULLs.
  if (sqlite3_column_count(sql_stmt) > 5 &&
      sqlite3_column_type(sql_stmt, 5) != SQLITE_NULL) {
    camera.SetSource(static_cast<CameraSource>(
        sqlite3_column_int64(sql_stmt, 5)));
  } else {
    camera.SetSource(CameraSource::UNKNOWN);
  }

  return camera;
}

Image ReadImageRow(sqlite3_stmt* sql_stmt) {
  Image image;

  image.SetImageId(static_cast<image_t>(sqlite3_column_int64(sql_stmt, 0)));
  image.SetName(std::string(
      reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1))));
  image.SetCameraId(static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 2)));

  // Upstream parity (a2bd15c2): the image-to-frame association is resolved
  // at read time by joining frame_data; NULL means the image is not yet part
  // of a frame.
  if (sqlite3_column_type(sql_stmt, 3) != SQLITE_NULL) {
    image.SetFrameId(static_cast<frame_t>(sqlite3_column_int64(sql_stmt, 3)));
  }

  return image;
}

}  // namespace


// W17.2b (upstream dbb41680 parity): the concrete SQL implementation lives
// behind the abstract Database interface; this class is the only
// implementation and is opened through SqliteDatabase::Open / OpenSqliteDatabase.
class SqliteDatabase : public Database {
 public:
  SqliteDatabase() = delete;
  explicit SqliteDatabase(const std::filesystem::path& path)
      : path_(path), database_(nullptr) {}

  ~SqliteDatabase() override { Close(); }

  // Open and close database. The same database should not be opened
  // concurrently in multiple threads or processes.
  static std::shared_ptr<Database> Open(const std::filesystem::path& path);

  void Close() override;

  bool ExistsCamera(const camera_t camera_id) const override;
  bool ExistsCameraSource(const camera_t camera_id,
                          const CameraSource source) const override;
  bool ExistsRig(const rig_t rig_id) const override;
  bool ExistsFrame(const frame_t frame_id) const override;
  bool ExistsImage(const image_t image_id) const override;
  bool ExistsImageWithName(std::string name) const override;
  bool ExistsKeypoints(const image_t image_id) const override;
  bool ExistsDescriptors(const image_t image_id) const override;
  bool ExistsMatches(const image_t image_id1,
                     const image_t image_id2) const override;
  bool ExistsInlierMatches(const image_t image_id1,
                           const image_t image_id2) const override;
  bool ExistsTwoViewGeometry(const image_t image_id1,
                             const image_t image_id2) const override;
  bool ExistsPosePrior(pose_prior_t pose_prior_id) const override;

  size_t NumCameras() const override;
  size_t NumRigs() const override;
  size_t NumFrames() const override;
  size_t NumImages() const override;
  size_t NumKeypoints() const override;
  size_t MaxNumKeypoints() const override;
  size_t NumKeypointsForImage(const image_t image_id) const override;
  size_t NumDescriptors() const override;
  size_t MaxNumDescriptors() const override;
  size_t NumDescriptorsForImage(const image_t image_id) const override;
  size_t NumMatches() const override;
  size_t NumInlierMatches() const override;
  size_t NumMatchedImagePairs() const override;
  size_t NumVerifiedImagePairs() const override;
  size_t NumPosePriors() const override;

  Camera ReadCamera(const camera_t camera_id,
                    const CameraSource source = CameraSource::BEST) const override;
  std::vector<Camera> ReadAllCameras(
      const CameraSource source = CameraSource::BEST) const override;
  std::map<CameraSource, Camera> ReadAllCameraSources(
      const camera_t camera_id) const override;
  NodeHashMap<camera_t, std::map<CameraSource, Camera>>
  ReadAllCameraSources() const override;
  Rig ReadRig(const rig_t rig_id) const override;
  std::optional<Rig> ReadRigWithSensor(sensor_t sensor_id) const override;
  std::vector<Rig> ReadAllRigs() const override;
  Frame ReadFrame(const frame_t frame_id) const override;
  std::vector<Frame> ReadAllFrames() const override;
  Image ReadImage(const image_t image_id) const override;
  Image ReadImageWithName(const std::string& name) const override;
  std::vector<Image> ReadAllImages() const override;
  FeatureKeypoints ReadKeypoints(const image_t image_id) const override;
  FeatureDescriptors ReadDescriptors(const image_t image_id) const override;
  FeatureMatches ReadMatches(const image_t image_id1,
                             const image_t image_id2) const override;
  std::vector<std::pair<image_pair_t, FeatureMatches>> ReadAllMatches()
          const override;
  std::vector<std::pair<image_pair_t, int>> ReadNumMatches() const override;
  TwoViewGeometry ReadTwoViewGeometry(const image_t image_id1,
                                      const image_t image_id2)
          const override;
  void ReadTwoViewGeometries(
          std::vector<image_pair_t>* image_pair_ids,
          std::vector<TwoViewGeometry>* two_view_geometries)
          const override;
  std::map<image_pair_t, TwoViewGeometry> ReadTwoViewGeometries()
          const override;
  void ReadTwoViewGeometryNumInliers(
          std::vector<std::pair<image_t, image_t>>* image_pairs,
          std::vector<int>* num_inliers) const override;
  PosePrior ReadPosePrior(pose_prior_t pose_prior_id) const override;
  std::vector<PosePrior> ReadAllPosePriors() const override;

  camera_t WriteCamera(const Camera& camera,
                       const bool use_camera_id = false) const override;
  rig_t WriteRig(const Rig& rig, const bool use_rig_id = false)
          const override;
  frame_t WriteFrame(const Frame& frame, const bool use_frame_id = false)
          const override;
  image_t WriteImage(const Image& image,
                     const bool use_image_id = false) const override;
  void WriteKeypoints(const image_t image_id,
                      const FeatureKeypoints& keypoints) const override;
  void WriteDescriptors(const image_t image_id,
                        const FeatureDescriptors& descriptors)
          const override;
  void WriteMatches(const image_t image_id1,
                    const image_t image_id2,
                    const FeatureMatches& matches) const override;
  void WriteTwoViewGeometry(const image_t image_id1,
                            const image_t image_id2,
                            const TwoViewGeometry& two_view_geometry)
          const override;
  void UpdateTwoViewGeometry(const image_t image_id1,
                             const image_t image_id2,
                             const TwoViewGeometry& two_view_geometry)
          const override;
  pose_prior_t WritePosePrior(const PosePrior& pose_prior,
                              bool use_pose_prior_id = false) override;

  void UpdateCamera(const Camera& camera) const override;
  void DeleteCameraSource(
      const camera_t camera_id,
      const CameraSource source = CameraSource::BEST) const override;
  void UpdateRig(const Rig& rig) const override;
  void UpdateFrame(const Frame& frame) const override;
  void UpdateImage(const Image& image) const override;
  void UpdateKeypoints(const image_t image_id,
                       const FeatureKeypoints& keypoints) const override;
  void UpdatePosePrior(const PosePrior& pose_prior) override;

  void DeleteMatches(const image_t image_id1,
                     const image_t image_id2) const override;
  void DeleteTwoViewGeometry(const image_t image_id1,
                             const image_t image_id2) const override;
  void DeleteInlierMatches(const image_t image_id1,
                           const image_t image_id2) const override;

  void ClearAllTables() const override;
  void ClearCameras() const override;
  void ClearRigs() const override;
  void ClearFrames() const override;
  void ClearImages() const override;
  void ClearDescriptors() const override;
  void ClearKeypoints() const override;
  void ClearMatches() const override;
  void ClearTwoViewGeometries() const override;
  void ClearPosePriors() const override;

 protected:
  void BeginTransaction() const override;
  void EndTransaction() const override;

 private:
  // Implementation of the opening logic (pragmas + migrations + prepared
  // statements), called by the static Open factory.
  void OpenSqlite();

  std::filesystem::path path_;

  // Prepare SQL statements once at construction of the database, and reuse
  // the statements for multiple queries by resetting their states.
  void PrepareSQLStatements();
  void FinalizeSQLStatements();

  // Create database tables, if not existing, called when opening a database.
  void CreateTables() const;
  void CreateCameraTable() const;
  void CreateRigTable() const;
  void CreateRigSensorsTable() const;
  void CreateFrameTable() const;
  void CreateFrameDataTable() const;
  void CreateImageTable() const;
  void CreateKeypointsTable() const;
  void CreateDescriptorsTable() const;
  void CreateMatchesTable() const;
  void CreateTwoViewGeometriesTable() const;
  void CreatePosePriorsTable() const;

  // Legacy-schema preparation before CreateTables() (upstream parity). Returns
  // whether the database carries the fork-legacy rig/frame/image schema,
  // whose tables were moved aside to be migrated by PostMigrateTables().
  bool PreMigrateTables() const;

  // Version-gated migrations and the user_version stamp (upstream parity).
  void PostMigrateTables(bool fork_legacy_schema) const;

  // Copies the moved-aside fork-legacy tables into the upstream schema.
  void MigrateForkLegacyTables() const;

  // Reads the raw PRAGMA user_version value.
  int ReadUserVersion() const;

  bool ExistsTable(const std::string& table_name) const;
  bool ExistsColumn(const std::string& table_name,
                    const std::string& column_name) const;

  bool ExistsRowId(sqlite3_stmt* sql_stmt,
                   const sqlite3_int64 row_id) const;
  bool ExistsRowString(sqlite3_stmt* sql_stmt,
                       const std::string& row_entry) const;

  size_t CountRows(const std::string& table) const;
  size_t CountRowsForEntry(sqlite3_stmt* sql_stmt,
                           const sqlite3_int64 row_id) const;
  size_t SumColumn(const std::string& column,
                   const std::string& table) const;
  size_t MaxColumn(const std::string& column,
                   const std::string& table) const;

  sqlite3* database_ = nullptr;

  // Ensure that only one database object at a time updates the schema of a
  // database. Since the schema is updated every time a database is opened,
  // this is to ensure that there are no race conditions ("database locked"
  // error messages) when the user actually only intends to read from the
  // database, which requires to open it.
  static std::mutex update_schema_mutex_;

  // A collection of all `sqlite3_stmt` objects for deletion in the
  // destructor.
  std::vector<sqlite3_stmt*> sql_stmts_;

  // num_*
  sqlite3_stmt* sql_stmt_num_cameras_ = nullptr;
  sqlite3_stmt* sql_stmt_num_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_num_descriptors_ = nullptr;

  // exists_*
  sqlite3_stmt* sql_stmt_exists_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_camera_source_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_image_id_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_image_name_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_two_view_geometry_ = nullptr;

  // add_*
  sqlite3_stmt* sql_stmt_add_camera_ = nullptr;
  // Upstream parity (f1ea91d0): explicit next-id allocation for the
  // composite-primary-key cameras table (no AUTOINCREMENT).
  sqlite3_stmt* sql_stmt_next_camera_id_ = nullptr;
  sqlite3_stmt* sql_stmt_add_image_ = nullptr;

  // update_*
  sqlite3_stmt* sql_stmt_update_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_update_image_ = nullptr;
  sqlite3_stmt* sql_stmt_update_keypoints_ = nullptr;

  // read_*
  sqlite3_stmt* sql_stmt_read_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_read_cameras_ = nullptr;
  sqlite3_stmt* sql_stmt_read_camera_source_ = nullptr;
  sqlite3_stmt* sql_stmt_read_cameras_for_source_ = nullptr;
  sqlite3_stmt* sql_stmt_read_camera_sources_for_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_read_all_camera_sources_ = nullptr;
  sqlite3_stmt* sql_stmt_read_image_id_ = nullptr;
  sqlite3_stmt* sql_stmt_read_image_name_ = nullptr;
  sqlite3_stmt* sql_stmt_read_images_ = nullptr;
  sqlite3_stmt* sql_stmt_read_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_read_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_read_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_read_matches_all_ = nullptr;
  sqlite3_stmt* sql_stmt_read_num_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_read_two_view_geometry_ = nullptr;
  sqlite3_stmt* sql_stmt_read_two_view_geometries_ = nullptr;
  sqlite3_stmt* sql_stmt_read_two_view_geometry_num_inliers_ = nullptr;

  // write_*
  sqlite3_stmt* sql_stmt_write_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_write_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_write_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_write_two_view_geometry_ = nullptr;

  // delete_*
  sqlite3_stmt* sql_stmt_delete_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_delete_camera_source_ = nullptr;
  sqlite3_stmt* sql_stmt_delete_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_delete_two_view_geometry_ = nullptr;

  // clear_*
  sqlite3_stmt* sql_stmt_write_pose_prior_ = nullptr;
  sqlite3_stmt* sql_stmt_read_pose_prior_ = nullptr;
  sqlite3_stmt* sql_stmt_read_pose_priors_ = nullptr;
  sqlite3_stmt* sql_stmt_update_pose_prior_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_pose_prior_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_pose_priors_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_cameras_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_images_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_two_view_geometries_ = nullptr;
};


std::mutex SqliteDatabase::update_schema_mutex_;

std::shared_ptr<Database> SqliteDatabase::Open(
    const std::filesystem::path& path) {
  auto database = std::make_shared<SqliteDatabase>(path);
  database->OpenSqlite();
  return database;
}

void SqliteDatabase::OpenSqlite() {
  Close();

  // SQLITE_OPEN_NOMUTEX specifies that the connection should not have a
  // mutex (so that we don't serialize the connection's operations).
  // Modifications to the database will still be serialized, but multiple
  // connections can read concurrently.
  SQLITE3_CALL(sqlite3_open_v2(
      path_.string().c_str(), &database_,
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

  const bool fork_legacy_schema = PreMigrateTables();
  CreateTables();
  PostMigrateTables(fork_legacy_schema);
  PrepareSQLStatements();
}

void SqliteDatabase::Close() {
  if (database_ != nullptr) {
    FinalizeSQLStatements();
    SQLITE3_EXEC(database_, "VACUUM", nullptr);
    sqlite3_close_v2(database_);
    database_ = nullptr;
  }
}

bool SqliteDatabase::ExistsCamera(const camera_t camera_id) const {
  return ExistsRowId(sql_stmt_exists_camera_, camera_id);
}

// Upstream parity (f1ea91d0): checks whether a specific calibration source
// exists for the camera (BEST is a query selector, not a source).
bool SqliteDatabase::ExistsCameraSource(const camera_t camera_id,
                                        const CameraSource source) const {
  THROW_CHECK_NE(source, CameraSource::BEST)
          << "Camera source cannot be BEST when checking calibration "
             "existence.";
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_exists_camera_source_, 1, camera_id));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_exists_camera_source_,
                         2,
                         static_cast<sqlite3_int64>(source)));
  const bool exists =
      SQLITE3_CALL(sqlite3_step(sql_stmt_exists_camera_source_)) == SQLITE_ROW;
  SQLITE3_CALL(sqlite3_reset(sql_stmt_exists_camera_source_));
  return exists;
}

bool SqliteDatabase::ExistsRig(const rig_t rig_id) const {
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT 1 FROM rigs WHERE rig_id=?;",
                                  -1, &stmt, nullptr));
  const bool exists = ExistsRowId(stmt, rig_id);
  SQLITE3_CALL(sqlite3_finalize(stmt));
  return exists;
}

bool SqliteDatabase::ExistsFrame(const frame_t frame_id) const {
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT 1 FROM frames WHERE frame_id=?;",
                                  -1, &stmt, nullptr));
  const bool exists = ExistsRowId(stmt, frame_id);
  SQLITE3_CALL(sqlite3_finalize(stmt));
  return exists;
}

bool SqliteDatabase::ExistsImage(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_image_id_, image_id);
}

bool SqliteDatabase::ExistsImageWithName(std::string name) const {
  return ExistsRowString(sql_stmt_exists_image_name_, name);
}

bool SqliteDatabase::ExistsKeypoints(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_keypoints_, image_id);
}

bool SqliteDatabase::ExistsDescriptors(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_descriptors_, image_id);
}

bool SqliteDatabase::ExistsMatches(const image_t image_id1,
                             const image_t image_id2) const {
  return ExistsRowId(sql_stmt_exists_matches_,
                     ImagePairToPairId(image_id1, image_id2));
}

bool SqliteDatabase::ExistsInlierMatches(const image_t image_id1,
                                   const image_t image_id2) const {
  // Upstream parity (d3ccaf35): the entry must exist AND carry a non-empty
  // inlier array (the fork previously aliased this to row existence).
  if (!ExistsRowId(sql_stmt_exists_two_view_geometry_,
                   ImagePairToPairId(image_id1, image_id2))) {
    return false;
  }
  return !ReadTwoViewGeometry(image_id1, image_id2).inlier_matches.empty();
}

bool SqliteDatabase::ExistsTwoViewGeometry(const image_t image_id1,
                                     const image_t image_id2) const {
  return ExistsRowId(sql_stmt_exists_two_view_geometry_,
                     ImagePairToPairId(image_id1, image_id2));
}

// Upstream parity (f1ea91d0): multiple calibrations per camera; the count
// is the number of distinct cameras.
size_t SqliteDatabase::NumCameras() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_num_cameras_));
  const size_t count = static_cast<size_t>(
      sqlite3_column_int64(sql_stmt_num_cameras_, 0));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_num_cameras_));
  return count;
}
size_t SqliteDatabase::NumRigs() const { return CountRows("rigs"); }
size_t SqliteDatabase::NumFrames() const { return CountRows("frames"); }

size_t SqliteDatabase::NumImages() const { return CountRows("images"); }

size_t SqliteDatabase::NumKeypoints() const { return SumColumn("rows", "keypoints"); }

size_t SqliteDatabase::MaxNumKeypoints() const {
  return MaxColumn("rows", "keypoints");
}

size_t SqliteDatabase::NumKeypointsForImage(const image_t image_id) const {
  return CountRowsForEntry(sql_stmt_num_keypoints_, image_id);
}

size_t SqliteDatabase::NumDescriptors() const {
  return SumColumn("rows", "descriptors");
}

size_t SqliteDatabase::MaxNumDescriptors() const {
  return MaxColumn("rows", "descriptors");
}

size_t SqliteDatabase::NumDescriptorsForImage(const image_t image_id) const {
  return CountRowsForEntry(sql_stmt_num_descriptors_, image_id);
}

size_t SqliteDatabase::NumMatches() const { return SumColumn("rows", "matches"); }

size_t SqliteDatabase::NumInlierMatches() const {
  return SumColumn("rows", "two_view_geometries");
}

size_t SqliteDatabase::NumMatchedImagePairs() const { return CountRows("matches"); }

size_t SqliteDatabase::NumVerifiedImagePairs() const {
  return CountRows("two_view_geometries");
}

// Upstream parity (f1ea91d0): `source` defaults to BEST, which selects the
// highest-priority calibration for the camera via the source-priority SQL
// ordering; an explicit source reads that exact calibration row.
Camera SqliteDatabase::ReadCamera(
    const camera_t camera_id, const CameraSource source) const {
  Camera camera;
  if (source == CameraSource::BEST) {
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_camera_, 1, camera_id));
    if (SQLITE3_CALL(sqlite3_step(sql_stmt_read_camera_)) == SQLITE_ROW) {
      camera = ReadCameraRow(sql_stmt_read_camera_);
    }
    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_camera_));
    return camera;
  }

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_camera_source_,
                                  1,
                                  camera_id));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_read_camera_source_,
                         2,
                         static_cast<sqlite3_int64>(source)));
  if (SQLITE3_CALL(sqlite3_step(sql_stmt_read_camera_source_)) == SQLITE_ROW) {
    camera = ReadCameraRow(sql_stmt_read_camera_source_);
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_camera_source_));
  return camera;
}

std::vector<Camera> SqliteDatabase::ReadAllCameras(
    const CameraSource source) const {
  std::vector<Camera> cameras;
  if (source == CameraSource::BEST) {
    while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_cameras_)) == SQLITE_ROW) {
      cameras.push_back(ReadCameraRow(sql_stmt_read_cameras_));
    }
    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_cameras_));
    return cameras;
  }

  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_read_cameras_for_source_,
                         1,
                         static_cast<sqlite3_int64>(source)));
  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_cameras_for_source_)) ==
         SQLITE_ROW) {
    cameras.push_back(ReadCameraRow(sql_stmt_read_cameras_for_source_));
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_cameras_for_source_));
  return cameras;
}

std::map<CameraSource, Camera> SqliteDatabase::ReadAllCameraSources(
    const camera_t camera_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_camera_sources_for_camera_,
                                  1,
                                  camera_id));
  std::map<CameraSource, Camera> calibrations;
  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_camera_sources_for_camera_)) ==
         SQLITE_ROW) {
    Camera camera = ReadCameraRow(sql_stmt_read_camera_sources_for_camera_);
    calibrations.emplace(camera.Source(), std::move(camera));
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_camera_sources_for_camera_));
  return calibrations;
}

NodeHashMap<camera_t, std::map<CameraSource, Camera>>
SqliteDatabase::ReadAllCameraSources() const {
  NodeHashMap<camera_t, std::map<CameraSource, Camera>> all_calibrations;
  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_all_camera_sources_)) ==
         SQLITE_ROW) {
    Camera camera = ReadCameraRow(sql_stmt_read_all_camera_sources_);
    all_calibrations[camera.CameraId()].emplace(camera.Source(),
                                                std::move(camera));
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_all_camera_sources_));
  return all_calibrations;
}

std::optional<Rig> SqliteDatabase::ReadRigWithSensor(
    const sensor_t sensor_id) const {
  // Non-reference sensor match.
  {
    sqlite3_stmt* stmt = nullptr;
    SQLITE3_CALL(sqlite3_prepare_v2(
        database_,
        "SELECT rig_id FROM rig_sensors WHERE sensor_id=? AND sensor_type=? "
        "LIMIT 1;",
        -1, &stmt, nullptr));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 2,
                                    static_cast<int>(sensor_id.type)));
    std::optional<rig_t> rig_id;
    if (SQLITE3_CALL(sqlite3_step(stmt)) == SQLITE_ROW) {
      rig_id = static_cast<rig_t>(sqlite3_column_int64(stmt, 0));
    }
    SQLITE3_CALL(sqlite3_finalize(stmt));
    if (rig_id.has_value()) {
      return ReadRig(*rig_id);
    }
  }
  // Reference sensor match.
  {
    sqlite3_stmt* stmt = nullptr;
    SQLITE3_CALL(sqlite3_prepare_v2(
        database_,
        "SELECT rig_id FROM rigs WHERE ref_sensor_id=? AND ref_sensor_type=? "
        "LIMIT 1;",
        -1, &stmt, nullptr));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 2,
                                    static_cast<int>(sensor_id.type)));
    std::optional<rig_t> rig_id;
    if (SQLITE3_CALL(sqlite3_step(stmt)) == SQLITE_ROW) {
      rig_id = static_cast<rig_t>(sqlite3_column_int64(stmt, 0));
    }
    SQLITE3_CALL(sqlite3_finalize(stmt));
    if (rig_id.has_value()) {
      return ReadRig(*rig_id);
    }
  }
  return std::nullopt;
}

Rig SqliteDatabase::ReadRig(const rig_t rig_id) const {
  sqlite3_stmt* ref_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(
      database_, "SELECT ref_sensor_id, ref_sensor_type FROM rigs "
      "WHERE rig_id=?;", -1, &ref_stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(ref_stmt, 1, rig_id));
  Rig rig;
  if (SQLITE3_CALL(sqlite3_step(ref_stmt)) != SQLITE_ROW) {
    SQLITE3_CALL(sqlite3_finalize(ref_stmt));
    return rig;
  }
  const sensor_t ref_sensor_id(
      static_cast<SensorType>(sqlite3_column_int(ref_stmt, 1)),
      static_cast<uint32_t>(sqlite3_column_int64(ref_stmt, 0)));
  SQLITE3_CALL(sqlite3_finalize(ref_stmt));
  rig.SetRigId(rig_id);
  rig.AddRefSensor(ref_sensor_id);
  // Upstream parity (a2bd15c2): rig_sensors only holds the non-reference
  // sensors; a NULL sensor_from_rig is an unknown extrinsic.
  sqlite3_stmt* sensor_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(
      database_, "SELECT sensor_id, sensor_type, sensor_from_rig FROM "
      "rig_sensors WHERE rig_id=? ORDER BY sensor_type, sensor_id;", -1,
      &sensor_stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(sensor_stmt, 1, rig_id));
  while (SQLITE3_CALL(sqlite3_step(sensor_stmt)) == SQLITE_ROW) {
    const sensor_t sensor_id(
        static_cast<SensorType>(sqlite3_column_int(sensor_stmt, 1)),
        static_cast<uint32_t>(sqlite3_column_int64(sensor_stmt, 0)));
    if (rig.HasSensor(sensor_id)) continue;
    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    if (SensorFromRigFromBlob(sensor_stmt, 2, &qvec, &tvec)) {
      rig.AddSensor(sensor_id, qvec, tvec);
    } else {
      rig.AddSensor(sensor_id, std::nullopt, std::nullopt);
    }
  }
  SQLITE3_CALL(sqlite3_finalize(sensor_stmt));
  return rig;
}

std::vector<Rig> SqliteDatabase::ReadAllRigs() const {
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

Frame SqliteDatabase::ReadFrame(const frame_t frame_id) const {
  // Upstream parity (a2bd15c2): frames persist only the rig association; the
  // pose is a reconstruction-level estimate and never stored.
  sqlite3_stmt* frame_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT rig_id FROM frames "
                                  "WHERE frame_id=?;", -1, &frame_stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(frame_stmt, 1, frame_id));
  Frame frame;
  if (SQLITE3_CALL(sqlite3_step(frame_stmt)) == SQLITE_ROW) {
    frame.SetFrameId(frame_id);
    frame.SetRigId(static_cast<rig_t>(sqlite3_column_int64(frame_stmt, 0)));
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
  return frame;
}

std::vector<Frame> SqliteDatabase::ReadAllFrames() const {
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

Image SqliteDatabase::ReadImage(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_image_id_, 1, image_id));

  Image image;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_id_));
  if (rc == SQLITE_ROW) {
    image = ReadImageRow(sql_stmt_read_image_id_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_id_));

  return image;
}

Image SqliteDatabase::ReadImageWithName(const std::string& name) const {
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

std::vector<Image> SqliteDatabase::ReadAllImages() const {
  std::vector<Image> images;
  images.reserve(NumImages());

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_images_)) == SQLITE_ROW) {
    images.push_back(ReadImageRow(sql_stmt_read_images_));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_images_));

  return images;
}

FeatureKeypoints SqliteDatabase::ReadKeypoints(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_));
  const FeatureKeypointsBlob blob = ReadDynamicMatrixBlob<FeatureKeypointsBlob>(
      sql_stmt_read_keypoints_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_));

  return FeatureKeypointsFromBlob(blob);
}

FeatureDescriptors SqliteDatabase::ReadDescriptors(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_descriptors_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_descriptors_));
  FeatureDescriptors descriptors;
  descriptors.data = ReadDynamicMatrixBlob<FeatureDescriptorsData>(
          sql_stmt_read_descriptors_, rc, 0);
  if (rc == SQLITE_ROW) {
    descriptors.type = static_cast<FeatureExtractorType>(
            sqlite3_column_int(sql_stmt_read_descriptors_, 3));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_descriptors_));
  return descriptors;
}

FeatureMatches SqliteDatabase::ReadMatches(image_t image_id1,
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

std::vector<std::pair<image_pair_t, FeatureMatches>> SqliteDatabase::ReadAllMatches()
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

std::vector<std::pair<image_pair_t, int>> SqliteDatabase::ReadNumMatches()
    const {
  std::vector<std::pair<image_pair_t, int>> num_matches;

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_num_matches_)) ==
         SQLITE_ROW) {
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_num_matches_, 0));
    const int rows =
        static_cast<int>(sqlite3_column_int64(sql_stmt_read_num_matches_, 1));
    num_matches.emplace_back(pair_id, rows);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_num_matches_));

  return num_matches;
}

// Serialize a camera to a little-endian binary blob for storage in a single
// BLOB column (upstream parity d3ccaf35; adapted to the fork's getter/setter
// Camera API).
std::string CameraToBlob(const Camera& camera) {
  std::ostringstream stream(std::ios::binary);
  WriteBinaryLittleEndian<camera_t>(&stream, camera.CameraId());
  WriteBinaryLittleEndian<int>(&stream, static_cast<int>(camera.ModelId()));
  WriteBinaryLittleEndian<uint64_t>(&stream, camera.Width());
  WriteBinaryLittleEndian<uint64_t>(&stream, camera.Height());
  WriteBinaryLittleEndian<uint8_t>(&stream,
                                   camera.HasPriorFocalLength() ? 1 : 0);
  WriteBinaryLittleEndian<uint64_t>(&stream, camera.Params().size());
  for (const double param : camera.Params()) {
    WriteBinaryLittleEndian<double>(&stream, param);
  }
  // Upstream parity (f1ea91d0): persist the calibration source after the
  // parameters so legacy blobs (without the trailing byte) stay readable.
  WriteBinaryLittleEndian<int8_t>(&stream,
                                  static_cast<int8_t>(camera.Source()));
  return stream.str();
}

Camera CameraFromBlob(const void* data, const size_t num_bytes) {
  const std::string bytes(static_cast<const char*>(data), num_bytes);
  std::istringstream stream(bytes, std::ios::binary);
  Camera camera;
  camera.SetCameraId(ReadBinaryLittleEndian<camera_t>(&stream));
  camera.SetModelId(
      static_cast<CameraModelId>(ReadBinaryLittleEndian<int>(&stream)));
  camera.SetWidth(ReadBinaryLittleEndian<uint64_t>(&stream));
  camera.SetHeight(ReadBinaryLittleEndian<uint64_t>(&stream));
  const bool has_prior = ReadBinaryLittleEndian<uint8_t>(&stream) != 0;
  const uint64_t num_params = ReadBinaryLittleEndian<uint64_t>(&stream);
  camera.Params().resize(num_params);
  ReadBinaryLittleEndian<double>(&stream, &camera.Params());
  // Upstream parity (f1ea91d0): legacy blobs without the trailing source
  // byte fall back to the prior flag (EXIF/GUESS).
  if (stream.peek() != EOF) {
    camera.SetSource(static_cast<CameraSource>(
        ReadBinaryLittleEndian<int8_t>(&stream)));
  } else {
    camera.SetSource(has_prior ? CameraSource::EXIF : CameraSource::GUESS);
  }
  return camera;
}

// Bind an optional camera to a statement column, writing NULL when unset. The
// returned blob must outlive the sqlite3_step call (bound with SQLITE_STATIC).
std::string WriteOptionalCamera(sqlite3_stmt* sql_stmt,
                                const std::optional<Camera>& camera,
                                const int col) {
  if (!camera.has_value()) {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt, col));
    return {};
  }
  std::string blob = CameraToBlob(*camera);
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt,
                                 col,
                                 blob.data(),
                                 static_cast<int>(blob.size()),
                                 SQLITE_STATIC));
  return blob;
}

std::optional<Camera> ReadOptionalCamera(sqlite3_stmt* sql_stmt,
                                         const int col) {
  if (sqlite3_column_type(sql_stmt, col) == SQLITE_NULL) {
    return std::nullopt;
  }
  return CameraFromBlob(
      sqlite3_column_blob(sql_stmt, col),
      static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col)));
}

TwoViewGeometry SqliteDatabase::ReadTwoViewGeometry(const image_t image_id1,
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

  // Upstream parity (a2bd15c2): NULL means not set.
  if (sqlite3_column_type(sql_stmt_read_two_view_geometry_, 4) !=
      SQLITE_NULL) {
    two_view_geometry.F = ReadStaticMatrixBlob<Eigen::Matrix3d>(
        sql_stmt_read_two_view_geometry_, rc, 4);
  }
  if (sqlite3_column_type(sql_stmt_read_two_view_geometry_, 5) !=
      SQLITE_NULL) {
    two_view_geometry.E = ReadStaticMatrixBlob<Eigen::Matrix3d>(
        sql_stmt_read_two_view_geometry_, rc, 5);
  }
  if (sqlite3_column_type(sql_stmt_read_two_view_geometry_, 6) !=
      SQLITE_NULL) {
    two_view_geometry.H = ReadStaticMatrixBlob<Eigen::Matrix3d>(
        sql_stmt_read_two_view_geometry_, rc, 6);
  }
  const bool has_qvec =
      sqlite3_column_type(sql_stmt_read_two_view_geometry_, 7) != SQLITE_NULL;
  const bool has_tvec =
      sqlite3_column_type(sql_stmt_read_two_view_geometry_, 8) != SQLITE_NULL;
  THROW_CHECK_EQ(has_qvec, has_tvec)
      << "qvec and tvec must both be NULL or both be non-NULL";
  if (has_qvec) {
    const Eigen::Vector4d quat_wxyz = ReadStaticMatrixBlob<Eigen::Vector4d>(
        sql_stmt_read_two_view_geometry_, rc, 7);
    two_view_geometry.cam2_from_cam1 =
        Rigid3d(Eigen::Quaterniond(quat_wxyz(0), quat_wxyz(1), quat_wxyz(2),
                                   quat_wxyz(3)),
                ReadStaticMatrixBlob<Eigen::Vector3d>(
                    sql_stmt_read_two_view_geometry_, rc, 8));
  }

  // Read the estimated cameras if present (NULL means not set).
  two_view_geometry.camera1 =
      ReadOptionalCamera(sql_stmt_read_two_view_geometry_, 9);
  two_view_geometry.camera2 =
      ReadOptionalCamera(sql_stmt_read_two_view_geometry_, 10);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_two_view_geometry_));

  two_view_geometry.inlier_matches = FeatureMatchesFromBlob(blob);
  // The write path stores the transposed matrix so that its column-major
  // memory equals the row-major original; undo that transpose here.
  if (two_view_geometry.F) {
    two_view_geometry.F->transposeInPlace();
  }
  if (two_view_geometry.E) {
    two_view_geometry.E->transposeInPlace();
  }
  if (two_view_geometry.H) {
    two_view_geometry.H->transposeInPlace();
  }

  if (SwapImagePair(image_id1, image_id2)) {
    two_view_geometry.Invert();
  }

  return two_view_geometry;
}

void SqliteDatabase::ReadTwoViewGeometries(
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

    // Upstream parity (a2bd15c2): NULL means not set.
    if (sqlite3_column_type(sql_stmt_read_two_view_geometries_, 5) !=
        SQLITE_NULL) {
      two_view_geometry.F = ReadStaticMatrixBlob<Eigen::Matrix3d>(
          sql_stmt_read_two_view_geometries_, rc, 5);
      two_view_geometry.F->transposeInPlace();
    }
    if (sqlite3_column_type(sql_stmt_read_two_view_geometries_, 6) !=
        SQLITE_NULL) {
      two_view_geometry.E = ReadStaticMatrixBlob<Eigen::Matrix3d>(
          sql_stmt_read_two_view_geometries_, rc, 6);
      two_view_geometry.E->transposeInPlace();
    }
    if (sqlite3_column_type(sql_stmt_read_two_view_geometries_, 7) !=
        SQLITE_NULL) {
      two_view_geometry.H = ReadStaticMatrixBlob<Eigen::Matrix3d>(
          sql_stmt_read_two_view_geometries_, rc, 7);
      two_view_geometry.H->transposeInPlace();
    }
    const bool has_qvec =
        sqlite3_column_type(sql_stmt_read_two_view_geometries_, 8) !=
        SQLITE_NULL;
    const bool has_tvec =
        sqlite3_column_type(sql_stmt_read_two_view_geometries_, 9) !=
        SQLITE_NULL;
    THROW_CHECK_EQ(has_qvec, has_tvec)
        << "qvec and tvec must both be NULL or both be non-NULL";
    if (has_qvec) {
      const Eigen::Vector4d quat_wxyz = ReadStaticMatrixBlob<Eigen::Vector4d>(
          sql_stmt_read_two_view_geometries_, rc, 8);
      two_view_geometry.cam2_from_cam1 =
          Rigid3d(Eigen::Quaterniond(quat_wxyz(0), quat_wxyz(1), quat_wxyz(2),
                                     quat_wxyz(3)),
                  ReadStaticMatrixBlob<Eigen::Vector3d>(
                      sql_stmt_read_two_view_geometries_, rc, 9));
    }
    two_view_geometry.camera1 =
        ReadOptionalCamera(sql_stmt_read_two_view_geometries_, 10);
    two_view_geometry.camera2 =
        ReadOptionalCamera(sql_stmt_read_two_view_geometries_, 11);
    two_view_geometries->push_back(two_view_geometry);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_two_view_geometries_));
}

std::map<image_pair_t, TwoViewGeometry> SqliteDatabase::ReadTwoViewGeometries()
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

void SqliteDatabase::ReadTwoViewGeometryNumInliers(
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

camera_t SqliteDatabase::WriteCamera(const Camera& camera,
                               const bool use_camera_id) const {
  // Upstream parity (f1ea91d0): BEST is a query selector, not a persisted
  // source; with auto-assignment the next id comes from the covering
  // primary-key B-tree (composite key has no AUTOINCREMENT).
  THROW_CHECK_NE(camera.Source(), CameraSource::BEST)
          << "Camera source cannot be BEST when writing to database.";
  camera_t camera_id = camera.CameraId();
  if (use_camera_id) {
    CHECK(!ExistsCamera(camera_id)) << "camera_id must be unique";
  } else {
    SQLITE3_CALL(sqlite3_step(sql_stmt_next_camera_id_));
    camera_id = static_cast<camera_t>(
        sqlite3_column_int64(sql_stmt_next_camera_id_, 0));
    SQLITE3_CALL(sqlite3_reset(sql_stmt_next_camera_id_));
  }

  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_add_camera_, 1, camera_id));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 2,
                                static_cast<int>(camera.ModelId())));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 3,
                                  static_cast<sqlite3_int64>(camera.Width())));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 4,
                                  static_cast<sqlite3_int64>(camera.Height())));

  const size_t num_params_bytes = sizeof(double) * camera.NumParams();
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt_add_camera_, 5, camera.ParamsData(),
                                 static_cast<int>(num_params_bytes),
                                 SQLITE_STATIC));

  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_add_camera_, 6,
                         static_cast<sqlite3_int64>(camera.Source())));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_camera_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_camera_));

  return camera_id;
}

namespace {

// Upstream WriteRigSensors (a2bd15c2): persists the non-reference sensors
// only; unknown extrinsics are stored as NULL.
void WriteRigSensors(sqlite3* database, const rig_t rig_id, const Rig& rig) {
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(
      database, "INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type, "
      "sensor_from_rig) VALUES(?, ?, ?, ?);", -1, &stmt, nullptr));
  for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, rig_id));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 2, sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int(stmt, 3, static_cast<int>(sensor_id.type)));
    // The blob must outlive sqlite3_step (bound with SQLITE_STATIC).
    std::string blob;
    if (rig.HasSensorFromRig(sensor_id)) {
      blob = SensorFromRigToBlob(rig.SensorFromRigQvec(sensor_id),
                                 rig.SensorFromRigTvec(sensor_id));
      SQLITE3_CALL(sqlite3_bind_blob(stmt, 4, blob.data(),
                                     static_cast<int>(blob.size()),
                                     SQLITE_STATIC));
    } else {
      SQLITE3_CALL(sqlite3_bind_null(stmt, 4));
    }
    SQLITE3_CALL(sqlite3_step(stmt));
    SQLITE3_CALL(sqlite3_reset(stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(stmt));
}

// Upstream WriteFrameData (a2bd15c2).
void WriteFrameData(sqlite3* database, const frame_t frame_id,
                    const Frame& frame) {
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(
      database, "INSERT INTO frame_data(frame_id, data_id, sensor_id, "
      "sensor_type) VALUES(?, ?, ?, ?);", -1, &stmt, nullptr));
  for (const data_t& data_id : frame.DataIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, frame_id));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 2, data_id.id));
    SQLITE3_CALL(sqlite3_bind_int64(stmt, 3, data_id.sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int(stmt, 4,
                                  static_cast<int>(data_id.sensor_id.type)));
    SQLITE3_CALL(sqlite3_step(stmt));
    SQLITE3_CALL(sqlite3_reset(stmt));
  }
  SQLITE3_CALL(sqlite3_finalize(stmt));
}

}  // namespace

rig_t SqliteDatabase::WriteRig(const Rig& rig, const bool use_rig_id) const {
  CHECK_GT(rig.NumSensors(), 0);
  sqlite3_stmt* rig_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) "
                                  "VALUES(?, ?, ?);", -1, &rig_stmt, nullptr));
  if (use_rig_id) {
    CHECK(!ExistsRig(rig.RigId()));
    SQLITE3_CALL(sqlite3_bind_int64(rig_stmt, 1, rig.RigId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(rig_stmt, 1));
  }
  SQLITE3_CALL(sqlite3_bind_int64(rig_stmt, 2, rig.RefSensorId().id));
  SQLITE3_CALL(sqlite3_bind_int(rig_stmt, 3, static_cast<int>(rig.RefSensorId().type)));
  SQLITE3_CALL(sqlite3_step(rig_stmt));
  SQLITE3_CALL(sqlite3_finalize(rig_stmt));
  const rig_t rig_id = static_cast<rig_t>(sqlite3_last_insert_rowid(database_));
  WriteRigSensors(database_, rig_id, rig);
  return rig_id;
}

frame_t SqliteDatabase::WriteFrame(const Frame& frame, const bool use_frame_id) const {
  CHECK(ExistsRig(frame.RigId()));
  sqlite3_stmt* frame_stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "INSERT INTO frames(frame_id, rig_id) "
                                  "VALUES(?, ?);", -1, &frame_stmt, nullptr));
  if (use_frame_id) {
    CHECK(!ExistsFrame(frame.FrameId()));
    SQLITE3_CALL(sqlite3_bind_int64(frame_stmt, 1, frame.FrameId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(frame_stmt, 1));
  }
  SQLITE3_CALL(sqlite3_bind_int64(frame_stmt, 2, frame.RigId()));
  SQLITE3_CALL(sqlite3_step(frame_stmt));
  SQLITE3_CALL(sqlite3_finalize(frame_stmt));
  const frame_t frame_id = static_cast<frame_t>(sqlite3_last_insert_rowid(database_));
  WriteFrameData(database_, frame_id, frame);
  return frame_id;
}

image_t SqliteDatabase::WriteImage(const Image& image,
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

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_image_));

  return static_cast<image_t>(sqlite3_last_insert_rowid(database_));
}

void SqliteDatabase::WriteKeypoints(const image_t image_id,
                              const FeatureKeypoints& keypoints) const {
  const FeatureKeypointsBlob blob = FeatureKeypointsToBlob(keypoints);

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_keypoints_, 1, image_id));
  WriteDynamicMatrixBlob(sql_stmt_write_keypoints_, blob, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_keypoints_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_keypoints_));
}

void SqliteDatabase::WriteDescriptors(const image_t image_id,
                                const FeatureDescriptors& descriptors) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_descriptors_, 1, image_id));
  WriteDynamicMatrixBlob(sql_stmt_write_descriptors_, descriptors.data, 2);
  SQLITE3_CALL(sqlite3_bind_int(sql_stmt_write_descriptors_, 5,
                                static_cast<int>(descriptors.type)));

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_descriptors_));
}

void SqliteDatabase::WriteMatches(const image_t image_id1, const image_t image_id2,
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

void SqliteDatabase::WriteTwoViewGeometry(
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

  // Upstream parity (a2bd15c2): write NULL for F/E/H/pose if not set,
  // otherwise transpose the matrices to obtain row-major data layout.
  // Important: the bound objects must live until `sqlite3_step` is called.
  Eigen::Matrix3d Ft, Et, Ht;
  if (two_view_geometry_ptr->F.has_value()) {
    Ft = two_view_geometry_ptr->F->transpose();
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Ft, 6);
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_two_view_geometry_, 6));
  }
  if (two_view_geometry_ptr->E.has_value()) {
    Et = two_view_geometry_ptr->E->transpose();
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Et, 7);
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_two_view_geometry_, 7));
  }
  if (two_view_geometry_ptr->H.has_value()) {
    Ht = two_view_geometry_ptr->H->transpose();
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Ht, 8);
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_two_view_geometry_, 8));
  }
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  if (two_view_geometry_ptr->cam2_from_cam1.has_value()) {
    const Eigen::Quaterniond& q =
        two_view_geometry_ptr->cam2_from_cam1->rotation();
    qvec << q.w(), q.x(), q.y(), q.z();
    tvec = two_view_geometry_ptr->cam2_from_cam1->translation();
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, qvec, 9);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, tvec, 10);
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_two_view_geometry_, 9));
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_two_view_geometry_, 10));
  }

  // Write NULL for camera1/camera2 if not set.
  const std::string camera1_blob =
      WriteOptionalCamera(sql_stmt_write_two_view_geometry_,
                          two_view_geometry_ptr->camera1,
                          11);
  const std::string camera2_blob =
      WriteOptionalCamera(sql_stmt_write_two_view_geometry_,
                          two_view_geometry_ptr->camera2,
                          12);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_two_view_geometry_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_two_view_geometry_));
}

// Upstream parity (f1ea91d0): insert-or-overwrite a calibration source for
// an existing camera. UNKNOWN updates the currently active (BEST) source in
// place; BEST is rejected.
void SqliteDatabase::UpdateCamera(const Camera& camera) const {
  CHECK(ExistsCamera(camera.CameraId())) << "Camera does not exist";

  Camera camera_to_update = camera;
  THROW_CHECK_NE(camera_to_update.Source(), CameraSource::BEST)
          << "Camera source cannot be BEST when updating database.";
  if (camera_to_update.Source() == CameraSource::UNKNOWN) {
    const Camera active_camera =
        ReadCamera(camera.CameraId(), CameraSource::BEST);
    camera_to_update.SetSource(active_camera.Source());
  }

  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 1,
                       camera_to_update.CameraId()));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 2,
                       static_cast<int>(camera_to_update.ModelId())));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 3,
                       static_cast<sqlite3_int64>(camera_to_update.Width())));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 4,
                       static_cast<sqlite3_int64>(camera_to_update.Height())));

  const size_t num_params_bytes =
      sizeof(double) * camera_to_update.NumParams();
  SQLITE3_CALL(
      sqlite3_bind_blob(sql_stmt_update_camera_, 5,
                        camera_to_update.ParamsData(),
                        static_cast<int>(num_params_bytes), SQLITE_STATIC));

  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 6,
                       static_cast<sqlite3_int64>(camera_to_update.Source())));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_camera_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_camera_));
}

// Upstream parity (f1ea91d0): delete a specific calibration source; BEST
// (the default) deletes all calibrations of the camera.
void SqliteDatabase::DeleteCameraSource(const camera_t camera_id,
                                        const CameraSource source) const {
  if (source == CameraSource::BEST) {
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_camera_, 1, camera_id));
    SQLITE3_CALL(sqlite3_step(sql_stmt_delete_camera_));
    SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_camera_));
    return;
  }
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_delete_camera_source_, 1, camera_id));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_delete_camera_source_,
                         2,
                         static_cast<sqlite3_int64>(source)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_camera_source_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_camera_source_));
}

void SqliteDatabase::UpdateRig(const Rig& rig) const {
  CHECK(ExistsRig(rig.RigId()));
  SQLITE3_EXEC(database_, StringPrintf("DELETE FROM rig_sensors WHERE rig_id=%u;", rig.RigId()).c_str(), nullptr);
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "UPDATE rigs SET ref_sensor_id=?, ref_sensor_type=? WHERE rig_id=?;", -1, &stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, rig.RefSensorId().id));
  SQLITE3_CALL(sqlite3_bind_int(stmt, 2, static_cast<int>(rig.RefSensorId().type)));
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 3, rig.RigId()));
  SQLITE3_CALL(sqlite3_step(stmt));
  SQLITE3_CALL(sqlite3_finalize(stmt));
  WriteRigSensors(database_, rig.RigId(), rig);
}

void SqliteDatabase::UpdateFrame(const Frame& frame) const {
  CHECK(ExistsFrame(frame.FrameId()));
  sqlite3_stmt* stmt = nullptr;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "UPDATE frames SET rig_id=? WHERE frame_id=?;", -1, &stmt, nullptr));
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 1, frame.RigId()));
  SQLITE3_CALL(sqlite3_bind_int64(stmt, 2, frame.FrameId()));
  SQLITE3_CALL(sqlite3_step(stmt));
  SQLITE3_CALL(sqlite3_finalize(stmt));
  SQLITE3_EXEC(database_, StringPrintf("DELETE FROM frame_data WHERE frame_id=%u;", frame.FrameId()).c_str(), nullptr);
  WriteFrameData(database_, frame.FrameId(), frame);
}

void SqliteDatabase::UpdateImage(const Image& image) const {
  SQLITE3_CALL(
      sqlite3_bind_text(sql_stmt_update_image_, 1, image.Name().c_str(),
                        static_cast<int>(image.Name().size()), SQLITE_STATIC));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_image_, 2, image.CameraId()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_image_, 3, image.ImageId()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_image_));
}

void SqliteDatabase::UpdateKeypoints(const image_t image_id,
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

// Upstream COLMAP dbb41680 API: update an existing two view geometry. The
// fork implements it as delete + write on the pair_id primary key, reusing
// the write path's swap-inversion and legacy blob bridging.
void SqliteDatabase::UpdateTwoViewGeometry(
        const image_t image_id1, const image_t image_id2,
        const TwoViewGeometry& two_view_geometry) const {
  // Full-row replacement: deleting via DeleteInlierMatches only clears the
  // inlier array (upstream semantics) and leaves the row in place, which
  // would trip the pair_id uniqueness on the write below.
  DeleteTwoViewGeometry(image_id1, image_id2);
  WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
}

void SqliteDatabase::DeleteMatches(const image_t image_id1,
                             const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_matches_, 1,
                                  static_cast<sqlite3_int64>(pair_id)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_matches_));
}

void SqliteDatabase::DeleteTwoViewGeometry(const image_t image_id1,
                                     const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_two_view_geometry_, 1,
                                  static_cast<sqlite3_int64>(pair_id)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_two_view_geometry_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_two_view_geometry_));
}

void SqliteDatabase::DeleteInlierMatches(const image_t image_id1,
                                   const image_t image_id2) const {
  // Upstream parity (d3ccaf35): clear the inlier array while keeping the
  // two-view geometry entry itself. A plain UPDATE avoids the fork's
  // delete+write UpdateTwoViewGeometry (which must not be re-entered) and
  // is unconditional - a missing row is a no-op, matching the upstream
  // Exists-guard semantics.
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  const std::string sql =
          "UPDATE two_view_geometries SET rows = 0, cols = 0, data = NULL"
          " WHERE pair_id = " +
          std::to_string(static_cast<long long>(pair_id)) + ";";
  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::ClearAllTables() const {
  ClearMatches();
  ClearTwoViewGeometries();
  ClearDescriptors();
  ClearKeypoints();
  ClearPosePriors();
  ClearFrames();
  ClearRigs();
  ClearImages();
  ClearCameras();
}

void SqliteDatabase::ClearCameras() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_cameras_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_cameras_));
}

void SqliteDatabase::ClearRigs() const {
  SQLITE3_EXEC(database_, "DELETE FROM rig_sensors; DELETE FROM rigs;", nullptr);
}

void SqliteDatabase::ClearFrames() const {
  SQLITE3_EXEC(database_, "DELETE FROM frame_data; DELETE FROM frames;", nullptr);
}

void SqliteDatabase::ClearImages() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_images_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_images_));
}

void SqliteDatabase::ClearDescriptors() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_descriptors_));
}

void SqliteDatabase::ClearKeypoints() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_keypoints_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_keypoints_));
}

void SqliteDatabase::ClearMatches() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_matches_));
}

void SqliteDatabase::ClearTwoViewGeometries() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_two_view_geometries_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_two_view_geometries_));
}
void SqliteDatabase::BeginTransaction() const {
  SQLITE3_EXEC(database_, "BEGIN TRANSACTION", nullptr);
}

void SqliteDatabase::EndTransaction() const {
  SQLITE3_EXEC(database_, "END TRANSACTION", nullptr);
}
void SqliteDatabase::PrepareSQLStatements() {
  sql_stmts_.clear();

  std::string sql;

  //////////////////////////////////////////////////////////////////////////////
  // num_*
  //////////////////////////////////////////////////////////////////////////////
  // Upstream parity (f1ea91d0): multiple calibrations per camera.
  sql = "SELECT COUNT(DISTINCT camera_id) FROM cameras;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_num_cameras_, 0));
  sql_stmts_.push_back(sql_stmt_num_cameras_);

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

  // Upstream parity (f1ea91d0).
  sql = "SELECT 1 FROM cameras WHERE camera_id = ? AND source = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_camera_source_, 0));
  sql_stmts_.push_back(sql_stmt_exists_camera_source_);

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
      "source) VALUES(?, ?, ?, ?, ?, ?);";
  SQLITE3_CALL(
      sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_camera_, 0));
  sql_stmts_.push_back(sql_stmt_add_camera_);

  // Upstream parity (f1ea91d0): composite primary key has no AUTOINCREMENT;
  // the next id comes from an explicit COALESCE(MAX)+1 allocation.
  sql = "SELECT COALESCE(MAX(camera_id), 0) + 1 FROM cameras;";
  SQLITE3_CALL(
      sqlite3_prepare_v2(database_, sql.c_str(), -1,
                         &sql_stmt_next_camera_id_, 0));
  sql_stmts_.push_back(sql_stmt_next_camera_id_);

  sql = "INSERT INTO images(image_id, name, camera_id) VALUES(?, ?, ?);";
  SQLITE3_CALL(
      sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_image_, 0));
  sql_stmts_.push_back(sql_stmt_add_image_);

  //////////////////////////////////////////////////////////////////////////////
  // update_*
  //////////////////////////////////////////////////////////////////////////////
  // Upstream parity (f1ea91d0): upsert of one calibration source.
  sql =
      "INSERT OR REPLACE INTO cameras(camera_id, model, width, height, "
      "params, source) VALUES(?, ?, ?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_camera_, 0));
  sql_stmts_.push_back(sql_stmt_update_camera_);

  sql = "UPDATE images SET name=?, camera_id=? WHERE image_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_image_, 0));
  sql_stmts_.push_back(sql_stmt_update_image_);

  //////////////////////////////////////////////////////////////////////////////
  // read_*
  //////////////////////////////////////////////////////////////////////////////
  // Upstream parity (f1ea91d0): source-priority ordering resolves BEST.
  // Relies on SQLite returning bare columns from the row matching the
  // single MAX() aggregate; do not add a second aggregate to this query.
  std::string camera_source_priority_sql = "CASE source";
  for (const CameraSource source :
       {CameraSource::UNKNOWN,
        CameraSource::USER,
        CameraSource::GUESS,
        CameraSource::EXIF,
        CameraSource::SINGLE_VIEW,
        CameraSource::VIEW_GRAPH}) {
    camera_source_priority_sql += StringPrintf(
        " WHEN %d THEN %d",
        static_cast<int>(source), CameraSourcePriority(source));
  }
  camera_source_priority_sql += " END";
  sql = StringPrintf(
      "SELECT camera_id, model, width, height, params, source, MAX(%s) "
      "FROM cameras GROUP BY camera_id ORDER BY camera_id ASC;",
      camera_source_priority_sql.c_str());
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_cameras_, 0));
  sql_stmts_.push_back(sql_stmt_read_cameras_);

  sql = StringPrintf(
      "SELECT camera_id, model, width, height, params, source "
      "FROM cameras WHERE camera_id = ? ORDER BY %s DESC LIMIT 1;",
      camera_source_priority_sql.c_str());
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_camera_, 0));
  sql_stmts_.push_back(sql_stmt_read_camera_);

  sql = "SELECT camera_id, model, width, height, params, source "
        "FROM cameras WHERE camera_id = ? AND source = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_camera_source_, 0));
  sql_stmts_.push_back(sql_stmt_read_camera_source_);

  sql = "SELECT camera_id, model, width, height, params, source "
        "FROM cameras WHERE source = ? ORDER BY camera_id ASC;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_cameras_for_source_, 0));
  sql_stmts_.push_back(sql_stmt_read_cameras_for_source_);

  sql = "SELECT camera_id, model, width, height, params, source "
        "FROM cameras WHERE camera_id = ? ORDER BY source ASC;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_camera_sources_for_camera_, 0));
  sql_stmts_.push_back(sql_stmt_read_camera_sources_for_camera_);

  sql = "SELECT camera_id, model, width, height, params, source "
        "FROM cameras ORDER BY camera_id ASC, source ASC;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_all_camera_sources_, 0));
  sql_stmts_.push_back(sql_stmt_read_all_camera_sources_);

  // Upstream parity (a2bd15c2): the image-to-frame association is resolved
  // at read time by joining frame_data on (data_id, sensor_type=CAMERA), not
  // stored in the images table.
  sql = "SELECT images.image_id, images.name, images.camera_id, "
        "frame_data.frame_id FROM images LEFT JOIN frame_data ON "
        "images.image_id = frame_data.data_id AND "
        "frame_data.sensor_type = 0 WHERE images.image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_image_id_, 0));
  sql_stmts_.push_back(sql_stmt_read_image_id_);

  sql = "SELECT images.image_id, images.name, images.camera_id, "
        "frame_data.frame_id FROM images LEFT JOIN frame_data ON "
        "images.image_id = frame_data.data_id AND "
        "frame_data.sensor_type = 0 WHERE images.name = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_image_name_, 0));
  sql_stmts_.push_back(sql_stmt_read_image_name_);

  sql = "SELECT images.image_id, images.name, images.camera_id, "
        "frame_data.frame_id FROM images LEFT JOIN frame_data ON "
        "images.image_id = frame_data.data_id AND "
        "frame_data.sensor_type = 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_images_, 0));
  sql_stmts_.push_back(sql_stmt_read_images_);

  sql = "SELECT rows, cols, data FROM keypoints WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_read_keypoints_);

  sql = "SELECT rows, cols, data, type FROM descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_read_descriptors_);

  sql = "SELECT rows, cols, data FROM matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_matches_, 0));
  sql_stmts_.push_back(sql_stmt_read_matches_);

  sql = "SELECT * FROM matches WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_matches_all_, 0));
  sql_stmts_.push_back(sql_stmt_read_matches_all_);

  sql = "SELECT pair_id, rows FROM matches WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_num_matches_, 0));
  sql_stmts_.push_back(sql_stmt_read_num_matches_);

  sql =
      "SELECT rows, cols, data, config, F, E, H, qvec, tvec, camera1, camera2 "
      "FROM two_view_geometries WHERE pair_id = ?;";
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

  // Column order matches ReadPosePriorRow below.
  sql = "SELECT pose_prior_id, corr_data_id, corr_sensor_id, "
        "corr_sensor_type, position, position_covariance, coordinate_system, "
        "gravity FROM pose_priors WHERE pose_prior_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_pose_prior_, 0));
  sql_stmts_.push_back(sql_stmt_read_pose_prior_);

  sql = "SELECT pose_prior_id, corr_data_id, corr_sensor_id, "
        "corr_sensor_type, position, position_covariance, coordinate_system, "
        "gravity FROM pose_priors;";
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

  sql = "UPDATE pose_priors SET corr_data_id=?, corr_sensor_id=?, "
        "corr_sensor_type=?, position=?, position_covariance=?, "
        "coordinate_system=?, gravity=? WHERE pose_prior_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_pose_prior_, 0));
  sql_stmts_.push_back(sql_stmt_update_pose_prior_);

  sql =
      "INSERT INTO descriptors(image_id, rows, cols, data, type) VALUES(?, ?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_write_descriptors_);

  sql = "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_matches_, 0));
  sql_stmts_.push_back(sql_stmt_write_matches_);

  sql = "INSERT INTO two_view_geometries(pair_id, rows, cols, data, config, F, "
      "E, H, qvec, tvec, camera1, camera2) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
      "?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_two_view_geometry_, 0));
  sql_stmts_.push_back(sql_stmt_write_two_view_geometry_);

  sql = "INSERT INTO pose_priors(pose_prior_id, corr_data_id, "
        "corr_sensor_id, corr_sensor_type, position, position_covariance, "
        "coordinate_system, gravity) VALUES(?, ?, ?, ?, ?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_pose_prior_, 0));
  sql_stmts_.push_back(sql_stmt_write_pose_prior_);

  //////////////////////////////////////////////////////////////////////////////
  // delete_*
  //////////////////////////////////////////////////////////////////////////////
  // Upstream parity (f1ea91d0): per-camera deletion; BEST deletes all
  // calibrations of the camera.
  sql = "DELETE FROM cameras WHERE camera_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_delete_camera_, 0));
  sql_stmts_.push_back(sql_stmt_delete_camera_);

  sql = "DELETE FROM cameras WHERE camera_id = ? AND source = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_delete_camera_source_, 0));
  sql_stmts_.push_back(sql_stmt_delete_camera_source_);

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

void SqliteDatabase::FinalizeSQLStatements() {
  for (const auto& sql_stmt : sql_stmts_) {
    SQLITE3_CALL(sqlite3_finalize(sql_stmt));
  }
}

void SqliteDatabase::CreateTables() const {
  CreateRigTable();
  CreateRigSensorsTable();
  CreateCameraTable();
  CreateFrameTable();
  CreateFrameDataTable();
  CreateImageTable();
  CreatePosePriorsTable();
  CreateKeypointsTable();
  CreateDescriptorsTable();
  CreateMatchesTable();
  CreateTwoViewGeometriesTable();
}

// The Create*Table statements below are verbatim upstream a2bd15c2 so that
// databases are interchangeable between the fork and upstream binaries.

void SqliteDatabase::CreateRigTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS rigs"
      "   (rig_id               INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    ref_sensor_id        INTEGER                             NOT NULL,"
      "    ref_sensor_type      INTEGER                             NOT "
      "NULL);"
      "CREATE UNIQUE INDEX IF NOT EXISTS rig_ref_sensor_assignment ON "
      "   rigs(ref_sensor_id, ref_sensor_type);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::CreateRigSensorsTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS rig_sensors"
      "   (rig_id               INTEGER                             NOT NULL,"
      "    sensor_id            INTEGER                             NOT NULL,"
      "    sensor_type          INTEGER                             NOT NULL,"
      "    sensor_from_rig      BLOB,"
      "FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);"
      "CREATE UNIQUE INDEX IF NOT EXISTS rig_sensor_assignment ON "
      "   rig_sensors(sensor_id, sensor_type);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::CreateCameraTable() const {
  // Upstream parity (f1ea91d0): composite primary key (camera_id, source);
  // no AUTOINCREMENT (SQLite limitation on composite keys), no
  // prior_focal_length column (subsumed by source).
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS cameras"
      "   (camera_id            INTEGER                             NOT NULL,"
      "    model                INTEGER                             NOT NULL,"
      "    width                INTEGER                             NOT NULL,"
      "    height               INTEGER                             NOT NULL,"
      "    params               BLOB,"
      "    source               INTEGER                             NOT NULL,"
      "    PRIMARY KEY(camera_id, source));";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::CreateFrameTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS frames"
      "   (frame_id             INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    rig_id               INTEGER                             NOT NULL,"
      "    FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::CreateFrameDataTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS frame_data"
      "   (frame_id             INTEGER                             NOT NULL,"
      "    data_id              INTEGER                             NOT NULL,"
      "    sensor_id            INTEGER                             NOT NULL,"
      "    sensor_type          INTEGER                             NOT NULL,"
      "    FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE "
      "CASCADE);"
      "CREATE UNIQUE INDEX IF NOT EXISTS frame_sensor_assignment ON "
      "   frame_data(data_id, sensor_type);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::CreateImageTable() const {
  const std::string sql = StringPrintf(
      "CREATE TABLE IF NOT EXISTS images"
      "   (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    name       TEXT                                NOT NULL UNIQUE,"
      "    camera_id  INTEGER                             NOT NULL,"
      "    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < %d));"
      "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);",
      kMaxNumImages);

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::CreateKeypointsTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS keypoints"
      "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows      INTEGER               NOT NULL,"
      "    cols      INTEGER               NOT NULL,"
      "    data      BLOB,"
      "    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE "
      "CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::CreateDescriptorsTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS descriptors"
      "   (image_id      INTEGER  PRIMARY KEY  NOT NULL,"
      "    type          INTEGER               NOT NULL,"
      "    rows          INTEGER               NOT NULL,"
      "    cols          INTEGER               NOT NULL,"
      "    data          BLOB,"
      "    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE "
      "CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::CreateMatchesTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS matches"
      "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows     INTEGER               NOT NULL,"
      "    cols     INTEGER               NOT NULL,"
      "    data     BLOB);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void SqliteDatabase::CreateTwoViewGeometriesTable() const {
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
        "    tvec     BLOB,"
        "    camera1  BLOB,"
        "    camera2  BLOB);";
    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }
}

void SqliteDatabase::CreatePosePriorsTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS pose_priors"
      "   (pose_prior_id              INTEGER  PRIMARY KEY  NOT NULL,"
      "    corr_data_id               INTEGER               NOT NULL,"
      "    corr_sensor_id             INTEGER               NOT NULL,"
      "    corr_sensor_type           INTEGER               NOT NULL,"
      "    position                   BLOB,"
      "    position_covariance        BLOB,"
      "    gravity                    BLOB,"
      "    coordinate_system          INTEGER               NOT NULL);"
      "CREATE UNIQUE INDEX IF NOT EXISTS pose_prior_data_assignment ON "
      "   pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
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

bool ExistsIndexImpl(sqlite3* db, const std::string& index_name) {
    sqlite3_stmt* stmt;
    SQLITE3_CALL(sqlite3_prepare_v2(
            db, "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?;",
            -1, &stmt, nullptr));
    SQLITE3_CALL(
            sqlite3_bind_text(stmt, 1, index_name.c_str(), -1, SQLITE_STATIC));
    const bool exists = SQLITE3_CALL(sqlite3_step(stmt)) == SQLITE_ROW;
    SQLITE3_CALL(sqlite3_finalize(stmt));
    return exists;
}

// The schema migration must fail closed: unlike SQLITE3_EXEC, which only
// logs, a failed statement aborts the open migration transaction.
void ExecOrThrow(sqlite3* db, const std::string& sql) {
    char* err_msg = nullptr;
    const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        const std::string message =
                err_msg != nullptr ? err_msg : sqlite3_errstr(rc);
        sqlite3_free(err_msg);
        throw std::runtime_error("Database migration failed: " + message +
                                 " [" + sql + "]");
    }
}

constexpr char kForkLegacySuffix[] = "_fork_legacy";

}  // namespace

bool SqliteDatabase::PreMigrateTables() const {
    // Legacy upstream databases keyed pose priors by image IDs; the table is
    // renamed so that CreateTables() can recreate it in the new schema.
    if (ExistsTable("pose_priors") &&
        ExistsColumnImpl(database_, "pose_priors", "image_id")) {
        SQLITE3_EXEC(database_,
                     "ALTER TABLE pose_priors RENAME TO pose_priors_old;",
                     nullptr);
    }

    // Fork-legacy schema (before the a2bd15c2 port): frame poses, image
    // priors, rig_cameras/frame_images side tables, reference sensors in
    // rig_sensors as qvec/tvec pairs, and data_id/sensor_* pose prior
    // columns. Each table is detected on its own, because a database written
    // by upstream and later opened by an older fork build only gained the
    // empty side tables.
    const std::vector<std::pair<std::string, bool>> tables = {
        {"rigs", ExistsColumn("rigs", "ref_camera_id")},
        {"rig_sensors", ExistsColumn("rig_sensors", "qvec")},
        {"rig_cameras", ExistsTable("rig_cameras")},
        {"frames", ExistsColumn("frames", "has_pose")},
        {"frame_data", ExistsTable("frame_data") &&
                           !ExistsIndexImpl(database_,
                                            "frame_sensor_assignment")},
        {"frame_images", ExistsTable("frame_images")},
        {"images", ExistsColumn("images", "prior_qw")},
        {"pose_priors", ExistsColumn("pose_priors", "data_id")},
    };
    bool fork_legacy_schema = false;
    for (const auto& [table, is_fork_legacy] : tables) {
        fork_legacy_schema |= is_fork_legacy;
    }
    if (!fork_legacy_schema) {
        return false;
    }

    LOG(INFO) << "Migrating fork-legacy database schema to upstream";
    // Move the fork-legacy tables aside; PostMigrateTables() copies them
    // into the tables recreated by CreateTables() and drops them. Foreign
    // keys are disabled and legacy_alter_table is enabled so that RENAME
    // does not rewrite the REFERENCES clauses of the kept tables (e.g.
    // keypoints -> images must keep naming the recreated images table) and
    // DROP does not cascade. Everything runs in one transaction, so a
    // failure leaves the database untouched.
    ExecOrThrow(database_, "PRAGMA foreign_keys=OFF;");
    ExecOrThrow(database_, "PRAGMA legacy_alter_table=ON;");
    ExecOrThrow(database_, "BEGIN TRANSACTION;");
    for (const auto& [table, is_fork_legacy] : tables) {
        if (!is_fork_legacy) {
            continue;
        }
        if (table == "images") {
            // The index moves with the renamed table and would otherwise
            // shadow the IF NOT EXISTS index of the recreated table.
            ExecOrThrow(database_, "DROP INDEX IF EXISTS index_name;");
        }
        ExecOrThrow(database_, "ALTER TABLE " + table + " RENAME TO " +
                                       table + kForkLegacySuffix + ";");
    }
    return true;
}

void SqliteDatabase::MigrateForkLegacyTables() const {
    auto legacy_table = [](const std::string& table) {
        return table + kForkLegacySuffix;
    };

    if (ExistsTable(legacy_table("rigs"))) {
        const std::string ref_sensor_id =
                ExistsColumn(legacy_table("rigs"), "ref_sensor_id")
                        ? "COALESCE(ref_sensor_id, ref_camera_id)"
                        : "ref_camera_id";
        const std::string ref_sensor_type =
                ExistsColumn(legacy_table("rigs"), "ref_sensor_type")
                        ? "COALESCE(ref_sensor_type, 0)"
                        : "0";
        ExecOrThrow(database_,
                    "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) "
                    "SELECT rig_id, " + ref_sensor_id + ", " +
                            ref_sensor_type + " FROM " +
                            legacy_table("rigs") + " ORDER BY rig_id;");
    }

    // Non-reference sensors only (upstream WriteRigSensors). The qvec/tvec
    // pair is re-encoded as the 7-double sensor_from_rig blob in C++, since
    // SQLite's || operator converts blobs to text. Cameras present in both
    // fork tables carry the same pose; the unique index keeps the first.
    {
        sqlite3_stmt* insert_stmt = nullptr;
        SQLITE3_CALL(sqlite3_prepare_v2(
                database_,
                "INSERT OR IGNORE INTO rig_sensors(rig_id, sensor_id, "
                "sensor_type, sensor_from_rig) SELECT ?1, ?2, ?3, ?4 WHERE NOT "
                "EXISTS (SELECT 1 FROM rigs WHERE rig_id = ?1 AND "
                "ref_sensor_id = ?2 AND ref_sensor_type = ?3);",
                -1, &insert_stmt, nullptr));
        auto copy_sensors = [this, insert_stmt](const std::string& sql) {
            sqlite3_stmt* read_stmt = nullptr;
            SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                            &read_stmt, nullptr));
            while (SQLITE3_CALL(sqlite3_step(read_stmt)) == SQLITE_ROW) {
                for (int col = 0; col < 3; ++col) {
                    SQLITE3_CALL(sqlite3_bind_int64(
                            insert_stmt, col + 1,
                            sqlite3_column_int64(read_stmt, col)));
                }
                Eigen::Vector4d qvec;
                Eigen::Vector3d tvec;
                // Must outlive sqlite3_step (bound with SQLITE_STATIC).
                std::string blob;
                if (sqlite3_column_type(read_stmt, 3) != SQLITE_NULL &&
                    sqlite3_column_type(read_stmt, 4) != SQLITE_NULL) {
                    CHECK(ReadPoseBlob(read_stmt, 3, 4, &qvec, &tvec));
                    blob = SensorFromRigToBlob(qvec, tvec);
                    SQLITE3_CALL(sqlite3_bind_blob(
                            insert_stmt, 4, blob.data(),
                            static_cast<int>(blob.size()), SQLITE_STATIC));
                } else {
                    SQLITE3_CALL(sqlite3_bind_null(insert_stmt, 4));
                }
                SQLITE3_CALL(sqlite3_step(insert_stmt));
                SQLITE3_CALL(sqlite3_reset(insert_stmt));
            }
            SQLITE3_CALL(sqlite3_finalize(read_stmt));
        };
        if (ExistsTable(legacy_table("rig_sensors"))) {
            copy_sensors("SELECT rig_id, sensor_id, sensor_type, qvec, tvec "
                         "FROM " + legacy_table("rig_sensors") +
                         " ORDER BY rig_id, sensor_type, sensor_id;");
        }
        if (ExistsTable(legacy_table("rig_cameras"))) {
            copy_sensors("SELECT rig_id, camera_id, 0, qvec, tvec FROM " +
                         legacy_table("rig_cameras") +
                         " ORDER BY rig_id, camera_id;");
        }
        SQLITE3_CALL(sqlite3_finalize(insert_stmt));
    }

    // Frame poses are reconstruction state and not part of the upstream
    // schema; they are dropped.
    if (ExistsTable(legacy_table("frames"))) {
        ExecOrThrow(database_,
                    "INSERT INTO frames(frame_id, rig_id) SELECT frame_id, "
                    "rig_id FROM " + legacy_table("frames") +
                            " ORDER BY frame_id;");
    }

    // Image priors had no production writer (only the database management
    // GUI) and are not part of the upstream schema; they are dropped.
    if (ExistsTable(legacy_table("images"))) {
        ExecOrThrow(database_,
                    "INSERT INTO images(image_id, name, camera_id) SELECT "
                    "image_id, name, camera_id FROM " +
                            legacy_table("images") + " ORDER BY image_id;");
    }

    if (ExistsTable(legacy_table("frame_data"))) {
        ExecOrThrow(database_,
                    "INSERT OR IGNORE INTO frame_data(frame_id, data_id, "
                    "sensor_id, sensor_type) SELECT frame_id, data_id, "
                    "sensor_id, sensor_type FROM " +
                            legacy_table("frame_data") + ";");
    }
    // frame_images rows are camera data of the image's own camera sensor.
    if (ExistsTable(legacy_table("frame_images"))) {
        ExecOrThrow(database_,
                    StringPrintf(
                            "INSERT OR IGNORE INTO frame_data(frame_id, "
                            "data_id, sensor_id, sensor_type) SELECT "
                            "fi.frame_id, fi.image_id, images.camera_id, %d "
                            "FROM %s AS fi JOIN images ON images.image_id = "
                            "fi.image_id;",
                            static_cast<int>(SensorType::CAMERA),
                            legacy_table("frame_images").c_str()));
    }

    if (ExistsTable(legacy_table("pose_priors"))) {
        ExecOrThrow(database_,
                    "INSERT OR IGNORE INTO pose_priors(pose_prior_id, "
                    "corr_data_id, corr_sensor_id, corr_sensor_type, "
                    "position, position_covariance, gravity, "
                    "coordinate_system) SELECT pose_prior_id, data_id, "
                    "sensor_id, sensor_type, position, position_covariance, "
                    "gravity, coordinate_system FROM " +
                            legacy_table("pose_priors") +
                            " ORDER BY pose_prior_id;");
    }

    for (const char* table : {"rigs", "rig_sensors", "rig_cameras", "frames",
                              "frame_data", "frame_images", "images",
                              "pose_priors"}) {
        if (ExistsTable(legacy_table(table))) {
            ExecOrThrow(database_,
                        "DROP TABLE " + legacy_table(table) + ";");
        }
    }

    // The fork-legacy two-view encoding wrote all-zero blobs for absent
    // F/E/H/pose and zero-length blobs for entries without inliers; upstream
    // only treats NULL as absent.
    const std::string zero_matrix = "X'" + std::string(2 * 9 * 8, '0') + "'";
    const std::string zero_qvec = "X'" + std::string(2 * 4 * 8, '0') + "'";
    for (const char* column : {"F", "E", "H"}) {
        ExecOrThrow(database_,
                    StringPrintf("UPDATE two_view_geometries SET %s = NULL "
                                 "WHERE length(%s) = 0 OR %s = %s;",
                                 column, column, column,
                                 zero_matrix.c_str()));
    }
    ExecOrThrow(database_,
                "UPDATE two_view_geometries SET qvec = NULL, tvec = NULL "
                "WHERE length(qvec) = 0 OR length(tvec) = 0 OR qvec = " +
                        zero_qvec + ";");

    ExecOrThrow(database_, "COMMIT;");
    ExecOrThrow(database_, "PRAGMA legacy_alter_table=OFF;");
    ExecOrThrow(database_, "PRAGMA foreign_keys=ON;");
}

int SqliteDatabase::ReadUserVersion() const {
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

void SqliteDatabase::PostMigrateTables(const bool fork_legacy_schema) const {
    // F2: descriptors.type is the single persistent source of the extractor
    // type. Old fork databases stored SIFT bytes in descriptors and learned
    // float32 descriptors in a second table with a fork-private enum. Migrate
    // once while the old table still exists, then remove the parallel source.
    if (!ExistsColumn("descriptors", "type")) {
        SQLITE3_EXEC(database_,
                     "ALTER TABLE descriptors ADD COLUMN type INTEGER NOT NULL "
                     "DEFAULT 0;",
                     nullptr);
    }
    if (ExistsTable("float_descriptors")) {
        sqlite3_stmt* invalid_type_stmt = nullptr;
        SQLITE3_CALL(sqlite3_prepare_v2(
                database_,
                "SELECT COUNT(*) FROM float_descriptors "
                "WHERE type NOT IN (0, 1, 2, 6, 7);",
                -1, &invalid_type_stmt, nullptr));
        const int invalid_rc = SQLITE3_CALL(sqlite3_step(invalid_type_stmt));
        const int invalid_count =
                invalid_rc == SQLITE_ROW
                        ? sqlite3_column_int(invalid_type_stmt, 0)
                        : 0;
        SQLITE3_CALL(sqlite3_finalize(invalid_type_stmt));
        if (invalid_count != 0) {
            throw std::runtime_error(
                    "Cannot migrate float_descriptors with an unsupported "
                    "legacy descriptor type.");
        }

        // This is O(total descriptor bytes) because SQLite copies each blob;
        // it runs once per legacy database and preserves raw float32 bytes.
        SQLITE3_EXEC(
                database_,
                "INSERT OR REPLACE INTO descriptors(image_id, rows, cols, data, type) "
                "SELECT image_id, rows, cols * 4, data, "
                "CASE type "
                "WHEN 0 THEN 0 "
                "WHEN 1 THEN 3 "
                "WHEN 2 THEN 4 "
                "WHEN 6 THEN 1 "
                "WHEN 7 THEN 2 END "
                "FROM float_descriptors;",
                nullptr);
        SQLITE3_EXEC(database_, "DROP TABLE float_descriptors;", nullptr);
    }

    // Upstream parity (a2bd15c2): old databases lack some two-view columns.
    for (const char* column :
         {"F", "E", "H", "qvec", "tvec", "camera1", "camera2"}) {
        if (!ExistsColumn("two_view_geometries", column)) {
            SQLITE3_EXEC(database_,
                         StringPrintf("ALTER TABLE two_view_geometries ADD "
                                      "COLUMN %s BLOB;",
                                      column)
                                 .c_str(),
                         nullptr);
        }
    }

    // Refuse databases written by a newer schema than this build supports.
    // Fork-legacy databases were stamped with the fork-private schema
    // revision 1 and are migrated instead.
    const int user_version = ReadUserVersion();
    if (!fork_legacy_schema && user_version > GetDatabaseVersionNumber()) {
        throw std::runtime_error(
                "Database schema version " + std::to_string(user_version) +
                " is newer than the supported version " +
                std::to_string(GetDatabaseVersionNumber()) + ".");
    }

    if (fork_legacy_schema) {
        MigrateForkLegacyTables();
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

    // Upstream parity (a2bd15c2): migrate image-keyed pose priors.
    if (ExistsTable("pose_priors_old")) {
        LOG(INFO) << "Migrating pose_priors table";
        sqlite3_stmt* sql_stmt;
        SQLITE3_CALL(sqlite3_prepare_v2(
                database_,
                "INSERT INTO pose_priors (pose_prior_id, corr_data_id, "
                "corr_sensor_id, corr_sensor_type, position, "
                "position_covariance, coordinate_system, gravity) "
                "SELECT pose_priors_old.image_id, pose_priors_old.image_id, "
                "images.camera_id, ?, position, position_covariance, "
                "coordinate_system, ? FROM pose_priors_old JOIN images ON "
                "pose_priors_old.image_id = images.image_id;",
                -1, &sql_stmt, nullptr));
        SQLITE3_CALL(sqlite3_bind_int64(
                sql_stmt, 1, static_cast<sqlite3_int64>(SensorType::CAMERA)));
        const PosePrior pose_prior;
        WriteStaticMatrixBlob(sql_stmt, pose_prior.gravity, 2);
        SQLITE3_CALL(sqlite3_step(sql_stmt));
        SQLITE3_CALL(sqlite3_finalize(sql_stmt));
        SQLITE3_EXEC(database_, "DROP TABLE pose_priors_old;", nullptr);
    }

    // Upstream parity (f1ea91d0): migrate the cameras table to the composite
    // primary key (camera_id, source) and recreate the images table without
    // the foreign key to cameras.
    if (user_version <= MakeDatabaseVersionNumber(4, 3, 0, 0)) {
      const bool needs_camera_rename =
          ExistsTable("cameras") && !ExistsColumn("cameras", "source");
      if (needs_camera_rename || ExistsTable("cameras_old") ||
          ExistsTable("images_old")) {
        ExecOrThrow(database_, "PRAGMA foreign_keys=OFF;");
        ExecOrThrow(database_, "PRAGMA legacy_alter_table=ON;");

        if (needs_camera_rename) {
          ExecOrThrow(database_, "ALTER TABLE cameras RENAME TO cameras_old;");
          CreateCameraTable();
        }

        if (ExistsTable("cameras_old")) {
          const std::string migrate_sql = StringPrintf(
              "INSERT OR IGNORE INTO cameras "
              "(camera_id, model, width, height, params, source) "
              "SELECT camera_id, model, width, height, params, "
              "CASE WHEN prior_focal_length != 0 THEN %d ELSE %d END "
              "FROM cameras_old;",
              static_cast<int>(CameraSource::EXIF),
              static_cast<int>(CameraSource::GUESS));
          ExecOrThrow(database_, migrate_sql.c_str());

          if (ExistsTable("images") && !ExistsTable("images_old")) {
            ExecOrThrow(database_, "DROP INDEX IF EXISTS index_name;");
            ExecOrThrow(database_,
                        "ALTER TABLE images RENAME TO images_old;");
            CreateImageTable();
          }

          ExecOrThrow(database_, "DROP TABLE cameras_old;");
        }

        if (ExistsTable("images_old")) {
          CreateImageTable();
          ExecOrThrow(database_,
                      "INSERT OR IGNORE INTO images (image_id, name, "
                      "camera_id) SELECT image_id, name, camera_id FROM "
                      "images_old;");
          ExecOrThrow(database_, "DROP TABLE images_old;");
        }

        ExecOrThrow(database_, "PRAGMA legacy_alter_table=OFF;");
        ExecOrThrow(database_, "PRAGMA foreign_keys=ON;");
      }
    }

    // Stamp the schema version.
    std::unique_lock<std::mutex> lock(update_schema_mutex_);
    const std::string update_user_version_sql =
            StringPrintf("PRAGMA user_version = %d;", GetDatabaseVersionNumber());
    SQLITE3_EXEC(database_, update_user_version_sql.c_str(), nullptr);
}

bool SqliteDatabase::ExistsTable(const std::string& table_name) const {
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

bool SqliteDatabase::ExistsColumn(const std::string& table_name,
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

bool SqliteDatabase::ExistsRowId(sqlite3_stmt* sql_stmt,
                           const sqlite3_int64 row_id) const {
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt, 1, static_cast<sqlite3_int64>(row_id)));

  const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

bool SqliteDatabase::ExistsRowString(sqlite3_stmt* sql_stmt,
                               const std::string& row_entry) const {
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt, 1, row_entry.c_str(),
                                 static_cast<int>(row_entry.size()),
                                 SQLITE_STATIC));

  const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

size_t SqliteDatabase::CountRows(const std::string& table) const {
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

size_t SqliteDatabase::CountRowsForEntry(sqlite3_stmt* sql_stmt,
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

size_t SqliteDatabase::SumColumn(const std::string& column,
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

size_t SqliteDatabase::MaxColumn(const std::string& column,
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

bool SqliteDatabase::ExistsPosePrior(pose_prior_t pose_prior_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_exists_pose_prior_, 1,
                                  static_cast<sqlite3_int64>(pose_prior_id)));
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_exists_pose_prior_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_exists_pose_prior_));
  return rc == SQLITE_ROW;
}

size_t SqliteDatabase::NumPosePriors() const {
  sqlite3_stmt* sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, "SELECT COUNT(*) FROM"
                                 " pose_priors;", -1, &sql_stmt, nullptr));
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  const size_t count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  SQLITE3_CALL(sqlite3_finalize(sql_stmt));
  THROW_CHECK_EQ(rc, SQLITE_ROW);
  return count;
}

PosePrior SqliteDatabase::ReadPosePrior(pose_prior_t pose_prior_id) const {
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

std::vector<PosePrior> SqliteDatabase::ReadAllPosePriors() const {
  std::vector<PosePrior> pose_priors;
  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_pose_priors_)) ==
         SQLITE_ROW) {
    pose_priors.push_back(ReadPosePriorRow(sql_stmt_read_pose_priors_));
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_pose_priors_));
  return pose_priors;
}

pose_prior_t SqliteDatabase::WritePosePrior(const PosePrior& pose_prior,
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

void SqliteDatabase::UpdatePosePrior(const PosePrior& pose_prior) {
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

void SqliteDatabase::ClearPosePriors() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_pose_priors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_pose_priors_));
}

std::shared_ptr<Database> OpenSqliteDatabase(
    const std::filesystem::path& path) {
  return SqliteDatabase::Open(path);
}

}  // namespace colmap
