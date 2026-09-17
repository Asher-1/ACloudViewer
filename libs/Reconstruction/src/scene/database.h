// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "feature/types.h"
#include "geometry/pose_prior.h"
#include "scene/camera.h"
#include "scene/frame.h"
#include "scene/image.h"
#include "scene/rig.h"
#include "scene/two_view_geometry.h"
#include "util/types.h"

namespace colmap {

// Database class to read and write images, features, cameras, matches, etc.
// from a SQLite database. The class is not thread-safe and must not be accessed
// concurrently. The class is optimized for single-thread speed and for optimal
// performance, wrap multiple method calls inside a leading `BeginTransaction`
// and trailing `EndTransaction`.
//
// W17.2b (upstream dbb41680 parity): Database is an abstract interface; the
// concrete SQL implementation lives in scene/database_sqlite.cc
// (SqliteDatabase) behind the Database::Open factory / OpenSqliteDatabase
// registered factory. The fork-specific float-descriptor surface is kept on
// the interface because the fork's float_descriptors table is a permanent
// local extension (W1 migration note).
class Database {
public:
    Database() = default;

    // Closes the database, if not closed before.
    virtual ~Database() = 0;

    NON_COPYABLE(Database)

    // Factory function to create a database implementation for a given path.
    // The factory should be robust to handle non-supported files and return a
    // runtime_error in that case.
    using Factory = std::function<std::shared_ptr<Database>(
            const std::filesystem::path&)>;

    // Register a factory to open a database implementation. Database factories
    // are tried in reverse order of registration. In other words, later
    // registrations are tried first.
    static void Register(Factory factory);

    // Open database and throw a runtime_error if none of the factories
    // succeeds.
    static std::shared_ptr<Database> Open(const std::filesystem::path& path);

    // Explicitly close the database before destruction.
    virtual void Close() = 0;

    const static int kSchemaVersion = 1;

    // The maximum number of images, that can be stored in the database.
    // This limitation arises due to the fact, that we generate unique IDs for
    // image pairs manually. Note: do not change this to
    // another type than `size_t`.
    const static size_t kMaxNumImages;

    // Check if entry already exists in database. For image pairs, the order of
    // `image_id1` and `image_id2` does not matter.
    virtual bool ExistsCamera(const camera_t camera_id) const = 0;
    virtual bool ExistsRig(const rig_t rig_id) const = 0;
    virtual bool ExistsFrame(const frame_t frame_id) const = 0;
    virtual bool ExistsImage(const image_t image_id) const = 0;
    virtual bool ExistsImageWithName(std::string name) const = 0;
    virtual bool ExistsKeypoints(const image_t image_id) const = 0;
    virtual bool ExistsDescriptors(const image_t image_id) const = 0;
    // Fork-specific surface (float_descriptors local table, W1).
    virtual bool ExistsFloatDescriptors(const image_t image_id) const = 0;
    virtual bool ExistsMatches(const image_t image_id1,
                               const image_t image_id2) const = 0;
    virtual bool ExistsInlierMatches(const image_t image_id1,
                                     const image_t image_id2) const = 0;
    // Upstream COLMAP dbb41680 API name. The legacy ExistsInlierMatches above
    // already queries the two_view_geometries table; both names share the
    // same prepared statement.
    virtual bool ExistsTwoViewGeometry(const image_t image_id1,
                                       const image_t image_id2) const = 0;
    // Upstream-parity pose-prior persistence (COLMAP 4.x).
    virtual bool ExistsPosePrior(pose_prior_t pose_prior_id) const = 0;

    // Number of rows in `cameras` table.
    virtual size_t NumCameras() const = 0;
    virtual size_t NumRigs() const = 0;
    virtual size_t NumFrames() const = 0;

    //  Number of rows in `images` table.
    virtual size_t NumImages() const = 0;

    // Sum of `rows` column in `keypoints` table, i.e. number of total
    // keypoints.
    virtual size_t NumKeypoints() const = 0;

    // The number of keypoints for the image with most features.
    virtual size_t MaxNumKeypoints() const = 0;

    // Number of keypoints for specific image.
    virtual size_t NumKeypointsForImage(const image_t image_id) const = 0;

    // Sum of `rows` column in `descriptors` table,
    // i.e. number of total descriptors.
    virtual size_t NumDescriptors() const = 0;

    // The number of descriptors for the image with most features.
    virtual size_t MaxNumDescriptors() const = 0;

    // Number of descriptors for specific image.
    virtual size_t NumDescriptorsForImage(const image_t image_id) const = 0;

    // Sum of `rows` column in `matches` table, i.e. number of total matches.
    virtual size_t NumMatches() const = 0;

    // Sum of `rows` column in `two_view_geometries` table,
    // i.e. number of total inlier matches.
    virtual size_t NumInlierMatches() const = 0;

    // Number of rows in `matches` table.
    virtual size_t NumMatchedImagePairs() const = 0;

    // Number of rows in `two_view_geometries` table.
    virtual size_t NumVerifiedImagePairs() const = 0;

    // Upstream-parity pose-prior persistence (COLMAP 4.x).
    virtual size_t NumPosePriors() const = 0;

    // Each image pair is assigned an unique ID in the `matches` and
    // `two_view_geometries` table. We intentionally avoid to store the pairs
    // in a separate table by using e.g. AUTOINCREMENT, since the overhead of
    // querying the unique pair ID is significant.
    inline static image_pair_t ImagePairToPairId(const image_t image_id1,
                                                 const image_t image_id2);

    inline static void PairIdToImagePair(const image_pair_t pair_id,
                                         image_t* image_id1,
                                         image_t* image_id2);

    // Upstream-parity overload returning the image pair by value.
    inline static std::pair<image_t, image_t> PairIdToImagePair(
            const image_pair_t pair_id) {
        std::pair<image_t, image_t> image_ids;
        PairIdToImagePair(pair_id, &image_ids.first, &image_ids.second);
        return image_ids;
    }

    // Return true if image pairs should be swapped. Used to enforce a specific
    // image order to generate unique image pair identifiers independent of the
    // order in which the image identifiers are used.
    inline static bool SwapImagePair(const image_t image_id1,
                                     const image_t image_id2);

    // Read an existing entry in the database. The user is responsible for
    // making sure that the entry actually exists. For image pairs, the order
    // of `image_id1` and `image_id2` does not matter.
    virtual Camera ReadCamera(const camera_t camera_id) const = 0;
    virtual std::vector<Camera> ReadAllCameras() const = 0;

    virtual Rig ReadRig(const rig_t rig_id) const = 0;
    // Upstream parity (d3ccaf35): find the rig that has the given sensor as
    // a non-reference or reference sensor, or nullopt if it doesn't exist.
    virtual std::optional<Rig> ReadRigWithSensor(sensor_t sensor_id) const = 0;
    virtual std::vector<Rig> ReadAllRigs() const = 0;
    virtual Frame ReadFrame(const frame_t frame_id) const = 0;
    virtual std::vector<Frame> ReadAllFrames() const = 0;

    virtual Image ReadImage(const image_t image_id) const = 0;
    virtual Image ReadImageWithName(const std::string& name) const = 0;
    virtual std::vector<Image> ReadAllImages() const = 0;

    virtual FeatureKeypoints ReadKeypoints(const image_t image_id) const = 0;
    virtual FeatureDescriptors ReadDescriptors(
            const image_t image_id) const = 0;
    // Fork-specific surface (float_descriptors local table, W1).
    virtual FeatureDescriptorsFloat ReadFloatDescriptors(
            const image_t image_id) const = 0;
    virtual FeatureDescriptorType ReadDescriptorType(
            const image_t image_id) const = 0;

    virtual FeatureMatches ReadMatches(const image_t image_id1,
                                       const image_t image_id2) const = 0;
    virtual std::vector<std::pair<image_pair_t, FeatureMatches>>
    ReadAllMatches() const = 0;

    virtual TwoViewGeometry ReadTwoViewGeometry(
            const image_t image_id1, const image_t image_id2) const = 0;
    virtual void ReadTwoViewGeometries(
            std::vector<image_pair_t>* image_pair_ids,
            std::vector<TwoViewGeometry>* two_view_geometries) const = 0;
    // Upstream-parity overload (COLMAP 4.x scene/database.h): all verified
    // pairs keyed by the image pair id.
    virtual std::map<image_pair_t, TwoViewGeometry> ReadTwoViewGeometries()
            const = 0;

    // Read all image pairs that have an entry in the `NumVerifiedImagePairs`
    // table with at least one inlier match and their number of inlier matches.
    virtual void ReadTwoViewGeometryNumInliers(
            std::vector<std::pair<image_t, image_t>>* image_pairs,
            std::vector<int>* num_inliers) const = 0;

    // Upstream-parity pose-prior persistence (COLMAP 4.x).
    virtual PosePrior ReadPosePrior(pose_prior_t pose_prior_id) const = 0;
    virtual std::vector<PosePrior> ReadAllPosePriors() const = 0;

    // Upstream COLMAP dbb41680 API: update an existing two view geometry.
    virtual void UpdateTwoViewGeometry(
            const image_t image_id1,
            const image_t image_id2,
            const TwoViewGeometry& two_view_geometry) const = 0;

    // Add new camera and return its database identifier. If `use_camera_id`
    // is false a new identifier is automatically generated.
    virtual camera_t WriteCamera(const Camera& camera,
                                 const bool use_camera_id = false) const = 0;
    virtual rig_t WriteRig(const Rig& rig,
                           const bool use_rig_id = false) const = 0;
    virtual frame_t WriteFrame(const Frame& frame,
                               const bool use_frame_id = false) const = 0;

    // Add new image and return its database identifier. If `use_image_id`
    // is false a new identifier is automatically generated.
    virtual image_t WriteImage(const Image& image,
                               const bool use_image_id = false) const = 0;

    // Write a new entry in the database. The user is responsible for making
    // sure that the entry does not yet exist. For image pairs, the order of
    // `image_id1` and `image_id2` does not matter.
    virtual void WriteKeypoints(const image_t image_id,
                                const FeatureKeypoints& keypoints) const = 0;
    virtual void WriteDescriptors(
            const image_t image_id,
            const FeatureDescriptors& descriptors) const = 0;
    // Fork-specific surface (float_descriptors local table, W1).
    virtual void WriteFloatDescriptors(
            const image_t image_id,
            const FeatureDescriptorsFloat& descriptors,
            FeatureDescriptorType type) const = 0;
    virtual void WriteMatches(const image_t image_id1,
                              const image_t image_id2,
                              const FeatureMatches& matches) const = 0;
    virtual void WriteTwoViewGeometry(
            const image_t image_id1,
            const image_t image_id2,
            const TwoViewGeometry& two_view_geometry) const = 0;
    // Upstream-parity pose-prior persistence (COLMAP 4.x).
    virtual pose_prior_t WritePosePrior(const PosePrior& pose_prior,
                                        bool use_pose_prior_id = false) = 0;

    // Update an existing camera in the database. The user is responsible for
    // making sure that the entry already exists.
    virtual void UpdateCamera(const Camera& camera) const = 0;
    virtual void UpdateRig(const Rig& rig) const = 0;
    virtual void UpdateFrame(const Frame& frame) const = 0;

    // Update an existing image in the database. The user is responsible for
    // making sure that the entry already exists.
    virtual void UpdateImage(const Image& image) const = 0;

    // Update an existing image's keypoints in the database. The user is
    // responsible for making sure that the entry already exists.
    virtual void UpdateKeypoints(const image_t image_id,
                                 const FeatureKeypoints& keypoints) const = 0;
    // Upstream-parity pose-prior persistence (COLMAP 4.x).
    virtual void UpdatePosePrior(const PosePrior& pose_prior) = 0;

    // Delete matches of an image pair.
    virtual void DeleteMatches(const image_t image_id1,
                               const image_t image_id2) const = 0;

    // Deletes a two-view geometry entry (upstream parity; the matches and
    // inlier matches tables are unaffected).
    virtual void DeleteTwoViewGeometry(const image_t image_id1,
                                       const image_t image_id2) const = 0;

    // Delete inlier matches of an image pair.
    virtual void DeleteInlierMatches(const image_t image_id1,
                                     const image_t image_id2) const = 0;

    // Clear all database tables
    virtual void ClearAllTables() const = 0;

    // Clear the entire cameras table
    virtual void ClearCameras() const = 0;
    virtual void ClearRigs() const = 0;
    virtual void ClearFrames() const = 0;

    // Clear the entire images, keypoints, and descriptors tables
    virtual void ClearImages() const = 0;

    // Clear the entire descriptors table
    virtual void ClearDescriptors() const = 0;

    // Clear the entire keypoints table
    virtual void ClearKeypoints() const = 0;

    // Clear the entire matches table.
    virtual void ClearMatches() const = 0;

    // Clear the entire inlier matches table.
    virtual void ClearTwoViewGeometries() const = 0;

    // Upstream-parity pose-prior persistence (COLMAP 4.x).
    virtual void ClearPosePriors() = 0;

    // Merge two databases into a single, new database.
    static void Merge(const Database& database1,
                      const Database& database2,
                      Database* merged_database);

protected:
    friend class DatabaseTransaction;

    // Combine multiple queries into one transaction by wrapping a code section
    // into a `BeginTransaction` and `EndTransaction`. You can create a scoped
    // transaction with `DatabaseTransaction` that ends when the transaction
    // object is destructed. Combining queries results in faster transaction
    // time due to reduced locking of the database etc.
    virtual void BeginTransaction() const = 0;
    virtual void EndTransaction() const = 0;

    // Used to ensure that only one transaction is active at the same time.
    std::mutex transaction_mutex_;

private:
    static std::vector<Factory> factories_;
};

// This class automatically manages the scope of a database transaction by
// calling `BeginTransaction` and `EndTransaction` during construction and
// destruction, respectively.
class DatabaseTransaction {
public:
    explicit DatabaseTransaction(Database* database);
    ~DatabaseTransaction();

private:
    NON_COPYABLE(DatabaseTransaction)
    NON_MOVABLE(DatabaseTransaction)
    Database* database_;
    std::unique_lock<std::mutex> database_lock_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

image_pair_t Database::ImagePairToPairId(const image_t image_id1,
                                         const image_t image_id2) {
    // image_t is unsigned: the >= 0 guards are vacuous, keep them only as
    // explicit casts so -Wsign-compare stays quiet.
    CHECK_GE(image_id1, static_cast<image_t>(0));
    CHECK_GE(image_id2, static_cast<image_t>(0));
    CHECK_LT(image_id1, kMaxNumImages);
    CHECK_LT(image_id2, kMaxNumImages);
    if (SwapImagePair(image_id1, image_id2)) {
        return static_cast<image_pair_t>(kMaxNumImages) * image_id2 + image_id1;
    } else {
        return static_cast<image_pair_t>(kMaxNumImages) * image_id1 + image_id2;
    }
}

void Database::PairIdToImagePair(const image_pair_t pair_id,
                                 image_t* image_id1,
                                 image_t* image_id2) {
    *image_id2 = static_cast<image_t>(pair_id % kMaxNumImages);
    *image_id1 = static_cast<image_t>((pair_id - *image_id2) / kMaxNumImages);
    CHECK_GE(*image_id1, static_cast<image_t>(0));
    CHECK_GE(*image_id2, static_cast<image_t>(0));
    CHECK_LT(*image_id1, kMaxNumImages);
    CHECK_LT(*image_id2, kMaxNumImages);
}

// Return true if image pairs should be swapped. Used to enforce a specific
// image order to generate unique image pair identifiers independent of the
// order in which the image identifiers are used.
bool Database::SwapImagePair(const image_t image_id1, const image_t image_id2) {
    return image_id1 > image_id2;
}

}  // namespace colmap
