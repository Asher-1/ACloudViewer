// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/camera.h"
#include "base/camera_models.h"
#include "base/correspondence_graph.h"
#include "base/database.h"
#include "base/image.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// A class that caches the contents of the database in memory, used to quickly
// create new reconstruction instances when multiple models are reconstructed.
class DatabaseCache {
public:
    DatabaseCache();

    // Get number of objects.
    inline size_t NumCameras() const;
    inline size_t NumImages() const;

    // Get specific objects.
    inline class Camera& Camera(const camera_t camera_id);
    inline const class Camera& Camera(const camera_t camera_id) const;
    inline class Image& Image(const image_t image_id);
    inline const class Image& Image(const image_t image_id) const;

    // Get all objects.
    inline const EIGEN_STL_UMAP(camera_t, class Camera) & Cameras() const;
    inline const EIGEN_STL_UMAP(image_t, class Image) & Images() const;

    // Check whether specific object exists.
    inline bool ExistsCamera(const camera_t camera_id) const;
    inline bool ExistsImage(const image_t image_id) const;

    // Get reference to correspondence graph.
    inline const class CorrespondenceGraph& CorrespondenceGraph() const;

    // Manually add data to cache.
    void AddCamera(const class Camera& camera);
    void AddImage(const class Image& image);

    // Load cameras, images, features, and matches from database.
    //
    // @param database              Source database from which to load data.
    // @param min_num_matches       Only load image pairs with a minimum number
    //                              of matches.
    // @param ignore_watermarks     Whether to ignore watermark image pairs.
    // @param image_names           Whether to use only load the data for a
    // subset
    //                              of the images. All images are used if empty.
    void Load(const Database& database,
              const size_t min_num_matches,
              const bool ignore_watermarks,
              const std::unordered_set<std::string>& image_names);

    // Find specific image by name. Note that this uses linear search.
    const class Image* FindImageWithName(const std::string& name) const;

private:
    class CorrespondenceGraph correspondence_graph_;

    EIGEN_STL_UMAP(camera_t, class Camera) cameras_;
    EIGEN_STL_UMAP(image_t, class Image) images_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t DatabaseCache::NumCameras() const { return cameras_.size(); }
size_t DatabaseCache::NumImages() const { return images_.size(); }

class Camera& DatabaseCache::Camera(const camera_t camera_id) {
    return cameras_.at(camera_id);
}

const class Camera& DatabaseCache::Camera(const camera_t camera_id) const {
    return cameras_.at(camera_id);
}

class Image& DatabaseCache::Image(const image_t image_id) {
    return images_.at(image_id);
}

const class Image& DatabaseCache::Image(const image_t image_id) const {
    return images_.at(image_id);
}

const EIGEN_STL_UMAP(camera_t, class Camera) & DatabaseCache::Cameras() const {
    return cameras_;
}

const EIGEN_STL_UMAP(image_t, class Image) & DatabaseCache::Images() const {
    return images_;
}

bool DatabaseCache::ExistsCamera(const camera_t camera_id) const {
    return cameras_.find(camera_id) != cameras_.end();
}

bool DatabaseCache::ExistsImage(const image_t image_id) const {
    return images_.find(image_id) != images_.end();
}

inline const class CorrespondenceGraph& DatabaseCache::CorrespondenceGraph()
        const {
    return correspondence_graph_;
}

}  // namespace colmap
