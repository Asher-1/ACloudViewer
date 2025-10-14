// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_MVS_WORKSPACE_H_
#define COLMAP_SRC_MVS_WORKSPACE_H_

#include "mvs/consistency_graph.h"
#include "mvs/depth_map.h"
#include "mvs/model.h"
#include "mvs/normal_map.h"
#include "util/bitmap.h"
#include "util/cache.h"
#include "util/misc.h"

namespace colmap {
namespace mvs {

class Workspace {
public:
    struct Options {
        // The maximum cache size in gigabytes.
        double cache_size = 32.0;

        // The number of threads to use when pre-loading workspace.
        int num_threads = -1;

        // Maximum image size in either dimension.
        int max_image_size = -1;

        // Whether to read image as RGB or gray scale.
        bool image_as_rgb = true;

        // Location and type of workspace.
        std::string workspace_path;
        std::string workspace_format;
        std::string input_type;
        std::string stereo_folder = "stereo";
    };

    Workspace(const Options& options);

    // Do nothing when we use a cache. Data is loaded as needed.
    virtual void Load(const std::vector<std::string>& image_names);

    inline const Options& GetOptions() const { return options_; }

    inline const Model& GetModel() const { return model_; }

    virtual const Bitmap& GetBitmap(const int image_idx);
    virtual const DepthMap& GetDepthMap(const int image_idx);
    virtual const NormalMap& GetNormalMap(const int image_idx);

    // Get paths to bitmap, depth map, normal map and consistency graph.
    std::string GetBitmapPath(const int image_idx) const;
    std::string GetDepthMapPath(const int image_idx) const;
    std::string GetNormalMapPath(const int image_idx) const;

    // Return whether bitmap, depth map, normal map, and consistency graph
    // exist.
    bool HasBitmap(const int image_idx) const;
    bool HasDepthMap(const int image_idx) const;
    bool HasNormalMap(const int image_idx) const;

protected:
    std::string GetFileName(const int image_idx) const;

    Options options_;
    Model model_;

private:
    std::string depth_map_path_;
    std::string normal_map_path_;
    std::vector<std::unique_ptr<Bitmap>> bitmaps_;
    std::vector<std::unique_ptr<DepthMap>> depth_maps_;
    std::vector<std::unique_ptr<NormalMap>> normal_maps_;
};

class CachedWorkspace : public Workspace {
public:
    CachedWorkspace(const Options& options);

    void Load(const std::vector<std::string>& image_names) override {}

    inline void ClearCache() { cache_.Clear(); }

    const Bitmap& GetBitmap(const int image_idx) override;
    const DepthMap& GetDepthMap(const int image_idx) override;
    const NormalMap& GetNormalMap(const int image_idx) override;

private:
    class CachedImage {
    public:
        CachedImage() {}
        CachedImage(CachedImage&& other);
        CachedImage& operator=(CachedImage&& other);
        inline size_t NumBytes() const { return num_bytes; }
        size_t num_bytes = 0;
        std::unique_ptr<Bitmap> bitmap;
        std::unique_ptr<DepthMap> depth_map;
        std::unique_ptr<NormalMap> normal_map;

    private:
        NON_COPYABLE(CachedImage)
    };

    MemoryConstrainedLRUCache<int, CachedImage> cache_;
};

// Import a PMVS workspace into the COLMAP workspace format. Only images in the
// provided option file name will be imported and used for reconstruction.
void ImportPMVSWorkspace(const Workspace& workspace,
                         const std::string& option_name);

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_WORKSPACE_H_
