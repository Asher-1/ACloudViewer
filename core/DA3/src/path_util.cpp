#include "path_util.hpp"

#include <cstdlib>
#include <filesystem>

namespace da {

namespace {

std::string joinCacheRoot(const std::filesystem::path& root) {
    return (root / "cloudViewer_data" / "extract" / "da3_models")
        .lexically_normal()
        .string();
}

}  // namespace

std::string default_model_cache_dir() {
    if (const char* data_root = std::getenv("CLOUDVIEWER_DATA_ROOT")) {
        if (data_root[0] != '\0') {
            return (std::filesystem::path(data_root) / "extract" / "da3_models")
                .lexically_normal()
                .string();
        }
    }

#if defined(_WIN32)
    if (const char* home = std::getenv("USERPROFILE")) {
        if (home[0] != '\0') {
            return joinCacheRoot(home);
        }
    }
    if (const char* local = std::getenv("LOCALAPPDATA")) {
        if (local[0] != '\0') {
            return (std::filesystem::path(local) / "cloudViewer" / "da3_models")
                .lexically_normal()
                .string();
        }
    }
#else
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return joinCacheRoot(home);
        }
    }
    if (const char* xdg_cache = std::getenv("XDG_CACHE_HOME")) {
        if (xdg_cache[0] != '\0') {
            return (std::filesystem::path(xdg_cache) / "cloudViewer" / "da3_models")
                .lexically_normal()
                .string();
        }
    }
#endif

    return (std::filesystem::temp_directory_path() / "cloudViewer_da3_models")
        .lexically_normal()
        .string();
}

}  // namespace da
