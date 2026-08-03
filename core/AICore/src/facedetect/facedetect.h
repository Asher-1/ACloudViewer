#ifndef FACEDETECT_H
#define FACEDETECT_H
#ifdef __cplusplus
extern "C" {
#endif
// Returns a static version string. Never null.
const char* facedetect_version(void);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
#include <vector>
namespace fd {

// Thin C++ convenience layer over the load-once pipeline. For repeated use (and
// for the flat C-API / LocalAI) prefer fd::Model directly to avoid reloading the
// GGUF on every call.

// Embed the primary face in an image file: detect -> align (norm_crop) ->
// ArcFace -> L2-normalized embedding (typically 512-d). Throws std::runtime_error
// on failure (model/image load, no face found, unsupported pack, ...).
std::vector<float> embed(const std::string& model_path, const std::string& image_path);

// Cosine distance (1 - cosine_similarity) between two faces' embeddings. <=
// threshold (default 0.35) means "same identity". Throws std::runtime_error on
// failure.
float verify(const std::string& model_path, const std::string& a,
             const std::string& b, float threshold = 0.35f);

} // namespace fd
#endif

#endif // FACEDETECT_H
