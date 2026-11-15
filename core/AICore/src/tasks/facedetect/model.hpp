#pragma once
#include "tasks/facedetect/model_loader.hpp"

#include "tasks/facedetect/detect.hpp"

#include "tasks/facedetect/image_io.hpp"

#include "tasks/facedetect/landmark_graph.hpp"


#include <memory>
#include <string>
#include <vector>

namespace fd {

// One analyzed face: detection + (optionally) the embedding and genderage
// attributes filled in by the pipeline.
struct Face {
    Detection det;                  // box + score + landmarks
    std::vector<float> embedding;   // L2-normalized, embed_dim (empty if not run)
    int   age    = -1;              // genderage age (-1 if not run)
    char  gender = '?';             // 'M' / 'F' / '?' (not run)
    float spoof_score = -1.0f;      // anti-spoof live-ness score (-1 if not run)
};

// Dense landmark output for one face (106-point 2D + optional 68-point 3D).
struct DenseLandmarkFace {
    Detection det;
    std::vector<LandmarkPoint> points_2d;
    std::vector<LandmarkPoint> points_3d;
};

// Load-once face-recognition context.
//
// Loads a GGUF model pack ONCE (owns the ModelLoader) and reuses it across many
// calls - this is what the flat C-API holds. The stage components (detector,
// aligner, recognizer, genderage, anti-spoof) are lightweight views over the
// ModelLoader, constructed per call; the expensive GGUF parse + weight mapping
// happens exactly once, in load().
class Model {
public:
    // Loads the GGUF pack at `gguf_path`. Returns nullptr on failure (no throw).
    static std::unique_ptr<Model> load(const std::string& gguf_path);

    // Detect all faces in an image (boxes + scores + landmarks), no embedding.
    // Throws std::runtime_error on failure.
    std::vector<Detection> detect(const Image& img) const;

    // Detect the primary face, align (norm_crop) and run ArcFace, returning the
    // L2-normalized embedding (length config().embed_dim). Throws on failure
    // (no face found, unsupported pack, ...).
    std::vector<float> embed(const Image& img) const;
    std::vector<float> embed(const Image& img, float min_detection_score) const;

    /** ArcFace/SFace embed using pre-detected 5-point landmarks on \p img. */
    std::vector<float> embed(const Image& img, const Detection& det) const;

    // Analyze every face: detect + genderage (and anti-spoof when present).
    // Throws std::runtime_error on failure.
    std::vector<Face> analyze(const Image& img) const;

    // Liveness verdict for one detected face: true when the MiniFASNet ensemble
    // judges it real (averaged "real" probability >= 0.5). Returns true (no veto)
    // when the pack carries no anti-spoof models. Throws on graph failure.
    bool is_real(const Image& img, const Detection& d) const;

    // Dense landmark heads (106-pt 2D + 68-pt 3D) on pre-detected faces.
    std::vector<DenseLandmarkFace> dense_landmarks(
            const Image& img, const std::vector<Detection>& dets) const;

    const FaceConfig& config() const { return loader_.config(); }
    const ModelLoader& loader() const { return loader_; }

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    Model() = default;
    ModelLoader loader_;
};

} // namespace fd
