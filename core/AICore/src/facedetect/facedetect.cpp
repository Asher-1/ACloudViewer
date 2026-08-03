#include "facedetect.h"
#include "model.hpp"
#include "image_io.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#define FACEDETECT_VERSION "0.0.1"

extern "C" const char* facedetect_version(void) { return FACEDETECT_VERSION; }

namespace fd {

// Thin convenience wrappers. For repeated use (and for the flat C-API / LocalAI)
// use fd::Model directly to avoid reloading the GGUF on every call.

std::vector<float> embed(const std::string& model_path, const std::string& image_path) {
    std::unique_ptr<Model> model = Model::load(model_path);
    if (!model) throw std::runtime_error("facedetect: failed to load model: " + model_path);
    Image img;
    if (!load_image_rgb(image_path, img))
        throw std::runtime_error("facedetect: failed to load image: " + image_path);
    return model->embed(img);
}

float verify(const std::string& model_path, const std::string& a,
             const std::string& b, float /*threshold*/) {
    std::vector<float> ea = embed(model_path, a);
    std::vector<float> eb = embed(model_path, b);
    // Cosine distance over the two L2-normalized embeddings.
    double dot = 0.0;
    const size_t n = ea.size() < eb.size() ? ea.size() : eb.size();
    for (size_t i = 0; i < n; ++i) dot += (double)ea[i] * (double)eb[i];
    return (float)(1.0 - dot);
}

} // namespace fd
