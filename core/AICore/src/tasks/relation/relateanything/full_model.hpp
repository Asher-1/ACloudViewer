#pragma once

#include "relateanything/options.hpp"
#include "relateanything/types.hpp"
#include <memory>

namespace relateanything {

enum class ImageFormat { rgb8 = 1, rgba8, gray8, bgr8, bgra8 };
// Borrowed through prepare_image; includes the real row stride and capacity.
struct ImageView {
    const std::uint8_t * data = nullptr;
    std::size_t width = 0, height = 0, row_stride_bytes = 0, data_bytes = 0;
    ImageFormat format = ImageFormat::rgb8;
};

struct ImageOptions {
    std::filesystem::path image;
    std::vector<float> boxes_xyxy;
    // Optional binary masks in original-image [N,H,W] order.
    std::vector<std::uint8_t> masks;
    std::vector<float> box_scores;
};

struct ImageInput {
    std::size_t width = 448;
    std::size_t height = 448;
    std::size_t object_count = 0;
    // RGB, CHW, ImageNet normalized after Pillow-compatible bilinear resize.
    std::vector<float> rgb_chw;
    std::vector<float> boxes_cxcywh;
    std::vector<float> coverage; // Optional [N,32,32] region coverage.
    std::vector<float> fill;
    std::vector<float> box_scores;
};

struct Relation {
    std::size_t subject = 0;
    std::size_t object = 0;
    std::size_t predicate = 0;
    float score = 0;
};

struct ImageOutput : PairOutput {
    std::vector<std::int32_t> valid_mask;
    std::vector<double> iteration_milliseconds;
    std::vector<Relation> relations;
    std::vector<Relation> spatial_relations;
    std::vector<Relation> semantic_relations;
    bool decomposed = false;
    // GGML-owned accelerator weight/graph buffers; excludes driver allocations.
    std::size_t backend_buffer_bytes = 0;
};

class FullModel {
public:
    explicit FullModel(const ModelOptions & options);
    ~FullModel();
    FullModel(const FullModel &) = delete;
    FullModel & operator=(const FullModel &) = delete;
    ImageOutput infer(const ImageInput & input);
    const std::vector<std::string> & predicates() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool is_image_input(const std::filesystem::path & path);
ImageInput prepare_image(const ImageOptions & options);
ImageInput prepare_image(const ImageView & image, const ImageOptions & regions);
ImageInput read_image_request(const std::filesystem::path & path);
ImageInput read_image_input(const std::filesystem::path & path);
void write_image_input(const std::filesystem::path & path, const ImageInput & input);
void write_image_output(const std::filesystem::path & path, const ImageOutput & output);
std::vector<float> encode_predicates(const ModelOptions & options, const std::vector<std::string> & texts);

} // namespace relateanything
