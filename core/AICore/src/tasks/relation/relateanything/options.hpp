#pragma once

#include "relateanything/backend_binding.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace relateanything {

struct ModelOptions {
    std::vector<std::string> predicate_texts;
    ra_backend_binding backend_binding{};
    std::size_t device_index = 0;
    std::filesystem::path model;
    std::filesystem::path dump_directory;
    std::filesystem::path vocabulary;
    std::filesystem::path vocabulary_names;
    std::filesystem::path predicate_type_map;
    std::filesystem::path text_student;
    std::filesystem::path tokenizer;
    std::filesystem::path texts;
    std::filesystem::path token_output;
    std::vector<std::string> text_templates;
    std::string backend = "cpu";
    std::size_t threads = 0;
    std::size_t max_boxes = 60;
    std::size_t topk = 128;
    bool decompose = false;
    bool flash_attention = true;
    bool compact_pairs = true;
    bool matmul_sampler = false;
    bool matmul_geometry = false;
    bool fused_vocabulary = true;
    bool cache_positions = true;
    bool retain_features = true;
    bool quant_f32_input = false;
    bool strict_f32 = false;
    bool strict_f32_auto = true;
    bool raw_logits = false;
    bool profile = false;

    void validate_model() const;
};

} // namespace relateanything
