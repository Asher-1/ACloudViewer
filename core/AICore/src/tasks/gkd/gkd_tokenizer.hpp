// GKDT ggml runtime - SimpleTokenizer (CLIP-style BPE) for dinotxt.
// Byte-exact reimplementation of network/clip_kd/simple_tokenizer.py for the
// inputs seen in practice (well-formed UTF-8 text). The vocab/merges ship
// inside the GGUF file, so no external files are needed at runtime.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace gkd {

class SimpleTokenizer {
public:
    // vocab: full token strings (index == token id), merges: "left right" pairs
    // ordered by rank. Both come from the GGUF `gkd.tokenizer.*` arrays.
    bool init(const std::vector<std::string>& vocab, const std::vector<std::string>& merges);

    // [SOT] + encode(text) + [EOT], zero-padded to context_length (77),
    // truncated with a trailing EOT when longer - identical to the Python
    // dinov3 Tokenizer.tokenize().
    std::vector<int32_t> tokenize(const std::string& text, int context_length) const;

    int32_t sot_token() const { return sot_id_; }
    int32_t eot_token() const { return eot_id_; }

private:
    std::vector<int32_t> encode(const std::string& text) const;
    std::vector<std::string> bpe(const std::string& token) const;  // split symbols

    std::unordered_map<std::string, int32_t> encoder_;
    std::unordered_map<std::string, int32_t> bpe_ranks_;  // key: "left\x01right"
    std::string byte_encoder_[256];
    mutable std::unordered_map<std::string, std::vector<std::string>> cache_;
    int32_t sot_id_ = -1;
    int32_t eot_id_ = -1;
};

}  // namespace gkd
