// BPE merge implementation adapted from sam3-ggml (MIT).
// Copyright (c) 2025-2026 Pierre-Antoine Bannier. See CLIP_TOKENIZER_LICENSE.
#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct clip_bpe_tokenizer {
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<int, std::string> decoder;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, int> merge_ranks;       // "a\x1fb" → rank
    std::unordered_map<uint8_t, std::string> byte_encoder;  // byte → unicode UTF-8
    std::unordered_map<std::string, std::string> cache;
    int sot_token = 49406;
    int eot_token = 49407;
};

static int clip_utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if (c < 0xC0) return 1;  // continuation (shouldn't start here)
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

static std::string clip_codepoint_to_utf8(int cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

static void clip_init_byte_encoder(std::unordered_map<uint8_t, std::string>& enc) {
    // Collect printable byte values
    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    // Corresponding codepoints (printable → identity)
    std::vector<int> cs(bs.begin(), bs.end());

    // Non-printable bytes get codepoints starting at 256
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    enc.clear();
    for (size_t i = 0; i < bs.size(); ++i) {
        enc[(uint8_t)bs[i]] = clip_codepoint_to_utf8(cs[i]);
    }
}

/*
** ── Merge key helper ─────────────────────────────────────────────────────────
*/

// Unit separator (0x1F) cannot appear in byte-encoded BPE tokens.
static inline std::string clip_merge_key(const std::string& a, const std::string& b) {
    std::string k;
    k.reserve(a.size() + 1 + b.size());
    k += a;
    k += '\x1f';
    k += b;
    return k;
}

static std::vector<std::string> clip_utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        int len = clip_utf8_len((uint8_t)s[i]);
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// Apply BPE merges to a byte-encoded word string.
// Returns space-separated BPE tokens (e.g. "he llo</w>").
static std::string clip_bpe_encode(clip_bpe_tokenizer& tok, const std::string& token) {
    auto cit = tok.cache.find(token);
    if (cit != tok.cache.end()) return cit->second;

    // Split into unicode chars, append </w> to last
    std::vector<std::string> word = clip_utf8_chars(token);
    if (word.empty()) return "";
    word.back() += "</w>";

    if (word.size() == 1) {
        tok.cache[token] = word[0];
        return word[0];
    }

    while (true) {
        // Find pair with lowest merge rank
        int best_rank = INT_MAX;
        std::string best_first, best_second;

        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = tok.merge_ranks.find(clip_merge_key(word[i], word[i + 1]));
            if (it != tok.merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_first = word[i];
                best_second = word[i + 1];
            }
        }

        if (best_rank == INT_MAX) break;

        // Merge all occurrences of this pair
        std::string merged = best_first + best_second;
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() &&
                word[i] == best_first && word[i + 1] == best_second) {
                new_word.push_back(merged);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = std::move(new_word);
        if (word.size() == 1) break;
    }

    // Join with spaces
    std::string result;
    for (size_t i = 0; i < word.size(); ++i) {
        if (i > 0) result += ' ';
        result += word[i];
    }
    tok.cache[token] = result;
    return result;
}
