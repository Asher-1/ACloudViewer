// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// GKDT ggml runtime - SimpleTokenizer (CLIP-style BPE) for dinotxt.
#include "tasks/gkd/gkd_tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "tasks/gkd/gkd_common.hpp"

namespace gkd {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------
namespace {

// Decode one UTF-8 code point starting at s[i]; advances i.
static uint32_t utf8_next(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
        i += 1;
        return c;
    }
    int len = 0;
    uint32_t cp = 0;
    if ((c & 0xE0) == 0xC0) {
        len = 2;
        cp = c & 0x1F;
    } else if ((c & 0xF0) == 0xE0) {
        len = 3;
        cp = c & 0x0F;
    } else if ((c & 0xF8) == 0xF0) {
        len = 4;
        cp = c & 0x07;
    } else {
        i += 1;
        return 0xFFFD;
    }
    if (i + len > s.size()) {
        i = s.size();
        return 0xFFFD;
    }
    for (int k = 1; k < len; k++) {
        unsigned char cc = (unsigned char)s[i + k];
        if ((cc & 0xC0) != 0x80) {
            i += 1;
            return 0xFFFD;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    i += len;
    return cp;
}

static void utf8_append(std::string& s, uint32_t cp) {
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
}

// Unicode category approximation used by the CLIP regex
//   <\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
// Covers ASCII exactly; well-known non-ASCII letter/digit/punctuation ranges
// are classified explicitly, and unknown code points default to letter (the
// overwhelming majority of unlisted scripts are letters).
enum class Cat { Space, Letter, Digit, Other };

static Cat category(uint32_t cp) {
    if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) return Cat::Space;
    if (cp < 0x80) {
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z'))
            return Cat::Letter;
        if (cp >= '0' && cp <= '9') return Cat::Digit;
        return Cat::Other;
    }
    // unicode whitespace
    if (cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
        cp == 0x3000)
        return Cat::Space;
    // combining marks: ftfy runs NFC before the regex, so marks attach to the
    // preceding letter; approximating that by folding marks into Letter.
    if ((cp >= 0x300 && cp <= 0x36F) || (cp >= 0x483 && cp <= 0x489) ||
        (cp >= 0x591 && cp <= 0x5C7) || (cp >= 0x1AB0 && cp <= 0x1AFF) ||
        (cp >= 0x1DC0 && cp <= 0x1DFF))
        return Cat::Letter;
    // digits / numbers (Nd and the common No ranges)
    if ((cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9) ||
        (cp >= 0x0966 && cp <= 0x096F) || (cp >= 0xFF10 && cp <= 0xFF19) ||
        cp == 0xB2 || cp == 0xB3 || cp == 0xB9 || (cp >= 0xBC && cp <= 0xBE) ||
        (cp >= 0x2070 && cp <= 0x2079) || (cp >= 0x2080 && cp <= 0x2089) ||
        (cp >= 0x2150 && cp <= 0x218F) || (cp >= 0x2460 && cp <= 0x2473) ||
        cp == 0x3007)
        return Cat::Digit;
    // punctuation / symbols / format characters -> [^\s\p{L}\p{N}]+
    if ((cp >= 0xA1 && cp <= 0xBF && cp != 0xAA && cp != 0xB5 && cp != 0xBA) ||
        cp == 0xAD || cp == 0xD7 || cp == 0xF7 ||
        (cp >= 0x2000 &&
         cp <= 0x206F) ||  // general punctuation (spaces handled above)
        (cp >= 0x207A && cp <= 0x207E) ||
        (cp >= 0x208A && cp <= 0x208E) ||
        (cp >= 0x20A0 && cp <= 0x20CF) ||  // currency
        (cp >= 0x2190 &&
         cp <= 0x2BFF) ||  // arrows, math operators, misc symbols, dingbats
        (cp >= 0x2E00 && cp <= 0x2E7F) ||  // supplemental punctuation
        (cp >= 0x3001 && cp <= 0x303F) ||  // CJK punctuation (3000 is space)
        (cp >= 0xFE10 && cp <= 0xFE19) ||
        (cp >= 0xFE30 && cp <= 0xFE52) || (cp >= 0xFE54 && cp <= 0xFE66) ||
        (cp >= 0xFE68 && cp <= 0xFE6B) || (cp >= 0xFF01 && cp <= 0xFF20) ||
        (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65) ||
        (cp >= 0xFFE0 && cp <= 0xFFEF))
        return Cat::Other;
    // letters: latin-1 supplement/extended, greek, cyrillic, hebrew, arabic,
    // devanagari, CJK, kana, hangul and general fallback
    if ((cp >= 0xC0 && cp <= 0x24F && cp != 0xD7 && cp != 0xF7) ||
        (cp >= 0x370 && cp <= 0x3FF) || (cp >= 0x400 && cp <= 0x4FF) ||
        (cp >= 0x531 && cp <= 0x58F) || (cp >= 0x590 && cp <= 0x6FF) ||
        (cp >= 0x900 && cp <= 0x97F) || (cp >= 0x1E00 && cp <= 0x1FFF) ||
        (cp >= 0x2C60 && cp <= 0x2C7F) || (cp >= 0x3040 && cp <= 0x30FF) ||
        (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) ||
        (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0xFF21 && cp <= 0xFF3A) || (cp >= 0xFF41 && cp <= 0xFF5A) ||
        (cp >= 0xFF66 && cp <= 0xFF9D) ||
        (cp >= 0x24B6 && cp <= 0x24E9))  // circled letters
        return Cat::Letter;
    return Cat::Letter;  // default: treat unknown as letter
}

// ---------------------------------------------------------------------------
// ftfy subset (the official tokenizer applies ftfy.fix_text before the regex).
// Implemented pieces, in the official fixer's order:
//   fix_encoding          latin-1 / sloppy-cp1252 mojibake repair (one layer,
//                         matching ftfy's default behaviour on our probes)
//   fix_latin_ligatures, fix_character_width, uncurl_quotes
//   fix_c1_controls       remaining C1 controls -> their cp1252 character
//   remove_control_chars  strip Cc except \t \n \r
// Not implemented: fix_encoding's exotic source encodings (1251/1250/macroman),
// replace_lossy_sequences detail, NFC normalization (approximated by folding
// combining marks into the preceding word in category()).
// ---------------------------------------------------------------------------

// reverse map for the sloppy bytes 0x80-0x9F (cp1252 interpretations)
static bool cp1252_reverse(uint32_t cp, unsigned char& byte) {
    static const struct {
        uint32_t cp;
        unsigned char b;
    } tbl[] = {{0x20AC, 0x80}, {0x201A, 0x82}, {0x0192, 0x83}, {0x201E, 0x84},
               {0x2026, 0x85}, {0x2020, 0x86}, {0x2021, 0x87}, {0x02C6, 0x88},
               {0x2030, 0x89}, {0x0160, 0x8A}, {0x2039, 0x8B}, {0x0152, 0x8C},
               {0x017D, 0x8E}, {0x2018, 0x91}, {0x2019, 0x92}, {0x201C, 0x93},
               {0x201D, 0x94}, {0x2022, 0x95}, {0x2013, 0x96}, {0x2014, 0x97},
               {0x02DC, 0x98}, {0x2122, 0x99}, {0x0161, 0x9A}, {0x203A, 0x9B},
               {0x0153, 0x9C}, {0x017E, 0x9E}, {0x0178, 0x9F}};
    for (const auto& e : tbl)
        if (e.cp == cp) {
            byte = e.b;
            return true;
        }
    return false;
}

// strict UTF-8 validation: rejects overlongs, surrogates and > U+10FFFF
static bool utf8_valid(const std::string& b) {
    size_t i = 0, n = b.size();
    while (i < n) {
        unsigned char c = (unsigned char)b[i];
        uint32_t cp;
        size_t len;
        if (c < 0x80) {
            i++;
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        } else
            return false;
        if (i + len > n) return false;
        for (size_t k = 1; k < len; k++) {
            unsigned char cc = (unsigned char)b[i + k];
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
            (len == 4 && cp < 0x10000))
            return false;                                // overlong
        if (cp >= 0xD800 && cp <= 0xDFFF) return false;  // surrogate
        if (cp > 0x10FFFF) return false;
        i += len;
    }
    return true;
}

// One layer of mojibake repair: every maximal run of latin-1/sloppy-cp1252
// encodable characters (all with cp >= 0x80) is re-encoded to bytes; when
// those bytes form valid UTF-8 that strictly reduces the number of
// latin-1-range code points, the run is replaced (the monotonic rule ftfy
// enforces through its badness scoring).
static void fix_encoding(std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    std::vector<uint32_t> run;  // codepoints of the current run
    std::string run_bytes;      // original utf-8 bytes of the run
    auto flush_run = [&]() {
        if (run.empty()) return;
        bool ok = true;
        std::string bytes;
        for (uint32_t cp : run) {
            if (cp <= 0xFF) {
                bytes += (char)(unsigned char)cp;
                continue;
            }
            unsigned char b;
            if (!cp1252_reverse(cp, b)) {
                ok = false;
                break;
            }
            bytes += (char)b;
        }
        bool applied = false;
        if (ok && utf8_valid(bytes)) {
            size_t j = 0, ncp = 0;
            while (j < bytes.size()) {
                utf8_next(bytes, j);
                ncp++;
            }
            if (ncp < run.size()) {  // strictly less mojibake-looking
                out += bytes;
                applied = true;
            }
        }
        if (!applied) out += run_bytes;
        run.clear();
        run_bytes.clear();
    };
    while (i < s.size()) {
        size_t start = i;
        uint32_t cp = utf8_next(s, i);
        unsigned char sink;
        bool in_run = cp >= 0x80 && (cp <= 0xFF || cp1252_reverse(cp, sink));
        if (in_run) {
            run.push_back(cp);
            run_bytes.append(s, start, i - start);
        } else {
            flush_run();
            out.append(s, start, i - start);
        }
    }
    flush_run();
    s = std::move(out);
}

static void fix_c1_controls(std::string& s) {
    // remaining C1 *characters* (U+0080..U+009F) -> their cp1252 character.
    // Operates on code points: bytes 0x80-0x9F inside multi-byte sequences
    // are continuation bytes, not C1 controls.
    static const struct {
        unsigned char b;
        uint32_t cp;
    } fwd[] = {{0x80, 0x20AC}, {0x82, 0x201A}, {0x83, 0x0192}, {0x84, 0x201E},
               {0x85, 0x2026}, {0x86, 0x2020}, {0x87, 0x2021}, {0x88, 0x02C6},
               {0x89, 0x2030}, {0x8A, 0x0160}, {0x8B, 0x2039}, {0x8C, 0x0152},
               {0x8E, 0x017D}, {0x91, 0x2018}, {0x92, 0x2019}, {0x93, 0x201C},
               {0x94, 0x201D}, {0x95, 0x2022}, {0x96, 0x2013}, {0x97, 0x2014},
               {0x98, 0x02DC}, {0x99, 0x2122}, {0x9A, 0x0161}, {0x9B, 0x203A},
               {0x9C, 0x0153}, {0x9E, 0x017E}, {0x9F, 0x0178}};
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        size_t start = i;
        uint32_t cp = utf8_next(s, i);
        if (cp >= 0x80 && cp <= 0x9F) {
            uint32_t rep = 0;
            for (const auto& e : fwd)
                if (e.b == (unsigned char)cp) {
                    rep = e.cp;
                    break;
                }
            if (rep) {
                if (rep < 0x80)
                    out += (char)rep;
                else if (rep < 0x800) {
                    out += (char)(0xC0 | (rep >> 6));
                    out += (char)(0x80 | (rep & 0x3F));
                } else {
                    out += (char)(0xE0 | (rep >> 12));
                    out += (char)(0x80 | ((rep >> 6) & 0x3F));
                    out += (char)(0x80 | (rep & 0x3F));
                }
                continue;
            }
        }
        out.append(s, start, i - start);
    }
    s = std::move(out);
}

static void remove_control_chars(std::string& s) {
    // strip Cc code points except \t \n \r (code-point level, like ftfy)
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        size_t start = i;
        uint32_t cp = utf8_next(s, i);
        bool is_cc = (cp < 0x20 && cp != '\t' && cp != '\n' && cp != '\r') ||
                     cp == 0x7F;
        if (!is_cc) out.append(s, start, i - start);
    }
    s = std::move(out);
}

static void ftfy_subset(std::string& s) {
    fix_encoding(s);
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = utf8_next(s, i);
        switch (cp) {
            case 0x2018:
            case 0x2019:
            case 0x201A:
            case 0x201B:
                out += '\'';
                break;
            case 0x201C:
            case 0x201D:
            case 0x201E:
            case 0x201F:
                out += '"';
                break;
            case 0xFB00:
                out += "ff";
                break;
            case 0xFB01:
                out += "fi";
                break;
            case 0xFB02:
                out += "fl";
                break;
            case 0xFB03:
                out += "ffi";
                break;
            case 0xFB04:
                out += "ffl";
                break;
            default:
                if (cp >= 0xFF01 && cp <= 0xFF5E) {
                    out += (char)(cp - 0xFF01 + 0x21);  // fullwidth -> ASCII
                } else {
                    // re-encode cp (shortest utf-8 form; matches the input
                    // bytes for all valid utf-8 that utf8_next accepts)
                    if (cp < 0x80)
                        out += (char)cp;
                    else if (cp < 0x800) {
                        out += (char)(0xC0 | (cp >> 6));
                        out += (char)(0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        out += (char)(0xE0 | (cp >> 12));
                        out += (char)(0x80 | ((cp >> 6) & 0x3F));
                        out += (char)(0x80 | (cp & 0x3F));
                    } else {
                        out += (char)(0xF0 | (cp >> 18));
                        out += (char)(0x80 | ((cp >> 12) & 0x3F));
                        out += (char)(0x80 | ((cp >> 6) & 0x3F));
                        out += (char)(0x80 | (cp & 0x3F));
                    }
                }
        }
    }
    fix_c1_controls(out);
    remove_control_chars(out);
    s = std::move(out);
}

// minimal html.unescape for the entities that matter in practice
static void html_unescape(std::string& s) {
    const std::pair<const char*, const char*> ents[] = {
            {"&amp;", "&"},   {"&lt;", "<"},   {"&gt;", ">"},
            {"&quot;", "\""}, {"&apos;", "'"}, {"&nbsp;", " "},
            {"&#39;", "'"},   {"&#38;", "&"},  {"&ndash;", "-"},
            {"&mdash;", "-"},
    };
    for (auto& [from, to] : ents) {
        size_t pos;
        while ((pos = s.find(from)) != std::string::npos) {
            s.replace(pos, std::strlen(from), to);
        }
    }
}

// lowercase (ASCII exact; full-width/simple unicode ranges folded)
static void lower_ascii(std::string& s) {
    for (auto& c : s) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// SimpleTokenizer
// ---------------------------------------------------------------------------
bool SimpleTokenizer::init(const std::vector<std::string>& vocab,
                           const std::vector<std::string>& merges) {
    if (vocab.empty() || merges.empty()) {
        GKD_LOG_ERROR("tokenizer: empty vocab or merges");
        return false;
    }
    for (size_t i = 0; i < vocab.size(); i++) {
        encoder_[vocab[i]] = (int32_t)i;
    }
    for (size_t r = 0; r < merges.size(); r++) {
        const std::string& m = merges[r];
        size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        bpe_ranks_[m.substr(0, sp) + '\x01' + m.substr(sp + 1)] = (int32_t)r;
    }
    sot_id_ = encoder_["<|startoftext|>"];
    eot_id_ = encoder_["<|endoftext|>"];
    if (sot_id_ < 0 || eot_id_ < 0) {
        GKD_LOG_ERROR("tokenizer: SOT/EOT tokens missing from vocab");
        return false;
    }
    // bytes_to_unicode()
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++) {
        byte_encoder_[bs[i]].clear();
        utf8_append(byte_encoder_[bs[i]], (uint32_t)cs[i]);
    }
    return true;
}

std::vector<std::string> SimpleTokenizer::bpe(const std::string& token) const {
    auto it = cache_.find(token);
    if (it != cache_.end()) return it->second;

    // word = tuple(token[:-1]) + (token[-1] + '</w>',)
    // Symbols are unicode characters of the byte-encoded token string.
    std::vector<std::string> word;
    {
        size_t i = 0;
        std::vector<std::string> chars;
        while (i < token.size()) {
            size_t start = i;
            utf8_next(token, i);
            chars.push_back(token.substr(start, i - start));
        }
        if (chars.empty()) return {};
        for (size_t k = 0; k + 1 < chars.size(); k++) word.push_back(chars[k]);
        word.push_back(chars.back() + "</w>");
    }
    if (word.size() == 1) {
        cache_[token] = word;
        return word;
    }

    auto get_rank = [&](const std::string& a, const std::string& b) -> int {
        auto r = bpe_ranks_.find(a + '\x01' + b);
        return r == bpe_ranks_.end() ? INT32_MAX : r->second;
    };

    while (true) {
        // bigram = min(pairs, key=rank)
        int best_rank = INT32_MAX;
        size_t best_i = SIZE_MAX;
        for (size_t i = 0; i + 1 < word.size(); i++) {
            int r = get_rank(word[i], word[i + 1]);
            if (r < best_rank) {
                best_rank = r;
                best_i = i;
            }
        }
        if (best_i == SIZE_MAX || best_rank == INT32_MAX) break;
        const std::string first = word[best_i];
        const std::string second = word[best_i + 1];

        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            size_t j = i;
            while (j < word.size() && word[j] != first) j++;
            new_word.insert(new_word.end(), word.begin() + i, word.begin() + j);
            i = j;
            if (i >= word.size()) break;
            if (word[i] == first && i < word.size() - 1 &&
                word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = std::move(new_word);
        if (word.size() == 1) break;
    }

    cache_[token] = word;
    return word;
}

std::vector<int32_t> SimpleTokenizer::encode(
        const std::string& text_raw) const {
    // text = whitespace_clean(basic_clean(ftfy.fix_text(text))).lower()
    std::string text = text_raw;
    ftfy_subset(text);
    html_unescape(text);
    // strip + collapse whitespace
    std::string collapsed;
    bool in_space = false;
    size_t start = 0, end = text.size();
    while (start < end && (unsigned char)text[start] <= ' ') start++;
    while (end > start && (unsigned char)text[end - 1] <= ' ') end--;
    for (size_t i = start; i < end; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c <= ' ') {
            if (!in_space) collapsed += ' ';
            in_space = true;
        } else {
            collapsed += text[i];
            in_space = false;
        }
    }
    lower_ascii(collapsed);

    // regex findall + byte-encode + BPE
    std::vector<int32_t> out;
    size_t i = 0;
    std::string byte_seq;  // byte-encoded current token
    auto emit = [&](const std::string& tok) {
        if (tok.empty()) return;
        byte_seq.clear();
        for (unsigned char b : tok) byte_seq += byte_encoder_[b];
        for (auto& sym : bpe(byte_seq)) {
            auto e = encoder_.find(sym);
            if (e != encoder_.end()) out.push_back(e->second);
        }
    };

    while (i < collapsed.size()) {
        size_t start = i;
        uint32_t cp = utf8_next(collapsed, i);
        Cat cat = category(cp);

        // contractions (case-insensitive, applied on lowercased text)
        if (cp == '\'' && start + 1 < collapsed.size()) {
            static const char* contr[] = {"'s", "'t",  "'re", "'ve",
                                          "'m", "'ll", "'d"};
            bool matched = false;
            for (const char* c : contr) {
                size_t len = std::strlen(c);
                if (collapsed.compare(start, len, c) == 0) {
                    emit(collapsed.substr(start, len));
                    i = start + len;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }
        if (cat == Cat::Letter) {
            while (i < collapsed.size()) {
                size_t j = i;
                if (category(utf8_next(collapsed, j)) != Cat::Letter) break;
                i = j;
            }
            emit(collapsed.substr(start, i - start));
            continue;
        }
        if (cat == Cat::Digit) {
            // [\p{N}] matches exactly ONE numeric character
            emit(collapsed.substr(start, i - start));
            continue;
        }
        if (cat == Cat::Other) {
            while (i < collapsed.size()) {
                size_t j = i;
                Cat c2 = category(utf8_next(collapsed, j));
                if (c2 == Cat::Letter || c2 == Cat::Digit || c2 == Cat::Space)
                    break;
                i = j;
            }
            emit(collapsed.substr(start, i - start));
            continue;
        }
        // Space: skipped by the regex
    }
    return out;
}

std::vector<int32_t> SimpleTokenizer::tokenize(const std::string& text,
                                               int context_length) const {
    std::vector<int32_t> tokens;
    tokens.push_back(sot_id_);
    std::vector<int32_t> ids = encode(text);
    tokens.insert(tokens.end(), ids.begin(), ids.end());
    tokens.push_back(eot_id_);
    if ((int)tokens.size() > context_length) {
        tokens.resize(context_length);
        tokens.back() = eot_id_;
    }
    std::vector<int32_t> result(context_length, 0);
    for (size_t k = 0; k < tokens.size() && k < (size_t)context_length; k++)
        result[k] = tokens[k];
    return result;
}

}  // namespace gkd
