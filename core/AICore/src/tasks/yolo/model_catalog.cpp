// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCryptographicHash>
#include <array>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "aicore/yolo_capi.h"
#include "gguf.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "yolo_gguf_models/";

// One catalog family (release filename stem without the -<quant> suffix).
// Filenames follow the yolo_gguf_models release exactly (61 variants x 3
// quantizations = 183 gguf assets, verified against the GitHub release
// expanded-assets listing).
struct VariantInfo {
    const char* name;     // release stem ("yolov8n", "yolo26n-depth", ...)
    const char* display;  // user-facing name
    const char* task;     // GGUF yolo.task ("text" = encoder-only tower)
    int text_input;       // text-conditioned head / text encoder tower
};

static constexpr const VariantInfo kVariants[] = {
        // Closed-set detection (COCO-80).
        {"yolov8n", "YOLOv8 Nano", "detect", 0},
        {"yolov8s", "YOLOv8 Small", "detect", 0},
        {"yolov8m", "YOLOv8 Medium", "detect", 0},
        {"yolov8l", "YOLOv8 Large", "detect", 0},
        {"yolov8x", "YOLOv8 XLarge", "detect", 0},
        {"yolo26n", "YOLO26 Nano", "detect", 0},
        {"yolo26s", "YOLO26 Small", "detect", 0},
        {"yolo26m", "YOLO26 Medium", "detect", 0},
        {"yolo26l", "YOLO26 Large", "detect", 0},
        {"yolo26x", "YOLO26 XLarge", "detect", 0},
        // Absolute depth (768 input).
        {"yolo26n-depth", "YOLO26 Nano Depth", "depth", 0},
        {"yolo26s-depth", "YOLO26 Small Depth", "depth", 0},
        {"yolo26m-depth", "YOLO26 Medium Depth", "depth", 0},
        {"yolo26l-depth", "YOLO26 Large Depth", "depth", 0},
        {"yolo26x-depth", "YOLO26 XLarge Depth", "depth", 0},
        // Instance segmentation.
        {"yolov8n-seg", "YOLOv8 Nano (Seg)", "segment", 0},
        {"yolov8s-seg", "YOLOv8 Small (Seg)", "segment", 0},
        {"yolov8m-seg", "YOLOv8 Medium (Seg)", "segment", 0},
        {"yolov8l-seg", "YOLOv8 Large (Seg)", "segment", 0},
        {"yolov8x-seg", "YOLOv8 XLarge (Seg)", "segment", 0},
        {"yolo26n-seg", "YOLO26 Nano (Seg)", "segment", 0},
        {"yolo26s-seg", "YOLO26 Small (Seg)", "segment", 0},
        {"yolo26m-seg", "YOLO26 Medium (Seg)", "segment", 0},
        {"yolo26l-seg", "YOLO26 Large (Seg)", "segment", 0},
        {"yolo26x-seg", "YOLO26 XLarge (Seg)", "segment", 0},
        // Keypoints (COCO-17 person pose).
        {"yolo26n-pose", "YOLO26 Nano (Pose)", "pose", 0},
        {"yolo26s-pose", "YOLO26 Small (Pose)", "pose", 0},
        {"yolo26m-pose", "YOLO26 Medium (Pose)", "pose", 0},
        {"yolo26l-pose", "YOLO26 Large (Pose)", "pose", 0},
        {"yolo26x-pose", "YOLO26 XLarge (Pose)", "pose", 0},
        // Oriented boxes (DOTA-15).
        {"yolo26n-obb", "YOLO26 Nano (OBB)", "obb", 0},
        {"yolo26s-obb", "YOLO26 Small (OBB)", "obb", 0},
        {"yolo26m-obb", "YOLO26 Medium (OBB)", "obb", 0},
        {"yolo26l-obb", "YOLO26 Large (OBB)", "obb", 0},
        {"yolo26x-obb", "YOLO26 XLarge (OBB)", "obb", 0},
        // Semantic segmentation (Cityscapes-19).
        {"yolo26n-sem", "YOLO26 Nano (Semantic)", "semantic", 0},
        {"yolo26s-sem", "YOLO26 Small (Semantic)", "semantic", 0},
        {"yolo26m-sem", "YOLO26 Medium (Semantic)", "semantic", 0},
        {"yolo26l-sem", "YOLO26 Large (Semantic)", "semantic", 0},
        {"yolo26x-sem", "YOLO26 XLarge (Semantic)", "semantic", 0},
        // Classification (ImageNet-1000, 224 input).
        {"yolo26n-cls", "YOLO26 Nano (Classify)", "classify", 0},
        {"yolo26s-cls", "YOLO26 Small (Classify)", "classify", 0},
        {"yolo26m-cls", "YOLO26 Medium (Classify)", "classify", 0},
        {"yolo26l-cls", "YOLO26 Large (Classify)", "classify", 0},
        {"yolo26x-cls", "YOLO26 XLarge (Classify)", "classify", 0},
        // Open-vocabulary detection (CLIP text embeddings).
        {"yolov8s-world", "YOLOv8 Small (World)", "detect", 1},
        {"yolov8m-world", "YOLOv8 Medium (World)", "detect", 1},
        {"yolov8l-world", "YOLOv8 Large (World)", "detect", 1},
        {"yolov8x-world", "YOLOv8 XLarge (World)", "detect", 1},
        // Open-vocabulary instance segmentation (MobileCLIP text tower).
        {"yoloe-26n-seg", "YOLOE26 Nano (Seg)", "segment", 1},
        {"yoloe-26s-seg", "YOLOE26 Small (Seg)", "segment", 1},
        {"yoloe-26m-seg", "YOLOE26 Medium (Seg)", "segment", 1},
        {"yoloe-26l-seg", "YOLOE26 Large (Seg)", "segment", 1},
        {"yoloe-26x-seg", "YOLOE26 XLarge (Seg)", "segment", 1},
        // Prompt-free YOLOE variants (image-derived vocabulary).
        {"yoloe-26n-seg-pf", "YOLOE26 Nano (Seg, Prompt-Free)", "segment", 1},
        {"yoloe-26s-seg-pf", "YOLOE26 Small (Seg, Prompt-Free)", "segment", 1},
        {"yoloe-26m-seg-pf", "YOLOE26 Medium (Seg, Prompt-Free)", "segment", 1},
        {"yoloe-26l-seg-pf", "YOLOE26 Large (Seg, Prompt-Free)", "segment", 1},
        {"yoloe-26x-seg-pf", "YOLOE26 XLarge (Seg, Prompt-Free)", "segment", 1},
        // Text-encoder towers (also usable standalone).
        {"clip-ViT-B-32", "CLIP ViT-B/32 (Text)", "text", 1},
        {"mobileclip2_b", "MobileCLIP2-B (Text)", "text", 1},
};

static constexpr int kVariantCount = sizeof(kVariants) / sizeof(kVariants[0]);

// 3 quantization suffixes.
static constexpr const char* kQuantSuffixes[] = {"f32", "f16", "q8_0"};

static constexpr int kQuantCount =
        sizeof(kQuantSuffixes) / sizeof(kQuantSuffixes[0]);

// Descriptive quant notes.
static constexpr const char* kQuantNotes[] = {
        "F32 \xe2\x80\x94 full precision reference",
        "F16 \xe2\x80\x94 half precision (recommended)",
        "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade",
};

// MSVC names the POSIX helper "_strdup"; keep a portable wrapper so the
// catalog builds warning-clean on all three platforms.
static char* dupString(const char* s) {
#ifdef _MSC_VER
    return _strdup(s);
#else
    return strdup(s);
#endif
}

struct ModelRow {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    const char* task;
    int depth_capable;
    int end2end;
    int text_input;
    int64_t expected_bytes; /* 0 = no official baseline */
    const char* sha256;     /* NULL = no official baseline */
};

// Official release digests (exact byte count + SHA-256) for the 33 original
// detect/depth assets, from the yolo_gguf_models release audit
// (2026-08-19, ultralytics-ggml-integration-plan.md 3.2). Later additions
// (segment/world/yoloe/pose/obb/sem/cls/depth s..x/text) have no published
// baseline yet — omitted, verify_model skips size/hash for them (still
// checks magic + task). Order matches kQuantSuffixes.
struct VariantDigest {
    const char* variant;
    int64_t bytes[3];  // f32, f16, q8_0
    const char* sha256[3];
};

static constexpr VariantDigest kDigests[] = {
        {"yolov8n",
         {12634464, 6342112, 3586144},
         {"b313ee45aaccec6543ebbaba9a87947a5d1e31a703982fa433adf28508447a84",
          "cef427cf9f87c6f4aa7c0a1758d7bf9386476bb4a8db6bddcbe0ac375a877b07",
          "53d9d3a8d078ead04ebd48fb473a8dc6f7a104cfff8387fd0c0d7d940839a4fa"}},
        {"yolov8s",
         {44656928, 22364736, 11915104},
         {"f4c38e042844f6ac1b572aa14cf3964c7ce47f8dda689a08f6f07997b8726e5b",
          "e41d5a5a5fcf4f7b9349cddd9d3d3af2753587ce2e4da626b2ad1ee4a576a69d",
          "21e9eeaf334de22abd1fef6a0aa8833ff9233592b9a836869d42ccd6c47fe757"}},
        {"yolov8m",
         {103583552, 51845376, 27709728},
         {"1466d8a4dc74dfcb6c54910eb1d7ec951c0081fffc9db6812ed5fc3a4fb2a2ab",
          "2f0d38e759deffd3f00018c4c1cf251e5003f6191d029b6889d375ac52019d9b",
          "70a5ee76ed7a46fb685dec6321cb764ac322e2440430e721fbf89cde9bda35f9"}},
        {"yolov8l",
         {174720864, 87431616, 46514784},
         {"25ec233d6ffd3f9ab04961cd3f2feda1e4a20d104a96a69558089a5762ad67e9",
          "e279885793c693933879ebf2e112b842d0f61b75c9b655dd28091199e9bf8325",
          "c77559bf9936c2825b8a32c41d55a16ec1704969d217f0456de4cf6313b89507"}},
        {"yolov8x",
         {272850144, 136507872, 73266464},
         {"9b9b880a7877aaeed73ad212f67555520580c440971238a93ba69ef9db7aea0a",
          "9d342a19cc00cdcbfa3a2e3454377f8faaa2493f84f31e9735ddfaee67b25d11",
          "ce168f3a0309a6c277b08e36aee4e92086699f8765b701f6bae3377dc9a485b8"}},
        {"yolo26n",
         {9682368, 4882048, 2704096},
         {"7f0b0aae5dd19b8fa5d2508db047d276ec0ce2649d4f22900f33165703d31b76",
          "303b1880135792e336be7598505d093915a3640f1b3ddffe057b532a2a7df400",
          "09768588eeaba5fd9f33a65ec675668a6410d2d636255091b0c2ed2a4b1b982b"}},
        {"yolo26s",
         {38031200, 19072928, 10205088},
         {"cb5fbf8f25100a8382a89ab9d8bfa99d126f2c591be49a872fcbba7ea511dca0",
          "051e3ba31b9ed510e44e899bde399d80143cb9df5e9c27acd06ac47acfa0fdac",
          "c191bc5a9dbfb948f4c0d4f93b53ff0254b9d447b2d693332bd6b0d069fe0689"}},
        {"yolo26m",
         {81695456, 40921952, 21831008},
         {"a54156c004a84381d6ac6baca9ffb5956e5ae884ef442bf3a694e9bf09e59dcf",
          "8653003dbc6ed933c5e19bafb5bba111d8812446daa976408431aada0a8e7023",
          "4fdd32e1b1b9ff4034546377d8ea272b698d2832b25804a54b34ea59f13ec0b7"}},
        {"yolo26l",
         {99303232, 49751488, 26547040},
         {"c94feb22c75b123ff1ca722a20a857274402fe796a21dda0e4574a294dd8acf3",
          "ed5a89f54d1795bf438033aaf3b2ae28890a9ed07e53ad1a9b4f06a0f687c6c8",
          "2e61102f589c76a97e331c75c4537c41ea0ff346130e355c4445ec20281c8aaf"}},
        {"yolo26x",
         {222977344, 111619840, 59612256},
         {"d3147c55065a447609af4854a068143f9c51d03fd019c37c3436d13d06fd102e",
          "527e17a761b4a996f55a889faac06e946db6fd857e70363ee164b5431fcf3ec6",
          "7d5a6e94e642034c63e7907914a0ebae5a6e615d595ad057528d086303671e02"}},
        {"yolo26n-depth",
         {20703040, 10390368, 5641440},
         {"13830a5e4d95e68fd165a5c298c82daadb5be47a686de56b82a5bd121e9b3ef7",
          "6ca6d946e774b28ce8a94e44d4eb20368fca61572a2934429c675cfb0868f795",
          "0d5795cd182c8c79c4c1a6f92e549f8e39841733f067d6b20764da568dfbac2f"}},
};

static_assert(sizeof(kDigests) / sizeof(kDigests[0]) == 11,
              "digest count mismatch");

static const VariantDigest* findDigest(const char* variant) {
    for (const auto& d : kDigests) {
        if (std::strcmp(d.variant, variant) == 0) return &d;
    }
    return nullptr;
}

// Build the flat model list at init time.
static std::vector<ModelRow> buildModels() {
    std::vector<ModelRow> rows;
    rows.reserve(kVariantCount * kQuantCount);
    for (int vi = 0; vi < kVariantCount; ++vi) {
        const VariantInfo& info = kVariants[vi];
        for (int qi = 0; qi < kQuantCount; ++qi) {
            std::string filename =
                    std::string(info.name) + "-" + kQuantSuffixes[qi] + ".gguf";
            std::string url = std::string(kDownloadBase) + filename;
            std::string display = std::string(info.display) + " \xe2\x80\x94 " +
                                  kQuantNotes[qi];
            rows.push_back({dupString(filename.c_str()), dupString(url.c_str()),
                            dupString(display.c_str()),
                            dupString(kQuantNotes[qi]),
                            "AGPL-3.0 (Ultralytics)", info.task,
                            std::strcmp(info.task, "depth") == 0 ? 1 : 0,
                            std::strncmp(info.name, "yolo26", 6) == 0 &&
                                            std::strcmp(info.task, "text") != 0
                                    ? 1
                                    : 0,
                            info.text_input, 0, nullptr});
            if (const VariantDigest* d = findDigest(info.name)) {
                rows.back().expected_bytes = d->bytes[qi];
                rows.back().sha256 = d->sha256[qi];
            }
        }
    }
    return rows;
}

static const std::vector<ModelRow> kModels = buildModels();

static int modelCount() { return static_cast<int>(kModels.size()); }

static bool roleMatches(enum aicore_yolo_model_role role, const ModelRow& row) {
    const bool closed_set = row.text_input == 0;
    switch (role) {
        case AICORE_YOLO_ROLE_DETECTION:
            return closed_set && row.task != nullptr &&
                   std::strcmp(row.task, "detect") == 0;
        case AICORE_YOLO_ROLE_DEPTH:
            return row.task != nullptr && std::strcmp(row.task, "depth") == 0;
        case AICORE_YOLO_ROLE_SEGMENT:
            return closed_set && row.task != nullptr &&
                   std::strcmp(row.task, "segment") == 0;
        case AICORE_YOLO_ROLE_POSE:
            return row.task != nullptr && std::strcmp(row.task, "pose") == 0;
        case AICORE_YOLO_ROLE_OBB:
            return row.task != nullptr && std::strcmp(row.task, "obb") == 0;
        case AICORE_YOLO_ROLE_CLASSIFY:
            return row.task != nullptr &&
                   std::strcmp(row.task, "classify") == 0;
        case AICORE_YOLO_ROLE_SEMANTIC:
            return row.task != nullptr &&
                   std::strcmp(row.task, "semantic") == 0;
        case AICORE_YOLO_ROLE_WORLD:
            return row.task != nullptr && row.text_input != 0 &&
                   std::strcmp(row.task, "detect") == 0;
        case AICORE_YOLO_ROLE_YOLOE:
            return row.task != nullptr && row.text_input != 0 &&
                   std::strcmp(row.task, "segment") == 0;
        case AICORE_YOLO_ROLE_TEXT:
            return row.task != nullptr && std::strcmp(row.task, "text") == 0;
        case AICORE_YOLO_ROLE_ANY:
        default:
            return true;
    }
}

static aicore_yolo_model_entry toEntry(const ModelRow& row) {
    return {row.filename,      row.download_url, row.display_name,
            row.quant_note,    row.license_note, row.task,
            row.depth_capable, row.end2end,      row.text_input};
}

// Static (non-thread-local) backing store: unlike the historical
// thread_local singleton, concurrently held entry pointers never overwrite
// each other.
static std::array<aicore_yolo_model_entry, 256> g_entry_store;

static const aicore_yolo_model_entry* entry_at(size_t index) {
    if (index >= kModels.size() || index >= g_entry_store.size())
        return nullptr;
    g_entry_store[index] = toEntry(kModels[index]);
    return &g_entry_store[index];
}

}  // namespace

AICORE_CAPI int aicore_yolo_model_count(enum aicore_yolo_model_role role) {
    int n = 0;
    for (const auto& row : kModels) {
        if (roleMatches(role, row)) ++n;
    }
    return n;
}

AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_model_at(
        int index, enum aicore_yolo_model_role role) {
    if (index < 0) return nullptr;
    int seen = -1;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (!roleMatches(role, kModels[i])) continue;
        ++seen;
        if (seen == index) return entry_at(i);
    }
    return nullptr;
}

AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_model_by_filename(
        const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return entry_at(i);
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_yolo_model_download_base(void) {
    return kDownloadBase;
}

namespace {

// Basename of a path (after the last '/' or '\\'), or the path itself.
static const char* pathBasename(const char* path) {
    const char* slash = std::strrchr(path, '/');
    const char* backslash = std::strrchr(path, '\\');
    const char* last =
            backslash && (!slash || backslash > slash) ? backslash : slash;
    return last ? last + 1 : path;
}

}  // namespace

AICORE_CAPI int aicore_yolo_verify_model(const char* path,
                                         aicore_yolo_verify_report* out) {
    if (out) *out = {};
    if (path == nullptr || path[0] == '\0') return -1;

    // 1. Basename must match a catalog entry.
    const char* base = pathBasename(path);
    const aicore_yolo_model_entry* entry = aicore_yolo_model_by_filename(base);
    if (!entry) return -1;  // out->filename_ok stays 0
    if (out) out->filename_ok = 1;

    // Filename is "<variant>-<quant>.gguf"; split at the LAST '-' of the
    // stem ("yolo26n-depth-f16.gguf" -> variant "yolo26n-depth", quant
    // "f16"). The prompt-free YOLOE variants ("-seg-pf") split the same way
    // (variant "yoloe-26n-seg-pf", quant "f16").
    const std::string file(base);
    const size_t dot = file.rfind(".gguf");
    const std::string stem =
            (dot != std::string::npos) ? file.substr(0, dot) : file;
    const size_t dash = stem.rfind('-');
    std::string variant_name = stem;
    std::string quant;
    if (dash != std::string::npos) {
        variant_name = stem.substr(0, dash);
        quant = stem.substr(dash + 1);
    }
    const VariantDigest* dig = findDigest(variant_name.c_str());
    const bool has_digest = dig != nullptr;
    int qi = 0;  // f32
    if (quant == "f16") qi = 1;
    if (quant == "q8_0") qi = 2;

    std::ifstream in(path, std::ios::binary);
    if (!in) return -1;
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    // 2. Exact byte count (skipped without a published baseline).
    if (has_digest && file_size != dig->bytes[qi]) {
        return -1;  // out->size_ok stays 0
    }
    if (out) out->size_ok = 1;

    // 3. SHA-256 (streamed; skipped without a published baseline).
    if (has_digest) {
        QCryptographicHash hash(QCryptographicHash::Sha256);
        char buf[1 << 16];
        while (in) {
            in.read(buf, sizeof(buf));
            const std::streamsize n = in.gcount();
            if (n > 0) hash.addData(buf, static_cast<int>(n));
        }
        const QByteArray hex = hash.result().toHex();
        if (std::strcmp(hex.constData(), dig->sha256[qi]) != 0) {
            return -1;  // out->hash_ok stays 0
        }
    }
    if (out) out->hash_ok = 1;

    // 4. GGUF magic.
    char magic[4] = {};
    in.clear();
    in.seekg(0, std::ios::beg);
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        return -1;  // out->magic_ok stays 0
    }
    if (out) out->magic_ok = 1;

    // 5. yolo.task metadata must match the catalog entry.
    gguf_init_params ip{};  // header only
    gguf_context* g = gguf_init_from_file(path, ip);
    if (!g) return -1;  // out->task_ok stays 0
    std::string task = "detect";
    const int kid = gguf_find_key(g, "yolo.task");
    if (kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_STRING) {
        const char* v = gguf_get_val_str(g, kid);
        if (v && v[0]) task = v;
    }
    gguf_free(g);
    if (entry->task == nullptr || task != entry->task) {
        return -1;  // out->task_ok stays 0
    }
    if (out) out->task_ok = 1;

    return 0;
}
