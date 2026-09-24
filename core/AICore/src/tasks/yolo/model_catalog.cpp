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
#include "common/gguf_file_io.hpp"
#include "gguf.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "yolo_gguf_models/";

// One catalog family (release filename stem without the -<quant> suffix).
// Filenames follow the yolo_gguf_models release exactly (77 variants x 3
// quantizations = 230 gguf assets — the multilingual bridge ships f16 +
// q8_0 only —, verified against the GitHub release expanded-assets listing;
// the 77 include the 5 native reid encoders published 2026-09-19; the
// obb/sem families ship both the 640 canonical speed graphs and the
// checkpoint-native 1024 resolution rebuilds).
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
        // Oriented boxes (DOTA-15); 640 canonical speed + 1024
        // checkpoint-native resolution rebuilds.
        {"yolo26n-obb", "YOLO26 Nano (OBB)", "obb", 0},
        {"yolo26s-obb", "YOLO26 Small (OBB)", "obb", 0},
        {"yolo26m-obb", "YOLO26 Medium (OBB)", "obb", 0},
        {"yolo26l-obb", "YOLO26 Large (OBB)", "obb", 0},
        {"yolo26x-obb", "YOLO26 XLarge (OBB)", "obb", 0},
        {"yolo26n-obb-1024", "YOLO26 Nano (OBB, 1024)", "obb", 0},
        {"yolo26s-obb-1024", "YOLO26 Small (OBB, 1024)", "obb", 0},
        {"yolo26m-obb-1024", "YOLO26 Medium (OBB, 1024)", "obb", 0},
        {"yolo26l-obb-1024", "YOLO26 Large (OBB, 1024)", "obb", 0},
        {"yolo26x-obb-1024", "YOLO26 XLarge (OBB, 1024)", "obb", 0},
        // Semantic segmentation (Cityscapes-19); same dual resolution as
        // the OBB family.
        {"yolo26n-sem", "YOLO26 Nano (Semantic)", "semantic", 0},
        {"yolo26s-sem", "YOLO26 Small (Semantic)", "semantic", 0},
        {"yolo26m-sem", "YOLO26 Medium (Semantic)", "semantic", 0},
        {"yolo26l-sem", "YOLO26 Large (Semantic)", "semantic", 0},
        {"yolo26x-sem", "YOLO26 XLarge (Semantic)", "semantic", 0},
        {"yolo26n-sem-1024", "YOLO26 Nano (Semantic, 1024)", "semantic", 0},
        {"yolo26s-sem-1024", "YOLO26 Small (Semantic, 1024)", "semantic", 0},
        {"yolo26m-sem-1024", "YOLO26 Medium (Semantic, 1024)", "semantic", 0},
        {"yolo26l-sem-1024", "YOLO26 Large (Semantic, 1024)", "semantic", 0},
        {"yolo26x-sem-1024", "YOLO26 XLarge (Semantic, 1024)", "semantic", 0},
        // Classification (ImageNet-1000, 224 input).
        {"yolo26n-cls", "YOLO26 Nano (Classify)", "classify", 0},
        {"yolo26s-cls", "YOLO26 Small (Classify)", "classify", 0},
        {"yolo26m-cls", "YOLO26 Medium (Classify)", "classify", 0},
        {"yolo26l-cls", "YOLO26 Large (Classify)", "classify", 0},
        {"yolo26x-cls", "YOLO26 XLarge (Classify)", "classify", 0},
        // (The legacy cls-tower ReID encoders were withdrawn 2026-09-19 in
        // favor of the native reid-yolo26{...} family below.)
        // Native ReID encoders converted from the official
        // yolo26{n,s,m,l,x}-reid.onnx assets (standalone ReID backbones with a
        // 512-d embedding head; task='reid' graphs). Authoritative encoder
        // family, published 2026-09-19; consumed through aicore/reid_capi.h.
        {"reid-yolo26n", "YOLO26 Nano ReID (official reid.onnx)", "classify",
         0},
        {"reid-yolo26s", "YOLO26 Small ReID (official reid.onnx)", "classify",
         0},
        {"reid-yolo26m", "YOLO26 Medium ReID (official reid.onnx)", "classify",
         0},
        {"reid-yolo26l", "YOLO26 Large ReID (official reid.onnx)", "classify",
         0},
        {"reid-yolo26x", "YOLO26 XLarge ReID (official reid.onnx)", "classify",
         0},
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
        // Multilingual bridge (sentence-transformers clip-ViT-B-32-
        // multilingual-v1: DistilBERT tower projected into the OpenAI CLIP
        // ViT-B/32 text space) — lets the YOLO-World head consume prompts in
        // 100+ languages with zero detector changes. F16 only (released).
        {"mclip-labse-vitb32", "Multilingual CLIP Bridge (Text)", "text", 1},
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
// (2026-08-19, ultralytics-ggml-integration-plan.md 3.2), and for the 30
// obb/sem 1024 resolution rebuilds (release audit 2026-09-18; hashes from
// the upstream cpp_ggml/models/gguf/SHA256SUMS). Later additions
// (segment/world/yoloe/pose/obb/sem 640/cls/depth s..x/text) have no published
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
        // obb/sem 1024 checkpoint-native resolution rebuilds (bytes from
        // the yolo_gguf_models release assets, hashes from the upstream
        // SHA256SUMS). Order f32, f16, q8_0.
        {"yolo26n-obb-1024",
         {9846880, 4965280, 2719232},
         {"a5ae3bcb7739355a33376b17289487e7c21ce3f029bae3bf964a1eb438283045",
          "ef59db803f10e563d39d372af9a707efcd7f5aaa362a1ac952632143cf533ed8",
          "aab9dd0c2a1ba79e9ff41ece44f8a9c797c2f719ef0bde62e7b1bd613c78390f"}},
        {"yolo26s-obb-1024",
         {39077440, 19597504, 10485120},
         {"aaf3914ddf3851226af05de6a13026e65a1d55a8b72a78e787322b87a6f8e670",
          "972fc7e869b7eecafdd038dab342e3fa2397af898abce6e526886ea71dc1c64f",
          "c8a82b57718d1fcb512a6ea0c04db941efb64e49a52241e288546f268173310f"}},
        {"yolo26m-obb-1024",
         {84891712, 42521920, 22682688},
         {"a433db64fd0d457a3c5aca0d0c2362d62d9fa2489c79eb37442bec3e16188404",
          "499b6a6d0edf732d350a23a5e45057f393054ce8c1ad1251dec6cdb3fcf4a3af",
          "8f3cedfc8898e17527fc67ea14ca4d781494ebfc8a0f881021f7d04a8cedc124"}},
        {"yolo26l-obb-1024",
         {102499488, 51351456, 27398752},
         {"3e73649a11e94a477542c48941abc6239c061e77037e16bcf90a7b87c91d3f37",
          "bd7eedb4c4585d2aa4ef602974a9b5d89e80febfc4575a93f100c4d94ab2dcb8",
          "2c08ad4436acc74f0a438e038e6e4936ffe27595023558f6ff8987ec6cc871b6"}},
        {"yolo26x-obb-1024",
         {230314272, 115290528, 61564416},
         {"9748beda87e0d5a8193372714f9ca466bd7072dc845a9231f96c1717f5e559a7",
          "970d903e2c8ec9c362d71686dd9ad27c71cc758970ed36297418cde8420d219c",
          "df1f9a960288bc56f78137e7f165f7e955907744faa32647bea7c726598b9dcd"}},
        {"yolo26n-sem-1024",
         {6239264, 3143744, 1715744},
         {"cb50a5185afce5928305c75c74e5fe597a78f7eee3ae132f20ab2481a28c6506",
          "37a635b5296c6f6de7d44a16fab3c2a579c27e9288f32b4de39c6f538b67de28",
          "145accdaebfae48f437b2c211ee09106ef4f1445f749df5a25851822837a31d2"}},
        {"yolo26s-sem-1024",
         {24810048, 12439168, 6646688},
         {"bb07fcd7bd34765a1ba993e970cb2edc7c8e530de3f8033eb344c6007695406d",
          "a454df76e453899dff4cd02a5094a385662accb11bb1d4249bf5251d9ba5d96a",
          "42d4dee7beb08572350f39a8cbfc61eedd72d03bdb52425d162f7cb0287f401a"}},
        {"yolo26m-sem-1024",
         {52505280, 26297664, 14015552},
         {"d90fafd1f1d4269ba421d63009966542f99dcab2728387b62c16476beaaa8a6f",
          "1b8e7474d14d19a6c79625063d2297eee49cac04cf5a5778e943f470d027db80",
          "b4a98f3b7f5da7405bd0e6a064d8f58b4eee69a3a989c1359d136b4214c286a5"}},
        {"yolo26l-sem-1024",
         {66698048, 33416128, 17819360},
         {"c7c0163536fb2d7cd2e7f25ce30893d25656ae9f9d7f88a668cff775f757d6f8",
          "547ecd08c4fa487ae785e9eaca3133b49b814fd6a25ac64cc058786912fc4a9b",
          "fbc89d1eb619114461a69bb9d0b1b26814d2db1009ca1b168a4fa8b43d82d2be"}},
        {"yolo26x-sem-1024",
         {149910336, 75042816, 40111424},
         {"280343eee856ea99c4c74fb9680c6782138ddf2ed2468b60ce286785a8db32fe",
          "70668741e3ca5fe05cc710af8ba9877bf8be110fefc439edd4925a11d60d1061",
          "61edbf355f3560e917037d0282535d617e19be0c7e8c3d41f5fd95559cf5789b"}},
        // reid-yolo26*-cls family: release + Hugging Face publication audit
        // (2026-09-19; digests measured from the published bytes).
        // reid-yolo26{n,s,m,l,x}: native encoders converted from the official
        // yolo26*-reid.onnx assets (2026-09-19; digests measured from the
        // published bytes).
        {"reid-yolo26n",
         {9741408, 4894976, 3253248},
         {"6e6630faf6e9b24cffa2a52985c77320d65918b8f328f16e2b6b0a07e186d828",
          "b8882416c8e1b14e10a22ac02d49af485c52f0ea940415c8151dabca7b4b8277",
          "c721be113ab6c770df5d2b9b5366e0c0c924cf027a8bb16cc8779bd4f38daa42"}},
        {"reid-yolo26s",
         {28333184, 14201280, 8200128},
         {"013c200e1f21f2479a31559d73b659e550d0e3a75559378bf8b9807c0e249419",
          "c9de79f801d3dc445e191bc64c994ccfd3480f738b036e9c7b0c684bcf530609",
          "fa41d98979a79e91781ea0871dfd6aca893fae65c1884b8decc19e479fab9f34"}},
        {"reid-yolo26m",
         {47967392, 24025888, 13422784},
         {"ddcf31eca0f1eef8d618a17a39310c00f5b6d1ef65a77f7d709cb3729777c601",
          "c16b179a411d0e833e7e0fcffa306d64bd8ef9a6864cae19cfb47be215dde3a8",
          "2d389d76479c9a90c7e5073da2d58609c30dcdeb6eba6c46ce5451bdb096beeb"}},
        {"reid-yolo26l",
         {59740928, 29932928, 16583648},
         {"00a64abdab9e6198fb500639f89cf5ea50e1c06a425e717df4d3aae6b93abc7d",
          "737944f648ed84988b5a093670fdd13f6e6fbf49e0a53f10b1b0c9a48b67d7fe",
          "6c6725a7029da9ab0418431ff0e0b6d22f8d5c2d94f50727fabd788e80410aeb"}},
        {"reid-yolo26x",
         {129024256, 64594112, 35176480},
         {"76ed20e776047a34ac92f14757e36e4be461e3522dc6e9c29f0672e2138fad6a",
          "bf7ab447f479d95cbcbfa91efc884db28b60753505e0223744f1055d469fd466",
          "8d42d32f371e91d4ba6f7c3f190b9032e76ba690488e274287f58877088c7377"}},
};

static_assert(sizeof(kDigests) / sizeof(kDigests[0]) == 26,
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
        // The multilingual bridge is published in F16 + Q8_0 (Q8_0 first:
        // it is the default — half the download at matching accuracy).
        const bool mclip_bridge =
                std::strcmp(info.name, "mclip-labse-vitb32") == 0;
        const char* license = mclip_bridge ? "MIT / Apache-2.0 (M-CLIP / LaBSE)"
                                           : "AGPL-3.0 (Ultralytics)";
        for (int qi = 0; qi < kQuantCount; ++qi) {
            // 0 = f32 (not published), 1 = f16, 2 = q8_0 (default).
            const int eqi =
                    mclip_bridge ? (qi == 0 ? 2 : (qi == 1 ? 1 : -1)) : qi;
            if (eqi < 0) continue;
            const int qi_eff = eqi;
            std::string filename = std::string(info.name) + "-" +
                                   kQuantSuffixes[qi_eff] + ".gguf";
            std::string url = std::string(kDownloadBase) + filename;
            std::string display = std::string(info.display) + " \xe2\x80\x94 " +
                                  kQuantNotes[qi_eff];
            rows.push_back({dupString(filename.c_str()), dupString(url.c_str()),
                            dupString(display.c_str()),
                            dupString(kQuantNotes[qi_eff]), license, info.task,
                            std::strcmp(info.task, "depth") == 0 ? 1 : 0,
                            std::strncmp(info.name, "yolo26", 6) == 0 &&
                                            std::strcmp(info.task, "text") != 0
                                    ? 1
                                    : 0,
                            info.text_input, 0, nullptr});
            if (const VariantDigest* d = findDigest(info.name)) {
                rows.back().expected_bytes = d->bytes[qi_eff];
                rows.back().sha256 = d->sha256[qi_eff];
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

AICORE_CAPI int aicore_yolo_model_default_index(
        enum aicore_yolo_model_role role) {
    // Per-role default declaration: the first row of the role-filtered
    // view whose note carries the visible "(recommended)" marker. Reading
    // the marker here (instead of hard-coding row offsets) keeps the
    // declaration and the user-facing label from ever drifting apart. The
    // returned index is relative to the role-filtered view, matching
    // aicore_yolo_model_at(index, role).
    int view = 0;
    for (const auto& row : kModels) {
        if (!roleMatches(role, row)) continue;
        if (row.quant_note && std::strstr(row.quant_note, "(recommended)")) {
            return view;
        }
        ++view;
    }
    return 0;
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
    // ctx=nullptr: header/metadata read only, no tensor mapping.
    gguf_context* g = ggml_common::open_gguf_file(path, /*no_alloc=*/false,
                                                  nullptr, "yolo_catalog");
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
