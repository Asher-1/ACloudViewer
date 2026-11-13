// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Per-call open-vocabulary switching on a resident YOLO-World context.
//
// Verifies aicore_yolo_detect_image_with_classes: same-class-count
// vocabulary switches reuse the loaded detector (no reload), alternate
// between a fixed set of vocabularies through the process-level text
// embedding cache, keep the result label table in sync with the active
// list, and reject wrong class counts with an actionable error.
//
// Skips (exit 77) without real assets:
//   AICORE_TEST_YOLO_GGUF        world-family detector GGUF
//   AICORE_TEST_YOLO_TEXT_MODEL  text-encoder GGUF (CLIP bridge)
//   AICORE_TEST_YOLO_IMAGE       any probe image
//   AICORE_TEST_YOLO_PARITY_DEVICE  optional device override
#include <aicore/image_view.h>
#include <aicore/yolo_capi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : fallback;
}

int fail(const char* what) {
    std::printf("[yolo-with-classes] FAIL: %s\n", what);
    return 1;
}

}  // namespace

int main() {
    const char* gguf = env_or("AICORE_TEST_YOLO_GGUF", nullptr);
    const char* text_model = env_or("AICORE_TEST_YOLO_TEXT_MODEL", nullptr);
    const char* image = env_or("AICORE_TEST_YOLO_IMAGE", nullptr);
    if (gguf == nullptr || text_model == nullptr || image == nullptr) {
        std::printf(
                "[yolo-with-classes] skipped: AICORE_TEST_YOLO_GGUF, "
                "AICORE_TEST_YOLO_TEXT_MODEL and AICORE_TEST_YOLO_IMAGE are "
                "required\n");
        return 77;
    }

    // Resolve a GPU device like the parity test; CPU works too (the test
    // is functional, not a timing gate).
    std::string device = "cpu";
    if (const char* forced =
                env_or("AICORE_TEST_YOLO_PARITY_DEVICE", nullptr)) {
        device = forced;
    } else if (aicore_yolo_warmup_backend("cuda") == 0) {
        device = "cuda";
    } else if (aicore_yolo_warmup_backend("vulkan") == 0) {
        device = "vulkan";
    }
    std::printf("[yolo-with-classes] device: %s\n", device.c_str());

    uint8_t* rgb = nullptr;
    int32_t w = 0, h = 0;
    if (aicore_yolo_load_path_rgb(image, &rgb, &w, &h) != 0 || rgb == nullptr) {
        std::printf("[yolo-with-classes] failed to load image %s\n", image);
        return 1;
    }
    aicore_image_view view{};
    view.data = rgb;
    view.width = w;
    view.height = h;
    view.row_stride_bytes = static_cast<size_t>(w) * 3;
    view.format = AICORE_IMAGE_RGB8;

    aicore_yolo_options* opts = aicore_yolo_options_new();
    if (opts == nullptr) return fail("options allocation");
    aicore_yolo_options_set_device(opts, device.c_str());
    aicore_yolo_options_set_text_model(opts, text_model);
    const char* first_vocab[] = {"person"};
    aicore_yolo_options_set_classes(opts, first_vocab, 1);

    aicore_yolo_ctx* ctx = aicore_yolo_load_opts(gguf, opts);
    aicore_yolo_options_free(opts);
    if (ctx == nullptr || aicore_yolo_is_ready(ctx) != 1) {
        std::printf(
                "[yolo-with-classes] skipped: world model not loadable "
                "(%s)\n",
                ctx ? aicore_yolo_last_error(ctx) : "context alloc");
        std::free(rgb);
        return 77;
    }

    // 1) Baseline detection with the load-time vocabulary.
    if (aicore_yolo_detect_image(ctx, &view) != 0) {
        std::printf("[yolo-with-classes] baseline detect failed: %s\n",
                    aicore_yolo_last_error(ctx));
        return fail("baseline detect");
    }

    // 2) Switch the vocabulary on the resident context (same count).
    const char* second_vocab[] = {"dog"};
    if (aicore_yolo_detect_image_with_classes(ctx, &view, second_vocab, 1) !=
        0) {
        std::printf("[yolo-with-classes] switch failed: %s\n",
                    aicore_yolo_last_error(ctx));
        return fail("vocabulary switch");
    }
    if (const int n = aicore_yolo_detection_count(ctx); n > 0) {
        const char* name = aicore_yolo_detection_class_name(ctx, 0);
        if (name != nullptr && std::strcmp(name, "dog") != 0) {
            std::printf("[yolo-with-classes] label table not switched: %s\n",
                        name);
            return fail("label table");
        }
    }

    // 3) Switch back: exercises the process-level embedding cache hit.
    if (aicore_yolo_detect_image_with_classes(ctx, &view, first_vocab, 1) !=
        0) {
        return fail("switch back");
    }

    // 4) Wrong class count must be rejected with an actionable error.
    const char* two[] = {"dog", "cat"};
    if (aicore_yolo_detect_image_with_classes(ctx, &view, two, 2) == 0) {
        return fail("wrong count accepted");
    }
    if (std::strstr(aicore_yolo_last_error(ctx), "count") == nullptr) {
        return fail("wrong-count error text");
    }

    // 5) After the rejected switch the context must still work with the
    // previously queued vocabulary (the label table is rolled back when
    // the re-queue fails, keeping label table and text input in sync).
    if (aicore_yolo_detect_image_with_classes(ctx, &view, first_vocab, 1) !=
        0) {
        std::printf("[yolo-with-classes] post-reject detect failed: %s\n",
                    aicore_yolo_last_error(ctx));
        return fail("detect after rejected switch");
    }

    aicore_yolo_free(ctx);
    aicore_yolo_free_buffer(rgb);
    std::printf("[yolo-with-classes] PASS\n");
    return 0;
}
