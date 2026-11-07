// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum aicore_image_format {
    AICORE_IMAGE_RGB8 = 1,
    AICORE_IMAGE_RGBA8 = 2,
    AICORE_IMAGE_GRAY8 = 3,
    AICORE_IMAGE_BGR8 = 4,
    AICORE_IMAGE_BGRA8 = 5
} aicore_image_format;

/** Borrowed, row-stride-aware image view used by in-memory task APIs. */
typedef struct aicore_image_view {
    const uint8_t* data;
    int32_t width;
    int32_t height;
    size_t row_stride_bytes;
    aicore_image_format format;
} aicore_image_view;

#ifdef __cplusplus
}
#endif
