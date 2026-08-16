// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Single translation unit for the shared stb_image implementation.
// The facedetect task uses stb_image for PNG/BMP fallback decoding.
// The rfdetr and rmbg tasks use Qt QImage instead (no stb dependency).

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
