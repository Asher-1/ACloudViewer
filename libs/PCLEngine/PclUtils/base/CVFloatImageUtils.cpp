// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "base/CVFloatImageUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>

// CV_CORE_LIB
#include <CVConst.h>

void PclUtils::FloatImageUtils::getColorForFloat(float value,
                                                 unsigned char& r,
                                                 unsigned char& g,
                                                 unsigned char& b) {
    if (std::isinf(value)) {
        if (value > 0.0f) {
            r = 150;
            g = 150;
            b = 200;
            return;
        }
        r = 150;
        g = 200;
        b = 150;
        return;
    }
    if (!std::isfinite(value)) {
        r = 200;
        g = 150;
        b = 150;
        return;
    }

    r = g = b = 0;
    value *= 10;
    if (value <= 1.0) {
        b = static_cast<unsigned char>(std::lrint(value * 200));
        r = static_cast<unsigned char>(std::lrint(value * 120));
    } else if (value <= 2.0) {
        b = static_cast<unsigned char>(200 + std::lrint((value - 1.0) * 55));
        r = static_cast<unsigned char>(120 - std::lrint((value - 1.0) * 120));
    } else if (value <= 3.0) {
        b = static_cast<unsigned char>(255 - std::lrint((value - 2.0) * 55));
        g = static_cast<unsigned char>(std::lrint((value - 2.0) * 200));
    } else if (value <= 4.0) {
        b = static_cast<unsigned char>(200 - std::lrint((value - 3.0) * 200));
        g = static_cast<unsigned char>(200 + std::lrint((value - 3.0) * 55));
    } else if (value <= 5.0) {
        g = static_cast<unsigned char>(255 - std::lrint((value - 4.0) * 100));
        r = static_cast<unsigned char>(std::lrint((value - 4.0) * 120));
    } else if (value <= 6.0) {
        r = static_cast<unsigned char>(100 + std::lrint((value - 5.0) * 155));
        g = static_cast<unsigned char>(120 - std::lrint((value - 5.0) * 120));
        b = static_cast<unsigned char>(120 - std::lrint((value - 5.0) * 120));
    } else if (value <= 7.0) {
        r = 255;
        g = static_cast<unsigned char>(std::lrint((value - 6.0) * 255));
    } else {
        r = 255;
        g = 255;
        b = static_cast<unsigned char>(std::lrint((value - 7.0) * 255.0 / 3.0));
    }
}

void PclUtils::FloatImageUtils::getColorForAngle(float value,
                                                 unsigned char& r,
                                                 unsigned char& g,
                                                 unsigned char& b) {
    if (std::isinf(value)) {
        if (value > 0.0f) {
            r = 150;
            g = 150;
            b = 200;
            return;
        }
        r = 150;
        g = 200;
        b = 150;
        return;
    }
    if (!std::isfinite(value)) {
        r = 200;
        g = 150;
        b = 150;
        return;
    }

    r = g = b = 0;
    if (value < -M_PI / 2.0f) {
        b = static_cast<unsigned char>(
                std::lrint(255 * (value + float(M_PI)) / (float(M_PI) / 2.0f)));
    } else if (value <= 0.0f) {
        b = 255;
        r = g = static_cast<unsigned char>(std::lrint(
                255 * (value + float(M_PI / 2)) / (float(M_PI) / 2.0f)));
    } else if (value <= M_PI / 2.0f) {
        g = 255;
        r = b = static_cast<unsigned char>(
                255 - std::lrint(255 * value / (float(M_PI) / 2.0f)));
    } else {
        g = static_cast<unsigned char>(
                255 -
                std::lrint(255 * (value - M_PI / 2.0f) / (float(M_PI) / 2.0f)));
    }
}

void PclUtils::FloatImageUtils::getColorForHalfAngle(float value,
                                                     unsigned char& r,
                                                     unsigned char& g,
                                                     unsigned char& b) {
    getColorForAngle(2.0f * value, r, g, b);
}

unsigned char* PclUtils::FloatImageUtils::getVisualImage(
        const float* float_image,
        int width,
        int height,
        float min_value,
        float max_value,
        bool gray_scale) {
    int size = width * height;
    int arraySize = 3 * size;
    auto* data = new unsigned char[arraySize];
    unsigned char* dataPtr = data;

    bool recalcMin = std::isinf(min_value), recalcMax = std::isinf(max_value);
    if (recalcMin) min_value = std::numeric_limits<float>::infinity();
    if (recalcMax) max_value = -std::numeric_limits<float>::infinity();

    if (recalcMin || recalcMax) {
        for (int i = 0; i < size; ++i) {
            float v = float_image[i];
            if (!std::isfinite(v)) continue;
            if (recalcMin) min_value = (std::min)(min_value, v);
            if (recalcMax) max_value = (std::max)(max_value, v);
        }
    }
    float factor = 1.0f / (max_value - min_value), offset = -min_value;

    for (int i = 0; i < size; ++i) {
        unsigned char &r = *(dataPtr++), &g = *(dataPtr++), &b = *(dataPtr++);
        float v = float_image[i];
        if (!std::isfinite(v)) {
            getColorForFloat(v, r, g, b);
            continue;
        }
        v = std::max(0.0f, std::min(1.0f, factor * (v + offset)));
        if (gray_scale) {
            r = g = b = static_cast<unsigned char>(std::lrint(v * 255));
        } else {
            getColorForFloat(v, r, g, b);
        }
    }
    return data;
}

unsigned char* PclUtils::FloatImageUtils::getVisualImage(
        const unsigned short* short_image,
        int width,
        int height,
        unsigned short min_value,
        unsigned short max_value,
        bool gray_scale) {
    int size = width * height;
    auto* data = new unsigned char[3 * size];
    unsigned char* dataPtr = data;

    float factor = 1.0f / static_cast<float>(max_value - min_value),
          offset = static_cast<float>(-min_value);

    for (int i = 0; i < size; ++i) {
        unsigned char &r = *(dataPtr++), &g = *(dataPtr++), &b = *(dataPtr++);
        float v = std::max(0.0f,
                           std::min(1.0f, factor * (short_image[i] + offset)));
        if (gray_scale) {
            r = g = b = static_cast<unsigned char>(std::lrint(v * 255));
        } else {
            getColorForFloat(v, r, g, b);
        }
    }
    return data;
}

unsigned char* PclUtils::FloatImageUtils::getVisualAngleImage(
        const float* angle_image, int width, int height) {
    int size = width * height;
    auto* data = new unsigned char[3 * size];
    unsigned char* dataPtr = data;
    for (int i = 0; i < size; ++i) {
        unsigned char &r = *(dataPtr++), &g = *(dataPtr++), &b = *(dataPtr++);
        getColorForAngle(angle_image[i], r, g, b);
    }
    return data;
}

unsigned char* PclUtils::FloatImageUtils::getVisualHalfAngleImage(
        const float* angle_image, int width, int height) {
    int size = width * height;
    auto* data = new unsigned char[3 * size];
    unsigned char* dataPtr = data;
    for (int i = 0; i < size; ++i) {
        unsigned char &r = *(dataPtr++), &g = *(dataPtr++), &b = *(dataPtr++);
        getColorForHalfAngle(angle_image[i], r, g, b);
    }
    return data;
}
