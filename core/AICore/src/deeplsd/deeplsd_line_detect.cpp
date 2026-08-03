// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "deeplsd_line_detect.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "lsd.h"

namespace deeplsd {
namespace {

constexpr double kLineNeighborhood = 5.0;
constexpr double kGradThresh = 3.0;
constexpr double kInvalidAngle = -1024.0;

inline size_t Idx(int x, int y, int width) {
    return static_cast<size_t>(y) * width + x;
}

void ComputeImageGrad(const std::vector<double>& img,
                      int width,
                      int height,
                      std::vector<double>& gradnorm,
                      std::vector<double>& gradangle) {
    const size_t n = static_cast<size_t>(width) * height;
    gradnorm.assign(n, 0.0);
    gradangle.assign(n, 0.0);

    std::vector<double> blur = img;
    const int ksize = 7;
    const int half = ksize / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            int count = 0;
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    const int sx = std::clamp(x + kx, 0, width - 1);
                    const int sy = std::clamp(y + ky, 0, height - 1);
                    sum += img[Idx(sx, sy, width)];
                    ++count;
                }
            }
            blur[Idx(x, y, width)] = sum / count;
        }
    }

    std::vector<double> dx(n, 0.0);
    std::vector<double> dy(n, 0.0);
    for (int y = 0; y < height; ++y) {
        for (int x = 1; x < width; ++x) {
            dx[Idx(x, y, width)] =
                    (blur[Idx(x, y, width)] - blur[Idx(x - 1, y, width)]) / 2.0;
        }
    }
    for (int y = 1; y < height; ++y) {
        for (int x = 1; x < width; ++x) {
            dy[Idx(x, y, width)] =
                    (dy[Idx(x, y - 1, width)] + dy[Idx(x, y, width)] +
                     dy[Idx(x - 1, y, width)] + dy[Idx(x - 1, y - 1, width)]) /
                    2.0;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        gradnorm[i] = std::hypot(dx[i], dy[i]);
        gradangle[i] = std::atan2(dy[i], dx[i]);
    }
}

void AlignWithGradAngle(const std::vector<double>& angle,
                        const std::vector<double>& img,
                        int width,
                        int height,
                        std::vector<double>& oriented,
                        std::vector<double>& img_grad_angle) {
    std::vector<double> gradnorm_tmp;
    ComputeImageGrad(img, width, height, gradnorm_tmp, img_grad_angle);
    oriented.resize(static_cast<size_t>(width) * height);

    const double pi = M_PI;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = Idx(x, y, width);
            const double pred_grad = std::fmod(angle[i], pi);
            const double pos_dist = std::min(
                    std::abs(img_grad_angle[i] - pred_grad),
                    2.0 * pi - std::abs(img_grad_angle[i] - pred_grad));
            const double neg_dist = std::min(
                    std::abs(img_grad_angle[i] - pred_grad + pi),
                    2.0 * pi - std::abs(img_grad_angle[i] - pred_grad + pi));
            oriented[i] =
                    pos_dist <= neg_dist ? pred_grad : pred_grad - pi;
        }
    }
}

void PreprocessAngle(const std::vector<double>& angle,
                     const std::vector<double>& img,
                     int width,
                     int height,
                     bool mask_border,
                     std::vector<double>& out) {
    std::vector<double> img_grad_angle;
    AlignWithGradAngle(angle, img, width, height, out, img_grad_angle);
    const double pi = M_PI;
    for (double& v : out) {
        v = std::fmod(v - pi / 2.0, 2.0 * pi);
    }
    if (mask_border) {
        for (int x = 0; x < width; ++x) {
            out[Idx(x, 0, width)] = kInvalidAngle;
        }
        for (int y = 0; y < height; ++y) {
            out[Idx(0, y, width)] = kInvalidAngle;
        }
    }
}

}  // namespace

bool DetectAfmLines(const uint8_t* gray,
                    int32_t width,
                    int32_t height,
                    int32_t row_stride,
                    const float* df_norm,
                    const float* angle_norm,
                    std::vector<LineSegment>* segments,
                    std::string* error) {
    if (segments == nullptr || gray == nullptr || df_norm == nullptr ||
        angle_norm == nullptr || width <= 0 || height <= 0) {
        if (error) {
            *error = "invalid DetectAfmLines arguments";
        }
        return false;
    }
    segments->clear();

    const size_t plane = static_cast<size_t>(width) * height;
    std::vector<double> img(plane);
    std::vector<double> df(plane);
    std::vector<double> line_level(plane);
    std::vector<double> gradnorm(plane);
    std::vector<double> gradangle(plane);

    const double pi = M_PI;
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const size_t i = Idx(x, y, width);
            img[i] = static_cast<double>(
                    gray[static_cast<size_t>(y) * row_stride + x]);
            df[i] = std::exp(-static_cast<double>(df_norm[i])) * kLineNeighborhood;
            line_level[i] = static_cast<double>(angle_norm[i]) * pi;
            gradnorm[i] = std::max(kLineNeighborhood - df[i], 0.0);
        }
    }

    std::vector<double> angle_input(plane);
    for (size_t i = 0; i < plane; ++i) {
        angle_input[i] = line_level[i] - pi / 2.0;
    }
    PreprocessAngle(angle_input, img, width, height, true, gradangle);
    for (size_t i = 0; i < plane; ++i) {
        if (gradnorm[i] < kGradThresh) {
            gradangle[i] = kInvalidAngle;
        }
    }

    int n_out = 0;
    double* raw = LineSegmentDetection(
            &n_out, img.data(), width, height,
            /*scale=*/1.0,
            /*sigma_scale=*/0.6,
            /*quant=*/2.0,
            /*ang_th=*/22.5,
            /*log_eps=*/0.0,
            /*density_th=*/0.0,
            /*n_bins=*/1024,
            /*grad_nfa=*/true,
            gradnorm.data(), gradangle.data());
    if (raw == nullptr || n_out <= 0) {
        if (raw != nullptr) {
            std::free(raw);
        }
        return true;
    }

    segments->reserve(static_cast<size_t>(n_out));
    for (int i = 0; i < n_out; ++i) {
        LineSegment seg;
        seg.x1 = static_cast<float>(raw[7 * i + 0]);
        seg.y1 = static_cast<float>(raw[7 * i + 1]);
        seg.x2 = static_cast<float>(raw[7 * i + 2]);
        seg.y2 = static_cast<float>(raw[7 * i + 3]);
        // LSD tuple: x1,y1,x2,y2,width,p,-log10(NFA). Index 5 is angle
        // precision p (~0.125), NOT match confidence. Use -log10(NFA) mapped
        // to [0,1] for UI filtering (higher = more significant segment).
        const float nfa_log = static_cast<float>(raw[7 * i + 6]);
        seg.score = std::clamp(nfa_log / 10.0f, 0.0f, 1.0f);
        segments->push_back(seg);
    }
    std::free(raw);
    return true;
}

}  // namespace deeplsd
