// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "visualization/rendering/Gradient.h"

#include <Image.h>
#include <Logging.h>

#include "visualization/rendering/Renderer.h"

namespace cloudViewer {
namespace visualization {
namespace rendering {

namespace {
static const int kNGradientPixels = 512;
static const int kNChannels = 4;

void SetPixel(std::vector<uint8_t>& img,
              int idx,
              const Eigen::Vector4f& color) {
    int i = kNChannels * idx;
    img[i] = uint8_t(std::round(255.0f * color[0]));
    img[i + 1] = uint8_t(std::round(255.0f * color[1]));
    img[i + 2] = uint8_t(std::round(255.0f * color[2]));
    img[i + 3] = uint8_t(std::round(255.0f * color[3]));
}

}  // namespace

Gradient::Gradient() {}

Gradient::Gradient(const std::vector<Gradient::Point>& points) {
    points_ = points;
}

Gradient::~Gradient() {}

const std::vector<Gradient::Point>& Gradient::GetPoints() const {
    return points_;
}

void Gradient::SetPoints(const std::vector<Gradient::Point>& points) {
    points_ = points;
    for (size_t i = 0; i < points_.size(); ++i) {
        if (points_[i].value < 0.0f || points_[i].value > 1.0f) {
            utility::LogWarning(
                    "Gradient point {} must be in range [0.0, 1.0], clamping",
                    points_[i].value);
            points_[i].value = std::max(0.0f, std::min(1.0f, points_[i].value));
        }
    }
    textures_.clear();
}

Gradient::Mode Gradient::GetMode() const { return mode_; }

void Gradient::SetMode(Mode mode) {
    if (mode != mode_) {
        mode_ = mode;
        textures_.clear();
    }
}

TextureHandle Gradient::GetTextureHandle(Renderer& renderer) {
    if (textures_.find(&renderer) == textures_.end()) {
        auto img = cloudViewer::make_shared<geometry::Image>();
        if (!points_.empty()) {
            int n_points = int(points_.size());
            if (mode_ == Mode::kGradient) {
                img->Prepare(kNGradientPixels, 1, kNChannels, 1);
                auto n = float(kNGradientPixels - 1);
                int idx = 0;
                for (int img_idx = 0; img_idx < kNGradientPixels; ++img_idx) {
                    auto x = float(img_idx) / n;
                    while (idx < n_points && x > points_[idx].value) {
                        idx += 1;
                    }

                    if (idx == 0) {
                        SetPixel(img->data_, img_idx, points_[0].color);
                    } else if (idx == n_points) {
                        SetPixel(img->data_, img_idx,
                                 points_[n_points - 1].color);
                    } else {
                        auto& p0 = points_[idx - 1];
                        auto& p1 = points_[idx];
                        auto dist = p1.value - p0.value;
                        // Calc weights between 0 and 1
                        auto w0 = 1.0f - (x - p0.value) / dist;
                        auto w1 = (x - p0.value) / dist;
                        auto r = w0 * p0.color[0] + w1 * p1.color[0];
                        auto g = w0 * p0.color[1] + w1 * p1.color[1];
                        auto b = w0 * p0.color[2] + w1 * p1.color[2];
                        auto a = w0 * p0.color[3] + w1 * p1.color[3];
                        SetPixel(img->data_, img_idx, {r, g, b, a});
                    }
                }
            } else {
                img->Prepare(n_points, 1, kNChannels, 1);
                for (int i = 0; i < n_points; ++i) {
                    SetPixel(img->data_, i, points_[i].color);
                }
            }
        } else {
            img->Prepare(1, 1, kNChannels, 1);
            SetPixel(img->data_, 0, {1.0f, 0.0, 1.0f, 1.0f});
        }

        textures_[&renderer] = renderer.AddTexture(img);
    }
    return textures_[&renderer];
}

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer
