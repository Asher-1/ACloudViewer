// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.erow.cn
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

#pragma once

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include <Image.h>
#include "visualization/rendering/Gradient.h"

namespace cloudViewer {
namespace visualization {
namespace rendering {

struct Material {
    std::string name;

    // Rendering attributes
    bool has_alpha = false;

    // PBR Material properties and maps
    Eigen::Vector4f base_color = Eigen::Vector4f(1.f, 1.f, 1.f, 1.f);
    float base_metallic = 0.f;
    float base_roughness = 1.f;
    float base_reflectance = 0.5f;
    float base_clearcoat = 0.f;
    float base_clearcoat_roughness = 0.f;
    float base_anisotropy = 0.f;

    float point_size = 3.f;

    std::shared_ptr<geometry::Image> albedo_img;
    std::shared_ptr<geometry::Image> normal_img;
    std::shared_ptr<geometry::Image> ao_img;
    std::shared_ptr<geometry::Image> metallic_img;
    std::shared_ptr<geometry::Image> roughness_img;
    std::shared_ptr<geometry::Image> reflectance_img;
    std::shared_ptr<geometry::Image> clearcoat_img;
    std::shared_ptr<geometry::Image> clearcoat_roughness_img;
    std::shared_ptr<geometry::Image> anisotropy_img;

    // Combined images
    std::shared_ptr<geometry::Image> ao_rough_metal_img;

    // Colormap (incompatible with other settings except point_size)
    // Values for 'value' must be in [0, 1] and the vector must be sorted
    // by increasing value. 'shader' must be "unlitGradient".
    std::shared_ptr<Gradient> gradient;
    float scalar_min = 0.0f;
    float scalar_max = 1.0f;

    // Generic material properties
    std::unordered_map<std::string, Eigen::Vector4f> generic_params;
    std::unordered_map<std::string, geometry::Image> generic_imgs;

    std::string shader;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer