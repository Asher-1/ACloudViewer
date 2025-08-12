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

#pragma once

#include <Image.h>
#include "cloudViewer/t/geometry/Image.h"

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "visualization/rendering/Gradient.h"

namespace cloudViewer {
namespace visualization {
namespace rendering {

struct Material {
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    std::string name;

    /// Returns true if the Material was not created with the default
    /// constructor and therefore has a valid shader name.
    bool IsValid() const { return !name.empty(); }

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

    // PBR material properties for refractive materials
    float thickness = 1.f;
    float transmission = 1.f;
    Eigen::Vector3f absorption_color =
            Eigen::Vector3f(1.f, 1.f, 1.f);  // linear color
    float absorption_distance = 1.f;

    float point_size = 3.f;
    float line_width = 1.f;  // only used with shader = "unlitLine"

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

    // Colors are assumed to be sRGB and tone-mapped accordingly.
    // If tone-mapping is disabled, then colors would be in linear RGB space,
    // in which case this should be set to false. If necessary, colors will be
    // linearized on the CPU.
    bool sRGB_color = true;

    // Unlike the material property sRGB_color which is used to indicate that
    // source colors are in sRGB colorspace, sRGB_vertex_color indicates that
    // per-vertex colors are in sRGB space and should be passed to the GPU as
    // sRGB color.
    bool sRGB_vertex_color = false;

    // Background image (shader = "unlitBackground")
    float aspect_ratio = 0.0f;  // 0: uses base_color; >0: uses albedo_img

    // Infinite ground plane
    float ground_plane_axis = 0.f;  // 0: XZ; >0: XY; <0: YZ

    // Generic material properties
    std::unordered_map<std::string, Eigen::Vector4f> generic_params;
    std::unordered_map<std::string, geometry::Image> generic_imgs;

    std::string shader = "defaultUnlit";

    // ---- Open3D v0.19 compatibility API (declarations) ----
    // Lifecycle / naming
    Material &SetDefaultProperties();
    Material &SetMaterialName(const std::string &mat_name);
    const std::string &GetMaterialName() const;

    // Scalar material properties (query-only, use fields to set)
    bool HasBaseColor() const;
    Eigen::Vector4f GetBaseColor() const;

    bool HasBaseRoughness() const;
    float GetBaseRoughness() const;

    bool HasBaseMetallic() const;
    float GetBaseMetallic() const;

    bool HasBaseReflectance() const;
    float GetBaseReflectance() const;

    bool HasBaseClearcoat() const;
    float GetBaseClearcoat() const;

    bool HasBaseClearcoatRoughness() const;
    float GetBaseClearcoatRoughness() const;

    bool HasAnisotropy() const;
    float GetAnisotropy() const;

    bool HasEmissiveColor() const;
    Eigen::Vector4f GetEmissiveColor() const;

    // Individual texture maps
    bool HasAlbedoMap() const;
    cloudViewer::t::geometry::Image GetAlbedoMap() const;
    Material &SetAlbedoMap(const cloudViewer::t::geometry::Image &img);

    bool HasNormalMap() const;
    cloudViewer::t::geometry::Image GetNormalMap() const;

    bool HasAOMap() const;
    cloudViewer::t::geometry::Image GetAOMap() const;

    bool HasMetallicMap() const;
    cloudViewer::t::geometry::Image GetMetallicMap() const;

    bool HasRoughnessMap() const;
    cloudViewer::t::geometry::Image GetRoughnessMap() const;

    bool HasReflectanceMap() const;
    cloudViewer::t::geometry::Image GetReflectanceMap() const;

    bool HasClearcoatMap() const;
    cloudViewer::t::geometry::Image GetClearcoatMap() const;

    bool HasClearcoatRoughnessMap() const;
    cloudViewer::t::geometry::Image GetClearcoatRoughnessMap() const;

    bool HasAnisotropyMap() const;
    cloudViewer::t::geometry::Image GetAnisotropyMap() const;

    bool HasAORoughnessMetalMap() const;
    cloudViewer::t::geometry::Image GetAORoughnessMetalMap() const;

    // Generic maps and utilities
    Material &SetTextureMap(const std::string &key, const geometry::Image &img);
    Material &SetTextureMap(const std::string &key,
                            const cloudViewer::t::geometry::Image &img);
    std::unordered_map<std::string, cloudViewer::t::geometry::Image>
    GetTextureMaps() const;

    Material &SetScalarProperty(const std::string &key, float value);

    // MaterialRecord ↔ Material compatibility (passthrough in this fork)
    static Material FromMaterialRecord(const Material &rec);
};

// ---- Inline compat helpers mirroring Open3D v0.19 Material API ----
inline Material &Material::SetDefaultProperties() {
    // Minimal defaults consistent with existing renderer expectations
    name = "defaultUnlit";
    shader = "defaultUnlit";
    has_alpha = false;
    sRGB_color = true;
    return *this;
}

inline Material &Material::SetMaterialName(const std::string &mat_name) {
    name = mat_name;
    return *this;
}

inline const std::string &Material::GetMaterialName() const { return name; }

inline bool Material::HasBaseColor() const { return true; }
inline Eigen::Vector4f Material::GetBaseColor() const { return base_color; }

inline bool Material::HasBaseRoughness() const { return true; }
inline float Material::GetBaseRoughness() const { return base_roughness; }

inline bool Material::HasBaseMetallic() const { return true; }
inline float Material::GetBaseMetallic() const { return base_metallic; }

inline bool Material::HasBaseReflectance() const { return true; }
inline float Material::GetBaseReflectance() const { return base_reflectance; }

inline bool Material::HasBaseClearcoat() const { return true; }
inline float Material::GetBaseClearcoat() const { return base_clearcoat; }

inline bool Material::HasBaseClearcoatRoughness() const { return true; }
inline float Material::GetBaseClearcoatRoughness() const {
    return base_clearcoat_roughness;
}

inline bool Material::HasAnisotropy() const { return true; }
inline float Material::GetAnisotropy() const { return base_anisotropy; }

// Emissive color compatibility (not exposed in fields yet → default black)
inline bool Material::HasEmissiveColor() const { return false; }
inline Eigen::Vector4f Material::GetEmissiveColor() const {
    return Eigen::Vector4f(0.f, 0.f, 0.f, 1.f);
}

// Individual texture map helpers (t::geometry::Image API)
inline bool Material::HasAlbedoMap() const { return static_cast<bool>(albedo_img); }
inline cloudViewer::t::geometry::Image Material::GetAlbedoMap() const {
    return albedo_img ? cloudViewer::t::geometry::Image::FromLegacy(*albedo_img)
                      : cloudViewer::t::geometry::Image();
}
inline Material &Material::SetAlbedoMap(const cloudViewer::t::geometry::Image &img) {
    albedo_img = std::make_shared<geometry::Image>(img.ToLegacy());
    return *this;
}

inline bool Material::HasNormalMap() const { return static_cast<bool>(normal_img); }
inline cloudViewer::t::geometry::Image Material::GetNormalMap() const {
    return normal_img ? cloudViewer::t::geometry::Image::FromLegacy(*normal_img)
                      : cloudViewer::t::geometry::Image();
}

inline bool Material::HasAOMap() const { return static_cast<bool>(ao_img); }
inline cloudViewer::t::geometry::Image Material::GetAOMap() const {
    return ao_img ? cloudViewer::t::geometry::Image::FromLegacy(*ao_img)
                  : cloudViewer::t::geometry::Image();
}

inline bool Material::HasMetallicMap() const { return static_cast<bool>(metallic_img); }
inline cloudViewer::t::geometry::Image Material::GetMetallicMap() const {
    return metallic_img ? cloudViewer::t::geometry::Image::FromLegacy(*metallic_img)
                        : cloudViewer::t::geometry::Image();
}

inline bool Material::HasRoughnessMap() const { return static_cast<bool>(roughness_img); }
inline cloudViewer::t::geometry::Image Material::GetRoughnessMap() const {
    return roughness_img ? cloudViewer::t::geometry::Image::FromLegacy(*roughness_img)
                         : cloudViewer::t::geometry::Image();
}

inline bool Material::HasReflectanceMap() const { return static_cast<bool>(reflectance_img); }
inline cloudViewer::t::geometry::Image Material::GetReflectanceMap() const {
    return reflectance_img
                   ? cloudViewer::t::geometry::Image::FromLegacy(*reflectance_img)
                   : cloudViewer::t::geometry::Image();
}

inline bool Material::HasClearcoatMap() const { return static_cast<bool>(clearcoat_img); }
inline cloudViewer::t::geometry::Image Material::GetClearcoatMap() const {
    return clearcoat_img ? cloudViewer::t::geometry::Image::FromLegacy(*clearcoat_img)
                         : cloudViewer::t::geometry::Image();
}

inline bool Material::HasClearcoatRoughnessMap() const {
    return static_cast<bool>(clearcoat_roughness_img);
}
inline cloudViewer::t::geometry::Image Material::GetClearcoatRoughnessMap() const {
    return clearcoat_roughness_img
                   ? cloudViewer::t::geometry::Image::FromLegacy(*clearcoat_roughness_img)
                   : cloudViewer::t::geometry::Image();
}

inline bool Material::HasAnisotropyMap() const { return static_cast<bool>(anisotropy_img); }
inline cloudViewer::t::geometry::Image Material::GetAnisotropyMap() const {
    return anisotropy_img
                   ? cloudViewer::t::geometry::Image::FromLegacy(*anisotropy_img)
                   : cloudViewer::t::geometry::Image();
}

inline bool Material::HasAORoughnessMetalMap() const {
    return static_cast<bool>(ao_rough_metal_img);
}
inline cloudViewer::t::geometry::Image Material::GetAORoughnessMetalMap() const {
    return ao_rough_metal_img
                   ? cloudViewer::t::geometry::Image::FromLegacy(*ao_rough_metal_img)
                   : cloudViewer::t::geometry::Image();
}

// Generic texture map API used by NPZ IO
inline Material &Material::SetTextureMap(const std::string &key,
                                         const geometry::Image &img) {
    if (key == "albedo") albedo_img = std::make_shared<geometry::Image>(img);
    else if (key == "normal") normal_img = std::make_shared<geometry::Image>(img);
    else if (key == "ao") ao_img = std::make_shared<geometry::Image>(img);
    else if (key == "metallic") metallic_img = std::make_shared<geometry::Image>(img);
    else if (key == "roughness") roughness_img = std::make_shared<geometry::Image>(img);
    else if (key == "reflectance") reflectance_img = std::make_shared<geometry::Image>(img);
    else if (key == "clearcoat") clearcoat_img = std::make_shared<geometry::Image>(img);
    else if (key == "clearcoat_roughness" || key == "clearCoatRoughness")
        clearcoat_roughness_img = std::make_shared<geometry::Image>(img);
    else if (key == "anisotropy") anisotropy_img = std::make_shared<geometry::Image>(img);
    else generic_imgs[key] = img;
    return *this;
}

inline Material &Material::SetTextureMap(const std::string &key,
                                         const cloudViewer::t::geometry::Image &img) {
    return SetTextureMap(key, img.ToLegacy());
}

inline std::unordered_map<std::string, cloudViewer::t::geometry::Image>
Material::GetTextureMaps() const {
    std::unordered_map<std::string, cloudViewer::t::geometry::Image> maps;
    if (albedo_img) maps["albedo"] = cloudViewer::t::geometry::Image::FromLegacy(*albedo_img);
    if (normal_img) maps["normal"] = cloudViewer::t::geometry::Image::FromLegacy(*normal_img);
    if (ao_img) maps["ao"] = cloudViewer::t::geometry::Image::FromLegacy(*ao_img);
    if (metallic_img) maps["metallic"] = cloudViewer::t::geometry::Image::FromLegacy(*metallic_img);
    if (roughness_img) maps["roughness"] = cloudViewer::t::geometry::Image::FromLegacy(*roughness_img);
    if (reflectance_img) maps["reflectance"] = cloudViewer::t::geometry::Image::FromLegacy(*reflectance_img);
    if (clearcoat_img) maps["clearcoat"] = cloudViewer::t::geometry::Image::FromLegacy(*clearcoat_img);
    if (clearcoat_roughness_img)
        maps["clearcoat_roughness"] = cloudViewer::t::geometry::Image::FromLegacy(*clearcoat_roughness_img);
    if (anisotropy_img) maps["anisotropy"] = cloudViewer::t::geometry::Image::FromLegacy(*anisotropy_img);
    if (ao_rough_metal_img)
        maps["ao_rough_metal"] = cloudViewer::t::geometry::Image::FromLegacy(*ao_rough_metal_img);
    for (const auto &kv : generic_imgs) {
        maps[kv.first] = cloudViewer::t::geometry::Image::FromLegacy(kv.second);
    }
    return maps;
}

inline Material &Material::SetScalarProperty(const std::string &key, float value) {
    if (key == "metallic") base_metallic = value;
    else if (key == "roughness") base_roughness = value;
    else if (key == "reflectance") base_reflectance = value;
    else if (key == "clearcoat") base_clearcoat = value;
    else if (key == "clearcoat_roughness" || key == "clearCoatRoughness")
        base_clearcoat_roughness = value;
    else if (key == "anisotropy") base_anisotropy = value;
    else generic_params[key] = Eigen::Vector4f(value, 0.f, 0.f, 0.f);
    return *this;
}

// From MaterialRecord compatibility: accept Material passthrough
inline Material Material::FromMaterialRecord(const Material &rec) { return rec; }

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer
