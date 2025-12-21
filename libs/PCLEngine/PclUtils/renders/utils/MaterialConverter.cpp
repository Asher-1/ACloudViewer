// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "renders/utils/MaterialConverter.h"

#include <CVLog.h>
#include <CVTools.h>
#include <Utils/sm2cc.h>
#include <ecvMaterial.h>
#include <ecvMaterialSet.h>
#include <pcl/io/obj_io.h>  // For pcl::TexMaterial

namespace PclUtils {
namespace renders {

VtkUtils::VtkMultiTextureRenderer::PBRMaterial
MaterialConverter::FromCCMaterial(const ccMaterial* ccMat) {
    VtkUtils::VtkMultiTextureRenderer::PBRMaterial pbr;

    if (!ccMat) {
        CVLog::Warning(
                "[MaterialConverter::FromCCMaterial] ccMaterial is null");
        return pbr;
    }

    pbr.name = ccMat->getName().toStdString();

    // Extract PBR texture paths
    using TexType = ccMaterial::TextureMapType;

    // Check for multiple map_Kd textures
    auto diffuse_textures = ccMat->getTextureFilenames(TexType::DIFFUSE);
    if (diffuse_textures.size() > 1) {
        pbr.hasMultipleMapKd = true;
        CVLog::Print(
                "[MaterialConverter::FromCCMaterial] Detected %zu map_Kd "
                "textures for material '%s', will use traditional "
                "multi-texture "
                "rendering",
                diffuse_textures.size(), pbr.name.c_str());
    }

    QString diffuse_path = ccMat->getTextureFilename(TexType::DIFFUSE);
    if (!diffuse_path.isEmpty()) {
        pbr.baseColorTexture = diffuse_path.toStdString();
    }

    QString ao_path = ccMat->getTextureFilename(TexType::AMBIENT);
    if (!ao_path.isEmpty()) {
        pbr.aoTexture = ao_path.toStdString();
    }

    QString normal_path = ccMat->getTextureFilename(TexType::NORMAL);
    if (!normal_path.isEmpty()) {
        pbr.normalTexture = normal_path.toStdString();
    }

    QString metallic_path = ccMat->getTextureFilename(TexType::METALLIC);
    if (!metallic_path.isEmpty()) {
        pbr.metallicTexture = metallic_path.toStdString();
    }

    QString roughness_path = ccMat->getTextureFilename(TexType::ROUGHNESS);
    if (!roughness_path.isEmpty()) {
        pbr.roughnessTexture = roughness_path.toStdString();
    }

    QString emissive_path = ccMat->getTextureFilename(TexType::EMISSIVE);
    if (!emissive_path.isEmpty()) {
        pbr.emissiveTexture = emissive_path.toStdString();
    }

    // Extract advanced PBR texture paths (VTK 9.2+)
    QString sheen_path = ccMat->getTextureFilename(TexType::SHEEN);
    if (!sheen_path.isEmpty()) {
        pbr.sheenTexture = sheen_path.toStdString();
    }

    QString clearcoat_path = ccMat->getTextureFilename(TexType::CLEARCOAT);
    if (!clearcoat_path.isEmpty()) {
        pbr.clearcoatTexture = clearcoat_path.toStdString();
    }

    QString clearcoat_roughness_path =
            ccMat->getTextureFilename(TexType::CLEARCOAT_ROUGHNESS);
    if (!clearcoat_roughness_path.isEmpty()) {
        pbr.clearcoatRoughnessTexture = clearcoat_roughness_path.toStdString();
    }

    QString anisotropy_path = ccMat->getTextureFilename(TexType::ANISOTROPY);
    if (!anisotropy_path.isEmpty()) {
        pbr.anisotropyTexture = anisotropy_path.toStdString();
    }

    // Extract material properties
    const ecvColor::Rgbaf& ambient = ccMat->getAmbient();
    const ecvColor::Rgbaf& diffuse = ccMat->getDiffuseFront();
    const ecvColor::Rgbaf& spec = ccMat->getSpecular();
    const ecvColor::Rgbaf& emission = ccMat->getEmission();

    // Clamp color values to [0,1] range
    pbr.baseColor[0] = std::max(0.0f, std::min(1.0f, diffuse.r));
    pbr.baseColor[1] = std::max(0.0f, std::min(1.0f, diffuse.g));
    pbr.baseColor[2] = std::max(0.0f, std::min(1.0f, diffuse.b));

    pbr.ambientColor[0] = std::max(0.0f, std::min(1.0f, ambient.r));
    pbr.ambientColor[1] = std::max(0.0f, std::min(1.0f, ambient.g));
    pbr.ambientColor[2] = std::max(0.0f, std::min(1.0f, ambient.b));

    pbr.diffuseColor[0] = std::max(0.0f, std::min(1.0f, diffuse.r));
    pbr.diffuseColor[1] = std::max(0.0f, std::min(1.0f, diffuse.g));
    pbr.diffuseColor[2] = std::max(0.0f, std::min(1.0f, diffuse.b));

    pbr.specularColor[0] = std::max(0.0f, std::min(1.0f, spec.r));
    pbr.specularColor[1] = std::max(0.0f, std::min(1.0f, spec.g));
    pbr.specularColor[2] = std::max(0.0f, std::min(1.0f, spec.b));

    // PBR scalar parameters with proper clamping
    pbr.metallic = std::max(0.0f, std::min(1.0f, ccMat->getMetallic()));
    pbr.roughness = std::max(0.0f, std::min(1.0f, ccMat->getRoughness()));
    pbr.ao = std::max(0.0f, std::min(2.0f, ccMat->getAmbientOcclusion()));

    // Shininess: prefer explicit Ns value, but if PBR parameters (roughness,
    // sheen) are present, calculate shininess from them to ensure consistency
    // Roughness and shininess are inversely related:
    // - roughness = 0.0 (smooth) -> high shininess
    // - roughness = 1.0 (rough) -> low shininess
    // Formula: shininess = base_shininess * (1 - roughness) * (1 +
    // sheen_factor)
    float baseShininess = ccMat->getShininessFront();
    bool hasExplicitShininess =
            (baseShininess !=
             50.0f);  // 50.0 is default, check if explicitly set
    bool hasPBRParams = (pbr.roughness != 0.5f || ccMat->getSheen() > 0.0f ||
                         ccMat->getClearcoat() > 0.0f);

    if (hasExplicitShininess) {
        // Use explicit Ns value from MTL file
        pbr.shininess = std::max(0.0f, std::min(128.0f, baseShininess));
    } else if (hasPBRParams) {
        // Calculate shininess from PBR parameters
        // Roughness: 0.0 = smooth (high shininess), 1.0 = rough (low shininess)
        float roughnessFactor = 1.0f - pbr.roughness;  // Invert roughness

        // Sheen: 0.0 = no sheen, 1.0 = high sheen (increases shininess)
        float sheenFactor = 1.0f + ccMat->getSheen() * 0.5f;  // Moderate boost

        // Clearcoat: 0.0 = no clearcoat, 1.0 = high clearcoat (increases
        // shininess)
        float clearcoatFactor =
                1.0f + ccMat->getClearcoat() * 0.3f;  // Moderate boost

        // Calculate effective shininess
        // Base shininess of 50.0 is reasonable default, adjust based on PBR
        // params
        float effectiveShininess =
                50.0f * roughnessFactor * sheenFactor * clearcoatFactor;
        pbr.shininess = std::max(0.0f, std::min(128.0f, effectiveShininess));

        CVLog::PrintDebug(
                "[MaterialConverter::FromCCMaterial] Calculated shininess from "
                "PBR params: "
                "base=50.00, roughness=%.2f (factor=%.2f), sheen=%.2f "
                "(factor=%.2f), "
                "clearcoat=%.2f (factor=%.2f) -> shininess=%.2f",
                pbr.roughness, roughnessFactor, ccMat->getSheen(), sheenFactor,
                ccMat->getClearcoat(), clearcoatFactor, pbr.shininess);
    } else {
        // Use default shininess (50.0)
        pbr.shininess = std::max(0.0f, std::min(128.0f, baseShininess));
    }

    // Opacity: prefer from opacity texture, then ambient alpha, then diffuse
    // alpha
    QString opacity_path = ccMat->getTextureFilename(TexType::OPACITY);
    if (!opacity_path.isEmpty()) {
        // If opacity texture exists, use default opacity value
        // The texture will modulate it
        pbr.opacity = std::max(0.0f, std::min(1.0f, ambient.a));
    } else {
        // Use alpha channel from ambient or diffuse
        pbr.opacity = std::max(0.0f, std::min(1.0f, ambient.a));
        if (pbr.opacity <= 0.0f) {
            pbr.opacity = std::max(0.0f, std::min(1.0f, diffuse.a));
        }
    }

    // Advanced PBR parameters
    pbr.sheen = std::max(0.0f, std::min(1.0f, ccMat->getSheen()));
    pbr.clearcoat = std::max(0.0f, std::min(1.0f, ccMat->getClearcoat()));
    pbr.clearcoatRoughness =
            std::max(0.0f, std::min(1.0f, ccMat->getClearcoatRoughness()));
    pbr.anisotropy = std::max(0.0f, std::min(1.0f, ccMat->getAnisotropy()));

    // Emissive color
    pbr.emissive[0] = std::max(0.0f, std::min(1.0f, emission.r));
    pbr.emissive[1] = std::max(0.0f, std::min(1.0f, emission.g));
    pbr.emissive[2] = std::max(0.0f, std::min(1.0f, emission.b));

    CVLog::PrintDebug(
            "[MaterialConverter::FromCCMaterial] Converted material '%s': "
            "ambient=(%.2f,%.2f,%.2f), diffuse=(%.2f,%.2f,%.2f), "
            "specular=(%.2f,%.2f,%.2f), shininess=%.2f, opacity=%.2f",
            pbr.name.c_str(), pbr.ambientColor[0], pbr.ambientColor[1],
            pbr.ambientColor[2], pbr.diffuseColor[0], pbr.diffuseColor[1],
            pbr.diffuseColor[2], pbr.specularColor[0], pbr.specularColor[1],
            pbr.specularColor[2], pbr.shininess, pbr.opacity);

    return pbr;
}

VtkUtils::VtkMultiTextureRenderer::PBRMaterial
MaterialConverter::FromPCLMaterial(const pcl::TexMaterial& pclMat) {
    CVLog::Print(
            "[MaterialConverter::FromPCLMaterial] ENTRY: tex_name=%s, "
            "tex_file=%s",
            pclMat.tex_name.c_str(), pclMat.tex_file.c_str());

    VtkUtils::VtkMultiTextureRenderer::PBRMaterial pbr;

    pbr.name = pclMat.tex_name;

    // pcl::TexMaterial only has one texture path field (tex_file), so multiple
    // textures are encoded into tex_name. Parse the encoding if present.
    // Format: materialName_PBR_MULTITEX|type0:path0|type1:path1|...
    // NOTE: This encoding is only used for pcl::TexMaterial compatibility.
    // For direct ccMaterial usage, use addTextureMeshFromCCMesh() which doesn't
    // need encoding.
    std::string tex_name = pclMat.tex_name;
    size_t pbr_pos = tex_name.find("_PBR_MULTITEX");

    if (pbr_pos != std::string::npos) {
        // Parse encoded texture paths
        size_t start = pbr_pos + 13;  // Length of "_PBR_MULTITEX"
        int diffuse_count = 0;
        while (start < tex_name.length() && tex_name[start] == '|') {
            size_t colon = tex_name.find(':', start + 1);
            size_t next_pipe = tex_name.find('|', colon);
            if (colon == std::string::npos) break;

            std::string type_str =
                    tex_name.substr(start + 1, colon - start - 1);
            std::string path =
                    (next_pipe != std::string::npos)
                            ? tex_name.substr(colon + 1, next_pipe - colon - 1)
                            : tex_name.substr(colon + 1);

            int type = std::stoi(type_str);

            switch (type) {
                case 0:  // DIFFUSE
                    pbr.baseColorTexture = path;
                    diffuse_count++;
                    break;
                case 1:  // AMBIENT (AO)
                    pbr.aoTexture = path;
                    break;
                case 3:  // NORMAL
                    pbr.normalTexture = path;
                    break;
                case 4:  // METALLIC
                    pbr.metallicTexture = path;
                    break;
                case 5:  // ROUGHNESS
                    pbr.roughnessTexture = path;
                    break;
                default:
                    break;
            }

            start = (next_pipe != std::string::npos) ? next_pipe
                                                     : tex_name.length();
        }

        if (diffuse_count > 1) {
            pbr.hasMultipleMapKd = true;
            CVLog::PrintDebug(
                    "[MaterialConverter::FromPCLMaterial] Detected %d map_Kd "
                    "textures for material '%s', will use traditional "
                    "multi-texture rendering",
                    diffuse_count, pbr.name.c_str());
        }

        // Extract real material name (remove encoded part)
        pbr.name = tex_name.substr(0, pbr_pos);
    } else {
        // Single texture mode - use tex_file directly
        if (!pclMat.tex_file.empty()) {
            pbr.baseColorTexture = pclMat.tex_file;
        }
    }

    // Extract material properties
    pbr.baseColor[0] = pclMat.tex_Kd.r;
    pbr.baseColor[1] = pclMat.tex_Kd.g;
    pbr.baseColor[2] = pclMat.tex_Kd.b;

    pbr.ambientColor[0] = pclMat.tex_Ka.r;
    pbr.ambientColor[1] = pclMat.tex_Ka.g;
    pbr.ambientColor[2] = pclMat.tex_Ka.b;

    pbr.diffuseColor[0] = pclMat.tex_Kd.r;
    pbr.diffuseColor[1] = pclMat.tex_Kd.g;
    pbr.diffuseColor[2] = pclMat.tex_Kd.b;

    pbr.specularColor[0] = pclMat.tex_Ks.r;
    pbr.specularColor[1] = pclMat.tex_Ks.g;
    pbr.specularColor[2] = pclMat.tex_Ks.b;

    pbr.shininess = std::max(0.0f, std::min(128.0f, pclMat.tex_Ns));
    pbr.opacity = std::max(0.0f, std::min(1.0f, pclMat.tex_d));

    // PBR scalar parameters (defaults for PCL materials)
    pbr.metallic = 0.0f;
    pbr.roughness = 0.5f;
    pbr.ao = 1.0f;

    // Emissive color (default to black)
    pbr.emissive[0] = 0.0f;
    pbr.emissive[1] = 0.0f;
    pbr.emissive[2] = 0.0f;

    CVLog::PrintDebug(
            "[MaterialConverter::FromPCLMaterial] Converted material '%s': "
            "ambient=(%.2f,%.2f,%.2f), diffuse=(%.2f,%.2f,%.2f), "
            "specular=(%.2f,%.2f,%.2f), shininess=%.2f, opacity=%.2f",
            pbr.name.c_str(), pbr.ambientColor[0], pbr.ambientColor[1],
            pbr.ambientColor[2], pbr.diffuseColor[0], pbr.diffuseColor[1],
            pbr.diffuseColor[2], pbr.specularColor[0], pbr.specularColor[1],
            pbr.specularColor[2], pbr.shininess, pbr.opacity);

    return pbr;
}

VtkUtils::VtkMultiTextureRenderer::PBRMaterial
MaterialConverter::FromMaterialSet(const ccMaterialSet* materials) {
    if (!materials || materials->empty()) {
        CVLog::Error(
                "[MaterialConverter::FromMaterialSet] No materials provided");
        VtkUtils::VtkMultiTextureRenderer::PBRMaterial pbr;
        return pbr;
    }

    auto firstMaterial = materials->at(0);
    if (!firstMaterial) {
        CVLog::Error(
                "[MaterialConverter::FromMaterialSet] First material is null!");
        VtkUtils::VtkMultiTextureRenderer::PBRMaterial pbr;
        return pbr;
    }

    // Convert first material (PBRRenderer only uses
    // first)
    VtkUtils::VtkMultiTextureRenderer::PBRMaterial pbr =
            FromCCMaterial(firstMaterial.data());

    // Check all materials for PBR textures to ensure accurate detection
    // This is important for TextureRenderManager::AnalyzeMaterials which uses
    // hasPBRTextures() to determine rendering mode
    using TexType = ccMaterial::TextureMapType;
    bool found_pbr_in_other_materials = false;

    // Check all materials (not just first) for PBR textures
    for (size_t i = 0; i < materials->size(); ++i) {
        auto mat = materials->at(i);
        if (!mat || mat == firstMaterial) continue;

        // Check if this material has any PBR texture types
        if (mat->hasTextureMap(TexType::NORMAL) ||
            mat->hasTextureMap(TexType::METALLIC) ||
            mat->hasTextureMap(TexType::ROUGHNESS) ||
            mat->hasTextureMap(TexType::AMBIENT)) {
            found_pbr_in_other_materials = true;
            break;
        }

        // Also check for base color texture (DIFFUSE) - it's also a PBR texture
        auto diffuse_textures = mat->getTextureFilenames(TexType::DIFFUSE);
        if (!diffuse_textures.empty()) {
            found_pbr_in_other_materials = true;
            break;
        }
    }

    // If we found PBR textures in other materials but first material's
    // conversion didn't detect them, ensure first material's baseColorTexture
    // is set (if it has a diffuse texture) so hasPBRTextures() returns true
    if (found_pbr_in_other_materials && !pbr.hasPBRTextures()) {
        auto firstDiffuse =
                firstMaterial->getTextureFilenames(TexType::DIFFUSE);
        if (!firstDiffuse.empty() && pbr.baseColorTexture.empty()) {
            pbr.baseColorTexture = firstDiffuse[0].toStdString();
            CVLog::PrintDebug(
                    "[MaterialConverter::FromMaterialSet] Found PBR textures "
                    "in "
                    "other materials, ensuring first material's "
                    "baseColorTexture "
                    "is set for accurate detection");
        }
    }

    return pbr;
}

bool MaterialConverter::HasPBREncoding(const ccMaterial* material) {
    if (!material) return false;

    // Check if material name contains _PBR_MULTITEX encoding
    // This indicates the material came from pcl::TexMaterial (pcl::TextureMesh)
    // and uses encoding due to pcl::TexMaterial structure limitations
    std::string mat_name = CVTools::FromQString(material->getName());
    if (mat_name.find("_PBR_MULTITEX") != std::string::npos) {
        return true;
    }

    // Also check if material has actual PBR texture types (for direct
    // ccMaterial usage) PBR textures: Normal, Metallic, Roughness, AO, etc.
    using TexType = ccMaterial::TextureMapType;

    if (material->hasTextureMap(TexType::NORMAL) ||
        material->hasTextureMap(TexType::METALLIC) ||
        material->hasTextureMap(TexType::ROUGHNESS) ||
        material->hasTextureMap(TexType::AMBIENT)) {
        return true;
    }

    return false;
}

bool MaterialConverter::HasMultipleMapKd(const ccMaterialSet* materials) {
    if (!materials || materials->empty()) return false;

    int count = 0;
    using TexType = ccMaterial::TextureMapType;
    for (size_t i = 0; i < materials->size(); ++i) {
        auto mat = materials->at(i);
        if (mat) {
            auto diffuse_textures = mat->getTextureFilenames(TexType::DIFFUSE);
            count += diffuse_textures.size();
        }
    }

    return count > 1;
}

int MaterialConverter::CountMapKd(const ccMaterialSet* materials) {
    if (!materials || materials->empty()) return 0;

    int count = 0;
    using TexType = ccMaterial::TextureMapType;
    for (size_t i = 0; i < materials->size(); ++i) {
        auto mat = materials->at(i);
        if (mat) {
            auto diffuse_textures = mat->getTextureFilenames(TexType::DIFFUSE);
            count += static_cast<int>(diffuse_textures.size());
        }
    }

    CVLog::PrintDebug(
            "[MaterialConverter::CountMapKd] Counted %d map_Kd textures",
            count);

    return count;
}

}  // namespace renders
}  // namespace PclUtils
