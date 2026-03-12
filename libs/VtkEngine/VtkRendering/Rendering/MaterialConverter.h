// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file MaterialConverter.h
 *  @brief Converts ccMaterial to PBRMaterial format
 */

#include <string>

#include "VtkMultiTextureRenderer.h"

// Forward declarations
class ccMaterial;
class ccMaterialSet;

namespace Visualization {
namespace renders {

/**
 * @brief Material converter utility
 *
 * Converts between different material formats:
 * - ccMaterial -> PBRMaterial
 * - ccMaterialSet -> PBRMaterial (first material)
 */
class MaterialConverter {
public:
    /**
     * @brief Convert ccMaterial to PBRMaterial
     */
    static VtkUtils::VtkMultiTextureRenderer::PBRMaterial FromCCMaterial(
            const ccMaterial* ccMat);

    /**
     * @brief Convert first material from ccMaterialSet to PBRMaterial
     */
    static VtkUtils::VtkMultiTextureRenderer::PBRMaterial FromMaterialSet(
            const ccMaterialSet* materials);

    /**
     * @brief Detect if material has PBR encoding
     */
    static bool HasPBREncoding(const ccMaterial* material);

    /**
     * @brief Detect if material set has multiple map_Kd textures
     */
    static bool HasMultipleMapKd(const ccMaterialSet* materials);

    /**
     * @brief Count map_Kd textures in material set
     */
    static int CountMapKd(const ccMaterialSet* materials);
};

}  // namespace renders
}  // namespace Visualization
