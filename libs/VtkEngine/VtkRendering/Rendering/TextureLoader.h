// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file TextureLoader.h
 *  @brief Loads textures from ccMaterial into VTK
 */

// Forward declarations
class vtkTexture;

// Include ccMaterial header for CShared typedef
#include <ecvMaterial.h>

namespace Visualization {
namespace renders {

/**
 * @brief Utility class for loading textures from various sources
 */
class TextureLoader {
public:
    /**
     * @brief Load texture from ccMaterial
     * @param material ccMaterial object
     * @param vtk_tex VTK texture object to load into
     * @return 1 on success, -1 on failure
     */
    static int LoadFromCCMaterial(ccMaterial::CShared material,
                                  vtkTexture* vtk_tex);
};

}  // namespace renders
}  // namespace Visualization
