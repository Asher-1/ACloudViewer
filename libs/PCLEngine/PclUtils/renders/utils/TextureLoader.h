// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <pcl/TextureMesh.h>

// Forward declarations
class vtkTexture;

// Include ccMaterial header for CShared typedef
#include <ecvMaterial.h>

namespace PclUtils {
namespace renders {

/**
 * @brief Utility class for loading textures from various sources
 */
class TextureLoader {
public:
    /**
     * @brief Load texture from pcl::TexMaterial
     * @param tex_mat PCL texture material
     * @param vtk_tex VTK texture object to load into
     * @return 1 on success, -1 on failure
     */
    static int LoadFromPCLMaterial(const pcl::TexMaterial& tex_mat,
                                   vtkTexture* vtk_tex);

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
}  // namespace PclUtils
