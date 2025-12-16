// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "renders/base/TextureRendererBase.h"

#include <CVLog.h>
#include <ecvMaterialSet.h>

// VTK
#include <vtkActor.h>
#include <vtkProperty.h>

namespace PclUtils {
namespace renders {

void TextureRendererBase::ClearTextures(vtkActor* actor) {
    if (!actor) return;
    vtkProperty* property = actor->GetProperty();
    if (property) {
        property->RemoveAllTextures();
    }
}

bool TextureRendererBase::ValidateActor(vtkActor* actor) const {
    if (!actor) {
        CVLog::Error("[TextureRendererBase::ValidateActor] Actor is null");
        return false;
    }
    return true;
}

bool TextureRendererBase::ValidateMaterials(
        const ccMaterialSet* materials) const {
    if (!materials) {
        CVLog::Error(
                "[TextureRendererBase::ValidateMaterials] Materials is null");
        return false;
    }
    if (materials->empty()) {
        CVLog::Warning(
                "[TextureRendererBase::ValidateMaterials] Materials is empty");
        return false;
    }
    return true;
}

}  // namespace renders
}  // namespace PclUtils
