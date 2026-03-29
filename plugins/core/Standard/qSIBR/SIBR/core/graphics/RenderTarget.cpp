// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/graphics/RenderTarget.hpp"
// #define HEADLESS

namespace sibr {
void blit(const IRenderTarget& src,
          const IRenderTarget& dst,
          GLbitfield mask,
          GLenum filter) {
#ifdef HEADLESS
    SIBR_ERR << "No named blit frame buffer in headless " << std::endl;
#else
    glBlitNamedFramebuffer(src.fbo(), dst.fbo(), 0, 0, src.w(), src.h(), 0, 0,
                           dst.w(), dst.h(), mask, filter);
#endif
}

void blit_and_flip(const IRenderTarget& src,
                   const IRenderTarget& dst,
                   GLbitfield mask,
                   GLenum filter) {
#ifdef HEADLESS
    SIBR_ERR << "No named blit frame buffer in headless " << std::endl;
#else
    glBlitNamedFramebuffer(src.fbo(), dst.fbo(), 0, 0, src.w(), src.h(), 0,
                           dst.h(), dst.w(), 0, mask, filter);
#endif
}

}  // namespace sibr
