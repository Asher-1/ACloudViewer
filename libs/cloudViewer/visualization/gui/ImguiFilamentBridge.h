// Altered from Filament's ImGuiHelper.cpp
// Filament code is from somewhere close to v1.4.3 and is:
/*
 * Copyright (C) 2018 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// CloudViewer alterations are:
// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: PixelBufferDescriptor assert unsigned is positive before subtracting
//       but MSVC can't figure that out.
// 4293:  Filament's utils/algorithm.h utils::details::clz() does strange
//        things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//        32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//        determine the if statement does not run.)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293)
#endif  // _MSC_VER

#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/Texture.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include <cstddef>  // <filament/Engine> recursive includes needs this, std::size_t especially
#include <memory>

struct ImDrawData;

namespace cloudViewer {

namespace visualization {
namespace rendering {
class FilamentRenderer;
}
}  // namespace visualization

namespace visualization {
namespace gui {

struct Size;
class Window;

// Translates ImGui's draw commands into Filament primitives, textures, vertex
// buffers, etc. Creates a UI-specific Scene object and populates it with a
// Renderable. Does not handle event processing; clients can simply call
// ImGui::GetIO() directly and set the mouse state.
class ImguiFilamentBridge {
public:
    ImguiFilamentBridge(visualization::rendering::FilamentRenderer* renderer,
                        const Size& window_size);
    ~ImguiFilamentBridge();

    // Helper method called after resolving fontPath; public so fonts can be
    // added by caller. Requires the appropriate ImGuiContext to be current
    void CreateAtlasTextureAlpha8(unsigned char* pixels,
                                  int width,
                                  int height,
                                  int bytes_per_px);

    // This populates the Filament View. Clients are responsible for
    // rendering the View. This should be called on every frame, regardless of
    // whether the Renderer wants to skip or not.
    void Update(ImDrawData* imguiData);

    void OnWindowResized(const Window& window);

private:
    void CreateBuffers(size_t num_required_buffers);
    void PopulateVertexData(size_t buffer_index,
                            size_t vb_size_in_bytes,
                            void* vb_data,
                            size_t ib_size_in_bytes,
                            void* ib_data);
    void CreateVertexBuffer(size_t buffer_index, size_t capacity);
    void CreateIndexBuffer(size_t buffer_index, size_t capacity);
    void SyncThreads();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
