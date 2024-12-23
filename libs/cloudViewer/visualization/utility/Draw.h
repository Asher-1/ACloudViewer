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

#include <vector>

#include "visualization/rendering/Model.h"
#include "visualization/visualizer/O3DVisualizer.h"

class ccHObject;

namespace cloudViewer {
namespace visualization {

struct DrawObject {
    std::string name;
    std::shared_ptr<ccHObject> geometry;
    std::shared_ptr<t::geometry::Geometry> tgeometry;
    std::shared_ptr<rendering::TriangleMeshModel> model;
    bool is_visible;

    DrawObject(const std::string &n,
               std::shared_ptr<ccHObject> g,
               bool vis = true);
    DrawObject(const std::string &n,
               std::shared_ptr<t::geometry::Geometry> tg,
               bool vis = true);
    DrawObject(const std::string &n,
               std::shared_ptr<rendering::TriangleMeshModel> m,
               bool vis = true);
};

struct DrawAction {
    std::string name;
    std::function<void(visualizer::O3DVisualizer &)> callback;
};

void Draw(const std::vector<std::shared_ptr<ccHObject>> &geometries,
          const std::string &window_name = "CloudViewer",
          int width = 1024,
          int height = 768,
          const std::vector<DrawAction> &actions = {});

void Draw(
        const std::vector<std::shared_ptr<t::geometry::Geometry>> &tgeometries,
        const std::string &window_name = "CloudViewer",
        int width = 1024,
        int height = 768,
        const std::vector<DrawAction> &actions = {});

void Draw(const std::vector<std::shared_ptr<rendering::TriangleMeshModel>>
                  &models,
          const std::string &window_name = "CloudViewer",
          int width = 1024,
          int height = 768,
          const std::vector<DrawAction> &actions = {});

void Draw(const std::vector<DrawObject> &objects,
          const std::string &window_name = "CloudViewer",
          int width = 1024,
          int height = 768,
          const std::vector<DrawAction> &actions = {});

}  // namespace visualization
}  // namespace cloudViewer
