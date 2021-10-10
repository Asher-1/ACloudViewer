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

#include "visualization/utility/Draw.h"

#include <Logging.h>

#include <sstream>

#include "visualization/gui/Application.h"
#include "visualization/rendering/Model.h"

namespace cloudViewer {
namespace visualization {

DrawObject::DrawObject(const std::string &n,
                       std::shared_ptr<ccHObject> g,
                       bool vis /*= true*/) {
    this->name = n;
    this->geometry = g;
    this->is_visible = vis;
}

DrawObject::DrawObject(const std::string &n,
                       std::shared_ptr<t::geometry::Geometry> tg,
                       bool vis /*= true*/) {
    this->name = n;
    this->tgeometry = tg;
    this->is_visible = vis;
}

DrawObject::DrawObject(const std::string &n,
                       std::shared_ptr<rendering::TriangleMeshModel> m,
                       bool vis /*= true*/) {
    this->name = n;
    this->model = m;
    this->is_visible = vis;
}

// ----------------------------------------------------------------------------
void Draw(const std::vector<std::shared_ptr<ccHObject>> &geometries,
          const std::string &window_name /*= "CloudViewer"*/,
          int width /*= 1024*/,
          int height /*= 768*/,
          const std::vector<DrawAction> &actions /*= {}*/) {
    std::vector<DrawObject> objs;
    objs.reserve(geometries.size());
    for (size_t i = 0; i < geometries.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objs.emplace_back(name.str(), geometries[i]);
    }
    Draw(objs, window_name, width, height, actions);
}

void Draw(
        const std::vector<std::shared_ptr<t::geometry::Geometry>> &tgeometries,
        const std::string &window_name /*= "CloudViewer"*/,
        int width /*= 1024*/,
        int height /*= 768*/,
        const std::vector<DrawAction> &actions /*= {}*/) {
    std::vector<DrawObject> objs;
    objs.reserve(tgeometries.size());
    for (size_t i = 0; i < tgeometries.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objs.emplace_back(name.str(), tgeometries[i]);
    }
    Draw(objs, window_name, width, height, actions);
}

void Draw(const std::vector<std::shared_ptr<rendering::TriangleMeshModel>>
                  &models,
          const std::string &window_name /*= "CloudViewer"*/,
          int width /*= 1024*/,
          int height /*= 768*/,
          const std::vector<DrawAction> &actions /*= {}*/) {
    std::vector<DrawObject> objs;
    objs.reserve(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objs.emplace_back(name.str(), models[i]);
    }
    Draw(objs, window_name, width, height, actions);
}

void Draw(const std::vector<DrawObject> &objects,
          const std::string &window_name /*= "CloudViewer"*/,
          int width /*= 1024*/,
          int height /*= 768*/,
          const std::vector<DrawAction> &actions /*= {}*/) {
    gui::Application::GetInstance().Initialize();
    auto draw = cloudViewer::make_shared<visualizer::O3DVisualizer>(
            window_name, width, height);
    for (auto &o : objects) {
        if (o.geometry) {
            draw->AddGeometry(o.name, o.geometry);
        } else if (o.tgeometry) {
            draw->AddGeometry(o.name, o.tgeometry);
        } else if (o.model) {
            draw->AddGeometry(o.name, o.model);
        } else {
            utility::LogWarning("Invalid object passed to Draw");
        }
        draw->ShowGeometry(o.name, o.is_visible);
    }

    for (auto &act : actions) {
        draw->AddAction(act.name, act.callback);
    }

    draw->ResetCameraToDefault();

    gui::Application::GetInstance().AddWindow(draw);
    draw.reset();  // so we don't hold onto the pointer after Run() cleans up
    gui::Application::GetInstance().Run();
}

}  // namespace visualization
}  // namespace cloudViewer
