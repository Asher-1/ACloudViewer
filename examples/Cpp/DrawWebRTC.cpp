// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
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

#include <cstdlib>

#include "CloudViewer.h"

using namespace cloudViewer;

// Create and add a window to gui::Application, but do not run it yet.
void AddDrawWindow(const std::vector<std::shared_ptr<ccHObject>> &geometries,
                   const std::string &window_name = "CloudViewer",
                   int width = 1024,
                   int height = 768,
                   const std::vector<visualization::DrawAction> &actions = {}) {
    std::vector<visualization::DrawObject> objects;
    objects.reserve(geometries.size());
    for (size_t i = 0; i < geometries.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objects.emplace_back(name.str(), geometries[i]);
    }

    auto &o3d_app = visualization::gui::Application::GetInstance();
    o3d_app.Initialize();

    auto draw =
            cloudViewer::make_shared<visualization::visualizer::O3DVisualizer>(
                    window_name, width, height);
    for (auto &o : objects) {
        if (o.geometry) {
            draw->AddGeometry(o.name, o.geometry);
        } else {
            draw->AddGeometry(o.name, o.tgeometry);
        }
        draw->ShowGeometry(o.name, o.is_visible);
    }
    for (auto &act : actions) {
        draw->AddAction(act.name, act.callback);
    }
    draw->ResetCameraToDefault();
    visualization::gui::Application::GetInstance().AddWindow(draw);
    draw.reset();  // so we don't hold onto the pointer after Run() cleans up
}

// Create a window with an empty box and a custom action button for adding a
// new visualization vindow.
void EmptyBox(const std::string& path) {
    const double pc_rad = 1.0;
    const double r = 0.4;

    auto big_bbox = cloudViewer::make_shared<ccBBox>(
            Eigen::Vector3d{-pc_rad, -3, -pc_rad},
            Eigen::Vector3d{6.0 + r, 1.0 + r, pc_rad});

    auto new_window_action =
            [&](visualization::visualizer::O3DVisualizer &o3dvis) {
                utility::LogInfo("new_window_action called");
                auto mesh = cloudViewer::make_shared<ccMesh>();
                mesh->createInternalCloud();
                io::ReadTriangleMesh(path + "/knot.ply", *mesh);
                mesh->computeVertexNormals();
                AddDrawWindow({mesh}, "CloudViewer pcd", 640, 480);
            };

    AddDrawWindow({big_bbox}, "CloudViewer EmptyBox", 800, 480,
                  {{"Load example mesh", new_window_action}});
}

// Create a window with various geometry objects.
void BoxWithObjects() {
    const double pc_rad = 1.0;
    const double r = 0.4;
    auto sphere_unlit = ccMesh::CreateSphere(r);
    sphere_unlit->translate({0.0, 1.0, 0.0});
    auto sphere_colored_unlit = ccMesh::CreateSphere(r);
    sphere_colored_unlit->paintUniformColor({1.0, 0.0, 0.0});
    sphere_colored_unlit->translate({2.0, 1.0, 0.0});
    auto sphere_lit = ccMesh::CreateSphere(r);
    sphere_lit->computeVertexNormals();
    sphere_lit->translate({4, 1, 0});
    auto sphere_colored_lit = ccMesh::CreateSphere(r);
    sphere_colored_lit->computeVertexNormals();
    sphere_colored_lit->paintUniformColor({0.0, 1.0, 0.0});
    sphere_colored_lit->translate({6, 1, 0});
    auto big_bbox = cloudViewer::make_shared<ccBBox>(
            Eigen::Vector3d{-pc_rad, -3, -pc_rad},
            Eigen::Vector3d{6.0 + r, 1.0 + r, pc_rad});
    auto bbox = sphere_unlit->getAxisAlignedBoundingBox();
    auto sphere_bbox = cloudViewer::make_shared<ccBBox>(bbox.getMinBound(),
                                                        bbox.getMaxBound());
    sphere_bbox->setColor({1.0, 0.5, 0.0});
    auto lines = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            sphere_lit->getAxisAlignedBoundingBox());
    auto lines_colored = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            sphere_colored_lit->getAxisAlignedBoundingBox());
    lines_colored->paintUniformColor({0.0, 0.0, 1.0});

    AddDrawWindow(
            {sphere_unlit, sphere_colored_unlit, sphere_lit, sphere_colored_lit,
             big_bbox, sphere_bbox, lines, lines_colored},
            "CloudViewer BoxWithObjects", 640, 480);
}

void PrintHelp() {
    using namespace cloudViewer;
    PrintCloudViewerVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > DrawWebRTC [filename]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char **argv) {
    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    if (!utility::filesystem::DirectoryExists(argv[1])) {
        utility::LogError("Invalid test_dir: {}", argv[1]);
    }
    visualization::webrtc_server::WebRTCWindowSystem::GetInstance()
            ->EnableWebRTC();

    // Uncomment this line to see more WebRTC loggings
//    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    EmptyBox(argv[1]);
    BoxWithObjects();
    visualization::gui::Application::GetInstance().Run();
}
