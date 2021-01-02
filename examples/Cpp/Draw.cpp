// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

double GetRandom() { return double(std::rand()) / double(RAND_MAX); }

std::shared_ptr<ccPointCloud> MakePointCloud(
        int npts, const Eigen::Vector3d center, double radius, bool colorize) {
    auto cloud = std::make_shared<ccPointCloud>();
    cloud->reserveThePointsTable(static_cast<unsigned>(npts));
    for (int i = 0; i < npts; ++i) {
        cloud->addEigenPoint({radius * GetRandom() + center.x(),
                              radius * GetRandom() + center.y(),
                              radius * GetRandom() + center.z()});
    }
    if (colorize) {
        cloud->reserveTheRGBTable();
        for (int i = 0; i < npts; ++i) {
            cloud->addEigenColor({GetRandom(), GetRandom(), GetRandom()});
        }
    }
    return cloud;
}

void SingleObject() {
    // No colors, no normals, should appear unlit black
    auto cube = ccMesh::CreateBox(1, 2, 4);
    visualization::Draw({cube});
}

void MultiObjects() {
    const double pc_rad = 1.0;
    auto pc_nocolor = MakePointCloud(100, {0.0, -2.0, 0.0}, pc_rad, false);
    auto pc_color = MakePointCloud(100, {3.0, -2.0, 0.0}, pc_rad, true);
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
    auto big_bbox = std::make_shared<ccBBox>(
            Eigen::Vector3d{-pc_rad, -3, -pc_rad},
            Eigen::Vector3d{6.0 + r, 1.0 + r, pc_rad});
    auto bbox = sphere_unlit->getAxisAlignedBoundingBox();
    auto sphere_bbox = std::make_shared<ccBBox>(
            bbox.getMinBound(), bbox.getMaxBound());
    sphere_bbox->setColor({1.0, 0.5, 0.0});
    auto lines = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            sphere_lit->getAxisAlignedBoundingBox());
    auto lines_colored = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            sphere_colored_lit->getAxisAlignedBoundingBox());
    lines_colored->paintUniformColor({0.0, 0.0, 1.0});

    visualization::Draw({pc_nocolor, pc_color, sphere_unlit,
                         sphere_colored_unlit, sphere_lit, sphere_colored_lit,
                         big_bbox, sphere_bbox, lines, lines_colored});
}

void Actions(const std::string test_dir) {
    const char *SOURCE_NAME = "Source";
    const char *RESULT_NAME = "Result (Poisson reconstruction)";
    const char *TRUTH_NAME = "Ground truth";

    auto bunny = std::make_shared<ccMesh>();
    bunny->createInternalCloud();
    io::ReadTriangleMesh(test_dir + "/Bunny.ply", *bunny);
    if (bunny->isEmpty()) {
        CVLib::utility::LogError(
                "Please download the Standford Bunny dataset using:\n"
                "   cd <cloudViewer_dir>/examples/python\n"
                "   python -c 'from cloudViewer_tutorial import *; "
                "get_bunny_mesh()'");
        return;
    }

    bunny->paintUniformColor({1, 0.75, 0});
    bunny->computeVertexNormals();
    auto cloud = std::make_shared<ccPointCloud>();

    cloud->addPoints(bunny->getVerticesPtr());
    cloud->addEigenNorms(bunny->getVertexNormals());
    cloud->paintUniformColor({0, 0.2, 1.0});

    auto make_mesh = [SOURCE_NAME, RESULT_NAME](
                             visualization::visualizer::O3DVisualizer &o3dvis) {
        std::shared_ptr<ccPointCloud> source =
                std::dynamic_pointer_cast<ccPointCloud>(
                        o3dvis.GetGeometry(SOURCE_NAME).geometry);
        auto mesh = std::get<0>(
                ccMesh::CreateFromPointCloudPoisson(*source));
        mesh->paintUniformColor({1, 1, 1});
        mesh->computeVertexNormals();
        o3dvis.AddGeometry(RESULT_NAME, mesh);
        o3dvis.ShowGeometry(SOURCE_NAME, false);
    };

    auto toggle_result =
            [TRUTH_NAME,
             RESULT_NAME](visualization::visualizer::O3DVisualizer &o3dvis) {
                bool truth_vis = o3dvis.GetGeometry(TRUTH_NAME).is_visible;
                o3dvis.ShowGeometry(TRUTH_NAME, !truth_vis);
                o3dvis.ShowGeometry(RESULT_NAME, truth_vis);
            };

    visualization::Draw({visualization::DrawObject(SOURCE_NAME, cloud),
                         visualization::DrawObject(TRUTH_NAME, bunny, false)},
                        "CloudViewer: Draw Example: Actions", 1024, 768,
                        {{"Create Mesh", make_mesh},
                         {"Toggle truth/result", toggle_result}});
}

Eigen::Matrix4d_u GetICPTransform(
        const ccPointCloud &source,
        const ccPointCloud &target,
        const std::vector<visualization::visualizer::O3DVisualizerSelections::
                                  SelectedIndex> &source_picked,
        const std::vector<visualization::visualizer::O3DVisualizerSelections::
                                  SelectedIndex> &target_picked) {
    std::vector<Eigen::Vector2i> indices;
    for (size_t i = 0; i < source_picked.size(); ++i) {
        indices.push_back({source_picked[i].index, target_picked[i].index});
    }

    // Estimate rough transformation using correspondences
    pipelines::registration::TransformationEstimationPointToPoint p2p;
    auto trans_init = p2p.ComputeTransformation(source, target, indices);

    // Point-to-point ICP for refinement
    const double max_dist = 0.03;  // 3cm distance threshold
    auto result = pipelines::registration::RegistrationICP(
            source, target, max_dist, trans_init);

    return result.transformation_;
}

void Selections(const std::string test_dir) {
    std::cout << "Selection example:" << std::endl;
    std::cout << "  One set:  pick three points from the source (yellow), "
              << std::endl;
    std::cout << "            then pick the same three points in the target"
                 "(blue) cloud"
              << std::endl;
    std::cout << "  Two sets: pick three points from the source cloud, "
              << std::endl;
    std::cout << "            then create a new selection set, and pick the"
              << std::endl;
    std::cout << "            three points from the target." << std::endl;

    const auto cloud0_path = test_dir + "/ICP/cloud_bin_0.pcd";
    const auto cloud1_path = test_dir + "/ICP/cloud_bin_2.pcd";
    auto source = std::make_shared<ccPointCloud>();
    io::ReadPointCloud(cloud0_path, *source);
    if (source->isEmpty()) {
        CVLib::utility::LogError("Could not open {}", cloud0_path);
        return;
    }
    auto target = std::make_shared<ccPointCloud>();
    io::ReadPointCloud(cloud1_path, *target);
    if (target->isEmpty()) {
        CVLib::utility::LogError("Could not open {}", cloud1_path);
        return;
    }
    source->paintUniformColor({1.000, 0.706, 0.000});
    target->paintUniformColor({0.000, 0.651, 0.929});

    const char *source_name = "Source (yellow)";
    const char *target_name = "Target (blue)";

    auto DoICPOneSet =
            [source, target, source_name,
             target_name](visualization::visualizer::O3DVisualizer &o3dvis) {
                auto sets = o3dvis.GetSelectionSets();
                if (sets.empty()) {
                     CVLib::utility::LogWarning(
                             "You must select points for correspondence before "
                             "running ICP!");
                     return;
                 }
                auto &source_picked_set = sets[0][source_name];
                auto &target_picked_set = sets[0][target_name];
                std::vector<visualization::visualizer::O3DVisualizerSelections::
                                    SelectedIndex>
                        source_picked(source_picked_set.begin(),
                                      source_picked_set.end());
                std::vector<visualization::visualizer::O3DVisualizerSelections::
                                    SelectedIndex>
                        target_picked(target_picked_set.begin(),
                                      target_picked_set.end());
                std::sort(source_picked.begin(), source_picked.end());
                std::sort(target_picked.begin(), target_picked.end());

                auto t = GetICPTransform(*source, *target, source_picked,
                                         target_picked);
                source->transform(t);

                // Update the source geometry
                o3dvis.RemoveGeometry(source_name);
                o3dvis.AddGeometry(source_name, source);
            };

    auto DoICPTwoSets =
            [source, target, source_name,
             target_name](visualization::visualizer::O3DVisualizer &o3dvis) {
                auto sets = o3dvis.GetSelectionSets();
                if (sets.size() < 2) {
                    CVLib::utility::LogWarning(
                            "You must have at least two sets of selected "
                            "points before running ICP!");
                    return;
                }
                auto &source_picked_set = sets[0][source_name];
                auto &target_picked_set = sets[1][target_name];
                std::vector<visualization::visualizer::O3DVisualizerSelections::
                                    SelectedIndex>
                        source_picked(source_picked_set.begin(),
                                      source_picked_set.end());
                std::vector<visualization::visualizer::O3DVisualizerSelections::
                                    SelectedIndex>
                        target_picked(target_picked_set.begin(),
                                      target_picked_set.end());
                std::sort(source_picked.begin(), source_picked.end());
                std::sort(target_picked.begin(), target_picked.end());

                auto t = GetICPTransform(*source, *target, source_picked,
                                         target_picked);
                source->transform(t);

                // Update the source geometry
                o3dvis.RemoveGeometry(source_name);
                o3dvis.AddGeometry(source_name, source);
            };

    visualization::Draw({visualization::DrawObject(source_name, source),
                         visualization::DrawObject(target_name, target)},
                        "CloudViewer: Draw example: Selection", 1024, 768,
                        {{"ICP Registration (one set)", DoICPOneSet},
                         {"ICP Registration (two sets)", DoICPTwoSets}});
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        CVLib::utility::LogError("missing input directionary!");
        return 0;
    }
    std::string TEST_DIR(argv[1]);
    if (!CVLib::utility::filesystem::DirectoryExists(TEST_DIR)) {
        CVLib::utility::LogError(
                "This example needs to be run from the <build>/bin/examples "
                "directory");
    }

    SingleObject();
    MultiObjects();
    Actions(TEST_DIR);
    Selections(TEST_DIR);
}
