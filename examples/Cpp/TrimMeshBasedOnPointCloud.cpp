// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CloudViewer.h"

void PrintHelp() {
    using namespace cloudViewer;
    PrintCloudViewerVersion();

    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > TrimMeshBasedOnPointCloud [options]");
    utility::LogInfo("      Trim a mesh baesd on distance to a point cloud.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4). Default: 2.");
    utility::LogInfo("    --in_mesh mesh_file       : Input mesh file. MUST HAVE.");
    utility::LogInfo("    --out_mesh mesh_file      : Output mesh file. MUST HAVE.");
    utility::LogInfo("    --pointcloud pcd_file     : Reference pointcloud file. MUST HAVE.");
    utility::LogInfo("    --distance d              : Maximum distance. MUST HAVE.");
    // clang-format on
}

int main(int argc, char* argv[]) {
    using namespace cloudViewer;

    if (argc < 4 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    auto in_mesh_file =
            utility::GetProgramOptionAsString(argc, argv, "--in_mesh");
    auto out_mesh_file =
            utility::GetProgramOptionAsString(argc, argv, "--out_mesh");
    auto pcd_file =
            utility::GetProgramOptionAsString(argc, argv, "--pointcloud");
    auto distance = utility::GetProgramOptionAsDouble(argc, argv, "--distance");
    if (distance <= 0.0) {
        utility::LogWarning("Illegal distance.");
        return 1;
    }
    if (in_mesh_file.empty() || out_mesh_file.empty() || pcd_file.empty()) {
        utility::LogWarning("Missing file names.");
        return 1;
    }
    auto mesh = io::CreateMeshFromFile(in_mesh_file);
    auto pcd = io::CreatePointCloudFromFile(pcd_file);
    if (mesh->IsEmpty() || pcd->IsEmpty()) {
        utility::LogWarning("Empty geometry.");
        return 1;
    }

    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*pcd);
    std::vector<bool> remove_vertex_mask(mesh->getVerticeSize(), false);
    utility::ConsoleProgressBar progress_bar(mesh->getVerticeSize(),
                                             "Prune vetices: ");
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#endif
    for (int i = 0; i < (int)mesh->getVerticeSize(); i++) {
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        int k = kdtree.SearchKNN(mesh->getVertice(static_cast<unsigned int>(i)),
                                 1, indices, dists);
        if (k == 0 || dists[0] > distance * distance) {
            remove_vertex_mask[i] = true;
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            ++progress_bar;
        }
    }

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
    assert(cloud);
    std::vector<int> index_old_to_new(cloud->size());

    bool has_vert_normal = cloud->hasNormals();
    bool has_vert_color = cloud->hasColors();
    size_t old_vertex_num = cloud->size();
    size_t k = 0;  // new index
    bool has_tri_normal = mesh->hasTriNormals();
    size_t old_triangle_num = mesh->size();
    size_t kt = 0;
    for (unsigned int i = 0; i < (unsigned int)old_vertex_num;
         i++) {  // old index
        if (!remove_vertex_mask[i]) {
            cloud->setPoint(k, *cloud->getPoint(i));
            if (has_vert_normal)
                cloud->setPointNormal(k, cloud->getPointNormal(i));
            if (has_vert_color)
                cloud->setPointColor(k, cloud->getPointColor(i));
            index_old_to_new[i] = (int)k;
            k++;
        } else {
            index_old_to_new[i] = -1;
        }
    }

    cloud->resize(static_cast<unsigned int>(k));
    if (has_vert_normal) cloud->resizeTheNormsTable();
    if (has_vert_color) cloud->resizeTheRGBTable();
    if (k < old_vertex_num) {
        for (size_t i = 0; i < old_triangle_num; i++) {
            Eigen::Vector3i triangle;
            mesh->getTriangleVertIndexes(i, triangle);
            triangle(0) = index_old_to_new[triangle(0)];
            triangle(1) = index_old_to_new[triangle(1)];
            triangle(2) = index_old_to_new[triangle(2)];
            if (triangle(0) != -1 && triangle(1) != -1 && triangle(2) != -1) {
                mesh->setTriangle(kt, triangle);
                if (has_tri_normal)
                    mesh->setTriangleNormalIndexes(
                            kt, mesh->getTriangleNormalIndexes(i));
                kt++;
            }
        }

        mesh->resize(kt);
        if (has_tri_normal) mesh->getTriNormsTable()->resize(kt);
    }
    utility::LogInfo(
            "[TrimMeshBasedOnPointCloud] {:d} vertices and {:d} triangles have "
            "been removed.",
            old_vertex_num - k, old_triangle_num - kt);
    io::WriteTriangleMesh(out_mesh_file, *mesh);

    visualization::DrawGeometries({mesh}, "Trimed-Mesh", 1600, 900);
    return 0;
}
