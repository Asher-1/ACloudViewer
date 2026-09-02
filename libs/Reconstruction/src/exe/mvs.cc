// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "exe/mvs.h"

#include "base/reconstruction.h"
#include "mvs/advancing_front_meshing.h"
#include "mvs/fusion.h"
#include "mvs/mesh_simplification.h"
#include "mvs/mesh_postprocessing.h"
#include "mvs/meshing.h"
#include "mvs/patch_match.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include "util/ply.h"

// CV_IO_LIB
#include <AutoIO.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

namespace colmap {

namespace {

bool ReadPlyMeshForSimplification(const std::string& path, PlyMesh* result) {
    CHECK_NOTNULL(result);
    ccMesh mesh;
    if (!mesh.CreateInternalCloud()) {
        return false;
    }
    cloudViewer::io::ReadTriangleMeshOptions options;
    options.print_progress = false;
    if (!cloudViewer::io::AutoReadMesh(path, mesh, options)) {
        return false;
    }

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    if (cloud == nullptr) {
        return false;
    }
    result->vertices.reserve(cloud->size());
    for (unsigned i = 0; i < cloud->size(); ++i) {
        const CCVector3* point = cloud->getPoint(i);
        CCVector3d global(point->x, point->y, point->z);
        if (cloud->isShifted()) {
            global = cloud->toGlobal3d(*point);
        }
        if (cloud->hasColors()) {
            const ecvColor::Rgb& color = cloud->getPointColor(i);
            result->vertices.emplace_back(
                    static_cast<float>(global.x), static_cast<float>(global.y),
                    static_cast<float>(global.z), color.r, color.g, color.b);
        } else {
            result->vertices.emplace_back(static_cast<float>(global.x),
                                          static_cast<float>(global.y),
                                          static_cast<float>(global.z));
        }
    }
    result->faces.reserve(mesh.size());
    for (unsigned i = 0; i < mesh.size(); ++i) {
        const cloudViewer::VerticesIndexes* triangle =
                mesh.getTriangleVertIndexes(i);
        result->faces.emplace_back(triangle->i1, triangle->i2, triangle->i3);
    }
    return true;
}

bool HasPlyExtension(const std::string& path) {
    std::string root;
    std::string extension;
    SplitFileExtension(path, &root, &extension);
    StringToLower(&extension);
    return extension == ".ply";
}

}  // namespace

int RunAdvancingFrontMesher(int argc, char** argv) {
#ifndef CGAL_ENABLED
    std::cerr << "ERROR: Advancing-front meshing requires CGAL, which is not "
                 "available on your system."
              << std::endl;
    return EXIT_FAILURE;
#else
    std::string input_path;
    std::string output_path;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path,
                              "Path to the dense workspace folder");
    options.AddRequiredOption("output_path", &output_path);
    options.AddAdvancingFrontMeshingOptions();
    options.AddMeshPostProcessingOptions();
    options.Parse(argc, argv);

    mvs::AdvancingFrontMeshing(*options.advancing_front_meshing, input_path,
                               output_path);
    if (options.mesh_post_processing->enabled &&
        !mvs::PostProcessMeshFile(output_path, output_path,
                                  *options.mesh_post_processing)) {
        LOG(ERROR) << "Failed to post-process output mesh: " << output_path;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
#endif
}

int RunDelaunayMesher(int argc, char** argv) {
#ifndef CGAL_ENABLED
    std::cerr << "ERROR: Delaunay meshing requires CGAL, which is not "
                 "available on your system."
              << std::endl;
    return EXIT_FAILURE;
#else   // CGAL_ENABLED
    std::string input_path;
    std::string input_type = "dense";
    std::string output_path;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path,
                              "Path to either the dense workspace folder or "
                              "the sparse reconstruction");
    options.AddDefaultOption("input_type", &input_type, "{dense, sparse}");
    options.AddRequiredOption("output_path", &output_path);
    options.AddDelaunayMeshingOptions();
    options.AddMeshPostProcessingOptions();
    options.Parse(argc, argv);

    StringToLower(&input_type);
    if (input_type == "sparse") {
        mvs::SparseDelaunayMeshing(*options.delaunay_meshing, input_path,
                                   output_path);
    } else if (input_type == "dense") {
        mvs::DenseDelaunayMeshing(*options.delaunay_meshing, input_path,
                                  output_path);
    } else {
        std::cout << "WARNING: Invalid input type - "
                     "supported values are 'sparse' and 'dense'."
                  << std::endl;
        return EXIT_FAILURE;
    }

    if (options.mesh_post_processing->enabled &&
        !mvs::PostProcessMeshFile(output_path, output_path,
                                  *options.mesh_post_processing)) {
        LOG(ERROR) << "Failed to post-process output mesh: " << output_path;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
#endif  // CGAL_ENABLED
}

int RunMeshSimplifier(int argc, char** argv) {
    std::string input_path;
    std::string output_path;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path,
                              "Path to input PLY mesh");
    options.AddRequiredOption("output_path", &output_path,
                              "Path to output PLY mesh");
    options.AddMeshSimplificationOptions();
    options.Parse(argc, argv);

    if (!ExistsFile(input_path) || !HasPlyExtension(input_path)) {
        LOG(ERROR) << "Input must be an existing PLY mesh: " << input_path;
        return EXIT_FAILURE;
    }
    if (!HasPlyExtension(output_path)) {
        LOG(ERROR) << "Output must use the .ply extension: " << output_path;
        return EXIT_FAILURE;
    }

    PlyMesh mesh;
    if (!ReadPlyMeshForSimplification(input_path, &mesh)) {
        LOG(ERROR) << "Failed to read input mesh: " << input_path;
        return EXIT_FAILURE;
    }
    LOG(INFO) << "Input mesh: " << mesh.vertices.size() << " vertices, "
              << mesh.faces.size() << " faces";
    const PlyMesh simplified =
            mvs::SimplifyMesh(mesh, *options.mesh_simplification);
    WriteBinaryPlyMesh(output_path, simplified);
    return EXIT_SUCCESS;
}

int RunPatchMatchStereo(int argc, char** argv) {
#ifndef CUDA_ENABLED
    std::cerr
            << "ERROR: Dense stereo reconstruction requires CUDA, which is not "
               "available on your system."
            << std::endl;
    return EXIT_FAILURE;
#else   // CUDA_ENABLED
    std::string workspace_path;
    std::string workspace_format = "COLMAP";
    std::string pmvs_option_name = "option-all";
    std::string config_path;

    OptionManager options;
    options.AddRequiredOption(
            "workspace_path", &workspace_path,
            "Path to the folder containing the undistorted images");
    options.AddDefaultOption("workspace_format", &workspace_format,
                             "{COLMAP, PMVS}");
    options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
    options.AddDefaultOption("config_path", &config_path);
    options.AddPatchMatchStereoOptions();
    options.Parse(argc, argv);

    StringToLower(&workspace_format);
    if (workspace_format != "colmap" && workspace_format != "pmvs") {
        std::cout
                << "WARNING: Invalid `workspace_format` - supported values are "
                   "'COLMAP' or 'PMVS'."
                << std::endl;
        return EXIT_FAILURE;
    }

    mvs::PatchMatchController controller(*options.patch_match_stereo,
                                         workspace_path, workspace_format,
                                         pmvs_option_name, config_path);

    controller.Start();
    controller.Wait();

    return EXIT_SUCCESS;
#endif  // CUDA_ENABLED
}

int RunPoissonMesher(int argc, char** argv) {
    std::string input_path;
    std::string output_path;

    OptionManager options;
    options.AddRequiredOption("input_path", &input_path);
    options.AddRequiredOption("output_path", &output_path);
    options.AddPoissonMeshingOptions();
    options.AddMeshPostProcessingOptions();
    options.Parse(argc, argv);

    CHECK(mvs::PoissonMeshing(*options.poisson_meshing, input_path,
                              output_path));
    if (options.mesh_post_processing->enabled &&
        !mvs::PostProcessMeshFile(output_path, output_path,
                                  *options.mesh_post_processing)) {
        LOG(ERROR) << "Failed to post-process output mesh: " << output_path;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RunStereoFuser(int argc, char** argv) {
    std::string workspace_path;
    std::string input_type = "geometric";
    std::string workspace_format = "COLMAP";
    std::string pmvs_option_name = "option-all";
    std::string output_type = "PLY";
    std::string output_path;
    std::string bbox_path;

    OptionManager options;
    options.AddRequiredOption("workspace_path", &workspace_path);
    options.AddDefaultOption("workspace_format", &workspace_format,
                             "{COLMAP, PMVS}");
    options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
    options.AddDefaultOption("input_type", &input_type,
                             "{photometric, geometric}");
    options.AddDefaultOption("output_type", &output_type, "{BIN, TXT, PLY}");
    options.AddRequiredOption("output_path", &output_path);
    options.AddDefaultOption("bbox_path", &bbox_path);
    options.AddStereoFusionOptions();
    options.Parse(argc, argv);

    StringToLower(&workspace_format);
    if (workspace_format != "colmap" && workspace_format != "pmvs") {
        std::cout
                << "WARNING: Invalid `workspace_format` - supported values are "
                   "'COLMAP' or 'PMVS'."
                << std::endl;
        return EXIT_FAILURE;
    }

    StringToLower(&input_type);
    if (input_type != "photometric" && input_type != "geometric") {
        std::cout << "WARNING: Invalid input type - supported values are "
                     "'photometric' and 'geometric'."
                  << std::endl;
        return EXIT_FAILURE;
    }

    if (!bbox_path.empty()) {
        std::ifstream file(bbox_path);
        if (file.is_open()) {
            auto& min_bound = options.stereo_fusion->bounding_box.first;
            auto& max_bound = options.stereo_fusion->bounding_box.second;
            file >> min_bound(0) >> min_bound(1) >> min_bound(2);
            file >> max_bound(0) >> max_bound(1) >> max_bound(2);
        } else {
            std::cout << "WARN: Invalid bounds path: \"" << bbox_path
                      << "\" - continuing without bounds check" << std::endl;
        }
    }

    mvs::StereoFusion fuser(*options.stereo_fusion, workspace_path,
                            workspace_format, pmvs_option_name, input_type);

    fuser.Start();
    fuser.Wait();

    Reconstruction reconstruction;

    // read data from sparse reconstruction
    if (workspace_format == "colmap") {
        reconstruction.Read(JoinPaths(workspace_path, "sparse"));
    }

    // overwrite sparse point cloud with dense point cloud from fuser
    reconstruction.ImportPLY(fuser.GetFusedPoints());

    std::cout << "Writing output: " << output_path << std::endl;

    // write output
    StringToLower(&output_type);
    if (output_type == "bin") {
        reconstruction.WriteBinary(output_path);
    } else if (output_type == "txt") {
        reconstruction.WriteText(output_path);
    } else if (output_type == "ply") {
        WriteBinaryPlyPoints(output_path, fuser.GetFusedPoints());
        mvs::WritePointsVisibility(output_path + ".vis",
                                   fuser.GetFusedPointsVisibility());
    } else {
        std::cerr << "ERROR: Invalid `output_type`" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

}  // namespace colmap
