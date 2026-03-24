// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/assets/InputCamera.hpp"
#include "core/assets/UVUnwrapper.hpp"
#include "core/graphics/Image.hpp"
#include "core/graphics/Mesh.hpp"
#include "core/imgproc/MeshTexturing.hpp"
#include "core/scene/BasicIBRScene.hpp"
#include "core/system/CommandLineArgs.hpp"

using namespace sibr;

struct TexturingAppArgs : virtual BasicIBRAppArgs {
    Arg<std::string> meshPath = {"mesh", ""};
    RequiredArg<std::string> output_path = {"output", "output texture path"};
    Arg<int> output_size = {"size", 8192, "texture side"};
    Arg<bool> flood_fill = {"flood", "perform flood fill"};
    Arg<bool> poisson_fill = {"poisson",
                              "perform Poisson filling (slow on large images)"};
    Arg<float> samples = {"samples", 1.0,
                          "%ge of total samples to be used for texturing"};
};

#ifdef SIBR_TOOL_EMBEDDED
extern "C" int sibr_tool_textureMesh(int ac, char** av) {
#else
int main(int ac, char** av) {
#endif
    sibr::CommandLineArgs::parseMainArgs(ac, av);

    TexturingAppArgs args;

    // Display help.
    if (!args.dataset_path.isInit() || !args.output_path.isInit()) {
        std::cout << "Usage: " << std::endl;
        std::cout << "\tRequired: --path path/to/dataset --output "
                     "path/to/output/file.png"
                  << std::endl;
        std::cout << "\tOptional: --size 8192 --flood (flood fill) --poisson "
                     "(poisson fill)"
                  << std::endl;
        return 0;
    }

    BasicIBRScene::SceneOptions opts;
    opts.renderTargets = false;
    if (!args.meshPath.get().empty()) {
        opts.mesh = false;
    }
    opts.texture = false;

    SIBR_LOG << "[Texturing] Loading data..." << std::endl;

    BasicIBRScene scene(args, opts);

    if (!scene.proxies()->hasProxy()) {
        sibr::Mesh::Ptr customMesh;
        customMesh.reset(new Mesh());
        customMesh->load(args.meshPath);
        scene.proxies()->replaceProxyPtr(customMesh);
    }

    auto meshPtr = scene.proxies()->proxyPtr();

    if (meshPtr->triangles().empty()) {
        SIBR_LOG << "[Texturing] Loaded proxy has "
                 << meshPtr->vertices().size()
                 << " vertices but 0 triangles (SfM sparse points). "
                    "Searching for a triangulated mesh..."
                 << std::endl;

        std::string dp = args.dataset_path.get();
        const std::vector<std::string> candidates = {
                dp + "/meshed-delaunay.ply",
                dp + "/meshed-poisson.ply",
                dp + "/mesh.ply",
                dp + "/mesh.obj",
                dp + "/recon.ply",
                dp + "/sfm_mvs_cm/recon.ply",
                dp + "/colmap/stereo/meshed-delaunay.ply",
                dp + "/capreal/mesh.ply",
                dp + "/capreal/mesh.obj",
        };
        bool found = false;
        for (const auto& c : candidates) {
            if (fileExists(c)) {
                Mesh::Ptr realMesh(new Mesh());
                realMesh->load(c);
                if (!realMesh->triangles().empty()) {
                    SIBR_LOG << "[Texturing] Using mesh: " << c << " ("
                             << realMesh->vertices().size() << " verts, "
                             << realMesh->triangles().size() << " tris)"
                             << std::endl;
                    scene.proxies()->replaceProxyPtr(realMesh);
                    meshPtr = realMesh;
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            SIBR_ERR << "[Texturing] No triangulated mesh found in dataset. "
                        "Use --mesh <path> to specify one."
                     << std::endl;
        }
    }

    if (!meshPtr->hasTexCoords()) {
        SIBR_LOG << "[Texturing] Mesh has no UVs, running UV unwrap first "
                    "(target size "
                 << args.output_size.get() << ")..." << std::endl;
        UVUnwrapper unwrapper(*meshPtr, args.output_size);
        Mesh::Ptr unwrapped = unwrapper.unwrap();
        scene.proxies()->replaceProxyPtr(unwrapped);
        meshPtr = unwrapped;
        SIBR_LOG << "[Texturing] UV unwrap complete ("
                 << meshPtr->vertices().size() << " verts)." << std::endl;
    }

    MeshTexturing texturer(args.output_size);
    texturer.setMesh(meshPtr);
    texturer.reproject(scene.cameras()->inputCameras(),
                       scene.images()->inputImages(), args.samples);

    // Export options.
    // UVs start at the bottom of the image, we have to flip.
    uint options = MeshTexturing::FLIP_VERTICAL;
    if (args.flood_fill) {
        options = options | MeshTexturing::FLOOD_FILL;
    }
    if (args.poisson_fill) {
        options = options | MeshTexturing::POISSON_FILL;
    }

    sibr::ImageRGB::Ptr result = texturer.getTexture(options);
    result->save(args.output_path);

    return 0;
}
