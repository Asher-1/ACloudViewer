// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <core/assets/UVUnwrapper.hpp>
#include <core/graphics/Mesh.hpp>
#include <core/system/CommandLineArgs.hpp>
#include <core/system/Config.hpp>

using namespace sibr;

/** Options for mesh unwrapping. */
struct UVMapperArgs : public AppArgs {
    RequiredArg<std::string> path = {"path", "path to the mesh"};
    Arg<std::string> output = {"output", "", "path to the output mesh"};
    Arg<int> size = {"size", 4096, "target UV map width (approx.)"};
    Arg<bool> visu = {"visu", "save visualisation"};
    Arg<std::string> textureName = {"texture-name",
                                    "TEXTURE_NAME_TO_PUT_IN_THE_FILE",
                                    "name of the texture to reference in the "
                                    "output mesh (Meshlab compatible)"};
};

#ifdef SIBR_TOOL_EMBEDDED
int sibr_tool_unwrapMesh(int ac, char** av) {
#else
int main(int ac, char** av) {
#endif
    CommandLineArgs::parseMainArgs(ac, av);
    UVMapperArgs args;
    std::string outputFile = args.output;
    if (outputFile.empty()) {
        outputFile = sibr::removeExtension(args.path.get()) + "_output.obj";
    }
    sibr::makeDirectory(sibr::parentDirectory(outputFile));

    // Load object file.
    Mesh mesh(false);
    if (sibr::getExtension(args.path) == "xml") {
        mesh.loadMtsXML(args.path);
    } else {
        mesh.load(args.path);
    }

    UVUnwrapper unwrapper(mesh, uint32_t(args.size));
    auto finalMesh = unwrapper.unwrap();
    finalMesh->save(outputFile, true, args.textureName);

    // Output debug vis.
    if (args.visu) {
        const std::string baseName = sibr::removeExtension(outputFile);
        const auto visuImgs = unwrapper.atlasVisualization();
        for (uint32_t i = 0; i < visuImgs.size(); i++) {
            const std::string fileName =
                    baseName + "_charts_atlas_" + std::to_string(i) + ".png";
            visuImgs[i]->save(fileName);
        }
    }
    return EXIT_SUCCESS;
}
