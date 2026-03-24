// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <core/assets/ImageListFile.hpp>
#include <core/raycaster/CameraRaycaster.hpp>
#include <core/scene/BasicIBRScene.hpp>
#include <core/system/CommandLineArgs.hpp>
#include <core/system/Utils.hpp>
#include <fstream>
#include <iostream>

/*
generate clipping_planes.txt file
*/
static const char* TAG = "[clippingPlanes]";

using namespace sibr;

#ifdef SIBR_TOOL_EMBEDDED
extern "C" int sibr_tool_clippingPlanes(int argc, const char** argv) {
#else
int main(const int argc, const char** argv) {
#endif
    CommandLineArgs::parseMainArgs(argc, argv);

    BasicIBRAppArgs myArgs;

    if (argc > 1 && argv[1][0] != '-') {
        myArgs.dataset_path = std::string(argv[1]);
    }

    std::string datasetPath = myArgs.dataset_path.get();
    if (datasetPath.empty() || !directoryExists(datasetPath)) {
        std::cerr << TAG << " Usage: clippingPlanes --path <dataset-path>\n";
        return 1;
    }

    BasicIBRScene::SceneOptions opts;
    opts.renderTargets = false;
    opts.texture = false;
    opts.images = false;

    BasicIBRScene scene(myArgs, opts);

    auto inCams = scene.cameras()->inputCameras();
    if (inCams.empty()) {
        std::cerr << TAG << " No cameras loaded from " << datasetPath << "\n";
        return 1;
    }

    if (!scene.proxies()->hasProxy() ||
        scene.proxies()->proxy().vertices().empty()) {
        std::cerr << TAG << " No proxy mesh loaded from " << datasetPath
                  << "\n";
        return 1;
    }

    const std::string clipping_planes_file_path =
            datasetPath + "/clipping_planes.txt";
    if (!sibr::fileExists(clipping_planes_file_path)) {
        std::vector<sibr::Vector2f> nearsFars;
        CameraRaycaster::computeClippingPlanes(scene.proxies()->proxy(), inCams,
                                               nearsFars);

        std::ofstream file(clipping_planes_file_path,
                           std::ios::trunc | std::ios::out);
        if (file) {
            for (const auto& nearFar : nearsFars) {
                if (nearFar[0] > 0 && nearFar[1] > 0) {
                    file << nearFar[0] << ' ' << nearFar[1] << std::endl;
                } else {
                    file << "0.1 100.0" << std::endl;
                }
            }
            file.close();
        } else {
            SIBR_WRG << " Could not save file '" << clipping_planes_file_path
                     << "'." << std::endl;
        }
    } else {
        std::cout << TAG << " clipping_planes.txt already exists, skipping.\n";
    }

    std::cout << TAG << " done!\n";
    return 0;
}