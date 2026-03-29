// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <core/assets/ImageListFile.hpp>
#include <core/raycaster/CameraRaycaster.hpp>
#include <core/scene/BasicIBRScene.hpp>
#include <core/scene/CalibratedCameras.hpp>
#include <core/scene/ParseData.hpp>
#include <core/system/CommandLineArgs.hpp>
#include <core/system/Utils.hpp>
#include <fstream>
#include <iostream>

#define PROGRAM_NAME "prepareColmap4Sibr"
using namespace sibr;

static const char* usage =
        ""
        "Usage: " PROGRAM_NAME
        " -path "
        "\n";

struct ColmapPreprocessArgs : public BasicIBRAppArgs {
    Arg<bool> fix_metadata = {"fix_metadata",
                              "Fix scene_metadata after crop and distort "};
};

#ifdef SIBR_TOOL_EMBEDDED
extern "C" int sibr_tool_prepareColmap4Sibr(int argc, const char** argv) {
#else
int main(const int argc, const char** argv) {
#endif
    CommandLineArgs::parseMainArgs(argc, argv);
    BasicIBRAppArgs myArgs;

    bool fix_metadata = CommandLineArgs::getGlobal().contains("fix_metadata");
    std::string pathScene = myArgs.dataset_path;

    std::vector<std::string> dirs = {"sfm_mvs_cm", "sibr_cm"};

    std::ofstream outputSceneMetadata;

    if (fix_metadata) {
        std::string cm_path = myArgs.dataset_path.get() + "/sibr_cm";
        myArgs.dataset_path = cm_path;

        BasicIBRScene::SceneOptions opts;
        opts.renderTargets = false;
        opts.mesh = false;
        BasicIBRScene cm_scene(myArgs, opts);

        std::vector<InputCamera::Ptr> cams = cm_scene.cameras()->inputCameras();

        std::string tmpFileName = cm_path + "/scene_metadata_tmp.txt";
        // done in a second pass, when everything has been created.
        outputSceneMetadata.open(tmpFileName);

        // overwrite previous version since image sizes have changed when
        // running sibr preprocessing
        outputSceneMetadata << "Scene Metadata File\n" << std::endl;

        if (outputSceneMetadata.bad())
            SIBR_ERR << "Problem writing new metadata file" << std::endl;

        SIBR_LOG << "Writing new scene_metadata.txt file "
                 << cm_path + "/scene_metadata.txt" << std::endl;

        outputSceneMetadata
                << "[list_images]\n<filename> <image_width> <image_height> "
                   "<near_clipping_plane> <far_clipping_plane>"
                << std::endl;

        for (int c = 0; c < cams.size(); c++) {
            InputCamera& camIm = *cams[c];

            std::string extensionFile =
                    boost::filesystem::extension(camIm.name());
            std::ostringstream ssZeroPad;
            ssZeroPad << std::setw(8) << std::setfill('0') << camIm.id();
            std::string newFileName = ssZeroPad.str() + extensionFile;
            // load image
            std::string imgpath = cm_path + "/images/" + camIm.name();
            sibr::ImageRGB im;
            if (!im.load(imgpath, false))
                SIBR_ERR << "Cant open image " << imgpath << std::endl;

            std::cerr << newFileName << " " << im.w() << " " << im.h() << " "
                      << camIm.znear() << " " << camIm.zfar() << std::endl;
            outputSceneMetadata << newFileName << " " << im.w() << " " << im.h()
                                << " " << camIm.znear() << " " << camIm.zfar()
                                << std::endl;
        }

        outputSceneMetadata
                << "\n// Always specify active/exclude images after list "
                   "images\n\n[exclude_images]\n<image1_idx> <image2_idx> ... "
                   "<image3_idx>"
                << std::endl;

        for (int i = 0; i < cm_scene.data()->activeImages().size(); i++) {
            if (!cm_scene.data()->activeImages()[i])
                outputSceneMetadata << i << " ";
        }
        outputSceneMetadata << "\n\n\n[other parameters]" << std::endl;
        outputSceneMetadata.close();

        std::string SMName = cm_path + "/scene_metadata.txt";

        SIBR_LOG << "Copying " << tmpFileName << " to " << SMName << std::endl;
        boost::filesystem::copy_file(
                tmpFileName, SMName,
                boost::filesystem::copy_option::overwrite_if_exists);
        boost::filesystem::remove(tmpFileName);

        return EXIT_SUCCESS;
    }
    std::cout << "[prepareColmap4Sibr] Creating bundle file for SIBR scene."
              << std::endl;
    std::cout << "[prepareColmap4Sibr] dataset_path = " << pathScene
              << std::endl;
    std::cerr << "[prepareColmap4Sibr] Loading scene..." << std::endl;
    std::cerr.flush();

    BasicIBRScene::SceneOptions sceneOpts;
    sceneOpts.renderTargets = false;
    sceneOpts.mesh = false;
    sceneOpts.texture = false;
    sceneOpts.images = false;
    BasicIBRScene scene(myArgs, sceneOpts);

    std::cerr << "[prepareColmap4Sibr] Scene constructed." << std::endl;
    std::cerr.flush();

    std::vector<InputCamera::Ptr> cams = scene.cameras()->inputCameras();
    const int maxCam = int(cams.size());
    const int minCam = 0;

    std::cout << "[prepareColmap4Sibr] Loaded " << maxCam << " cameras."
              << std::endl;

    for (auto dir : dirs) {
        std::cout << "[prepareColmap4Sibr] Creating dir: " << dir << std::endl;
        if (!directoryExists(pathScene + "/" + dir.c_str())) {
            makeDirectory(pathScene + "/" + dir.c_str());
        }
    }

    std::ofstream outputBundleCam;
    std::ofstream outputListIm;

    outputBundleCam.open(pathScene + "/sfm_mvs_cm/bundle.out");
    outputListIm.open(pathScene + "/sfm_mvs_cm/list_images.txt");
    outputBundleCam << "# Bundle file v0.3" << std::endl;
    outputBundleCam << maxCam << " " << 0 << std::endl;

    outputSceneMetadata.open(pathScene + "/sibr_cm/scene_metadata.txt");
    outputSceneMetadata << "Scene Metadata File\n" << std::endl;
    outputSceneMetadata
            << "[list_images]\n<filename> <image_width> <image_height> "
               "<near_clipping_plane> <far_clipping_plane>"
            << std::endl;

    std::sort(cams.begin(), cams.end(),
              [](const InputCamera::Ptr& a, const InputCamera::Ptr& b) {
                  return a->id() < b->id();
              });

    for (int c = minCam; c < maxCam; c++) {
        InputCamera& camIm = *cams[c];

        std::string extensionFile = boost::filesystem::extension(camIm.name());
        std::ostringstream ssZeroPad;
        ssZeroPad << std::setw(8) << std::setfill('0') << camIm.id();
        std::string newFileName = ssZeroPad.str() + extensionFile;

        std::string srcImage =
                pathScene + "/colmap/stereo/images/" + camIm.name();
        if (!boost::filesystem::exists(srcImage)) {
            srcImage = pathScene + "/images/" + camIm.name();
        }
        if (boost::filesystem::exists(srcImage)) {
            boost::filesystem::copy_file(
                    srcImage, pathScene + "/sfm_mvs_cm/" + newFileName,
                    boost::filesystem::copy_option::overwrite_if_exists);
        } else {
            std::cerr << "[prepareColmap4Sibr] Warning: image not found: "
                      << camIm.name() << std::endl;
        }
        outputBundleCam << camIm.toBundleString(false, true);
        outputListIm << newFileName << " " << camIm.w() << " " << camIm.h()
                     << std::endl;
        outputSceneMetadata << newFileName << " " << camIm.w() << " "
                            << camIm.h() << " " << camIm.znear() << " "
                            << camIm.zfar() << std::endl;
    }

    outputSceneMetadata << "\n// Always specify active/exclude images after "
                           "list images\n\n[exclude_images]\n<image1_idx> "
                           "<image2_idx> ... <image3_idx>"
                        << std::endl;

    for (int i = 0; i < scene.data()->activeImages().size(); i++) {
        if (!scene.data()->activeImages()[i]) outputSceneMetadata << i << " ";
    }
    outputSceneMetadata << "\n\n\n[other parameters]" << std::endl;

    outputBundleCam.close();
    outputListIm.close();
    outputSceneMetadata.close();

    std::vector<std::vector<std::string>> meshPathList = {
            {"/meshed-delaunay.ply", "/sfm_mvs_cm/recon.ply"},
            {"/meshed-poisson.ply", "/sfm_mvs_cm/recon.ply"},
            {"/mesh.ply", "/sfm_mvs_cm/recon.ply"},
            {"/mesh.obj", "/sfm_mvs_cm/recon.ply"},
            {"/fused.ply", "/sfm_mvs_cm/recon.ply"},
            {"/capreal/mesh.ply", "/sfm_mvs_cm/recon.ply"},
            {"/capreal/mesh.obj", "/sfm_mvs_cm/recon.ply"},
            {"/capreal/mesh.mtl", "/sfm_mvs_cm/"},
            {"/capreal/texture.png", "/sfm_mvs_cm/"},
            {"/capreal/mesh_u1_v1.png", "/sfm_mvs_cm/"},
            {"/colmap/stereo/meshed-delaunay.ply", "/sfm_mvs_cm/recon.ply"},
    };

    bool success = false;
    for (const std::vector<std::string>& meshPaths : meshPathList) {
        if (boost::filesystem::exists(pathScene + meshPaths[0])) {
            sibr::copyFile(pathScene + meshPaths[0], pathScene + meshPaths[1],
                           true);
            std::cout << "[prepareColmap4Sibr] Copied " << meshPaths[0]
                      << " -> " << meshPaths[1] << std::endl;
            success = true;
        }
    }
    if (!success) {
        std::cerr << "[prepareColmap4Sibr] Warning: no proxy geometry found "
                     "in any of these locations:"
                  << std::endl;
        for (const std::vector<std::string>& meshPaths : meshPathList)
            std::cerr << "  " << pathScene + meshPaths[0] << std::endl;
        std::cerr << "[prepareColmap4Sibr] Camera data was still exported "
                     "successfully."
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
