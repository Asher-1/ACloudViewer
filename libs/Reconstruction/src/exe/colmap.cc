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

#include <QGuiApplication>
#include <memory>

#include "exe/database.h"
#include "exe/feature.h"
#include "exe/gui.h"
#include "exe/image.h"
#include "exe/model.h"
#include "exe/mvs.h"
#include "exe/sfm.h"
#include "exe/vocab_tree.h"
#include "util/cancellation.h"
#include "util/version.h"

namespace {

using command_func_t = std::function<int(int, char**)>;

// Commands that stream results to disk while running mark themselves with
// kSupportsGracefulShutdown: a first SIGINT/SIGTERM then requests a
// cooperative stop (through colmap::ScopedSignalHandler) so that partial
// results are finalized and flushed, and a second signal terminates the
// process immediately (upstream dbb41680 parity).
constexpr bool kSupportsGracefulShutdown = true;

struct Command {
    Command(std::string name,
            command_func_t function,
            const bool supports_graceful_shutdown = false)
        : name(std::move(name)),
          function(std::move(function)),
          supports_graceful_shutdown(supports_graceful_shutdown) {}

    std::string name;
    command_func_t function;
    bool supports_graceful_shutdown;
};

int ShowHelp(const std::vector<Command>& commands) {
    std::cout << colmap::StringPrintf(
                         "%s -- Structure-from-Motion and Multi-View Stereo\n"
                         "              (%s)",
                         colmap::GetVersionInfo().c_str(),
                         colmap::GetBuildInfo().c_str())
              << std::endl
              << std::endl;

    std::cout << "Usage:" << std::endl;
    std::cout << "  colmap [command] [options]" << std::endl << std::endl;

    std::cout << "Documentation:" << std::endl;
    std::cout << "  https://colmap.github.io/" << std::endl << std::endl;

    std::cout << "Example usage:" << std::endl;
    std::cout << "  colmap help [ -h, --help ]" << std::endl;
    std::cout << "  colmap gui" << std::endl;
    std::cout << "  colmap gui -h [ --help ]" << std::endl;
    std::cout << "  colmap automatic_reconstructor -h [ --help ]" << std::endl;
    std::cout << "  colmap automatic_reconstructor --image_path IMAGES "
                 "--workspace_path WORKSPACE"
              << std::endl;
    std::cout
            << "  colmap feature_extractor --image_path IMAGES --database_path "
               "DATABASE"
            << std::endl;
    std::cout << "  colmap exhaustive_matcher --database_path DATABASE"
              << std::endl;
    std::cout << "  colmap mapper --image_path IMAGES --database_path DATABASE "
                 "--output_path MODEL"
              << std::endl;
    std::cout << "  ..." << std::endl << std::endl;

    std::cout << "Available commands:" << std::endl;
    std::cout << "  help" << std::endl;
    for (const auto& command : commands) {
        std::cout << "  " << command.name << std::endl;
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace colmap;

    InitializeGlog(argv);
#ifdef GUI_ENABLED
    Q_INIT_RESOURCE(resources);
#endif

    std::vector<Command> commands;
    commands.emplace_back("gui", &RunGraphicalUserInterface);
    commands.emplace_back("automatic_reconstructor",
                          &RunAutomaticReconstructor);
    commands.emplace_back("advancing_front_mesher", &RunAdvancingFrontMesher);
    commands.emplace_back("bundle_adjuster", &RunBundleAdjuster,
                          kSupportsGracefulShutdown);
    commands.emplace_back("color_extractor", &RunColorExtractor);
    commands.emplace_back("database_cleaner", &RunDatabaseCleaner);
    commands.emplace_back("database_creator", &RunDatabaseCreator);
    commands.emplace_back("database_merger", &RunDatabaseMerger);
    commands.emplace_back("rig_configurator", &RunRigConfigurator);
    commands.emplace_back("delaunay_mesher", &RunDelaunayMesher);
    commands.emplace_back("exhaustive_matcher", &RunExhaustiveMatcher,
                          kSupportsGracefulShutdown);
    commands.emplace_back("feature_extractor", &RunFeatureExtractor,
                          kSupportsGracefulShutdown);
    commands.emplace_back("feature_importer", &RunFeatureImporter,
                          kSupportsGracefulShutdown);
    commands.emplace_back("hierarchical_mapper", &RunHierarchicalMapper);
    commands.emplace_back("global_mapper", &RunGlobalMapper);
    commands.emplace_back("rotation_averager", &RunRotationAverager);
    commands.emplace_back("view_graph_calibrator", &RunViewGraphCalibrator);
    commands.emplace_back("image_deleter", &RunImageDeleter);
    commands.emplace_back("image_filterer", &RunImageFilterer);
    commands.emplace_back("image_rectifier", &RunImageRectifier);
    commands.emplace_back("image_registrator", &RunImageRegistrator,
                          kSupportsGracefulShutdown);
    commands.emplace_back("image_texturer", &RunImageTexturer);
    commands.emplace_back("image_undistorter", &RunImageUndistorter,
                          kSupportsGracefulShutdown);
    commands.emplace_back("image_undistorter_standalone",
                          &RunImageUndistorterStandalone,
                          kSupportsGracefulShutdown);
    commands.emplace_back("mesh_texturer", &RunMeshTexturer);
    commands.emplace_back("mesh_simplifier", &RunMeshSimplifier);
    commands.emplace_back("pose_prior_mapper", &RunPosePriorMapper);
    commands.emplace_back("mapper", &RunMapper,
                          kSupportsGracefulShutdown);
    commands.emplace_back("matches_importer", &RunMatchesImporter,
                          kSupportsGracefulShutdown);
    commands.emplace_back("model_aligner", &RunModelAligner);
    commands.emplace_back("model_clusterer", &RunModelClusterer);
    commands.emplace_back("model_analyzer", &RunModelAnalyzer);
    commands.emplace_back("model_comparer", &RunModelComparer);
    commands.emplace_back("model_converter", &RunModelConverter);
    commands.emplace_back("model_cropper", &RunModelCropper);
    commands.emplace_back("model_merger", &RunModelMerger);
    commands.emplace_back("model_orientation_aligner",
                          &RunModelOrientationAligner);
    commands.emplace_back("model_splitter", &RunModelSplitter);
    commands.emplace_back("model_transformer", &RunModelTransformer);
    commands.emplace_back("patch_match_stereo", &RunPatchMatchStereo,
                          kSupportsGracefulShutdown);
    commands.emplace_back("point_filtering", &RunPointFiltering);
    commands.emplace_back("point_triangulator", &RunPointTriangulator,
                          kSupportsGracefulShutdown);
    commands.emplace_back("poisson_mesher", &RunPoissonMesher);
    commands.emplace_back("project_generator", &RunProjectGenerator);
    commands.emplace_back("sequential_matcher", &RunSequentialMatcher,
                          kSupportsGracefulShutdown);
    commands.emplace_back("spatial_matcher", &RunSpatialMatcher,
                          kSupportsGracefulShutdown);
    commands.emplace_back("stereo_fusion", &RunStereoFuser,
                          kSupportsGracefulShutdown);
    commands.emplace_back("transitive_matcher", &RunTransitiveMatcher,
                          kSupportsGracefulShutdown);
    commands.emplace_back("vocab_tree_builder", &RunVocabTreeBuilder);
    commands.emplace_back("vocab_tree_matcher", &RunVocabTreeMatcher,
                          kSupportsGracefulShutdown);
    commands.emplace_back("vocab_tree_retriever", &RunVocabTreeRetriever);

    if (argc == 1) {
#ifndef Q_OS_MAC
        return ShowHelp(commands);
#endif
    }
    // On macOS with argc==1 (Finder double-click), default to "gui".
    // On all platforms with argc>1, use the provided command.
    const std::string command = (argc > 1) ? argv[1] : "gui";
    if (command == "help" || command == "-h" || command == "--help") {
        return ShowHelp(commands);
    } else {
        const Command* matched_command = nullptr;
        for (const auto& command_entry : commands) {
            if (command == command_entry.name) {
                matched_command = &command_entry;
                break;
            }
        }
        if (matched_command == nullptr) {
            std::cerr << StringPrintf(
                                 "ERROR: Command `%s` not recognized. To list "
                                 "the "
                                 "available commands, run `colmap help`.",
                                 command.c_str())
                      << std::endl;
            return EXIT_FAILURE;
        } else {
            int command_argc = argc - 1;
            char** command_argv = &argv[1];
            command_argv[0] = argv[0];

            // The "gui" command creates its own QApplication; for all other
            // commands we need a lightweight QGuiApplication so that the
            // texturing pipeline can use QImageReader / QPainter on QImage.
            std::unique_ptr<QGuiApplication> headless_app;
            if (command != "gui") {
                qputenv("QT_QPA_PLATFORM", "offscreen");
                headless_app.reset(new QGuiApplication(argc, argv));
            }

            std::unique_ptr<colmap::ScopedSignalHandler> signal_handler;
            if (matched_command->supports_graceful_shutdown) {
                signal_handler =
                        std::make_unique<colmap::ScopedSignalHandler>();
            }

            const int exit_code =
                    matched_command->function(command_argc, command_argv);
            if (signal_handler && signal_handler->ReceivedSignal() != 0) {
                LOG(INFO) << "Graceful shutdown completed after receiving "
                             "signal "
                          << signal_handler->ReceivedSignal();
                return signal_handler->GetExitCode();
            }
            return exit_code;
        }
    }

    return ShowHelp(commands);
}
