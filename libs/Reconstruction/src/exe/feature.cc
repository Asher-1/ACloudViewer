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

#include "exe/feature.h"

#include "base/camera_models.h"
#include "base/image_reader.h"
#include "exe/gui.h"
#include "feature/extraction.h"
#include "feature/loma.h"
#include "feature/matching.h"
#include "util/misc.h"
#include "util/opengl_utils.h"
#include "util/option_manager.h"

namespace colmap {
namespace {

bool VerifyCameraParams(const std::string& camera_model,
                        const std::string& params) {
    if (!ExistsCameraModelWithName(camera_model)) {
        std::cerr << "ERROR: Camera model does not exist" << std::endl;
        return false;
    }

    const std::vector<double> camera_params = CSVToVector<double>(params);
    const int camera_model_id = CameraModelNameToId(camera_model);

    if (camera_params.size() > 0 &&
        !CameraModelVerifyParams(camera_model_id, camera_params)) {
        std::cerr << "ERROR: Invalid camera parameters" << std::endl;
        return false;
    }
    return true;
}

bool VerifySiftGPUParams(const bool use_gpu) {
#if !defined(CUDA_ENABLED) && !defined(OPENGL_ENABLED)
    if (use_gpu) {
        std::cerr
                << "ERROR: Cannot use Sift GPU without CUDA or OpenGL support; "
                   "set SiftExtraction.use_gpu or SiftMatching.use_gpu to "
                   "false."
                << std::endl;
        return false;
    }
#endif
    return true;
}

bool UsesLomaMatching(const SiftMatchingOptions& options) {
    return options.use_loma || !options.loma_matcher_model_path.empty();
}

// This enum can be used as optional input for feature_extractor and
// feature_importer to ensure that the camera flags of ImageReader are set in an
// exclusive and unambigous way. The table below explains the corespondence of
// each setting with the flags
//
// -----------------------------------------------------------------------------------
// |            |                         ImageReaderOptions | | CameraMode |
// single_camera | single_camera_per_folder | single_camera_per_image |
// |------------|---------------|--------------------------|-------------------------|
// | AUTO       | false         | false                    | false | | SINGLE |
// true          | false                    | false                   | |
// PER_FOLDER | false         | true                     | false | | PER_IMAGE
// | false         | false                    | true                    |
// -----------------------------------------------------------------------------------
//
// Note: When using AUTO mode a camera model will be uniquely identified by the
// following 5 parameters from EXIF tags:
// 1. Camera Make
// 2. Camera Model
// 3. Focal Length
// 4. Image Width
// 5. Image Height
//
// If any of the tags is missing then a camera model is considered invalid and a
// new camera is created similar to the PER_IMAGE mode.
//
// If these considered fields are not sufficient to uniquely identify a camera
// then using the AUTO mode will lead to incorrect setup for the cameras, e.g.
// the same camera is used with same focal length but different principal point
// between captures. In these cases it is recommended to either use the
// PER_FOLDER or PER_IMAGE settings.
enum class CameraMode { AUTO = 0, SINGLE = 1, PER_FOLDER = 2, PER_IMAGE = 3 };

void UpdateImageReaderOptionsFromCameraMode(ImageReaderOptions& options,
                                            CameraMode mode) {
    switch (mode) {
        case CameraMode::AUTO:
            options.single_camera = false;
            options.single_camera_per_folder = false;
            options.single_camera_per_image = false;
            break;
        case CameraMode::SINGLE:
            options.single_camera = true;
            options.single_camera_per_folder = false;
            options.single_camera_per_image = false;
            break;
        case CameraMode::PER_FOLDER:
            options.single_camera = false;
            options.single_camera_per_folder = true;
            options.single_camera_per_image = false;
            break;
        case CameraMode::PER_IMAGE:
            options.single_camera = false;
            options.single_camera_per_folder = false;
            options.single_camera_per_image = true;
            break;
    }
}

}  // namespace

int RunFeatureExtractor(int argc, char** argv) {
    std::string image_list_path;
    std::string feature_type = "sift";
    std::string loma_detector_model;
    std::string loma_descriptor_model;
    std::string loma_device = "auto";
    int camera_mode = -1;

    OptionManager options;
    options.AddDatabaseOptions();
    options.AddImageOptions();
    options.AddDefaultOption("camera_mode", &camera_mode);
    options.AddDefaultOption("image_list_path", &image_list_path);
    options.AddDefaultOption("feature_type", &feature_type,
                             "sift, loma_b, or loma_b128");
    options.AddDefaultOption("loma_detector_model", &loma_detector_model);
    options.AddDefaultOption("loma_descriptor_model", &loma_descriptor_model);
    options.AddDefaultOption("loma_device", &loma_device);
    options.AddExtractionOptions();
    options.Parse(argc, argv);

    ImageReaderOptions reader_options = *options.image_reader;
    reader_options.database_path = *options.database_path;
    reader_options.image_path = *options.image_path;

    if (camera_mode >= 0) {
        UpdateImageReaderOptionsFromCameraMode(reader_options,
                                               (CameraMode)camera_mode);
    }

    if (!image_list_path.empty()) {
        reader_options.image_list = ReadTextFileLines(image_list_path);
        if (reader_options.image_list.empty()) {
            return EXIT_SUCCESS;
        }
    }

    if (!ExistsCameraModelWithName(reader_options.camera_model)) {
        std::cerr << "ERROR: Camera model does not exist" << std::endl;
    }

    if (!VerifyCameraParams(reader_options.camera_model,
                            reader_options.camera_params)) {
        return EXIT_FAILURE;
    }

    if (feature_type != "sift" && feature_type != "loma_b" &&
        feature_type != "loma_b128") {
        std::cerr << "ERROR: feature_type must be sift, loma_b, or loma_b128"
                  << std::endl;
        return EXIT_FAILURE;
    }
    if (feature_type == "sift" &&
        !VerifySiftGPUParams(options.sift_extraction->use_gpu)) {
        return EXIT_FAILURE;
    }

    const bool use_loma =
            feature_type != "sift" || options.sift_extraction->use_loma ||
            !loma_detector_model.empty() || !loma_descriptor_model.empty() ||
            !options.sift_extraction->loma_detector_model_path.empty() ||
            !options.sift_extraction->loma_descriptor_model_path.empty();
    if (use_loma) {
        if (feature_type == "sift") feature_type = "loma_b";
        LomaExtractionOptions loma_options;
        loma_options.detector_model_path =
                loma_detector_model.empty()
                        ? options.sift_extraction->loma_detector_model_path
                        : loma_detector_model;
        loma_options.descriptor_model_path =
                loma_descriptor_model.empty()
                        ? options.sift_extraction->loma_descriptor_model_path
                        : loma_descriptor_model;
        loma_options.device = loma_device == "auto"
                                      ? options.sift_extraction->loma_device
                                      : loma_device;
        loma_options.max_num_features =
                options.sift_extraction->max_num_features;
        loma_options.descriptor_type =
                feature_type == "loma_b" ? FeatureDescriptorType::kLomaG
                                         : FeatureDescriptorType::kLomaB128;
        LomaFeatureExtractor feature_extractor(reader_options, loma_options);
        feature_extractor.Start();
        feature_extractor.Wait();
        return feature_extractor.Succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::unique_ptr<QApplication> app;
    if (options.sift_extraction->use_gpu && kUseOpenGL) {
        app.reset(new QApplication(argc, argv));
    }

    SiftFeatureExtractor feature_extractor(reader_options,
                                           *options.sift_extraction);

    if (options.sift_extraction->use_gpu && kUseOpenGL) {
        RunThreadWithOpenGLContext(&feature_extractor);
    } else {
        feature_extractor.Start();
        feature_extractor.Wait();
    }

    return EXIT_SUCCESS;
}

int RunFeatureImporter(int argc, char** argv) {
    std::string import_path;
    std::string image_list_path;
    int camera_mode = -1;

    OptionManager options;
    options.AddDatabaseOptions();
    options.AddImageOptions();
    options.AddDefaultOption("camera_mode", &camera_mode);
    options.AddRequiredOption("import_path", &import_path);
    options.AddDefaultOption("image_list_path", &image_list_path);
    options.AddExtractionOptions();
    options.Parse(argc, argv);

    ImageReaderOptions reader_options = *options.image_reader;
    reader_options.database_path = *options.database_path;
    reader_options.image_path = *options.image_path;

    if (camera_mode >= 0) {
        UpdateImageReaderOptionsFromCameraMode(reader_options,
                                               (CameraMode)camera_mode);
    }

    if (!image_list_path.empty()) {
        reader_options.image_list = ReadTextFileLines(image_list_path);
        if (reader_options.image_list.empty()) {
            return EXIT_SUCCESS;
        }
    }

    if (!VerifyCameraParams(reader_options.camera_model,
                            reader_options.camera_params)) {
        return EXIT_FAILURE;
    }

    FeatureImporter feature_importer(reader_options, import_path);
    feature_importer.Start();
    feature_importer.Wait();

    return EXIT_SUCCESS;
}

int RunMatchesImporter(int argc, char** argv) {
    std::string match_list_path;
    std::string match_type = "pairs";

    OptionManager options;
    options.AddDatabaseOptions();
    options.AddRequiredOption("match_list_path", &match_list_path);
    options.AddDefaultOption("match_type", &match_type,
                             "{'pairs', 'raw', 'inliers'}");
    options.AddMatchingOptions();
    options.Parse(argc, argv);

    if (!UsesLomaMatching(*options.sift_matching) &&
        !VerifySiftGPUParams(options.sift_matching->use_gpu)) {
        return EXIT_FAILURE;
    }

    std::unique_ptr<QApplication> app;
    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        app.reset(new QApplication(argc, argv));
    }

    std::unique_ptr<Thread> feature_matcher;
    if (match_type == "pairs") {
        ImagePairsMatchingOptions matcher_options;
        matcher_options.match_list_path = match_list_path;
        feature_matcher.reset(new ImagePairsFeatureMatcher(
                matcher_options, *options.sift_matching,
                *options.database_path));
    } else if (match_type == "raw" || match_type == "inliers") {
        FeaturePairsMatchingOptions matcher_options;
        matcher_options.match_list_path = match_list_path;
        matcher_options.verify_matches = match_type == "raw";
        feature_matcher.reset(new FeaturePairsFeatureMatcher(
                matcher_options, *options.sift_matching,
                *options.database_path));
    } else {
        std::cerr << "ERROR: Invalid `match_type`";
        return EXIT_FAILURE;
    }

    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        RunThreadWithOpenGLContext(feature_matcher.get());
    } else {
        feature_matcher->Start();
        feature_matcher->Wait();
    }

    return EXIT_SUCCESS;
}

int RunExhaustiveMatcher(int argc, char** argv) {
    OptionManager options;
    options.AddDatabaseOptions();
    options.AddExhaustiveMatchingOptions();
    options.Parse(argc, argv);

    if (!UsesLomaMatching(*options.sift_matching) &&
        !VerifySiftGPUParams(options.sift_matching->use_gpu)) {
        return EXIT_FAILURE;
    }

    std::unique_ptr<QApplication> app;
    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        app.reset(new QApplication(argc, argv));
    }

    ExhaustiveFeatureMatcher feature_matcher(*options.exhaustive_matching,
                                             *options.sift_matching,
                                             *options.database_path);

    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        RunThreadWithOpenGLContext(&feature_matcher);
    } else {
        feature_matcher.Start();
        feature_matcher.Wait();
    }

    return EXIT_SUCCESS;
}

int RunSequentialMatcher(int argc, char** argv) {
    OptionManager options;
    options.AddDatabaseOptions();
    options.AddSequentialMatchingOptions();
    options.Parse(argc, argv);

    if (!UsesLomaMatching(*options.sift_matching) &&
        !VerifySiftGPUParams(options.sift_matching->use_gpu)) {
        return EXIT_FAILURE;
    }

    std::unique_ptr<QApplication> app;
    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        app.reset(new QApplication(argc, argv));
    }

    SequentialFeatureMatcher feature_matcher(*options.sequential_matching,
                                             *options.sift_matching,
                                             *options.database_path);

    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        RunThreadWithOpenGLContext(&feature_matcher);
    } else {
        feature_matcher.Start();
        feature_matcher.Wait();
    }

    return EXIT_SUCCESS;
}

int RunSpatialMatcher(int argc, char** argv) {
    OptionManager options;
    options.AddDatabaseOptions();
    options.AddSpatialMatchingOptions();
    options.Parse(argc, argv);

    if (!UsesLomaMatching(*options.sift_matching) &&
        !VerifySiftGPUParams(options.sift_matching->use_gpu)) {
        return EXIT_FAILURE;
    }

    std::unique_ptr<QApplication> app;
    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        app.reset(new QApplication(argc, argv));
    }

    SpatialFeatureMatcher feature_matcher(*options.spatial_matching,
                                          *options.sift_matching,
                                          *options.database_path);

    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        RunThreadWithOpenGLContext(&feature_matcher);
    } else {
        feature_matcher.Start();
        feature_matcher.Wait();
    }

    return EXIT_SUCCESS;
}

int RunTransitiveMatcher(int argc, char** argv) {
    OptionManager options;
    options.AddDatabaseOptions();
    options.AddTransitiveMatchingOptions();
    options.Parse(argc, argv);

    if (!UsesLomaMatching(*options.sift_matching) &&
        !VerifySiftGPUParams(options.sift_matching->use_gpu)) {
        return EXIT_FAILURE;
    }

    std::unique_ptr<QApplication> app;
    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        app.reset(new QApplication(argc, argv));
    }

    TransitiveFeatureMatcher feature_matcher(*options.transitive_matching,
                                             *options.sift_matching,
                                             *options.database_path);

    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        RunThreadWithOpenGLContext(&feature_matcher);
    } else {
        feature_matcher.Start();
        feature_matcher.Wait();
    }

    return EXIT_SUCCESS;
}

int RunVocabTreeMatcher(int argc, char** argv) {
    OptionManager options;
    options.AddDatabaseOptions();
    options.AddVocabTreeMatchingOptions();
    options.Parse(argc, argv);

    if (!UsesLomaMatching(*options.sift_matching) &&
        !VerifySiftGPUParams(options.sift_matching->use_gpu)) {
        return EXIT_FAILURE;
    }

    std::unique_ptr<QApplication> app;
    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        app.reset(new QApplication(argc, argv));
    }

    VocabTreeFeatureMatcher feature_matcher(*options.vocab_tree_matching,
                                            *options.sift_matching,
                                            *options.database_path);

    if (!UsesLomaMatching(*options.sift_matching) &&
        options.sift_matching->use_gpu && kUseOpenGL) {
        RunThreadWithOpenGLContext(&feature_matcher);
    } else {
        feature_matcher.Start();
        feature_matcher.Wait();
    }

    return EXIT_SUCCESS;
}

}  // namespace colmap
