// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#define TEST_NAME "controllers/da3_depth_controller"
#include "util/testing.h"

#include "controllers/da3_depth_controller.h"

#include <filesystem>

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDA3ModelTypeEnums) {
    BOOST_CHECK(static_cast<int>(DA3ModelType::BASE) == 0);
    BOOST_CHECK(static_cast<int>(DA3ModelType::LARGE) == 1);
    BOOST_CHECK(static_cast<int>(DA3ModelType::GIANT) == 2);
    BOOST_CHECK(static_cast<int>(DA3ModelType::NESTED_METRIC) == 3);
    BOOST_CHECK(static_cast<int>(DA3ModelType::NESTED_ANYVIEW) == 4);
}

BOOST_AUTO_TEST_CASE(TestDA3QuantTypeEnums) {
    BOOST_CHECK(static_cast<int>(DA3QuantType::F32) == 0);
    BOOST_CHECK(static_cast<int>(DA3QuantType::F16) == 1);
    BOOST_CHECK(static_cast<int>(DA3QuantType::Q8_0) == 2);
    BOOST_CHECK(static_cast<int>(DA3QuantType::Q4_K) == 3);
}

BOOST_AUTO_TEST_CASE(TestSparseModelModeEnums) {
    BOOST_CHECK(static_cast<int>(SparseModelMode::COLMAP_NATIVE) == 0);
    BOOST_CHECK(static_cast<int>(SparseModelMode::DA3_DEPTH_POSE) == 1);
}

BOOST_AUTO_TEST_CASE(TestStereoPipelineModeEnums) {
    BOOST_CHECK(static_cast<int>(StereoPipelineMode::COLMAP_PATCH_MATCH) == 0);
    BOOST_CHECK(static_cast<int>(StereoPipelineMode::DA3_DEPTH_INFERENCE) == 1);
}

BOOST_AUTO_TEST_CASE(TestDA3ModelFilename) {
    BOOST_CHECK_EQUAL(DA3ModelFilename(DA3ModelType::BASE, DA3QuantType::Q8_0),
                      "depth-anything-base-q8_0.gguf");
    BOOST_CHECK_EQUAL(DA3ModelFilename(DA3ModelType::BASE, DA3QuantType::F32),
                      "depth-anything-base-f32.gguf");
    BOOST_CHECK_EQUAL(DA3ModelFilename(DA3ModelType::BASE, DA3QuantType::F16),
                      "depth-anything-base-f16.gguf");
    BOOST_CHECK_EQUAL(DA3ModelFilename(DA3ModelType::BASE, DA3QuantType::Q4_K),
                      "depth-anything-base-q4_k.gguf");
    BOOST_CHECK_EQUAL(DA3ModelFilename(DA3ModelType::LARGE, DA3QuantType::Q8_0),
                      "depth-anything-large-q8_0.gguf");
    BOOST_CHECK_EQUAL(DA3ModelFilename(DA3ModelType::GIANT, DA3QuantType::Q8_0),
                      "depth-anything-giant-q8_0.gguf");
    BOOST_CHECK_EQUAL(DA3ModelFilename(DA3ModelType::NESTED_METRIC, DA3QuantType::F32),
                      "depth-anything-nested-metric.gguf");
    BOOST_CHECK_EQUAL(DA3ModelFilename(DA3ModelType::NESTED_ANYVIEW, DA3QuantType::Q8_0),
                      "depth-anything-nested-anyview-q8_0.gguf");
}

BOOST_AUTO_TEST_CASE(TestDA3ModelFilenameFallback) {
    // LARGE + F16 is not in the explicit map, should use fallback naming convention
    const std::string name = DA3ModelFilename(DA3ModelType::LARGE, DA3QuantType::F16);
    BOOST_CHECK_EQUAL(name, "depth-anything-large-f16.gguf");
}

BOOST_AUTO_TEST_CASE(TestDA3ModelDownloadURL) {
    const std::string url = DA3ModelDownloadURL(DA3ModelType::BASE, DA3QuantType::Q8_0);
    BOOST_CHECK_EQUAL(
        url,
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/"
        "depth-anything-base-q8_0.gguf");
}

BOOST_AUTO_TEST_CASE(TestDA3ModelDownloadURI) {
    const std::string uri = DA3ModelDownloadURI(DA3ModelType::LARGE, DA3QuantType::F32);
    // URI should contain the download URL
    BOOST_CHECK(uri.find("depth-anything-large-f32.gguf") != std::string::npos);
    BOOST_CHECK(uri.find("https://") == 0);
}

BOOST_AUTO_TEST_CASE(TestDA3ModelSupportsStereo) {
    BOOST_CHECK(!DA3ModelSupportsStereo(DA3ModelType::BASE));
    BOOST_CHECK(!DA3ModelSupportsStereo(DA3ModelType::LARGE));
    BOOST_CHECK(!DA3ModelSupportsStereo(DA3ModelType::GIANT));
    BOOST_CHECK(DA3ModelSupportsStereo(DA3ModelType::NESTED_METRIC));
    BOOST_CHECK(DA3ModelSupportsStereo(DA3ModelType::NESTED_ANYVIEW));
    BOOST_CHECK(DA3ModelIsNested(DA3ModelType::NESTED_ANYVIEW));
    BOOST_CHECK(!DA3ModelIsNested(DA3ModelType::BASE));
}

BOOST_AUTO_TEST_CASE(TestDA3ConfigDefaults) {
    DA3Config config;
    BOOST_CHECK(config.model_type == DA3ModelType::BASE);
    BOOST_CHECK(config.quant_type == DA3QuantType::Q8_0);
    BOOST_CHECK(config.model_path.empty());
    BOOST_CHECK(config.metric_model_path.empty());
    BOOST_CHECK_EQUAL(config.num_threads, -1);
    BOOST_CHECK(config.sparse_mode == SparseModelMode::COLMAP_NATIVE);
    BOOST_CHECK(config.stereo_mode == StereoPipelineMode::COLMAP_PATCH_MATCH);
}

BOOST_AUTO_TEST_CASE(TestResolveModelPathWithExplicitPath) {
    DA3Config config;
    config.model_path = "/nonexistent/path/model.gguf";
    // Should return empty since the file doesn't exist
    const std::string resolved = DA3DepthController::ResolveModelPath(config);
    BOOST_CHECK(resolved.empty() || resolved == config.model_path);
}

BOOST_AUTO_TEST_CASE(TestResolveModelPathWithExistingFile) {
    // Create a temporary file to test the "existing file" path
    auto temp_dir = std::filesystem::temp_directory_path() / "da3_test";
    std::filesystem::create_directories(temp_dir);
    auto temp_file = temp_dir / "test_model.gguf";

    // Write a dummy file
    {
        std::ofstream ofs(temp_file, std::ios::binary);
        ofs << "GGUF_DUMMY";
    }

    DA3Config config;
    config.model_path = temp_file.string();
    const std::string resolved = DA3DepthController::ResolveModelPath(config);
    BOOST_CHECK_EQUAL(resolved, temp_file.string());

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

BOOST_AUTO_TEST_CASE(TestDA3DepthControllerConstruction) {
    DA3Config config;
    config.model_type = DA3ModelType::BASE;
    config.quant_type = DA3QuantType::Q8_0;
    config.sparse_mode = SparseModelMode::DA3_DEPTH_POSE;

    auto temp_dir = std::filesystem::temp_directory_path() / "da3_ctrl_test";
    std::filesystem::create_directories(temp_dir);

    DA3DepthController controller(config, (std::filesystem::temp_directory_path() / "images").string(),
                                    temp_dir.string());

    bool callback_called = false;
    controller.SetProgressCallback(
        [&](int current, int total, const std::string& status) {
            callback_called = true;
        });

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

BOOST_AUTO_TEST_CASE(TestDA3ModelFilenameAllCombinations) {
    // Verify all known model/quant combinations produce non-empty filenames
    const std::vector<DA3ModelType> models = {
        DA3ModelType::BASE, DA3ModelType::LARGE, DA3ModelType::GIANT,
        DA3ModelType::NESTED_METRIC, DA3ModelType::NESTED_ANYVIEW
    };
    const std::vector<DA3QuantType> quants = {
        DA3QuantType::F32, DA3QuantType::F16, DA3QuantType::Q8_0, DA3QuantType::Q4_K
    };

    for (auto m : models) {
        for (auto q : quants) {
            const std::string name = DA3ModelFilename(m, q);
            BOOST_CHECK(!name.empty());
            BOOST_CHECK(name.find(".gguf") != std::string::npos);
            BOOST_CHECK(name.find("depth-anything-") == 0);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestDA3ModelDownloadURLAllCombinations) {
    const std::string base_url =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/";

    const std::vector<DA3ModelType> models = {
        DA3ModelType::BASE, DA3ModelType::LARGE, DA3ModelType::GIANT
    };
    const std::vector<DA3QuantType> quants = {
        DA3QuantType::F32, DA3QuantType::Q8_0, DA3QuantType::Q4_K
    };

    for (auto m : models) {
        for (auto q : quants) {
            const std::string url = DA3ModelDownloadURL(m, q);
            BOOST_CHECK(url.find(base_url) == 0);
            BOOST_CHECK(url.find(".gguf") != std::string::npos);
        }
    }
}
