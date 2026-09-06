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
#include <fstream>

using namespace colmap;

TEST(controllers_da3_depth_controller, TestDA3ModelTypeEnums) {
    EXPECT_TRUE(static_cast<int>(DA3ModelType::BASE) == 0);
    EXPECT_TRUE(static_cast<int>(DA3ModelType::LARGE) == 1);
    EXPECT_TRUE(static_cast<int>(DA3ModelType::GIANT) == 2);
    EXPECT_TRUE(static_cast<int>(DA3ModelType::NESTED_METRIC) == 3);
    EXPECT_TRUE(static_cast<int>(DA3ModelType::NESTED_ANYVIEW) == 4);
}

TEST(controllers_da3_depth_controller, TestDA3QuantTypeEnums) {
    EXPECT_TRUE(static_cast<int>(DA3QuantType::F32) == 0);
    EXPECT_TRUE(static_cast<int>(DA3QuantType::F16) == 1);
    EXPECT_TRUE(static_cast<int>(DA3QuantType::Q8_0) == 2);
    EXPECT_TRUE(static_cast<int>(DA3QuantType::Q4_K) == 3);
}

TEST(controllers_da3_depth_controller, TestSparseModelModeEnums) {
    EXPECT_TRUE(static_cast<int>(SparseModelMode::COLMAP_NATIVE) == 0);
    EXPECT_TRUE(static_cast<int>(SparseModelMode::DA3_DEPTH_POSE) == 1);
}

TEST(controllers_da3_depth_controller, TestStereoPipelineModeEnums) {
    EXPECT_TRUE(static_cast<int>(StereoPipelineMode::COLMAP_PATCH_MATCH) == 0);
    EXPECT_TRUE(static_cast<int>(StereoPipelineMode::DA3_DEPTH_INFERENCE) == 1);
}

TEST(controllers_da3_depth_controller, TestDA3ModelFilename) {
    EXPECT_EQ(DA3ModelFilename(DA3ModelType::BASE, DA3QuantType::Q8_0),
                      "depth-anything-base-q8_0.gguf");
    EXPECT_EQ(DA3ModelFilename(DA3ModelType::BASE, DA3QuantType::F32),
                      "depth-anything-base-f32.gguf");
    EXPECT_EQ(DA3ModelFilename(DA3ModelType::BASE, DA3QuantType::F16),
                      "depth-anything-base-f16.gguf");
    EXPECT_EQ(DA3ModelFilename(DA3ModelType::BASE, DA3QuantType::Q4_K),
                      "depth-anything-base-q4_k.gguf");
    EXPECT_EQ(DA3ModelFilename(DA3ModelType::LARGE, DA3QuantType::Q8_0),
                      "depth-anything-large-q8_0.gguf");
    EXPECT_EQ(DA3ModelFilename(DA3ModelType::GIANT, DA3QuantType::Q8_0),
                      "depth-anything-giant-q8_0.gguf");
    EXPECT_EQ(DA3ModelFilename(DA3ModelType::NESTED_METRIC, DA3QuantType::F32),
                      "depth-anything-nested-metric.gguf");
    EXPECT_EQ(DA3ModelFilename(DA3ModelType::NESTED_ANYVIEW, DA3QuantType::Q8_0),
                      "depth-anything-nested-anyview-q8_0.gguf");
}

TEST(controllers_da3_depth_controller, TestDA3ModelFilenameFallback) {
    // LARGE + F16 is not in the explicit map, should use fallback naming convention
    const std::string name = DA3ModelFilename(DA3ModelType::LARGE, DA3QuantType::F16);
    EXPECT_EQ(name, "depth-anything-large-f16.gguf");
}

TEST(controllers_da3_depth_controller, TestDA3ModelDownloadURL) {
    const std::string url = DA3ModelDownloadURL(DA3ModelType::BASE, DA3QuantType::Q8_0);
    EXPECT_EQ(
        url,
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/"
        "depth-anything-base-q8_0.gguf");
}

TEST(controllers_da3_depth_controller, TestDA3ModelDownloadURI) {
    const std::string uri = DA3ModelDownloadURI(DA3ModelType::LARGE, DA3QuantType::F32);
    // URI should contain the download URL
    EXPECT_TRUE(uri.find("depth-anything-large-f32.gguf") != std::string::npos);
    EXPECT_TRUE(uri.find("https://") == 0);
}

TEST(controllers_da3_depth_controller, TestDA3ModelSupportsStereo) {
    EXPECT_FALSE(DA3ModelSupportsStereo(DA3ModelType::BASE));
    EXPECT_FALSE(DA3ModelSupportsStereo(DA3ModelType::LARGE));
    EXPECT_FALSE(DA3ModelSupportsStereo(DA3ModelType::GIANT));
    EXPECT_TRUE(DA3ModelSupportsStereo(DA3ModelType::NESTED_METRIC));
    EXPECT_TRUE(DA3ModelSupportsStereo(DA3ModelType::NESTED_ANYVIEW));
    EXPECT_TRUE(DA3ModelIsNested(DA3ModelType::NESTED_ANYVIEW));
    EXPECT_FALSE(DA3ModelIsNested(DA3ModelType::BASE));
}

TEST(controllers_da3_depth_controller, TestDA3ConfigDefaults) {
    DA3Config config;
    EXPECT_TRUE(config.model_type == DA3ModelType::BASE);
    EXPECT_TRUE(config.quant_type == DA3QuantType::Q8_0);
    EXPECT_TRUE(config.model_path.empty());
    EXPECT_TRUE(config.metric_model_path.empty());
    EXPECT_EQ(config.num_threads, -1);
    EXPECT_TRUE(config.sparse_mode == SparseModelMode::COLMAP_NATIVE);
    EXPECT_TRUE(config.stereo_mode == StereoPipelineMode::COLMAP_PATCH_MATCH);
}

TEST(controllers_da3_depth_controller, TestResolveModelPathWithExplicitPath) {
    DA3Config config;
    config.model_path = "/nonexistent/path/model.gguf";
    // Should return empty since the file doesn't exist
    const std::string resolved = DA3DepthController::ResolveModelPath(config);
    EXPECT_TRUE(resolved.empty() || resolved == config.model_path);
}

TEST(controllers_da3_depth_controller, TestResolveModelPathWithExistingFile) {
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
    EXPECT_EQ(resolved, temp_file.string());

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

TEST(controllers_da3_depth_controller, TestDA3DepthControllerConstruction) {
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

TEST(controllers_da3_depth_controller, TestDA3ModelFilenameAllCombinations) {
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
            EXPECT_FALSE(name.empty());
            EXPECT_TRUE(name.find(".gguf") != std::string::npos);
            EXPECT_TRUE(name.find("depth-anything-") == 0);
        }
    }
}

TEST(controllers_da3_depth_controller, TestDA3ModelDownloadURLAllCombinations) {
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
            EXPECT_TRUE(url.find(base_url) == 0);
            EXPECT_TRUE(url.find(".gguf") != std::string::npos);
        }
    }
}
