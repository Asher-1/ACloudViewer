// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QTemporaryDir>
#include <memory>
#include <string>

#include "gguf.h"
#include "src/common/gguf_file_io.hpp"

namespace {
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Ggml = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

class GgufFileIo : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(directory.isValid());
        path = directory.filePath("synthetic.gguf").toUtf8().toStdString();
        Gguf data(gguf_init_empty(), gguf_free);
        ASSERT_NE(data, nullptr);
        gguf_set_val_str(data.get(), "general.architecture", "test-arch");
        gguf_set_val_u32(data.get(), "test.answer", 42);
        gguf_write_to_file(data.get(), path.c_str(), false);
    }

    QTemporaryDir directory;
    std::string path;
};

TEST_F(GgufFileIo, OpensMetadataAndPreservesKeys) {
    ggml_context* raw = nullptr;
    std::string error;
    Gguf data(ggml_common::open_gguf_file(path, true, &raw, "test", &error),
              gguf_free);
    Ggml metadata(raw, ggml_free);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(metadata, nullptr);
    EXPECT_TRUE(error.empty());
    const auto architecture = gguf_find_key(data.get(), "general.architecture");
    ASSERT_GE(architecture, 0);
    EXPECT_STREQ(gguf_get_val_str(data.get(), architecture), "test-arch");
}

TEST_F(GgufFileIo, OpensAllocatedStorageWithoutErrorOutput) {
    ggml_context* raw = nullptr;
    Gguf data(ggml_common::open_gguf_file(path, false, &raw, "test", nullptr),
              gguf_free);
    Ggml metadata(raw, ggml_free);
    EXPECT_NE(data, nullptr);
    EXPECT_NE(metadata, nullptr);
}

TEST_F(GgufFileIo, RepeatedOpensHaveIndependentOwnership) {
    for (int round = 0; round < 2; ++round) {
        SCOPED_TRACE(round);
        ggml_context* raw = nullptr;
        Gguf data(
                ggml_common::open_gguf_file(path, true, &raw, "test", nullptr),
                gguf_free);
        Ggml metadata(raw, ggml_free);
        EXPECT_NE(data, nullptr);
    }
}

TEST_F(GgufFileIo, MissingFileReportsTaskAndPath) {
    const auto missing = path + ".does-not-exist";
    ggml_context* raw = nullptr;
    std::string error;
    Gguf data(ggml_common::open_gguf_file(missing, true, &raw, "some_task",
                                          &error),
              gguf_free);
    Ggml metadata(raw, ggml_free);
    EXPECT_EQ(data, nullptr);
    EXPECT_EQ(error, "some_task: failed to open GGUF '" + missing + "'");
}

TEST_F(GgufFileIo, MissingContextUsesGenericErrorPrefix) {
    std::string error;
    Gguf data(ggml_common::open_gguf_file(path + ".missing2", true, nullptr,
                                          nullptr, &error),
              gguf_free);
    EXPECT_EQ(error.rfind("gguf: failed to open GGUF '", 0), 0u);
}
}  // namespace
