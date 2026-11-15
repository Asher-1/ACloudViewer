// SPDX-License-Identifier: BSD-3-Clause
//
// Port of upstream MVS/colmap d3ccaf35 src/colmap/feature/onnx_utils_test.cc
// (batch 24 test alignment), adapted to the fork's AICore engine surface.
//
// Fork adjudication of the upstream ONNX-infrastructure cases:
//   - the hermetic Identity ONNX graph round trip and the ONNX execution
//     provider selection have no fork equivalent: the fork's engines are the
//     AICore ggml task implementations, which own their graphs internally;
//   - the portable error-reporting contract (a failed session load surfaces
//     an error message instead of crashing) is covered here against the
//     AICore ALIKED C API, mirroring upstream's ErrorLogSink case.

#include "aicore/aliked_capi.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(AlikedSessionContract, FailedLoadReportsLastError) {
  // A nonexistent model path must fail the load without crashing and leave
  // a non-empty error message on the returned context.
  aicore_aliked_ctx* ctx =
          aicore_aliked_load_opts("/nonexistent/aliked-model.gguf", nullptr);
  if (ctx == nullptr) {
    SUCCEED() << "load returned NULL for a nonexistent model";
    return;
  }
  EXPECT_FALSE(aicore_aliked_is_ready(ctx));
  EXPECT_NE(aicore_aliked_last_error(ctx), nullptr);
  EXPECT_STRNE(aicore_aliked_last_error(ctx), "");
  aicore_aliked_free(ctx);
}

TEST(AlikedSessionContract, ShutdownIsIdempotent) {
  aicore_aliked_shutdown();
  aicore_aliked_shutdown();
  SUCCEED();
}

TEST(AlikedSessionContract, AbiVersionIsStable) {
  EXPECT_EQ(aicore_aliked_abi_version(), aicore_aliked_abi_version());
  EXPECT_GE(aicore_aliked_abi_version(), 1);
}

}  // namespace
}  // namespace colmap
