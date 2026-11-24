// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "tasks/facedetect/backend.hpp"

TEST(BackendRegistry, CompatibleSessionsShareBackendAndRetainPeerOwnership) {
    std::string error;
    auto first = aicore::runtime::acquire_backend_lease("cpu", 1, &error);
    ASSERT_TRUE(first) << error;
    EXPECT_EQ(ggml_common::flash_attention_sequence_limit(first.handle()),
              INT64_MAX);
    auto second = aicore::runtime::acquire_backend_lease("cpu", 1, &error);
    ASSERT_TRUE(second) << error;
    EXPECT_EQ(first.handle(), second.handle());
    fd::Backend face_session(1, "cpu");
    EXPECT_EQ(face_session.handle(), second.handle());
    first.reset();
    ASSERT_TRUE(second);
    ASSERT_NE(second.handle(), nullptr);
    const auto lock = second.lock();
    EXPECT_TRUE(lock.owns_lock());
}

TEST(BackendRegistry,
     ThreadConfigurationsRemainDistinctAndGroupLocksDeduplicate) {
    std::string error;
    auto first = aicore::runtime::acquire_backend_lease("cpu", 1, &error);
    ASSERT_TRUE(first) << error;
    const auto lock = first.lock();
    ASSERT_TRUE(lock.owns_lock());
    auto second = aicore::runtime::acquire_backend_lease("cpu", 2, &error);
    ASSERT_TRUE(second) << error;
    EXPECT_NE(first.handle(), second.handle());
    const std::vector<aicore::runtime::BackendLease> group = {first, second,
                                                              first};
    const auto group_lock = aicore::runtime::lock_backend_leases(group);
    EXPECT_TRUE(group_lock.owns_lock());
}

TEST(BackendRegistry, ParallelWorkersDoNotShareSerializedBackends) {
    std::string error;
    auto shared = aicore::runtime::acquire_backend_lease("cpu", 1, &error);
    ASSERT_TRUE(shared) << error;
    auto first =
            aicore::runtime::acquire_parallel_backend_lease("cpu", 1, &error);
    ASSERT_TRUE(first) << error;
    auto second =
            aicore::runtime::acquire_parallel_backend_lease("cpu", 1, &error);
    ASSERT_TRUE(second) << error;
    EXPECT_NE(first.handle(), second.handle());
    EXPECT_NE(first.handle(), shared.handle());
}
