// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// NPZ writer contract: stored ZIP members, NumPy headers, little-endian f32
// payloads and CRC-32. No model or backend is needed.
#include <gtest/gtest.h>

#include <QTemporaryDir>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "tasks/querysplat/querysplat_npz.hpp"

namespace {

uint16_t read_u16(const unsigned char* p) {
    return uint16_t(p[0]) | uint16_t(p[1]) << 8;
}
uint32_t read_u32(const unsigned char* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 |
           uint32_t(p[3]) << 24;
}

class QuerySplatNpz : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(directory.isValid())
                << directory.errorString().toStdString();
        path = directory.filePath("cameras.npz").toStdString();
    }

    QTemporaryDir directory;
    std::string path;
    const float pose[8] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f};
    const float depth[6] = {0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f};
};

TEST_F(QuerySplatNpz, StoresOrderedNumpyArraysAndZipDirectory) {
    ASSERT_TRUE(qs::write_npz(path, {{"w2c", {2, 2, 2}, pose, 8},
                                     {"depth", {1, 2, 3}, depth, 6}}));

    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(
            std::fopen(path.c_str(), "rb"), &std::fclose);
    ASSERT_NE(file, nullptr);
    std::vector<unsigned char> blob;
    unsigned char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), file.get())) > 0)
        blob.insert(blob.end(), buf, buf + n);
    ASSERT_EQ(std::ferror(file.get()), 0);
    ASSERT_EQ(std::fclose(file.release()), 0);
    ASSERT_GE(blob.size(), 30u);
    EXPECT_EQ(blob[0], 'P');
    EXPECT_EQ(blob[1], 'K');

    // Member 0: local header and NumPy payload for w2c.npy.
    const unsigned char* p = blob.data();
    EXPECT_EQ(read_u32(p), 0x04034b50u);
    EXPECT_EQ(read_u16(p + 8), 0);  // Stored, not deflated.
    const uint16_t name0_len = read_u16(p + 26);
    const uint32_t data0_len = read_u32(p + 18);
    ASSERT_GE(blob.size(), 30u + name0_len + 10u);
    const std::string name0(reinterpret_cast<const char*>(p + 30), name0_len);
    EXPECT_EQ(name0, "w2c.npy");

    const unsigned char* npy = p + 30 + name0_len;
    EXPECT_EQ(std::memcmp(npy, "\x93NUMPY\x01\x00", 8), 0);
    const uint16_t hdr_len = read_u16(npy + 8);
    ASSERT_GE(blob.size(),
              30u + name0_len + 10u + hdr_len + sizeof(pose) + 30u);
    const std::string hdr(reinterpret_cast<const char*>(npy + 10), hdr_len);
    EXPECT_NE(hdr.find("'<f4'"), std::string::npos);
    EXPECT_NE(hdr.find("'fortran_order': False"), std::string::npos);
    EXPECT_NE(hdr.find("(2, 2, 2)"), std::string::npos);
    EXPECT_EQ((10 + hdr_len) % 64, 0);
    const unsigned char* payload = npy + 10 + hdr_len;
    EXPECT_EQ(data0_len, 10u + hdr_len + sizeof(pose));
    EXPECT_EQ(std::memcmp(payload, pose, sizeof(pose)), 0);
    EXPECT_NE(read_u32(p + 14), 0u);

    // Member 1 starts immediately after member 0's payload.
    const unsigned char* p1 = payload + sizeof(pose);
    EXPECT_EQ(read_u32(p1), 0x04034b50u);
    const uint16_t name1_len = read_u16(p1 + 26);
    ASSERT_GE(blob.size(), size_t(p1 - p) + 30u + name1_len);
    const std::string name1(reinterpret_cast<const char*>(p1 + 30), name1_len);
    EXPECT_EQ(name1, "depth.npy");

    ASSERT_GT(blob.size(), 22u);
    const unsigned char* tail = blob.data() + blob.size() - 22;
    EXPECT_EQ(read_u32(tail), 0x06054b50u);
    EXPECT_EQ(read_u16(tail + 10), 2);
    int cd_entries = 0;
    for (size_t i = 0; i + 4 <= blob.size(); ++i) {
        if (read_u32(blob.data() + i) == 0x02014b50u) cd_entries++;
    }
    EXPECT_EQ(cd_entries, 2);
}

TEST(QuerySplatNpzChecksum, MatchesStandardIeeeCrc32Vector) {
    const unsigned char check9[] = {'1', '2', '3', '4', '5',
                                    '6', '7', '8', '9'};
    EXPECT_EQ(qs::npz_detail::crc32(check9, sizeof(check9)), 0xCBF43926u);
}

TEST_F(QuerySplatNpz, RejectsEmptyArrayList) {
    ASSERT_TRUE(qs::write_npz(path, {{"w2c", {2, 2, 2}, pose, 8},
                                     {"depth", {1, 2, 3}, depth, 6}}));
    EXPECT_FALSE(qs::write_npz(path, {}));
}

TEST_F(QuerySplatNpz, RejectsRegularFileAsDirectoryComponent) {
    ASSERT_TRUE(qs::write_npz(path, {{"w2c", {2, 2, 2}, pose, 8},
                                     {"depth", {1, 2, 3}, depth, 6}}));
    // Deterministic path error, independent of the test user's permissions.
    EXPECT_FALSE(qs::write_npz(path + "/forbidden.npz", {{"x", {1}, pose, 1}}));
}

}  // namespace
