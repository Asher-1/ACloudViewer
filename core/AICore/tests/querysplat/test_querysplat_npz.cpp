// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// npz writer contract test: the querysplat exports (cameras / input depths)
// hand .npz files to python analysis, so the store-method ZIP + .npy payload
// layout is verified byte-by-byte here — member order, magic numbers, header
// dict fields, little-endian f32 payload, CRC-32 — plus the failure paths
// (empty array list, unwritable target). No model or backend needed.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "querysplat_npz.hpp"
#include "tests/common/test_macros.hpp"

namespace {

uint16_t read_u16(const unsigned char* p) {
    return uint16_t(p[0]) | uint16_t(p[1]) << 8;
}
uint32_t read_u32(const unsigned char* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 |
           uint32_t(p[3]) << 24;
}

}  // namespace

int main() {
    int failures = 0;

    // ---- happy path: one 1-D and one 3-D array -------------------------
    const float pose[8] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f};
    const float depth[6] = {0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f};
    const std::string path = "/tmp/aicore_test_npz/cameras.npz";
    std::system("mkdir -p /tmp/aicore_test_npz");
    AICORE_CHECK(qs::write_npz(path, {{"w2c", {2, 2, 2}, pose, 8},
                                      {"depth", {1, 2, 3}, depth, 6}}));

    std::FILE* f = std::fopen(path.c_str(), "rb");
    AICORE_CHECK(f != nullptr);
    std::vector<unsigned char> blob;
    if (f) {
        unsigned char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
            blob.insert(blob.end(), buf, buf + n);
        std::fclose(f);
    }
    AICORE_CHECK(blob.size() > 4);
    AICORE_CHECK(blob[0] == 'P' && blob[1] == 'K');  // ZIP local header

    // member 0: local file header + npy payload for "w2c.npy"
    const unsigned char* p = blob.data();
    AICORE_CHECK(read_u32(p) == 0x04034b50u);  // local file header signature
    AICORE_CHECK(read_u16(p + 8) == 0);        // method: store (no deflate)
    const uint16_t name0_len = read_u16(p + 26);
    const uint32_t data0_len = read_u32(p + 18);  // compressed == uncompressed
    const std::string name0(reinterpret_cast<const char*>(p + 30), name0_len);
    AICORE_CHECK(name0 == "w2c.npy");

    const unsigned char* npy = p + 30 + name0_len;
    AICORE_CHECK(std::memcmp(npy, "\x93NUMPY\x01\x00", 8) == 0);
    const uint16_t hdr_len = read_u16(npy + 8);
    const std::string hdr(reinterpret_cast<const char*>(npy + 10), hdr_len);
    AICORE_CHECK(hdr.find("'<f4'") != std::string::npos);
    AICORE_CHECK(hdr.find("'fortran_order': False") != std::string::npos);
    AICORE_CHECK(hdr.find("(2, 2, 2)") != std::string::npos);
    // header block is padded to a 64-byte boundary (numpy requirement)
    AICORE_CHECK((10 + hdr_len) % 64 == 0);
    // payload: little-endian f32, byte-identical to the input
    const unsigned char* payload = npy + 10 + hdr_len;
    AICORE_CHECK(data0_len == 10 + hdr_len + 8 * 4);
    AICORE_CHECK(std::memcmp(payload, pose, 8 * 4) == 0);

    // CRC field: non-zero for a non-empty member, and the writer's IEEE
    // CRC-32 is pinned against the standard zlib reference vector
    {
        const unsigned char check9[] = {'1', '2', '3', '4', '5',
                                        '6', '7', '8', '9'};
        AICORE_CHECK(read_u32(p + 14) != 0);
        AICORE_CHECK(qs::npz_detail::crc32(check9, sizeof(check9)) ==
                     0xCBF43926u);
    }

    // member 1 starts right after member 0's payload
    const unsigned char* p1 = payload + 8 * 4;
    AICORE_CHECK(read_u32(p1) == 0x04034b50u);
    const uint16_t name1_len = read_u16(p1 + 26);
    const std::string name1(reinterpret_cast<const char*>(p1 + 30), name1_len);
    AICORE_CHECK(name1 == "depth.npy");

    // central directory + EOCD at the tail
    AICORE_CHECK(blob.size() > 22);
    const unsigned char* tail = blob.data() + blob.size() - 22;
    AICORE_CHECK(read_u32(tail) == 0x06054b50u);  // EOCD
    AICORE_CHECK(read_u16(tail + 10) == 2);       // 2 members
    // central directory signature appears twice (one per member)
    int cd_entries = 0;
    for (size_t i = 0; i + 4 <= blob.size(); ++i) {
        if (read_u32(blob.data() + i) == 0x02014b50u) cd_entries++;
    }
    AICORE_CHECK(cd_entries == 2);

    // ---- failure paths --------------------------------------------------
    AICORE_CHECK(!qs::write_npz("/tmp/aicore_test_npz", {}));  // no arrays
    // a regular FILE used as a directory component: fopen fails with
    // ENOTDIR deterministically (unlike permission-based targets, which
    // depend on the test user)
    AICORE_CHECK(!qs::write_npz(  // unwritable target
            "/tmp/aicore_test_npz/cameras.npz/forbidden.npz",
            {{"x", {1}, pose, 1}}));

    std::printf(
            failures == 0 ? "NPZ: all checks passed\n" : "NPZ: %d failure(s)\n",
            failures);
    return failures == 0 ? 0 : 1;
}
