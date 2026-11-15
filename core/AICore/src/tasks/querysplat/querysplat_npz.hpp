// Minimal .npz (numpy) writer for the querysplat compatibility exports.
//
// The official chain saves the predicted cameras and the raw VGGT input
// depths as .npz archives next to their JSON/PLY artifacts so python
// analysis can torch.load them directly. AICore's inference path stays
// typed (aicore_querysplat_result_* accessors); these exports are the
// file boundary only (see the AICore plugin contract).
//
// Format: an uncompressed (store-method) ZIP whose members are .npy files
// — magic + 2-byte version + 2-byte header length + a python-literal dict
// padded to a 64-byte boundary + raw little-endian f32 payload. No third
// party dependency; verified against numpy.load in the export tests.
#pragma once

#include "FileSystem.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qs {

struct NpzArray {
    std::string name;             // member name without the .npy suffix
    std::vector<int64_t> shape;   // numpy shape (at least 1 dim)
    const float* data;            // row-major (C order) f32 payload
    int64_t count = 0;            // number of elements
};

namespace npz_detail {

inline void put_u16(std::string& s, uint16_t v) {
    s.push_back(char(v & 0xff));
    s.push_back(char((v >> 8) & 0xff));
}
inline void put_u32(std::string& s, uint32_t v) {
    for (int i = 0; i < 4; i++) s.push_back(char((v >> (8 * i)) & 0xff));
}

inline uint32_t crc32(const unsigned char* p, size_t n) {
    // IEEE CRC-32 (reflected, poly 0xEDB88320), zlib-compatible.
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++)
        c = table[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

inline std::string npy_header(const std::vector<int64_t>& shape) {
    std::string sh;
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) sh += ", ";
        sh += std::to_string(shape[i]);
    }
    if (shape.size() == 1) sh += ",";  // numpy 1-tuple literal
    std::string dict =
            "{'descr': '<f4', 'fortran_order': False, 'shape': (" + sh +
            "), }";
    // magic(6) + version(2) + hlen(2) + dict + nspaces + "\n" aligned to a
    // 64-byte boundary (numpy's ARRAY_ALIGN writer convention; matches the
    // byte layout of np.lib.format for the same dict). The padding here used
    // to run one space short and shift every following member.
    const size_t nspaces = 64 - ((11 + dict.size()) % 64);
    dict.append(nspaces, ' ');
    dict.push_back('\n');
    std::string out;
    out += "\x93NUMPY";
    out.push_back(char(1));
    out.push_back(char(0));
    put_u16(out, uint16_t(dict.size()));
    out += dict;
    return out;
}

}  // namespace npz_detail

// Writes an uncompressed .npz with one f32 .npy member per entry.
// Returns false on any file/IO failure.
inline bool write_npz(const std::string& path,
                      const std::vector<NpzArray>& arrays) {
    if (arrays.empty()) return false;
    FILE* f = cloudViewer::utility::filesystem::FOpen(path.c_str(), "wb");
    if (!f) return false;

    std::string body;      // local records + payloads
    std::string central;   // central directory
    std::vector<uint32_t> offsets;
    offsets.reserve(arrays.size());
    for (const NpzArray& a : arrays) {
        const std::string member = a.name + ".npy";
        const std::string header = npz_detail::npy_header(a.shape);
        const size_t payload_bytes = size_t(a.count) * sizeof(float);
        // CRC covers the npy header + payload together (the member blob).
        std::vector<unsigned char> blob(header.size() + payload_bytes);
        std::memcpy(blob.data(), header.data(), header.size());
        if (payload_bytes)
            std::memcpy(blob.data() + header.size(), a.data, payload_bytes);
        const uint32_t crc_all =
                npz_detail::crc32(blob.data(), blob.size());
        offsets.push_back(uint32_t(body.size()));

        std::string local;
        npz_detail::put_u32(local, 0x04034b50u);
        npz_detail::put_u16(local, 20);    // version needed
        npz_detail::put_u16(local, 0);     // flags
        npz_detail::put_u16(local, 0);     // method: store
        npz_detail::put_u16(local, 0);     // mod time
        npz_detail::put_u16(local, 0);     // mod date
        npz_detail::put_u32(local, crc_all);
        npz_detail::put_u32(local, uint32_t(blob.size()));
        npz_detail::put_u32(local, uint32_t(blob.size()));
        npz_detail::put_u16(local, uint16_t(member.size()));
        npz_detail::put_u16(local, 0);     // extra len
        local += member;
        body += local;
        body.append(reinterpret_cast<const char*>(blob.data()), blob.size());

        std::string cd;
        npz_detail::put_u32(cd, 0x02014b50u);
        npz_detail::put_u16(cd, 20);       // version made by
        npz_detail::put_u16(cd, 20);       // version needed
        npz_detail::put_u16(cd, 0);        // flags
        npz_detail::put_u16(cd, 0);        // method
        npz_detail::put_u16(cd, 0);        // time
        npz_detail::put_u16(cd, 0);        // date
        npz_detail::put_u32(cd, crc_all);
        npz_detail::put_u32(cd, uint32_t(blob.size()));
        npz_detail::put_u32(cd, uint32_t(blob.size()));
        npz_detail::put_u16(cd, uint16_t(member.size()));
        npz_detail::put_u16(cd, 0);        // extra
        npz_detail::put_u16(cd, 0);        // comment
        npz_detail::put_u16(cd, 0);        // disk start
        npz_detail::put_u16(cd, 0);        // internal attrs
        npz_detail::put_u32(cd, 0);        // external attrs
        npz_detail::put_u32(cd, offsets.back());
        cd += member;
        central += cd;
    }

    const uint32_t cd_offset = uint32_t(body.size());
    std::string eocd;
    npz_detail::put_u32(eocd, 0x06054b50u);
    npz_detail::put_u16(eocd, 0);
    npz_detail::put_u16(eocd, 0);
    npz_detail::put_u16(eocd, uint16_t(arrays.size()));
    npz_detail::put_u16(eocd, uint16_t(arrays.size()));
    npz_detail::put_u32(eocd, uint32_t(central.size()));
    npz_detail::put_u32(eocd, cd_offset);
    npz_detail::put_u16(eocd, 0);

    const std::string all = body + central + eocd;
    const bool ok = std::fwrite(all.data(), 1, all.size(), f) == all.size();
    std::fclose(f);
    return ok;
}

}  // namespace qs
