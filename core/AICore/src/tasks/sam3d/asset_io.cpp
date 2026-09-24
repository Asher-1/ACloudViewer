// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "asset_io.hpp"

#include <QtCore/QBuffer>
#include <QtCore/QString>
#include <QtGui/QImage>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>

namespace sam3d {
namespace {

constexpr uint32_t kGlbMagic = 0x46546C67;  // glTF
constexpr uint32_t kGlbVersion = 2;
constexpr uint32_t kJsonChunk = 0x4E4F534A;  // JSON
constexpr uint32_t kBinChunk = 0x004E4942;   // BIN\0

void append_bytes(std::vector<uint8_t>& dst, const void* data, size_t size) {
    const auto* first = static_cast<const uint8_t*>(data);
    dst.insert(dst.end(), first, first + size);
}

template <typename T>
void append_values(std::vector<uint8_t>& dst, const std::vector<T>& values) {
    if (!values.empty())
        append_bytes(dst, values.data(), values.size() * sizeof(T));
}

void pad4(std::vector<uint8_t>& dst, uint8_t value = 0) {
    while (dst.size() % 4 != 0) dst.push_back(value);
}

void append_u32(std::vector<uint8_t>& dst, uint32_t value) {
    append_bytes(dst, &value, sizeof(value));
}

// ACloudViewer adaptation: the source runtime decoded JPEG through libjpeg to
// pin the RGB output colorspace. AICore has no libjpeg dependency and the
// in-memory image-view bridge is the production input, so every encoded
// format (PNG/JPEG/...) decodes through QImage with the non-premultiplied
// RGBA8888 layout, matching the Qt decode path the qSAM3D plugin uses.

bool png_from_rgba(const RgbaImage& image,
                   std::vector<uint8_t>& out,
                   std::string& error) {
    if (image.width <= 0 || image.height <= 0 ||
        image.rgba.size() !=
                static_cast<size_t>(image.width) * image.height * 4) {
        error = "PBR texture must be a non-empty RGBA image";
        return false;
    }
    QImage texture(image.rgba.data(), image.width, image.height,
                   static_cast<qsizetype>(image.width) * 4,
                   QImage::Format_RGBA8888);
    QByteArray png_bytes;
    QBuffer buffer(&png_bytes);
    buffer.open(QIODevice::WriteOnly);
    if (!texture.save(&buffer, "PNG")) {
        error = "QImage failed to encode PBR texture";
        return false;
    }
    out.assign(png_bytes.cbegin(), png_bytes.cend());
    return true;
}

std::string json_number(float value) {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(7);
    stream << value;
    return stream.str();
}

}  // namespace

bool load_rgba_image(const std::string& path,
                     RgbaImage& out,
                     std::string& error) {
    const QImage source(QString::fromStdString(path));
    if (source.isNull()) {
        error = "cannot decode image '" + path + "'";
        return false;
    }
    // Format_RGBA8888 is non-premultiplied byte-order RGBA; the premultiplied
    // ARGB32 round-trip perturbs semi-transparent pixels.
    const QImage image = source.convertToFormat(QImage::Format_RGBA8888);
    const int width = image.width();
    const int height = image.height();
    out.width = width;
    out.height = height;
    out.rgba.resize(static_cast<size_t>(width) * height * 4);
    const int stride = image.bytesPerLine();
    if (stride == width * 4) {
        std::memcpy(out.rgba.data(), image.constBits(),
                    static_cast<size_t>(width) * height * 4);
    } else {
        for (int y = 0; y < height; ++y) {
            std::memcpy(out.rgba.data() + static_cast<size_t>(y) * width * 4,
                        image.constBits() + static_cast<qsizetype>(y) * stride,
                        static_cast<size_t>(width) * 4);
        }
    }
    return true;
}

bool write_rgba_png(const std::string& path,
                    const RgbaImage& image,
                    std::string& error) {
    if (image.width <= 0 || image.height <= 0 ||
        image.rgba.size() !=
                static_cast<size_t>(image.width) * image.height * 4) {
        error = "PNG output must be a non-empty RGBA image";
        return false;
    }
    const QImage wrapper(image.rgba.data(), image.width, image.height,
                         static_cast<qsizetype>(image.width) * 4,
                         QImage::Format_RGBA8888);
    if (!wrapper.save(QString::fromStdString(path), "PNG")) {
        error = "QImage failed to write PNG '" + path + "'";
        return false;
    }
    return true;
}

namespace {

// Returns true when at least one (position, uv) byte pair backs more than
// one vertex; `out` then receives the first-occurrence-wins deduplicated
// copy with remapped indices. A legal xatlas seam vertex has distinct uv
// coordinates on its two charts, so only exact (position, uv) duplicates
// - chart corners that share one smooth vertex - are merged here; the
// official to_glb writer keeps (position, uv) unique, and matching that
// vertex table is what the controlled mesh/UV contract compares.
bool chart_corner_duplicates(const NativeMesh& mesh, NativeMesh& out) {
    const size_t vertex_count = mesh.positions.size() / 3;
    if (mesh.texcoords.size() != vertex_count * 2 || vertex_count == 0)
        return false;
    std::map<std::array<uint8_t, 20>, uint32_t>
            first;  // 12 position + 8 uv bytes
    std::vector<uint32_t> remap(vertex_count);
    bool duplicated = false;
    for (uint32_t i = 0; i < vertex_count; ++i) {
        std::array<uint8_t, 20> key{};
        std::memcpy(key.data(), &mesh.positions[i * 3], 12);
        std::memcpy(key.data() + 12, &mesh.texcoords[i * 2], 8);
        auto result = first.emplace(key, i);
        remap[i] = result.first->second;
        duplicated = duplicated || !result.second;
    }
    if (!duplicated) return false;
    std::unordered_map<uint32_t, uint32_t> compact;
    out = NativeMesh{};
    out.positions.reserve(first.size() * 3);
    if (!mesh.normals.empty()) out.normals.reserve(first.size() * 3);
    out.texcoords.reserve(first.size() * 2);
    if (!mesh.colors.empty()) out.colors.reserve(first.size() * 3);
    if (!mesh.vertex_attributes.empty()) {
        const size_t stride = mesh.vertex_attributes.size() / vertex_count;
        out.vertex_attributes.reserve(first.size() * stride);
    }
    for (uint32_t i = 0; i < vertex_count; ++i) {
        if (remap[i] != i) continue;  // a later duplicate of an earlier corner
        compact.emplace(i, static_cast<uint32_t>(out.positions.size() / 3));
        out.positions.insert(out.positions.end(),
                             mesh.positions.begin() + i * 3,
                             mesh.positions.begin() + i * 3 + 3);
        out.texcoords.insert(out.texcoords.end(),
                             mesh.texcoords.begin() + i * 2,
                             mesh.texcoords.begin() + i * 2 + 2);
        if (!mesh.normals.empty()) {
            out.normals.insert(out.normals.end(), mesh.normals.begin() + i * 3,
                               mesh.normals.begin() + i * 3 + 3);
        }
        if (!mesh.colors.empty()) {
            out.colors.insert(out.colors.end(), mesh.colors.begin() + i * 3,
                              mesh.colors.begin() + i * 3 + 3);
        }
        if (!mesh.vertex_attributes.empty()) {
            const size_t stride = mesh.vertex_attributes.size() / vertex_count;
            out.vertex_attributes.insert(
                    out.vertex_attributes.end(),
                    mesh.vertex_attributes.begin() + i * stride,
                    mesh.vertex_attributes.begin() + (i + 1) * stride);
        }
    }
    out.indices.reserve(mesh.indices.size());
    for (uint32_t index : mesh.indices)
        out.indices.push_back(compact.at(remap[index]));
    out.material = mesh.material;
    return true;
}

}  // namespace

bool write_pbr_glb(const std::string& path,
                   const NativeMesh& mesh,
                   std::string& error) {
    NativeMesh deduped;
    if (chart_corner_duplicates(mesh, deduped)) {
        // Recursing once is safe: the deduplicated mesh cannot trip the
        // detector again.
        return write_pbr_glb(path, deduped, error);
    }
    const size_t vertex_count = mesh.positions.size() / 3;
    if (vertex_count == 0 || mesh.positions.size() % 3 != 0) {
        error = "GLB mesh requires non-empty xyz positions";
        return false;
    }
    if (mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        error = "GLB mesh requires triangle indices";
        return false;
    }
    for (uint32_t index : mesh.indices) {
        if (index >= vertex_count) {
            error = "GLB index is outside the vertex array";
            return false;
        }
    }
    if (!mesh.normals.empty() && mesh.normals.size() != mesh.positions.size()) {
        error = "GLB normals must contain one xyz tuple per vertex";
        return false;
    }
    if (!mesh.texcoords.empty() && mesh.texcoords.size() != vertex_count * 2) {
        error = "GLB texture coordinates must contain one uv tuple per vertex";
        return false;
    }
    if (!mesh.colors.empty() && mesh.colors.size() != vertex_count * 3) {
        error = "GLB vertex colors must contain one rgb tuple per vertex";
        return false;
    }

    std::vector<uint8_t> png;
    const bool has_texture = !mesh.material.base_color_texture.rgba.empty();
    if (has_texture && mesh.texcoords.empty()) {
        error = "GLB base-color texture requires one uv tuple per vertex";
        return false;
    }
    if (has_texture &&
        !png_from_rgba(mesh.material.base_color_texture, png, error))
        return false;

    std::vector<uint8_t> bin;
    const uint32_t positions_offset = static_cast<uint32_t>(bin.size());
    append_values(bin, mesh.positions);
    pad4(bin);
    const uint32_t normals_offset = static_cast<uint32_t>(bin.size());
    if (!mesh.normals.empty()) append_values(bin, mesh.normals);
    pad4(bin);
    const uint32_t texcoords_offset = static_cast<uint32_t>(bin.size());
    if (!mesh.texcoords.empty()) append_values(bin, mesh.texcoords);
    pad4(bin);
    const uint32_t colors_offset = static_cast<uint32_t>(bin.size());
    if (!mesh.colors.empty()) append_values(bin, mesh.colors);
    pad4(bin);
    const uint32_t indices_offset = static_cast<uint32_t>(bin.size());
    append_values(bin, mesh.indices);
    pad4(bin);
    const uint32_t texture_offset = static_cast<uint32_t>(bin.size());
    if (has_texture) append_values(bin, png);
    pad4(bin);

    std::array<float, 3> min_pos{std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
    std::array<float, 3> max_pos{-std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity()};
    for (size_t i = 0; i < mesh.positions.size(); i += 3) {
        for (size_t c = 0; c < 3; ++c) {
            min_pos[c] = std::min(min_pos[c], mesh.positions[i + c]);
            max_pos[c] = std::max(max_pos[c], mesh.positions[i + c]);
        }
    }

    std::ostringstream json;
    json << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"sam3d-ggml-"
            "native\"},"
         << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
         << "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0";
    int accessor_count = 1;
    if (!mesh.normals.empty()) json << ",\"NORMAL\":" << accessor_count++;
    if (!mesh.texcoords.empty()) json << ",\"TEXCOORD_0\":" << accessor_count++;
    if (!mesh.colors.empty()) json << ",\"COLOR_0\":" << accessor_count++;
    const int indices_accessor = accessor_count++;
    json << "},\"indices\":" << indices_accessor
         << ",\"material\":0,\"mode\":4}]}],";

    json << "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":["
         << json_number(mesh.material.base_color[0]) << ","
         << json_number(mesh.material.base_color[1]) << ","
         << json_number(mesh.material.base_color[2]) << ","
         << json_number(mesh.material.base_color[3])
         << "],\"metallicFactor\":" << json_number(mesh.material.metallic)
         << ",\"roughnessFactor\":" << json_number(mesh.material.roughness);
    if (has_texture) json << ",\"baseColorTexture\":{\"index\":0}";
    json << "}}],";
    const int texture_view = 2 + (!mesh.normals.empty() ? 1 : 0) +
                             (!mesh.texcoords.empty() ? 1 : 0) +
                             (!mesh.colors.empty() ? 1 : 0);
    if (has_texture) {
        json << "\"textures\":[{\"source\":0}],\"images\":[{\"bufferView\":"
             << texture_view << ",\"mimeType\":\"image/png\"}],";
    }
    json << "\"buffers\":[{\"byteLength\":" << bin.size()
         << "}],\"bufferViews\":["
         << "{\"buffer\":0,\"byteOffset\":" << positions_offset
         << ",\"byteLength\":" << mesh.positions.size() * sizeof(float)
         << ",\"target\":34962}";
    int view_index = 1;
    if (!mesh.normals.empty()) {
        json << ",{\"buffer\":0,\"byteOffset\":" << normals_offset
             << ",\"byteLength\":" << mesh.normals.size() * sizeof(float)
             << ",\"target\":34962}";
        ++view_index;
    }
    if (!mesh.texcoords.empty()) {
        json << ",{\"buffer\":0,\"byteOffset\":" << texcoords_offset
             << ",\"byteLength\":" << mesh.texcoords.size() * sizeof(float)
             << ",\"target\":34962}";
        ++view_index;
    }
    if (!mesh.colors.empty()) {
        json << ",{\"buffer\":0,\"byteOffset\":" << colors_offset
             << ",\"byteLength\":" << mesh.colors.size() * sizeof(float)
             << ",\"target\":34962}";
        ++view_index;
    }
    const int indices_view = view_index++;
    json << ",{\"buffer\":0,\"byteOffset\":" << indices_offset
         << ",\"byteLength\":" << mesh.indices.size() * sizeof(uint32_t)
         << ",\"target\":34963}";
    if (has_texture) {
        json << ",{\"buffer\":0,\"byteOffset\":" << texture_offset
             << ",\"byteLength\":" << png.size() << "}";
    }
    json << "],\"accessors\":["
         << "{\"bufferView\":0,\"componentType\":5126,\"count\":"
         << vertex_count << ",\"type\":\"VEC3\",\"min\":["
         << json_number(min_pos[0]) << "," << json_number(min_pos[1]) << ","
         << json_number(min_pos[2]) << "],\"max\":[" << json_number(max_pos[0])
         << "," << json_number(max_pos[1]) << "," << json_number(max_pos[2])
         << "]}";
    int next_view = 1;
    if (!mesh.normals.empty()) {
        json << ",{\"bufferView\":" << next_view++
             << ",\"componentType\":5126,\"count\":" << vertex_count
             << ",\"type\":\"VEC3\"}";
    }
    if (!mesh.texcoords.empty()) {
        json << ",{\"bufferView\":" << next_view++
             << ",\"componentType\":5126,\"count\":" << vertex_count
             << ",\"type\":\"VEC2\"}";
    }
    if (!mesh.colors.empty()) {
        json << ",{\"bufferView\":" << next_view++
             << ",\"componentType\":5126,\"count\":" << vertex_count
             << ",\"type\":\"VEC3\"}";
    }
    json << ",{\"bufferView\":" << indices_view
         << ",\"componentType\":5125,\"count\":" << mesh.indices.size()
         << ",\"type\":\"SCALAR\"}]}";

    std::string json_bytes = json.str();
    std::vector<uint8_t> glb;
    std::vector<uint8_t> json_chunk(json_bytes.begin(), json_bytes.end());
    pad4(json_chunk, ' ');
    append_u32(glb, kGlbMagic);
    append_u32(glb, kGlbVersion);
    append_u32(glb, static_cast<uint32_t>(12 + 8 + json_chunk.size() + 8 +
                                          bin.size()));
    append_u32(glb, static_cast<uint32_t>(json_chunk.size()));
    append_u32(glb, kJsonChunk);
    append_values(glb, json_chunk);
    append_u32(glb, static_cast<uint32_t>(bin.size()));
    append_u32(glb, kBinChunk);
    append_values(glb, bin);

    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        error = "cannot open GLB output '" + path + "'";
        return false;
    }
    const size_t written = std::fwrite(glb.data(), 1, glb.size(), file);
    const int close_status = std::fclose(file);
    if (written != glb.size() || close_status != 0) {
        error = "failed to write complete GLB output '" + path + "'";
        return false;
    }
    return true;
}

}  // namespace sam3d
