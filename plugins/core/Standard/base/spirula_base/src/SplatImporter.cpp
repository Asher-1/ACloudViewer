// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SplatImporter.h"

// Light-weight reader for the standard 3DGS PLY that Spirula Studio exports
// (the same file the upstream web viewer and SuperSplat consume). Deliberately
// narrow: no third-party dependency, no ascii/big-endian support, only the
// properties a splat viewer needs.

#include <FileIOFilter.h>
#include <ecvDrawableObject.h>
#include <ecvGlobalShiftManager.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace spirula {

namespace {

constexpr float kSH0Coefficient = 0.28209479177387814f;
// Low-opacity gaussians fill empty space; 0.1 keeps thin structures while
// dropping the bulk of the transmissive shell (same role as the upstream
// viewer's opacity prune, just fixed for a dialog-less import).
constexpr float kOpacityThreshold = 0.1f;
// Display cap, matching qFreeSplatter's behavior for gaussian clouds; the
// full count goes to metadata.
constexpr int kMaxDisplayPoints = 500000;

struct PropertySpec {
    std::string name;
    int sizeBytes = 0;
};

int plyTypeSize(const std::string& type) {
    if (type == "char" || type == "int8" || type == "uchar" ||
        type == "uint8") {
        return 1;
    }
    if (type == "short" || type == "int16" || type == "ushort" ||
        type == "uint16") {
        return 2;
    }
    if (type == "int" || type == "int32" || type == "uint" ||
        type == "uint32") {
        return 4;
    }
    if (type == "float" || type == "float32" || type == "double" ||
        type == "float64") {
        return 4;  // double fixed up by the caller (both read as 8 bytes only
                   // for float64; see propertyOffset)
    }
    return 0;
}

int propertySizeBytes(const std::string& type) {
    if (type == "double" || type == "float64") {
        return 8;
    }
    return plyTypeSize(type);
}

struct PlyLayout {
    bool ok = false;
    long long vertexCount = 0;
    long long dataOffset = 0;
    int stride = 0;
    int offset[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};  // x y z
    // fdc0 fdc1 fdc2 opacity scale0 scale1 scale2
    bool hasScale = false;
};

PlyLayout parsePlyHeader(const QString& plyPath, QString* errorOut) {
    PlyLayout layout;
    QFile file(plyPath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (errorOut) {
            *errorOut = QStringLiteral("Cannot open '%1'.").arg(plyPath);
        }
        return layout;
    }

    bool littleEndian = false;
    bool inVertexElement = false;
    long long offsetCursor = 0;
    QByteArray line;

    while (!(line = file.readLine()).isEmpty()) {
        const std::string token =
                std::string(line.constData(), static_cast<size_t>(line.size()));
        if (token.rfind("format binary_little_endian", 0) == 0) {
            littleEndian = true;
        } else if (token.rfind("format binary_big_endian", 0) == 0) {
            if (errorOut) {
                *errorOut = QStringLiteral(
                        "big-endian PLY is not supported by the light splat "
                        "importer (re-export as binary little-endian).");
            }
            return layout;
        } else if (token.rfind("format ascii", 0) == 0) {
            if (errorOut) {
                *errorOut = QStringLiteral(
                        "ASCII PLY is not supported by the light splat "
                        "importer (re-export as binary little-endian).");
            }
            return layout;
        } else if (token.rfind("element ", 0) == 0) {
            inVertexElement = token.find(" vertex ") != std::string::npos;
            if (inVertexElement) {
                layout.vertexCount = std::atoll(token.c_str() + 15);
            }
        } else if (token.rfind("property ", 0) == 0 && inVertexElement) {
            char type[64] = {0};
            char name[64] = {0};
            if (sscanf(token.c_str(), "property %63s %63s", type, name) != 2) {
                continue;
            }
            const int size = propertySizeBytes(type);
            if (size <= 0) {
                if (errorOut) {
                    *errorOut = QStringLiteral("Unsupported PLY type '%1'.")
                                        .arg(type);
                }
                return layout;
            }
            const std::string key(name);
            int slot = -1;
            if (key == "x") {
                slot = 0;
            } else if (key == "y") {
                slot = 1;
            } else if (key == "z") {
                slot = 2;
            } else if (key == "f_dc_0") {
                slot = 3;
            } else if (key == "f_dc_1") {
                slot = 4;
            } else if (key == "f_dc_2") {
                slot = 5;
            } else if (key == "opacity") {
                slot = 6;
            } else if (key == "scale_0") {
                slot = 7;
                layout.hasScale = true;
            } else if (key == "scale_1") {
                slot = 8;
            } else if (key == "scale_2") {
                slot = 9;
            }
            if (slot >= 0) {
                layout.offset[slot] = static_cast<int>(offsetCursor);
            }
            offsetCursor += size;
        } else if (token.rfind("end_header", 0) == 0) {
            layout.dataOffset = file.pos();
            break;
        }
    }

    layout.stride = static_cast<int>(offsetCursor);
    layout.ok = littleEndian && layout.vertexCount > 0 && layout.stride > 0 &&
                layout.offset[0] >= 0 && layout.offset[1] >= 0 &&
                layout.offset[2] >= 0 && layout.offset[3] >= 0 &&
                layout.offset[4] >= 0 && layout.offset[5] >= 0 &&
                layout.offset[6] >= 0;
    if (layout.ok && layout.stride > 0) {
        // dataOffset already points past "end_header\n"
    }
    if (!layout.ok && errorOut && errorOut->isEmpty()) {
        *errorOut = QStringLiteral(
                            "'%1' is not a binary little-endian 3DGS PLY with "
                            "x/y/z, "
                            "f_dc and opacity properties.")
                            .arg(plyPath);
    }
    return layout;
}

struct ExtractedVertex {
    float x;
    float y;
    float z;
    unsigned char r;
    unsigned char g;
    unsigned char b;
    float opacity;
    float scale;
};

//! Single pass over the vertex block; temp buffer ~24 B/vertex (10M gaussians
//! ~= 240 MB) so the opacity prune needs no second disk scan.
bool extractVertices(const QString& plyPath,
                     const PlyLayout& layout,
                     std::vector<ExtractedVertex>& out,
                     QString* errorOut) {
    QFile file(plyPath);
    if (!file.open(QIODevice::ReadOnly) || !file.seek(layout.dataOffset)) {
        if (errorOut) {
            *errorOut = QStringLiteral("Cannot read vertex data of '%1'.")
                                .arg(plyPath);
        }
        return false;
    }

    std::vector<unsigned char> raw;
    raw.resize(layout.stride);
    out.reserve(static_cast<size_t>(layout.vertexCount));
    out.resize(static_cast<size_t>(layout.vertexCount));

    for (long long i = 0; i < layout.vertexCount; ++i) {
        if (file.read(reinterpret_cast<char*>(raw.data()), layout.stride) !=
            layout.stride) {
            if (errorOut) {
                *errorOut = QStringLiteral("Truncated PLY vertex data.");
            }
            return false;
        }
        const auto readF = [&](int slot) -> float {
            const int off = layout.offset[slot];
            float v = 0.0f;
            std::memcpy(&v, raw.data() + off, sizeof(float));
            return v;
        };

        ExtractedVertex& v = out[static_cast<size_t>(i)];
        v.x = readF(0);
        v.y = readF(1);
        v.z = readF(2);
        const float fdc0 = readF(3);
        const float fdc1 = readF(4);
        const float fdc2 = readF(5);
        // SH0 DC term -> view-independent color
        v.r = static_cast<unsigned char>(
                std::clamp(0.5f + kSH0Coefficient * fdc0, 0.0f, 1.0f) * 255.0f);
        v.g = static_cast<unsigned char>(
                std::clamp(0.5f + kSH0Coefficient * fdc1, 0.0f, 1.0f) * 255.0f);
        v.b = static_cast<unsigned char>(
                std::clamp(0.5f + kSH0Coefficient * fdc2, 0.0f, 1.0f) * 255.0f);
        const float rawOpacity = readF(6);
        v.opacity = 1.0f / (1.0f + std::exp(-rawOpacity));
        // 3DGS stores scales in log space; the isotropic splat size is the
        // largest axis in linear (meters). Missing axes fall back to scale_0.
        if (layout.hasScale) {
            const float s0 = std::exp(readF(7));
            const float s1 = layout.offset[8] >= 0 ? std::exp(readF(8)) : s0;
            const float s2 = layout.offset[9] >= 0 ? std::exp(readF(9)) : s0;
            v.scale = std::max(s0, std::max(s1, s2));
        } else {
            v.scale = 0.0f;
        }
    }
    return true;
}

}  // namespace

QString resolveSplatPath(const QString& picked) {
    QFileInfo info(picked);
    if (!info.isDir()) {
        return picked;
    }
    // Run directory or step-*.ckpt: upstream --resume resolves either to its
    // splat.ply, so the import dialog accepts them under the same rule.
    const QDir dir(picked);
    const QString splat = dir.filePath(QStringLiteral("splat.ply"));
    return QFileInfo::exists(splat) ? splat : QString();
}

ccHObject* loadSplatPointCloud(const QString& plyPath, QString* errorOut) {
    const PlyLayout layout = parsePlyHeader(plyPath, errorOut);
    if (!layout.ok) {
        return nullptr;
    }

    std::vector<ExtractedVertex> vertices;
    if (!extractVertices(plyPath, layout, vertices, errorOut)) {
        return nullptr;
    }

    std::vector<size_t> validIndices;
    validIndices.reserve(vertices.size() / 2);
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (vertices[i].opacity > kOpacityThreshold) {
            validIndices.push_back(i);
        }
    }
    if (validIndices.empty()) {
        if (errorOut) {
            *errorOut =
                    QStringLiteral(
                            "No gaussians above the opacity threshold (%1).")
                            .arg(kOpacityThreshold);
        }
        return nullptr;
    }

    int step = 1;
    int displayCount = static_cast<int>(validIndices.size());
    if (displayCount > kMaxDisplayPoints) {
        // Keep the DB interactive for 10M-gaussian runs; the full count is
        // stored in metadata.
        step = (displayCount + kMaxDisplayPoints - 1) / kMaxDisplayPoints;
        displayCount = (displayCount + step - 1) / step;
    }

    auto* cloud = new ccPointCloud(QFileInfo(plyPath).baseName());
    if (!cloud->reserve(static_cast<unsigned>(displayCount))) {
        delete cloud;
        if (errorOut) {
            *errorOut = QStringLiteral("Not enough memory for %1 points.")
                                .arg(displayCount);
        }
        return nullptr;
    }
    const bool hasColors = cloud->reserveTheRGBTable();

    auto* opacitySF = new ccScalarField("Opacity");
    auto* scaleSF = new ccScalarField("Scale");
    const bool hasOpacitySF =
            opacitySF->reserveSafe(static_cast<unsigned>(displayCount));
    const bool hasScaleSF =
            scaleSF->reserveSafe(static_cast<unsigned>(displayCount));

    for (int vi = 0; vi < displayCount; ++vi) {
        const ExtractedVertex& v = vertices[validIndices[vi * step]];
        cloud->addPoint(CCVector3(v.x, v.y, v.z));
        if (hasColors) {
            cloud->addRGBColor(v.r, v.g, v.b);
        }
        if (hasOpacitySF) {
            opacitySF->addElement(static_cast<ScalarType>(v.opacity));
        }
        if (hasScaleSF) {
            scaleSF->addElement(static_cast<ScalarType>(v.scale));
        }
    }

    if (hasOpacitySF) {
        opacitySF->computeMinAndMax();
        cloud->addScalarField(opacitySF);
    } else {
        opacitySF->release();
    }
    if (hasScaleSF) {
        scaleSF->computeMinAndMax();
        cloud->addScalarField(scaleSF);
    } else {
        scaleSF->release();
    }

    if (hasColors) {
        cloud->showColors(true);
    }
    cloud->setVisible(true);
    cloud->setEnabled(true);
    // Auto-enable the isotropic point-gaussian splat rendering: per-point
    // size comes from the "Scale" field through the mapper's ScaleArray, so
    // the uniform radius acts as a global multiplier (1.0 = physical meters).
    // Entity state, not display state — both the plugin dialog and the
    // spirula.* RPC import path pick it up unchanged.
    cloud->setPointGaussianEnabled(true);
    cloud->setPointGaussianShaderPreset(
            static_cast<int>(ccDrawableObject::PG_PLAIN_CIRCLE));
    cloud->setPointGaussianRadius(1.0);
    cloud->setPointGaussianScaleArray("Scale");
    cloud->setPointGaussianEmissive(false);
    cloud->setMetaData(QStringLiteral("Spirula/Source"), plyPath);
    // Marks this cloud as a gaussian-splat model: qSpirula auto-enables the
    // point-gaussian splat rendering for it (per-point size = Scale field).
    cloud->setMetaData(QStringLiteral("Spirula/GaussianModel"), true);
    cloud->setMetaData(QStringLiteral("Spirula/TotalGaussians"),
                       static_cast<unsigned long long>(validIndices.size()));
    cloud->setMetaData(QStringLiteral("Spirula/ImportedGaussians"),
                       displayCount);
    return cloud;
}

ccHObject* loadMesh(const QString& plyPath, QString* errorOut) {
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::NO_DIALOG_AUTO_SHIFT;
    CC_FILE_ERROR err = CC_FERR_NO_ERROR;
    ccHObject* loaded = FileIOFilter::LoadFromFile(plyPath, params, err);
    if (!loaded && errorOut) {
        *errorOut = QStringLiteral("PLY load failed for '%1' (error %2).")
                            .arg(plyPath)
                            .arg(static_cast<int>(err));
    }
    return loaded;
}

}  // namespace spirula
