// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QImageReader>
#include <QRegularExpression>
#include <QVector3D>

// CV_CORE_LIB
#include <CVTools.h>

// CV_DB_LIB
#include "IoUtils.h"
#include "assimp/material.h"
#include "assimp/mesh.h"
#include "assimp/metadata.h"
#include "assimp/scene.h"
#include "ecvHObjectCaster.h"
#include "ecvMaterialSet.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"

namespace {
QImage _getEmbeddedTexture(unsigned int inTextureIndex,
                           const aiScene *inScene) {
    QImage image;

    if (inScene->mNumTextures == 0) {
        CVLog::Warning(
                QStringLiteral("[qMeshIO] Scene requests embedded texture, but "
                               "there are none"));
        return image;
    }

    auto texture = inScene->mTextures[inTextureIndex];

    // From assimp: "If mHeight is zero the texture is compressed"
    bool isCompressed = (texture->mHeight == 0);

    if (!isCompressed) {
        CVLog::Warning(
                QStringLiteral("[qMeshIO] Uncompressed embedded textures not "
                               "yet implemented"));
        return image;
    }

    // From assimp: "mWidth specifies the size of the memory area pcData is
    // pointing to, in bytes"
    auto dataSize = static_cast<const int32_t>(texture->mWidth);

    const QByteArray imageDataByteArray(
            reinterpret_cast<const char *>(texture->pcData), dataSize);

    return QImage::fromData(imageDataByteArray);
}

QImage _getTextureFromFile(const QString &inPath,
                           const QString &inTexturePath) {
    QString cPath = QStringLiteral("%1/%2").arg(inPath, inTexturePath);

    cPath = CVTools::ToNativeSeparators(cPath);

    if (!QFile::exists(cPath)) {
        CVLog::Warning(QStringLiteral("[qMeshIO] Material not found: '%1'")
                               .arg(cPath));
        return {};
    }

    QImageReader reader(cPath);

    QImage image = reader.read();
    if (image.isNull()) {
        CVLog::Warning(
                QString("[_getTextureFromFile] failed to read image %1, %2")
                        .arg(cPath)
                        .arg(reader.errorString()));
    }
    return image;
}

inline ecvColor::Rgbaf _convertColour(const aiColor4D &inColour) {
    return ecvColor::Rgbaf{inColour.r, inColour.g, inColour.b, inColour.a};
}

// Map all the material properties we know about from assimp
void _assignMaterialProperties(aiMaterial *inAIMaterial,
                               ccMaterial::Shared &inCCMaterial) {
    aiColor4D colour;

    if (inAIMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, colour) == AI_SUCCESS) {
        inCCMaterial->setDiffuse(_convertColour(colour));
    }

    if (inAIMaterial->Get(AI_MATKEY_COLOR_AMBIENT, colour) == AI_SUCCESS) {
        inCCMaterial->setAmbient(_convertColour(colour));
    }

    if (inAIMaterial->Get(AI_MATKEY_COLOR_SPECULAR, colour) == AI_SUCCESS) {
        inCCMaterial->setSpecular(_convertColour(colour));
    }

    if (inAIMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, colour) == AI_SUCCESS) {
        inCCMaterial->setEmission(_convertColour(colour));
    }

    ai_real property;

    if (inAIMaterial->Get(AI_MATKEY_SHININESS, property) == AI_SUCCESS) {
        inCCMaterial->setShininess(property);
    }

    if (inAIMaterial->Get(AI_MATKEY_OPACITY, property) == AI_SUCCESS) {
        inCCMaterial->setTransparency(property);
    }
}
}  // namespace

namespace IoUtils {
ccMaterialSet *createMaterialSetForMesh(const aiMesh *inMesh,
                                        const QString &inPath,
                                        const aiScene *inScene) {
    if (inScene->mNumMaterials == 0) {
        return nullptr;
    }

    unsigned int index = inMesh->mMaterialIndex;
    const auto aiMaterial = inScene->mMaterials[index];

    const aiString cName = aiMaterial->GetName();

    auto newMaterial = ccMaterial::Shared(new ccMaterial(cName.C_Str()));

    CVLog::PrintDebug(QStringLiteral("[qMeshIO] Creating material '%1'")
                              .arg(newMaterial->getName()));

    // ========================================================================
    // Load all PBR texture types
    // ========================================================================

    // Helper lambda to load a texture for a specific type
    auto loadTexture = [&](aiTextureType aiType,
                           ccMaterial::TextureMapType ccType,
                           const char *typeName) {
        if (aiMaterial->GetTextureCount(aiType) > 0) {
            aiString texturePath;

            if (aiMaterial->GetTexture(aiType, 0, &texturePath) == AI_SUCCESS) {
                static QRegularExpression sRegExp("^\\*(?<index>[0-9]+)$");
                auto match = sRegExp.match(texturePath.C_Str());

                QImage image;
                QString path =
                        CVTools::ToNativeSeparators(QStringLiteral("%1/%2").arg(
                                inPath, texturePath.C_Str()));

                // Try different loading methods
                if (match.hasMatch()) {
                    // Embedded texture by index
                    const QString cIndex = match.captured("index");
                    image = _getEmbeddedTexture(cIndex.toUInt(), inScene);
                } else if (!QFile::exists(path) && inScene->HasTextures()) {
                    // Find embedded texture by name
                    unsigned int cIndex = 0;
                    for (unsigned int i = 0; i < inScene->mNumTextures; ++i) {
                        aiString textureName = inScene->mTextures[i]->mFilename;
                        if (textureName == texturePath) {
                            cIndex = i;
                            break;
                        }
                    }
                    image = _getEmbeddedTexture(cIndex, inScene);
                } else {
                    // Load from file
                    image = _getTextureFromFile(inPath, texturePath.C_Str());
                }

                if (!image.isNull()) {
                    // Use new multi-texture API
                    if (newMaterial->loadAndSetTextureMap(ccType, path)) {
                        CVLog::PrintDebug(
                                QStringLiteral(
                                        "[qMeshIO] Loaded %1 texture: %2")
                                        .arg(typeName, path));
                    }

                    // For diffuse, also set legacy texture for backward
                    // compatibility
                    if (ccType == ccMaterial::TextureMapType::DIFFUSE) {
                        newMaterial->setTexture(image, path, false);
                    }
                } else {
                    CVLog::Warning(
                            QStringLiteral(
                                    "[qMeshIO] Failed to load %1 texture: %2")
                                    .arg(typeName, path));
                }
            }
        }
    };

    // Load all supported texture types
    loadTexture(aiTextureType_DIFFUSE, ccMaterial::TextureMapType::DIFFUSE,
                "Diffuse");
    loadTexture(aiTextureType_AMBIENT, ccMaterial::TextureMapType::AMBIENT,
                "Ambient/AO");
    loadTexture(aiTextureType_SPECULAR, ccMaterial::TextureMapType::SPECULAR,
                "Specular");
    loadTexture(aiTextureType_NORMALS, ccMaterial::TextureMapType::NORMAL,
                "Normal");
    loadTexture(aiTextureType_HEIGHT, ccMaterial::TextureMapType::NORMAL,
                "Height/Normal");  // Height maps can be used as normals
    loadTexture(aiTextureType_EMISSIVE, ccMaterial::TextureMapType::EMISSIVE,
                "Emissive");
    loadTexture(aiTextureType_OPACITY, ccMaterial::TextureMapType::OPACITY,
                "Opacity");
    loadTexture(aiTextureType_DISPLACEMENT,
                ccMaterial::TextureMapType::DISPLACEMENT, "Displacement");
    loadTexture(aiTextureType_REFLECTION,
                ccMaterial::TextureMapType::REFLECTION, "Reflection");
    loadTexture(aiTextureType_SHININESS, ccMaterial::TextureMapType::SHININESS,
                "Shininess");
    loadTexture(aiTextureType_METALNESS, ccMaterial::TextureMapType::METALLIC,
                "Metallic");
    loadTexture(aiTextureType_DIFFUSE_ROUGHNESS,
                ccMaterial::TextureMapType::ROUGHNESS, "Roughness");

    _assignMaterialProperties(aiMaterial, newMaterial);

    ccMaterialSet *materialSet = new ccMaterialSet("Materials");

    materialSet->addMaterial(newMaterial);

    return materialSet;
}

ccMesh *newCCMeshFromAIMesh(const aiMesh *inMesh) {
    auto newPC = new ccPointCloud("Vertices");
    auto newMesh = new ccMesh(newPC);

    QString name(inMesh->mName.C_Str());

    if (name.isEmpty()) {
        name = QStringLiteral("Mesh");
    }

    CVLog::Print(QStringLiteral("[qMeshIO] Mesh '%1' has %2 verts & %3 faces")
                         .arg(name,
                              QLocale::system().toString(inMesh->mNumVertices),
                              QLocale::system().toString(inMesh->mNumFaces)));

    if (!inMesh->HasPositions() || !inMesh->HasFaces()) {
        CVLog::Warning(
                QStringLiteral(
                        "[qMeshIO] Mesh '%1' does not have vertices or faces")
                        .arg(name));

        delete newPC;
        delete newMesh;

        return nullptr;
    }

    // reserve memory for points and mesh (because we need to do this before
    // other memory allocations)
    newPC->reserveThePointsTable(inMesh->mNumVertices);

    if (inMesh->HasFaces()) {
        newMesh->reserve(inMesh->mNumFaces);
    }

    // vertex colors
    bool hasVertexColors = inMesh->HasVertexColors(0);
    if (hasVertexColors) {
        bool allocated = newPC->reserveTheRGBTable();
        if (!allocated) {
            CVLog::Warning(
                    QStringLiteral(
                            "[qMeshIO] Cannot allocate colors for mesh '%1'")
                            .arg(name));
        }
    }

    // normals
    if (inMesh->HasNormals()) {
        bool allocated = newPC->reserveTheNormsTable();

        if (!allocated) {
            CVLog::Warning(
                    QStringLiteral(
                            "[qMeshIO] Cannot allocate normals for mesh '%1'")
                            .arg(name));
        }
    }

    // texture coordinates
    bool hasTextureCoordinates = inMesh->HasTextureCoords(0);

    TextureCoordsContainer *texCoords = nullptr;

    if (hasTextureCoordinates) {
        texCoords = new TextureCoordsContainer;

        texCoords->reserve(inMesh->mNumVertices);

        bool allocated = texCoords->isAllocated();

        allocated &= newMesh->reservePerTriangleTexCoordIndexes();
        allocated &= newMesh->reservePerTriangleMtlIndexes();

        if (!allocated) {
            delete texCoords;
            hasTextureCoordinates = false;
            CVLog::Warning(QStringLiteral("[qMeshIO] Cannot allocate texture "
                                          "coordinates for mesh '%1'")
                                   .arg(name));
        } else {
            newMesh->setTexCoordinatesTable(texCoords);
        }
    }

    // vertices
    for (unsigned int i = 0; i < inMesh->mNumVertices; ++i) {
        const aiVector3D &point = inMesh->mVertices[i];

        CCVector3 point2(static_cast<PointCoordinateType>(point.x),
                         static_cast<PointCoordinateType>(point.y),
                         static_cast<PointCoordinateType>(point.z));

        newPC->addPoint(point2);

        // colors
        if (newPC->hasColors()) {
            const aiColor4D &colors = inMesh->mColors[0][i];

            ecvColor::Rgb color(static_cast<ColorCompType>(colors.r * 255),
                                static_cast<ColorCompType>(colors.g * 255),
                                static_cast<ColorCompType>(colors.b * 255));

            newPC->addRGBColor(color);
        }

        // normals
        if (newPC->hasNormals()) {
            const aiVector3D &normal = inMesh->mNormals[i];

            CCVector3 normal2(static_cast<PointCoordinateType>(normal.x),
                              static_cast<PointCoordinateType>(normal.y),
                              static_cast<PointCoordinateType>(normal.z));

            newPC->addNorm(normal2);
        }

        // texture coordinates
        if (hasTextureCoordinates) {
            const aiVector3D &texCoord = inMesh->mTextureCoords[0][i];

            const TexCoords2D coord{texCoord.x, texCoord.y};

            texCoords->addElement(coord);
        }
    }

    newPC->setEnabled(false);

    // faces
    if (inMesh->HasFaces()) {
        newMesh->reserve(inMesh->mNumFaces);

        for (unsigned int i = 0; i < inMesh->mNumFaces; ++i) {
            const aiFace &face = inMesh->mFaces[i];

            if (face.mNumIndices != 3) {
                continue;
            }

            newMesh->addTriangle(face.mIndices[0], face.mIndices[1],
                                 face.mIndices[2]);

            // texture coordinates
            if (hasTextureCoordinates) {
                newMesh->addTriangleMtlIndex(0);

                newMesh->addTriangleTexCoordIndexes(
                        static_cast<int>(face.mIndices[0]),
                        static_cast<int>(face.mIndices[1]),
                        static_cast<int>(face.mIndices[2]));
            }
        }
    }

    if (newMesh->size() == 0) {
        CVLog::Warning(
                QStringLiteral("[qMeshIO] Mesh '%1' does not have any faces")
                        .arg(name));

        delete newPC;
        delete newMesh;

        return nullptr;
    }

    newMesh->setName(name);
    newMesh->setVisible(true);

    if (!newPC->hasNormals()) {
        CVLog::Warning(
                QStringLiteral("[qMeshIO] Mesh '%1' does not have normals - "
                               "will compute them per vertex automatically!")
                        .arg(name));

        newMesh->computeNormals(true);
    }

    newMesh->showNormals(true);
    newMesh->showColors(hasVertexColors);
    newMesh->addChild(newPC);

    return newMesh;
}

ccGLMatrix convertMatrix(const aiMatrix4x4 &inAssimpMatrix) {
    const int cWidth = 4;
    const int cHeight = 4;

    PointCoordinateType data[OPENGL_MATRIX_SIZE];

    for (unsigned int i = 0; i < cWidth; ++i) {
        for (unsigned int j = 0; j < cHeight; ++j) {
            data[j * cHeight + i] =
                    static_cast<PointCoordinateType>(inAssimpMatrix[i][j]);
        }
    }

    return ccGLMatrix(data);
}

QVariant convertMetaValueToVariant(aiMetadata *inData,
                                   unsigned int inValueIndex) {
    QVariant metaValue;

    switch (inData->mValues[inValueIndex].mType) {
        case AI_BOOL: {
            bool value = false;

            inData->Get<bool>(inValueIndex, value);

            metaValue = value;
            break;
        }

        case AI_INT32: {
            int32_t value = 0;

            inData->Get<int32_t>(inValueIndex, value);

            metaValue = value;
            break;
        }

        case AI_UINT64: {
            uint64_t value = 0;

            inData->Get<uint64_t>(inValueIndex, value);

            metaValue = static_cast<qulonglong>(value);
            break;
        }

        case AI_FLOAT: {
            float value = 0;

            inData->Get<float>(inValueIndex, value);

            metaValue = value;
            break;
        }

        case AI_DOUBLE: {
            double value = 0;

            inData->Get<double>(inValueIndex, value);

            metaValue = value;
            break;
        }

        case AI_AISTRING: {
            aiString value;

            inData->Get<aiString>(inValueIndex, value);

            metaValue = value.C_Str();
            break;
        }

        case AI_AIVECTOR3D: {
            aiVector3D value;

            inData->Get<aiVector3D>(inValueIndex, value);

            metaValue = QVector3D(value.x, value.y, value.z);
            break;
        }

        case AI_META_MAX:
        case FORCE_32BIT: {
            // This is necessary to avoid a warning.
            // Assimp doesn't use enum type specifiers.
            // It uses this odd trick w/FORCE_32BIT to force the type of the
            // enum.
            break;
        }
    }

    return metaValue;
}
}  // namespace IoUtils
