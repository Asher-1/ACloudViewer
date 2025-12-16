// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvMaterialSet.h"

#include "ecvViewportParameters.h"

// Qt
#include <QFileInfo>
#include <QImage>
#include <QSet>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

// System
#include <set>

ccMaterialSet::ccMaterialSet(const QString& name /*=QString()*/)
    : std::vector<ccMaterial::CShared>(), CCShareable(), ccHObject(name) {
    setFlagState(CC_LOCKED, true);
}

int ccMaterialSet::findMaterialByName(QString mtlName) const {
    CVLog::PrintDebug(QString("[ccMaterialSet::findMaterialByName] Query: ") +
                      mtlName);

    int i = 0;
    for (ccMaterialSet::const_iterator it = begin(); it != end(); ++it, ++i) {
        ccMaterial::CShared mtl = *it;
        CVLog::PrintDebug(
                QString("\tmaterial #%1 name: %2").arg(i).arg(mtl->getName()));
        if (mtl->getName() == mtlName) return i;
    }

    return -1;
}

int ccMaterialSet::findMaterialByUniqueID(QString uniqueID) const {
    CVLog::PrintDebug(
            QString("[ccMaterialSet::findMaterialByUniqueID] Query: ") +
            uniqueID);

    int i = 0;
    for (ccMaterialSet::const_iterator it = begin(); it != end(); ++it, ++i) {
        ccMaterial::CShared mtl = *it;
        CVLog::PrintDebug(QString("\tmaterial #%1 ID: %2")
                                  .arg(i)
                                  .arg(mtl->getUniqueIdentifier()));
        if (mtl->getUniqueIdentifier() == uniqueID) return i;
    }

    return -1;
}

int ccMaterialSet::addMaterial(ccMaterial::CShared mtl,
                               bool allowDuplicateNames /*=false*/) {
    if (!mtl) {
        // invalid input material
        return -1;
    }

    // material already exists?
    int previousIndex = findMaterialByName(mtl->getName());
    // DGM: warning, the materials may have the same name, but they may be
    // different in reality (other texture, etc.)!
    if (previousIndex >= 0) {
        const ccMaterial::CShared& previousMtl = (*this)[previousIndex];
        if (!previousMtl->compare(*mtl)) {
            // in fact the material is a bit different
            previousIndex = -1;
            if (!allowDuplicateNames) {
                // generate a new name
                static const unsigned MAX_ATTEMPTS = 100;
                for (unsigned i = 1; i < MAX_ATTEMPTS; i++) {
                    QString newMtlName =
                            previousMtl->getName() + QString("_%1").arg(i);
                    if (findMaterialByName(newMtlName) < 0) {
                        // we duplicate the material and we change its name
                        ccMaterial::Shared newMtl(new ccMaterial(*mtl));
                        newMtl->setName(newMtlName);
                        mtl = newMtl;
                        break;
                    }
                }
            }
        }
    }
    if (previousIndex >= 0 && !allowDuplicateNames) return previousIndex;

    try {
        push_back(mtl);
    } catch (const std::bad_alloc&) {
        // not enough memory
        return -1;
    }

    return static_cast<int>(size()) - 1;
}

// MTL PARSER INSPIRED FROM KIXOR.NET "objloader"
// (http://www.kixor.net/dev/objloader/)
bool ccMaterialSet::ParseMTL(QString path,
                             const QString& filename,
                             ccMaterialSet& materials,
                             QStringList& errors) {
    // open mtl file
    QString fullPathFilename = path + QString('/') + filename;
    QFile file(fullPathFilename);
    if (!file.open(QFile::ReadOnly)) {
        errors << QString("Error reading file: %1").arg(filename);
        return false;
    }

    // update path (if the input filename has already a relative path)
    path = QFileInfo(fullPathFilename).absolutePath();

    QTextStream stream(&file);

    QString currentLine = stream.readLine();
    unsigned currentLineIndex = 0;
    ccMaterial::Shared currentMaterial(0);
    while (!currentLine.isNull()) {
        ++currentLineIndex;

        QStringList tokens = qtCompatSplitRegex(currentLine, "\\s+",
                                                QtCompat::SkipEmptyParts);

        // skip comments & empty lines
        if (tokens.empty() ||
            tokens.front().startsWith('/', Qt::CaseInsensitive) ||
            tokens.front().startsWith('#', Qt::CaseInsensitive)) {
            currentLine = stream.readLine();
            continue;
        }

        // start material
        if (tokens.front() == "newmtl") {
            // push the previous material (if any)
            if (currentMaterial) {
                materials.addMaterial(currentMaterial);
                currentMaterial = ccMaterial::Shared(0);
            }

            // get the name
            QString materialName =
                    currentLine.mid(7).trimmed();  // we must take the whole
                                                   // line! (see OBJ filter)
            if (materialName.isEmpty()) materialName = "undefined";
            currentMaterial = ccMaterial::Shared(new ccMaterial(materialName));

        } else if (currentMaterial)  // we already have a "current" material
        {
            // ambient
            if (tokens.front() == "Ka") {
                if (tokens.size() > 3) {
                    ecvColor::Rgbaf ambient(tokens[1].toFloat(),
                                            tokens[2].toFloat(),
                                            tokens[3].toFloat(), 1.0f);
                    currentMaterial->setAmbient(ambient);
                }
            }

            // diff
            else if (tokens.front() == "Kd") {
                if (tokens.size() > 3) {
                    ecvColor::Rgbaf diffuse(tokens[1].toFloat(),
                                            tokens[2].toFloat(),
                                            tokens[3].toFloat(), 1.0f);
                    currentMaterial->setDiffuse(diffuse);
                }
            }

            // specular
            else if (tokens.front() == "Ks") {
                if (tokens.size() > 3) {
                    ecvColor::Rgbaf specular(tokens[1].toFloat(),
                                             tokens[2].toFloat(),
                                             tokens[3].toFloat(), 1.0f);
                    currentMaterial->setSpecular(specular);
                }
            }

            // emission
            else if (tokens.front() == "Ke") {
                if (tokens.size() > 3) {
                    ecvColor::Rgbaf emission(tokens[1].toFloat(),
                                             tokens[2].toFloat(),
                                             tokens[3].toFloat(), 1.0f);
                    currentMaterial->setEmission(emission);
                }
            }

            // shiny
            else if (tokens.front() == "Ns") {
                if (tokens.size() > 1)
                    currentMaterial->setShininess(tokens[1].toFloat());
            }
            // transparency (inverse of)
            else if (tokens.front() == "d") {
                if (tokens.size() > 1) {
                    float d = tokens[1].toFloat();
                    if (d == 0) {
                        CVLog::Warning(
                                QString("Material %1 'alpha' (=d) value is 0 "
                                        "(= fully transparent)")
                                        .arg(currentMaterial->getName()));
                    }
                    currentMaterial->setTransparency(d);
                }
            }
            // transparency
            else if (tokens.front() == "Tr") {
                if (tokens.size() > 1) {
                    float tr = tokens[1].toFloat();
                    if (tr == 1.0f) {
                        CVLog::Warning(
                                QString("Material %1 'transparency' (=Tr) "
                                        "value is 1 (= fully transparent)")
                                        .arg(currentMaterial->getName()));
                    }
                    currentMaterial->setTransparency(1.0f -
                                                     tokens[1].toFloat());
                }
            }
            // reflection
            else if (tokens.front() == "r") {
                // ignored
                // if (tokens.size() > 1)
                //	currentMaterial->reflect = tokens[1].toFloat();
            }
            // glossy
            else if (tokens.front() == "sharpness") {
                // ignored
                // if (tokens.size() > 1)
                //	currentMaterial->glossy = tokens[1].toFloat();
            }
            // refract index
            else if (tokens.front() == "Ni") {
                // ignored
                // if (tokens.size() > 1)
                //	currentMaterial->refract_index = tokens[1].toFloat();
            }
            // illumination type
            else if (tokens.front() == "illum") {
                if (tokens.size() > 1)
                    currentMaterial->setIllum(tokens[1].toInt());
            }
            // texture maps - support standard OBJ and PBR extensions
            else if (tokens.front() == "map_Ka" || tokens.front() == "map_Kd" ||
                     tokens.front() == "map_Ks" || tokens.front() == "map_Ke" ||
                     tokens.front() == "map_d" || tokens.front() == "map_Ns" ||
                     tokens.front() == "map_Bump" ||
                     tokens.front() == "map_bump" || tokens.front() == "bump" ||
                     tokens.front() == "norm" || tokens.front() == "map_Pr" ||
                     tokens.front() == "map_Pm" || tokens.front() == "map_Ps" ||
                     tokens.front() == "map_Pc" ||
                     tokens.front() == "map_Pcr" ||
                     tokens.front() == "map_aniso" ||
                     tokens.front() == "map_disp" || tokens.front() == "disp" ||
                     tokens.front() == "refl") {
                // Determine texture map type
                ccMaterial::TextureMapType mapType;
                QString mapCommand = tokens.front();

                if (mapCommand == "map_Kd") {
                    mapType = ccMaterial::TextureMapType::DIFFUSE;
                } else if (mapCommand == "map_Ka") {
                    mapType = ccMaterial::TextureMapType::AMBIENT;
                } else if (mapCommand == "map_Ks") {
                    mapType = ccMaterial::TextureMapType::SPECULAR;
                } else if (mapCommand == "map_Ke") {
                    mapType = ccMaterial::TextureMapType::EMISSIVE;
                } else if (mapCommand == "map_d") {
                    mapType = ccMaterial::TextureMapType::OPACITY;
                } else if (mapCommand == "map_Ns") {
                    // map_Ns is shininess/glossiness map
                    // Can be mapped to roughness (inverted) or used as-is
                    mapType = ccMaterial::TextureMapType::SHININESS;
                } else if (mapCommand == "map_Bump" ||
                           mapCommand == "map_bump" || mapCommand == "bump" ||
                           mapCommand == "norm") {
                    mapType = ccMaterial::TextureMapType::NORMAL;
                } else if (mapCommand == "map_Pr") {
                    mapType = ccMaterial::TextureMapType::ROUGHNESS;
                } else if (mapCommand == "map_Pm") {
                    mapType = ccMaterial::TextureMapType::METALLIC;
                } else if (mapCommand == "map_Ps") {
                    mapType = ccMaterial::TextureMapType::SHEEN;
                } else if (mapCommand == "map_Pc") {
                    mapType = ccMaterial::TextureMapType::CLEARCOAT;
                } else if (mapCommand == "map_Pcr") {
                    mapType = ccMaterial::TextureMapType::CLEARCOAT_ROUGHNESS;
                } else if (mapCommand == "map_aniso") {
                    mapType = ccMaterial::TextureMapType::ANISOTROPY;
                } else if (mapCommand == "map_disp" || mapCommand == "disp") {
                    mapType = ccMaterial::TextureMapType::DISPLACEMENT;
                } else if (mapCommand == "refl") {
                    mapType = ccMaterial::TextureMapType::REFLECTION;
                } else {
                    // Fallback to diffuse
                    mapType = ccMaterial::TextureMapType::DIFFUSE;
                }

                // Extract texture filename
                // DGM: in case there's hidden or space characters at the
                // beginning of the line...
                int shift = currentLine.indexOf(mapCommand, 0);
                QString textureFilename =
                        (shift + mapCommand.length() + 1 < currentLine.size()
                                 ? currentLine
                                           .mid(shift + mapCommand.length() + 1)
                                           .trimmed()
                                 : QString());

                // Filter out MTL texture options (e.g., -bm, -blendu, -blendv,
                // -boost, -mm, -o, -s, -t, -texres, -clamp, -imfchan) These
                // options start with '-' and may have numeric parameters
                QStringList parts =
                        textureFilename.split(' ', Qt::SkipEmptyParts);
                QString actualFilename;
                for (const QString& part : parts) {
                    // Skip options that start with '-' and their numeric
                    // parameters
                    if (part.startsWith('-')) {
                        continue;  // Skip option flag
                    }
                    // Check if this is a numeric parameter following an option
                    bool isNumber = false;
                    part.toDouble(&isNumber);
                    if (isNumber && actualFilename.isEmpty()) {
                        continue;  // Skip numeric parameter
                    }
                    // This should be the actual filename
                    actualFilename = part;
                    break;
                }

                // If we found a filename after filtering, use it
                if (!actualFilename.isEmpty()) {
                    textureFilename = actualFilename;
                }

                // remove any quotes around the filename (Photoscan 1.4 bug)
                if (textureFilename.startsWith("\"")) {
                    textureFilename =
                            textureFilename.right(textureFilename.size() - 1);
                }
                if (textureFilename.endsWith("\"")) {
                    textureFilename =
                            textureFilename.left(textureFilename.size() - 1);
                }

                // Normalize path separators (convert backslashes to forward
                // slashes)
                textureFilename = textureFilename.replace('\\', '/');

                QString fullTexName = path + QString('/') + textureFilename;

                // Load texture using new multi-texture API
                if (!currentMaterial->loadAndSetTextureMap(mapType,
                                                           fullTexName)) {
                    errors << QString("Failed to load texture file: %1 (type: "
                                      "%2)")
                                      .arg(fullTexName)
                                      .arg(mapCommand);
                }
            }
            // PBR scalar parameters
            else if (tokens.front() == "Pm") {
                // Metallic factor
                if (tokens.size() > 1)
                    currentMaterial->setMetallic(tokens[1].toFloat());
            } else if (tokens.front() == "Pr") {
                // Roughness factor
                if (tokens.size() > 1)
                    currentMaterial->setRoughness(tokens[1].toFloat());
            } else if (tokens.front() == "Ps") {
                // Sheen factor
                if (tokens.size() > 1)
                    currentMaterial->setSheen(tokens[1].toFloat());
            } else if (tokens.front() == "Pc") {
                // Clearcoat factor
                if (tokens.size() > 1)
                    currentMaterial->setClearcoat(tokens[1].toFloat());
            } else if (tokens.front() == "Pcr") {
                // Clearcoat roughness
                if (tokens.size() > 1)
                    currentMaterial->setClearcoatRoughness(tokens[1].toFloat());
            } else if (tokens.front() == "aniso") {
                // Anisotropy factor
                if (tokens.size() > 1)
                    currentMaterial->setAnisotropy(tokens[1].toFloat());
            } else if (tokens.front() == "Pa") {
                // Ambient Occlusion factor
                if (tokens.size() > 1)
                    currentMaterial->setAmbientOcclusion(tokens[1].toFloat());
            } else {
                errors << QString("Unknown command '%1' at line %2")
                                  .arg(tokens.front())
                                  .arg(currentLineIndex);
            }
        }

        currentLine = stream.readLine();
    }

    file.close();

    // don't forget to push the last material!
    if (currentMaterial) materials.addMaterial(currentMaterial);

    return true;
}

bool ccMaterialSet::saveAsMTL(QString path,
                              const QString& baseFilename,
                              QStringList& errors) const {
    // open mtl file
    QString filename = path + QString('/') + baseFilename + QString(".mtl");
    QFile file(filename);
    if (!file.open(QFile::WriteOnly)) {
        errors << QString("Error writing file: %1").arg(filename);
        return false;
    }
    QTextStream stream(&file);

    stream << "# Generated by CLOUDVIEWER " << endl;

    // texture filenames already used
    QMap<QString, QString> absFilenamesSaved;
    QSet<QString> filenamesUsed;

    // Helper function to save texture and return relative filename
    auto saveTextureFile = [&](const QString& absFilename,
                               size_t matIndex) -> QString {
        if (absFilename.isEmpty()) return QString();

        // if the file has not already been saved
        if (!absFilenamesSaved.contains(absFilename)) {
            QFileInfo fileInfo(absFilename);
            QString texName = fileInfo.fileName();
            if (texName.isEmpty()) {
                // Generate a name if filename is empty
                texName = QString("tex_%1.jpg").arg(matIndex);
            } else if (fileInfo.suffix().isEmpty()) {
                texName += QString(".jpg");
            }

            // make sure that the local filename is unique!
            QString originalTexName = texName;
            int counter = 0;
            while (filenamesUsed.contains(texName)) {
                QString baseName = fileInfo.completeBaseName();
                QString suffix =
                        fileInfo.suffix().isEmpty() ? "jpg" : fileInfo.suffix();
                texName = QString("%1_%2.%3")
                                  .arg(baseName)
                                  .arg(counter++)
                                  .arg(suffix);
            }
            filenamesUsed.insert(texName);

            QString destFilename = path + QString('/') + texName;
            QImage texture = ccMaterial::GetTexture(absFilename);
            if (!texture.isNull() && texture.save(destFilename)) {
                // Normalize path separators (use forward slashes for MTL
                // format)
                QString normalizedTexName = texName.replace('\\', '/');
                absFilenamesSaved[absFilename] = normalizedTexName;
            } else {
                errors << QString("Failed to save the texture file '%1' to "
                                  "'%2'!")
                                  .arg(absFilename, destFilename);
                return QString();
            }
        }

        if (absFilenamesSaved.contains(absFilename)) {
            // Normalize path separators (use forward slashes for MTL format)
            QString relativeFilename = absFilenamesSaved[absFilename];
            return relativeFilename.replace('\\', '/');
        }
        return QString();
    };

    size_t matIndex = 0;
    for (ccMaterialSet::const_iterator it = begin(); it != end();
         ++it, ++matIndex) {
        ccMaterial::CShared mtl = *it;
        stream << endl << "newmtl " << mtl->getName() << endl;

        // Basic material colors
        const ecvColor::Rgbaf& Ka = mtl->getAmbient();
        const ecvColor::Rgbaf& Kd = mtl->getDiffuseFront();
        const ecvColor::Rgbaf& Ks = mtl->getSpecular();
        const ecvColor::Rgbaf& Ke = mtl->getEmission();

        stream << "Ka " << Ka.r << " " << Ka.g << " " << Ka.b << endl;
        stream << "Kd " << Kd.r << " " << Kd.g << " " << Kd.b << endl;
        stream << "Ks " << Ks.r << " " << Ks.g << " " << Ks.b << endl;

        // Emission color (if not black/default)
        if (Ke.r > 0.0f || Ke.g > 0.0f || Ke.b > 0.0f) {
            stream << "Ke " << Ke.r << " " << Ke.g << " " << Ke.b << endl;
        }

        // Transparency (use diffuse alpha, or ambient if diffuse is not set)
        float transparency = Kd.a;
        if (transparency < 1.0f) {
            stream << "d " << transparency << endl;
        }

        // Illumination model
        stream << "illum " << mtl->getIllum() << endl;

        // Shininess
        stream << "Ns " << mtl->getShininessFront() << endl;

        // PBR scalar parameters (only save if non-default values)
        if (mtl->getMetallic() > 0.0f) {
            stream << "Pm " << mtl->getMetallic() << endl;
        }
        if (mtl->getRoughness() != 0.5f) {
            stream << "Pr " << mtl->getRoughness() << endl;
        }
        if (mtl->getSheen() > 0.0f) {
            stream << "Ps " << mtl->getSheen() << endl;
        }
        if (mtl->getClearcoat() > 0.0f) {
            stream << "Pc " << mtl->getClearcoat() << endl;
        }
        if (mtl->getClearcoatRoughness() > 0.0f) {
            stream << "Pcr " << mtl->getClearcoatRoughness() << endl;
        }
        if (mtl->getAnisotropy() > 0.0f) {
            stream << "aniso " << mtl->getAnisotropy() << endl;
        }
        if (mtl->getAmbientOcclusion() > 0.0f) {
            stream << "Pa " << mtl->getAmbientOcclusion() << endl;
        }

        // Save all texture maps using the multi-texture API
        auto allTextures = mtl->getAllTextureFilenames();
        for (const auto& texPair : allTextures) {
            ccMaterial::TextureMapType mapType = texPair.first;
            QString absFilename = texPair.second;

            if (absFilename.isEmpty()) continue;

            QString relativeFilename = saveTextureFile(absFilename, matIndex);
            if (relativeFilename.isEmpty()) continue;

            // Map texture type to MTL command
            QString mapCommand;
            switch (mapType) {
                case ccMaterial::TextureMapType::DIFFUSE:
                    mapCommand = "map_Kd";
                    break;
                case ccMaterial::TextureMapType::AMBIENT:
                    mapCommand = "map_Ka";
                    break;
                case ccMaterial::TextureMapType::SPECULAR:
                    mapCommand = "map_Ks";
                    break;
                case ccMaterial::TextureMapType::EMISSIVE:
                    mapCommand = "map_Ke";
                    break;
                case ccMaterial::TextureMapType::OPACITY:
                    mapCommand = "map_d";
                    break;
                case ccMaterial::TextureMapType::SHININESS:
                    mapCommand = "map_Ns";
                    break;
                case ccMaterial::TextureMapType::NORMAL:
                    mapCommand = "map_Bump";
                    break;
                case ccMaterial::TextureMapType::ROUGHNESS:
                    mapCommand = "map_Pr";
                    break;
                case ccMaterial::TextureMapType::METALLIC:
                    mapCommand = "map_Pm";
                    break;
                case ccMaterial::TextureMapType::SHEEN:
                    mapCommand = "map_Ps";
                    break;
                case ccMaterial::TextureMapType::CLEARCOAT:
                    mapCommand = "map_Pc";
                    break;
                case ccMaterial::TextureMapType::CLEARCOAT_ROUGHNESS:
                    mapCommand = "map_Pcr";
                    break;
                case ccMaterial::TextureMapType::ANISOTROPY:
                    mapCommand = "map_aniso";
                    break;
                case ccMaterial::TextureMapType::DISPLACEMENT:
                    mapCommand = "map_disp";
                    break;
                case ccMaterial::TextureMapType::REFLECTION:
                    mapCommand = "refl";
                    break;
                default:
                    // Fallback to diffuse
                    mapCommand = "map_Kd";
                    break;
            }

            stream << mapCommand << " " << relativeFilename << endl;
        }

        // Legacy texture support (for backward compatibility)
        // Only save if not already saved via multi-texture API
        if (mtl->hasTexture() &&
            !mtl->hasTextureMap(ccMaterial::TextureMapType::DIFFUSE)) {
            QString absFilename = mtl->getTextureFilename();
            QString relativeFilename = saveTextureFile(absFilename, matIndex);
            if (!relativeFilename.isEmpty()) {
                stream << "map_Kd " << relativeFilename << endl;
            }
        }
    }

    file.close();

    return true;
}

bool ccMaterialSet::append(const ccMaterialSet& source) {
    try {
        for (ccMaterialSet::const_iterator it = source.begin();
             it != source.end(); ++it) {
            ccMaterial::CShared mtl = *it;
            if (addMaterial(mtl) <= 0) {
                CVLog::WarningDebug(
                        QString("[ccMaterialSet::append] Material %1 couldn't "
                                "be added to material set and will be ignored")
                                .arg(mtl->getName()));
            }
        }
    } catch (... /*const std::bad_alloc&*/)  // out of memory
    {
        CVLog::Warning("[ccMaterialSet::append] Not enough memory");
        return false;
    }

    return true;
}

ccMaterialSet* ccMaterialSet::clone() const {
    ccMaterialSet* cloneSet = new ccMaterialSet(getName());
    if (!cloneSet->append(*this)) {
        CVLog::Warning("[ccMaterialSet::clone] Not enough memory");
        cloneSet->release();
        cloneSet = 0;
    }

    return cloneSet;
}

bool ccMaterialSet::toFile_MeOnly(QFile& out) const {
    if (!ccHObject::toFile_MeOnly(out)) return false;

    // Materials count (dataVersion>=20)
    uint32_t count = (uint32_t)size();
    if (out.write((const char*)&count, 4) < 0) return WriteError();

    // texture filenames
    std::set<QString> texFilenames;

    // Write each material
    for (ccMaterialSet::const_iterator it = begin(); it != end(); ++it) {
        ccMaterial::CShared mtl = *it;
        mtl->toFile(out);

        // remember its texture as well (if any)
        QString texFilename = mtl->getTextureFilename();
        if (!texFilename.isEmpty()) texFilenames.insert(texFilename);
    }

    // now save the number of textures (dataVersion>=37)
    QDataStream outStream(&out);
    outStream << static_cast<uint32_t>(texFilenames.size());
    // and save the textures (dataVersion>=37)
    {
        for (std::set<QString>::const_iterator it = texFilenames.begin();
             it != texFilenames.end(); ++it) {
            outStream << *it;                          // name
            outStream << ccMaterial::GetTexture(*it);  // then image
        }
    }

    return true;
}

bool ccMaterialSet::fromFile_MeOnly(QFile& in,
                                    short dataVersion,
                                    int flags,
                                    LoadedIDMap& oldToNewIDMap) {
    if (!ccHObject::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    // Materials count (dataVersion>=20)
    uint32_t count = 0;
    ;
    if (in.read((char*)&count, 4) < 0) return ReadError();
    if (count == 0) return true;

    // Load each material
    {
        for (uint32_t i = 0; i < count; ++i) {
            ccMaterial::Shared mtl(new ccMaterial);
            if (!mtl->fromFile(in, dataVersion, flags, oldToNewIDMap))
                return false;
            addMaterial(
                    mtl,
                    true);  // if we load a file, we can't allow that materials
                            // are not in the same order as before!
        }
    }

    if (dataVersion >= 37) {
        QDataStream inStream(&in);

        // now load the number of textures (dataVersion>=37)
        uint32_t texCount = 0;
        inStream >> texCount;
        // and load the textures (dataVersion>=37)
        {
            for (uint32_t i = 0; i < texCount; ++i) {
                QString filename;
                inStream >> filename;
                QImage image;
                inStream >> image;
                ccMaterial::AddTexture(image, filename);
            }
        }
    }

    return true;
}
