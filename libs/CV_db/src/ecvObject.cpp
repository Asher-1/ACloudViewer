// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvObject.h"

// Qt
#include <QtCompat.h>

#include <QSettings>

#ifdef USE_VLD
// VLD
#include <vld.h>
#endif

// System
#include <stdint.h>

// clang-format off
/** Versions:
    V1.0 = prior to 05/04/2012 = old version
    V2.0 - 05/04/2012 - upgrade to serialized version with version tracking
    V2.1 - 07/02/2012 - points & 2D labels upgraded
    V2.2 - 11/26/2012 - object name is now a QString
    V2.3 - 02/07/2013 - attribute 'm_selectionBehavior' added to ccHObject class
    v2.4 - 02/22/2013 - per-cloud point size + whether name is displayed in 3D or not
    v2.5 - 03/16/2013 - ccViewportParameters structure modified
    v2.6 - 04/03/2013 - strictly positive scalar field removed and 'hidden' values marker is now NaN
    v2.7 - 04/12/2013 - Customizable color scales
    v2.8 - 07/12/2013 - Polylines are now supported
    v2.9 - 08/14/2013 - ccMeshGroup removed, ccSubMesh added
    v3.0 - 08/30/2013 - QObject's meta data structure added
    v3.1 - 09/25/2013 - ccPolyline width added
    v3.2 - 10/11/2013 - ccFacet (2D polygons) are now supported
    v3.3 - 12/19/2013 - global scale information is now saved for point clouds
    v3.4 - 01/09/2014 - ccIndexedTransformation and ccIndexedTransformationBuffer added + CC_CLASS_ENUM is now coded on 64 bits
    v3.5 - 02/13/2014 - ccSensor class updated
    v3.6 - 05/30/2014 - ccGLWindow and associated structures (viewport, etc.) now use double precision
    v3.7 - 08/24/2014 - Textures are stored and saved as a single DB with only references to them in each material (key = absolute filename)
    v3.8 - 09/14/2014 - GBL and camera sensors structures have evolved
    v3.9 - 01/30/2015 - Shift & scale information are now saved for polylines (+ separate interface)
    v4.0 - 08/06/2015 - Custom labels added to color scales
    v4.1 - 09/01/2015 - Scan grids added to point clouds
    v4.2 - 10/07/2015 - Global shift added to the ccScalarField structure
    v4.3 - 01/07/2016 - Additional intrinsic parameters of a camera sensor (optical center)
    v4.4 - 07/07/2016 - Full WaveForm data added to point clouds
    v4.5 - 10/06/2016 - Transformation history is now saved
    v4.6 - 11/03/2016 - Null normal vector code added
    v4.7 - 12/22/2016 - Return index added to ccWaveform
    v4.8 - 10/19/2018 - The CC_CAMERA_BIT and CC_QUADRIC_BIT were wrongly defined
    v4.9 - 03/31/2019 - Point labels can now be picked on meshes
    v5.0 - 10/06/2019 - Point labels can now target the entity center
    v5.1 - 03/29/2019 - New camera management (viewports have changed)
    v5.2 - 11/30/2020 - New ccCoordinateSystem added
    v5.3 - 10/02/2022 - ccViewportParameters new members (near and far clipping planes)
    v5.4 - 01/29/2023 - ccColorScale custom labels can be overridden by a string
    v5.5 - 11/10/2024 - Scalar fields with 'double' offset and names as std::string
    v5.6 - 02/18/2025 - Circle entity
    v5.7 - 10/01/2025 - Disc entity
    v5.8 - 07/06/2026 - PBR material parameters and multi-texture map serialization for ccMaterial
    v5.9 - 11/25/2026 - Object metadata uses the fixed Qt 5.12 stream format
**/
// clang-format on

const unsigned c_currentDBVersion = 59;  // 5.9

//! Default unique ID generator (using the system persistent settings as we did
//! previously proved to be not reliable)
static ccUniqueIDGenerator::Shared s_uniqueIDGenerator(new ccUniqueIDGenerator);

void ccObject::SetUniqueIDGenerator(ccUniqueIDGenerator::Shared generator) {
    if (generator == s_uniqueIDGenerator) return;

    // we hope that the previous generator has not been used!
    assert(!s_uniqueIDGenerator || s_uniqueIDGenerator->getLast() == 0);
    s_uniqueIDGenerator = generator;
}

ccUniqueIDGenerator::Shared ccObject::GetUniqueIDGenerator() {
    return s_uniqueIDGenerator;
}

unsigned ccObject::GetCurrentDBVersion() { return c_currentDBVersion; }

unsigned ccObject::GetNextUniqueID() {
    if (!s_uniqueIDGenerator) {
        assert(false);
        s_uniqueIDGenerator =
                ccUniqueIDGenerator::Shared(new ccUniqueIDGenerator);
    }
    return s_uniqueIDGenerator->fetchOne();
}

unsigned ccObject::GetLastUniqueID() {
    return s_uniqueIDGenerator ? s_uniqueIDGenerator->getLast() : 0;
}

ccObject::ccObject(QString name)
    : m_name(name.isEmpty() ? "unnamed" : name),
      m_baseName(m_name),
      m_filePath(QString(m_baseName) + ".bin"),
      m_removeFlag(false),
      m_flags(CC_ENABLED),
      m_uniqueID(GetNextUniqueID()) {}

ccObject::ccObject(const ccObject& object)
    : m_name(object.m_name),
      m_baseName(object.m_baseName),
      m_filePath(object.m_filePath),
      m_removeFlag(false),
      m_flags(object.m_flags),
      m_uniqueID(GetNextUniqueID()) {}

void ccObject::setUniqueID(unsigned ID) {
    m_uniqueID = ID;

    // updates last unique ID
    if (s_uniqueIDGenerator)
        s_uniqueIDGenerator->update(m_uniqueID);
    else
        assert(false);
}

void ccObject::setFlagState(CV_OBJECT_FLAG flag, bool state) {
    if (state)
        m_flags |= unsigned(flag);
    else
        m_flags &= (~unsigned(flag));
}

bool ccObject::toFile(QFile& out, short dataVersion) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));

    // Version validation
    if (dataVersion < 34) {
        assert(false);
        return false;
    }

    // class ID (dataVersion>=20)
    // DGM: on 64 bits since version 34
    uint64_t classID = static_cast<uint64_t>(getClassID());
    if (out.write((const char*)&classID, 8) != 8) return WriteError();

    // unique ID (dataVersion>=20)
    // DGM: this ID will be useful to recreate dynamic links between entities!
    uint32_t uniqueID = (uint32_t)m_uniqueID;
    if (out.write((const char*)&uniqueID, 4) != 4) return WriteError();

    // name (dataVersion>=22)
    {
        QDataStream outStream(&out);
        if (dataVersion >= 59)
            outStream.setVersion(qtCompatDataStreamVersion());
        outStream << m_name;
        if (outStream.status() != QDataStream::Ok) return WriteError();
    }

    // flags (dataVersion>=20)
    uint32_t objFlags = (uint32_t)m_flags;
    if (out.write((const char*)&objFlags, 4) != 4) return WriteError();

    // meta data (dataVersion>=30)
    {
        // check for valid pieces of meta-data
        // DGM: some pieces of meta-data can't be properly streamed (the ones
        // relying on 'Q_DECLARE_METATYPE' calls typically)
        uint32_t validMetaDataCount = 0;
        for (QVariantMap::const_iterator it = m_metaData.begin();
             it != m_metaData.end(); ++it) {
            if (!it.key().contains(".nosave")) {
                ++validMetaDataCount;
            }
        }

        // count
        if (out.write((const char*)&validMetaDataCount, 4) != 4)
            return WriteError();

        //"key + value" pairs
        QDataStream outStream(&out);
        if (dataVersion >= 59)
            outStream.setVersion(qtCompatDataStreamVersion());
        for (QVariantMap::const_iterator it = m_metaData.begin();
             it != m_metaData.end(); ++it) {
            if (!it.key().contains(".nosave")) {
                outStream << it.key();
                outStream << it.value();
            }
        }
        if (outStream.status() != QDataStream::Ok) return WriteError();
    }

    return true;
}

short ccObject::minimumFileVersion() const {
    // Only persisted metadata needs the new format. Keep metadata-free files
    // and transient .nosave entries compatible with the old minimum version.
    for (auto it = m_metaData.cbegin(); it != m_metaData.cend(); ++it) {
        if (!it.key().contains(".nosave")) return 59;
    }
    return 34;
}

CV_CLASS_ENUM ccObject::ReadClassIDFromFile(QFile& in, short dataVersion) {
    assert(in.isOpen() && (in.openMode() & QIODevice::ReadOnly));

    // class ID (on 32 bits between version 2.0 and 3.3, then 64 bits from
    // version 3.4)
    CV_CLASS_ENUM classID = CV_TYPES::OBJECT;
    if (dataVersion < 34) {
        uint32_t _classID = 0;
        if (in.read((char*)&_classID, 4) != 4) return ReadError();
        classID = static_cast<CV_CLASS_ENUM>(_classID);
    } else {
        uint64_t _classID = 0;
        if (in.read((char*)&_classID, 8) != 8) return ReadError();
        classID = static_cast<CV_CLASS_ENUM>(_classID);
    }

    return classID;
}

QVariant ccObject::getMetaData(const QString& key) const {
    return m_metaData.value(key, QVariant());
}

bool ccObject::removeMetaData(const QString& key) {
    return m_metaData.remove(key) != 0;
}

void ccObject::setMetaData(const QString& key, const QVariant& data) {
    m_metaData.insert(key, data);
}

void ccObject::setMetaData(const QVariantMap& dataset,
                           bool overwrite /*=false*/) {
    for (QVariantMap::const_iterator it = dataset.begin(); it != dataset.end();
         ++it) {
        if (overwrite || !m_metaData.contains(it.key())) {
            m_metaData[it.key()] = it.value();
        }
    }
}

bool ccObject::hasMetaData(const QString& key) const {
    return m_metaData.contains(key);
}

bool ccObject::fromFile(QFile& in,
                        short dataVersion,
                        int flags,
                        LoadedIDMap& oldToNewIDMap) {
    assert(in.isOpen() && (in.openMode() & QIODevice::ReadOnly));

    if (dataVersion < 20) return CorruptError();

    // DGM: if we are here, we assume the class ID has already been read!
    // Call ccObject::readClassIDFromFile if necessary
    ////class ID (dataVersion>=20)
    // uint32_t classID = 0;
    // if (in.read((char*)&classID,4) < 0)
    //	return ReadError();

    // unique ID (dataVersion>=20)
    uint32_t uniqueID = 0;
    if (in.read((char*)&uniqueID, 4) != 4) return ReadError();
    // DGM: this ID will be useful to recreate dynamic links between entities
    // later!
    if (oldToNewIDMap.contains(uniqueID)) {
        CVLog::Warning(QString("Malformed file: uniqueID #%1 is used several "
                               "times! (not that unique ;)")
                               .arg(uniqueID));
    }
    oldToNewIDMap.insert(uniqueID, m_uniqueID);

    // name
    if (dataVersion < 22)  // old style
    {
        char name[256];
        if (in.read(name, 256) != 256) return ReadError();
        setName(name);
    } else  //(dataVersion>=22)
    {
        QDataStream inStream(&in);
        if (dataVersion >= 59) inStream.setVersion(qtCompatDataStreamVersion());
        inStream >> m_name;
        if (inStream.status() != QDataStream::Ok) return ReadError();
    }

    // flags (dataVersion>=20)
    uint32_t objFlags = 0;
    if (in.read((char*)&objFlags, 4) != 4) return ReadError();
    m_flags = (unsigned)objFlags;

    // meta data (dataVersion>=30)
    if (dataVersion >= 30) {
        // count
        uint32_t metaDataCount = 0;
        if (in.read((char*)&metaDataCount, 4) != 4) return ReadError();

        //"key + value" pairs
        QDataStream inStream(&in);
        if (dataVersion >= 59) inStream.setVersion(qtCompatDataStreamVersion());
        for (uint32_t i = 0; i < metaDataCount; ++i) {
            QString key;
            QVariant value;
            inStream >> key;
            if (inStream.status() != QDataStream::Ok) return ReadError();
            const qint64 valuePosition = in.pos();
            inStream >> value;
            if (inStream.status() == QDataStream::Ok && !value.isValid()) {
                // Qt5 can silently turn a foreign user tag into an invalid
                // QVariant without setting an error. Only wire tag 0 denotes
                // a genuinely invalid QVariant; preserve that ordinary value.
                const qint64 valueEnd = in.pos();
                if (!in.seek(valuePosition)) return ReadError();
                quint32 storedType = 0;
                inStream >> storedType;
                if (storedType != 0) {
                    inStream.setStatus(QDataStream::ReadCorruptData);
                } else if (!in.seek(valueEnd)) {
                    return ReadError();
                }
            }
            if (inStream.status() == QDataStream::Ok) {
                // Registered LasVlr stream operators work on both Qt majors.
                // Do not discard them, or an ordinary value with this key.
                setMetaData(key, value);
                continue;
            }

            // A missing LAS plugin cannot construct its historical user type.
            // Retain the old ability to open such files, but validate the
            // actual QVariant header and every length rather than skipping 16
            // bytes.
            if (key != "LAS.vlrs" ||
                QMetaType::isRegistered(value.userType()) ||
                !in.seek(valuePosition)) {
                return ReadError();
            }
            inStream.resetStatus();
            quint32 typeId = 0;
            quint8 isNull = 0;
            quint32 typeNameSize = 0;
            inStream >> typeId >> isNull >> typeNameSize;
            const char expectedTypeName[] = "LasVlr";
            char typeName[sizeof(expectedTypeName)] = {};
            // Historical Qt4/Qt5/Qt6 user-type wire tags.
            if (inStream.status() != QDataStream::Ok ||
                (typeId != 127 && typeId != 1024 && typeId != 65536) ||
                isNull > 1 || typeNameSize != sizeof(typeName) ||
                inStream.readRawData(typeName, sizeof(typeName)) !=
                        sizeof(typeName) ||
                QByteArray(typeName, sizeof(typeName)) !=
                        QByteArray(expectedTypeName,
                                   sizeof(expectedTypeName))) {
                return CorruptError();
            }

            quint64 vlrCount = 0;
            inStream >> vlrCount;
            // Each VLR needs a 54-byte header even when its payload is empty.
            if (inStream.status() != QDataStream::Ok ||
                vlrCount > quint64(in.size() - in.pos()) / 54) {
                return CorruptError();
            }
            for (quint64 j = 0; j < vlrCount; ++j) {
                if (inStream.skipRawData(20) != 20) return ReadError();
                quint16 payloadSize = 0;
                inStream >> payloadSize;
                const int remainingRecord = 32 + int(payloadSize);
                if (inStream.status() != QDataStream::Ok ||
                    remainingRecord > in.size() - in.pos() ||
                    inStream.skipRawData(remainingRecord) != remainingRecord) {
                    return ReadError();
                }
            }
            quint64 extraFieldCount = 0;
            inStream >> extraFieldCount;
            // The existing legacy fallback describes 272-byte raw ABI records.
            // This is not a new portable LAS serialization format.
            if (inStream.status() != QDataStream::Ok ||
                extraFieldCount > quint64(in.size() - in.pos()) / 272 ||
                !in.seek(in.pos() + qint64(extraFieldCount * 272))) {
                return CorruptError();
            }
            CVLog::Warning("LAS VLR metadata skipped: its type is unavailable");
        }
    }

    return true;
}
