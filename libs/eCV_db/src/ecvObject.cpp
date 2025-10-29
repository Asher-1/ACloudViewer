// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvObject.h"

// Qt
#include <QSettings>

#ifdef USE_VLD
// VLD
#include <vld.h>
#endif

// System
#include <stdint.h>

const unsigned c_currentDBVersion = 48;  // 4.8

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

bool ccObject::toFile(QFile& out) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));

    // class ID (dataVersion>=20)
    // DGM: on 64 bits since version 34
    uint64_t classID = static_cast<uint64_t>(getClassID());
    if (out.write((const char*)&classID, 8) < 0) return WriteError();

    // unique ID (dataVersion>=20)
    // DGM: this ID will be useful to recreate dynamic links between entities!
    uint32_t uniqueID = (uint32_t)m_uniqueID;
    if (out.write((const char*)&uniqueID, 4) < 0) return WriteError();

    // name (dataVersion>=22)
    {
        QDataStream outStream(&out);
        outStream << m_name;
    }

    // flags (dataVersion>=20)
    uint32_t objFlags = (uint32_t)m_flags;
    if (out.write((const char*)&objFlags, 4) < 0) return WriteError();

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
        if (out.write((const char*)&validMetaDataCount, 4) < 0)
            return WriteError();

        //"key + value" pairs
        QDataStream outStream(&out);
        for (QVariantMap::const_iterator it = m_metaData.begin();
             it != m_metaData.end(); ++it) {
            if (!it.key().contains(".nosave")) {
                outStream << it.key();
                outStream << it.value();
            }
        }
    }

    return true;
}

CV_CLASS_ENUM ccObject::ReadClassIDFromFile(QFile& in, short dataVersion) {
    assert(in.isOpen() && (in.openMode() & QIODevice::ReadOnly));

    // class ID (on 32 bits between version 2.0 and 3.3, then 64 bits from
    // version 3.4)
    CV_CLASS_ENUM classID = CV_TYPES::OBJECT;
    if (dataVersion < 34) {
        uint32_t _classID = 0;
        if (in.read((char*)&_classID, 4) < 0) return ReadError();
        classID = static_cast<CV_CLASS_ENUM>(_classID);
    } else {
        uint64_t _classID = 0;
        if (in.read((char*)&_classID, 8) < 0) return ReadError();
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
    if (in.read((char*)&uniqueID, 4) < 0) return ReadError();
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
        if (in.read(name, 256) < 0) return ReadError();
        setName(name);
    } else  //(dataVersion>=22)
    {
        QDataStream inStream(&in);
        inStream >> m_name;
    }

    // flags (dataVersion>=20)
    uint32_t objFlags = 0;
    if (in.read((char*)&objFlags, 4) < 0) return ReadError();
    m_flags = (unsigned)objFlags;

    // meta data (dataVersion>=30)
    if (dataVersion >= 30) {
        // count
        uint32_t metaDataCount = 0;
        if (in.read((char*)&metaDataCount, 4) < 0) return ReadError();

        //"key + value" pairs
        for (uint32_t i = 0; i < metaDataCount; ++i) {
            QDataStream inStream(&in);
            QString key;
            QVariant value;
            inStream >> key;
            inStream >> value;
            setMetaData(key, value);
        }
    }

    return true;
}
