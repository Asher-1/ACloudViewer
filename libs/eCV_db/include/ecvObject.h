// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "CVTypes.h"
#include "eCV_db.h"
#include "ecvColorTypes.h"
#include "ecvSerializableObject.h"

// QT
#include <QSharedPointer>
#include <QVariant>

//! Unique ID generator (should be unique for the whole application instance -
//! with plugins, etc.)
class ECV_DB_LIB_API ccUniqueIDGenerator {
public:
    static constexpr unsigned InvalidUniqueID = 0xFFFFFFFF;
    static constexpr unsigned MinUniqueID = 0x00000100;

    //! Shared type
    using Shared = QSharedPointer<ccUniqueIDGenerator>;

    //! Default constructor
    ccUniqueIDGenerator() : m_lastUniqueID(MinUniqueID) {}

    //! Resets the unique ID
    void reset() { m_lastUniqueID = MinUniqueID; }
    //! Returns a (new) unique ID
    unsigned fetchOne() { return ++m_lastUniqueID; }
    //! Returns the value of the last generated unique ID
    unsigned getLast() const { return m_lastUniqueID; }
    //! Updates the value of the last generated unique ID with the current one
    void update(unsigned ID) {
        if (ID > m_lastUniqueID) m_lastUniqueID = ID;
    }

protected:
    unsigned m_lastUniqueID;
};

//! Generic "CLOUDVIEWER  Object" template
class ECV_DB_LIB_API ccObject : public ccSerializableObject {
public:
    //! Default constructor
    /** \param name object name (optional)
    \param uniqueID unique ID (handle with care! Will be auto generated if equal
    to ccUniqueIDGenerator::InvalidUniqueID)
    **/
    ccObject(QString name = QString());

    //! Copy constructor
    ccObject(const ccObject& object);

    //! Returns current database version
    static unsigned GetCurrentDBVersion();
    //! Sets the unique ID generator
    static void SetUniqueIDGenerator(ccUniqueIDGenerator::Shared generator);
    //! Returns the unique ID generator
    static ccUniqueIDGenerator::Shared GetUniqueIDGenerator();

    //! Returns class ID
    virtual CV_CLASS_ENUM getClassID() const = 0;

    //! Returns object name
    virtual inline QString getName() const { return m_name; }

    //! Sets object name
    virtual inline void setName(const QString& name) { m_name = name; }

    //! Sets removeFlag
    virtual inline void setRemoveFlag(bool removeFlag) {
        m_removeFlag = removeFlag;
    }

    //! Returns removeFlag
    virtual inline bool getRemoveFlag() { return m_removeFlag; }

    //! Returns object unique ID
    virtual inline unsigned getUniqueID() const { return m_uniqueID; }

    //! Changes unique ID
    /** WARNING: HANDLE WITH CARE!
            Updates persistent settings (last unique ID) if necessary.
    **/
    virtual void setUniqueID(unsigned ID);

    //! Returns whether the object is enabled or not
    /** Shortcut to access flag CC_ENABLED
     **/
    virtual inline bool isEnabled() const { return getFlagState(CC_ENABLED); }

    //! Sets the "enabled" property
    /** Shortcut to modify flag CC_ENABLED
     **/
    virtual inline void setEnabled(bool state) {
        setFlagState(CC_ENABLED, state);
    }

    //! Toggles the "enabled" property
    virtual inline void toggleActivation() { setEnabled(!isEnabled()); }

    //! Returns whether the object is locked  or not
    /** Shortcut to access flag CC_LOCKED
     **/
    virtual inline bool isLocked() const { return getFlagState(CC_LOCKED); }

    //! Sets the "enabled" property
    /** Shortcut to modify flag CC_LOCKED
     **/
    virtual inline void setLocked(bool state) {
        setFlagState(CC_LOCKED, state);
    }

    // shortcuts
    inline bool isLeaf() const { return (getClassID() & CC_LEAF_BIT) != 0; }
    inline bool isCustom() const { return (getClassID() & CC_CUSTOM_BIT) != 0; }
    inline bool isHierarchy() const {
        return (getClassID() & CC_HIERARCH_BIT) != 0;
    }

    inline bool isKindOf(CV_CLASS_ENUM type) const {
        return (getClassID() & type) == type;
    }
    inline bool isA(CV_CLASS_ENUM type) const { return (getClassID() == type); }

    //! Returns a new unassigned unique ID
    /** Unique IDs are handled with persistent settings
            in order to assure consistency between main app
            and plugins!
    **/
    static unsigned GetNextUniqueID();

    //! Returns last assigned unique ID
    /** Unique IDs are handled with persistent settings
            in order to assure consistency between main app
            and plugins!
    **/
    static unsigned GetLastUniqueID();

    //! Helper: reads out class ID from a binary stream
    /** Must be called before 'fromFile'!
     **/
    static CV_CLASS_ENUM ReadClassIDFromFile(QFile& in, short dataVersion);

    //! Returns a given associated meta data
    /** \param key meta data unique identifier (case sensitive)
            \return meta data (if any) or an invalid QVariant
    **/
    QVariant getMetaData(const QString& key) const;

    //! Removes a given associated meta-data
    /** \param key meta-data unique identifier (case sensitive)
            \return success
    **/
    bool removeMetaData(const QString& key);

    //! Sets a meta-data element
    /** \param key meta-data unique identifier (case sensitive)
            \param data data
    **/
    void setMetaData(const QString& key, const QVariant& data);

    //! Sets several meta-data elements at a time
    /** \param dataset meta-data set
            \param overwrite whether existing meta-data elements should be
    replaced by the input ones (with the same key) or not
    **/
    void setMetaData(const QVariantMap& dataset, bool overwrite = false);

    //! Returns whether a meta-data element with the given key exists or not
    /** \param key meta-data unique identifier (case sensitive)
            \return whether the element exists or not
    **/
    bool hasMetaData(const QString& key) const;

    //! Returns meta-data map (const only)
    const QVariantMap& metaData() const { return m_metaData; }

    inline void setBaseName(const QString& baseName) { m_baseName = baseName; }
    inline QString getBaseName() const { return m_baseName; }

    inline void setFullPath(const QString& fullPaht) { m_filePath = fullPaht; }
    inline QString getFullPath() const { return m_filePath; }

protected:
    //! Returns flag state
    virtual inline bool getFlagState(CV_OBJECT_FLAG flag) const {
        return (m_flags & flag);
    }

    //! Sets flag state
    /** \param flag object flag to set
            \param state flag state
    **/
    virtual void setFlagState(CV_OBJECT_FLAG flag, bool state);

    // inherited from ccSerializableObject
    bool toFile(QFile& out) const override;

    //! Reimplemented from ccSerializableObject::fromFile
    /** Be sure to call ccObject::ReadClassIDFromFile (once)
            before calling this method, as the classID is voluntarily
            skipped (in order to let the user instantiate the object first)
    **/
    bool fromFile(QFile& in,
                  short dataVersion,
                  int flags,
                  LoadedIDMap& oldToNewIDMap) override;

    //! Object name
    QString m_name;

    QString m_baseName;
    QString m_filePath;

    bool m_removeFlag;

    //! Object flags
    unsigned m_flags;

    //! Associated meta-data
    QVariantMap m_metaData;

private:
    //! Object unique ID
    unsigned m_uniqueID;
};
