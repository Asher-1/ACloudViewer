// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef SETTING_MANAGER_H
#define SETTING_MANAGER_H

#include <QCoreApplication>
#include <QSettings>
#include <QSharedPointer>
#include <QTextCodec>

#include "CVAppCommon.h"

//! ecvSettingManager
class QDialog;
class QMainWindow;
class QDockWidget;
class CVAPPCOMMON_LIB_API ecvSettingManager : public QSettings {
    Q_OBJECT
    typedef QSettings Superclass;

public:
    //! Destructor
    ~ecvSettingManager() override {}

    //! Returns the (unique) static instance
    /** \param autoInit automatically initialize the console instance (with no
     *widget!) if not done already
     **/
    static ecvSettingManager *TheInstance(bool autoInit = true);

    //! Releases unique instance
    static void ReleaseInstance();

    static void Init(const QString &path);  //
    static void setValue(const QString &section,
                         const QString &key,
                         const QVariant &value);                        //
    static void removeNode(const QString &section);                     //
    static void removeKey(const QString &section, const QString &key);  //
    static QVariant getValue(const QString &section,
                             const QString &key,
                             const QVariant &defaultValue = QVariant());  //

    virtual void saveState(const QMainWindow &window, const QString &key);
    virtual void saveState(const QDialog &dialog, const QString &key);

    virtual void restoreState(const QString &key, QMainWindow &window);
    virtual void restoreState(const QString &key, QDialog &dialog);

    /**
     * Calling this method will cause the modified signal to be emitted.
     */
    virtual void alertSettingsModified();

    /**
     * Creates a new backup file for the current settings.
     * If `filename` is empty, then a backup file name will automatically be
     * picked. On success returns the backup file name, on failure an empty
     * string is returned.
     */
    QString backup(const QString &filename = QString());

public:
    virtual void clear();
    virtual void sync();
    virtual Status status() const;
    virtual bool isAtomicSyncRequired() const;
    virtual void setAtomicSyncRequired(bool enable);

    virtual void beginGroup(const QString &prefix);
    virtual void endGroup();
    virtual QString group() const;

    virtual int beginReadArray(const QString &prefix);
    virtual void beginWriteArray(const QString &prefix, int size = -1);
    virtual void endArray();
    virtual void setArrayIndex(int i);

    virtual QStringList allKeys() const;
    virtual QStringList childKeys() const;
    virtual QStringList childGroups() const;
    virtual bool isWritable() const;

    virtual void setValue(const QString &key, const QVariant &value);
    virtual QVariant value(const QString &key,
                           const QVariant &defaultValue = QVariant()) const;

    virtual void remove(const QString &key);
    virtual bool contains(const QString &key) const;

    virtual void setFallbacksEnabled(bool b);
    virtual bool fallbacksEnabled() const;

    virtual QString fileName() const;
    virtual Format format() const;
    virtual Scope scope() const;
    virtual QString organizationName() const;
    virtual QString applicationName() const;

#if QT_CONFIG(textcodec)
    void setIniCodec(QTextCodec *codec);
    void setIniCodec(const char *codecName);
    QTextCodec *iniCoxdec() const;
#endif
protected:
    /**
     * ensure that when window state is being loaded, if dock windows are
     * beyond the viewport, we correct them.
     */
    virtual void sanityCheckDock(QDockWidget *docke_widget);

private:
    //! Default constructor
    /** Constructor is protected to avoid using this object as a non static
     *class.
     **/
    ecvSettingManager() {}

    QSharedPointer<QSettings> m_iniFile;

signals:
    void modified();
};

#endif
