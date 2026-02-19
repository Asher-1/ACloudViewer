// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCoreApplication>
#include <QSettings>
#include <QSharedPointer>
// Qt5/Qt6 Compatibility
#include <QtCompat.h>

#include "CVAppCommon.h"

/**
 * @class ecvSettingManager
 * @brief Application settings manager (singleton)
 * 
 * Extends QSettings to provide centralized management of application
 * settings with additional features:
 * - Window and dialog state persistence (position, size, dock widgets)
 * - Hierarchical key-value storage with section support
 * - Settings backup/restore functionality
 * - Automatic sanity checking for UI element positions
 * - Modified signal for change notifications
 * 
 * Settings are typically stored in INI format on all platforms.
 * Uses singleton pattern for global access throughout the application.
 * 
 * @see QSettings
 * @see ecvOptions
 */
class QDialog;
class QMainWindow;
class QDockWidget;
class CVAPPCOMMON_LIB_API ecvSettingManager : public QSettings {
    Q_OBJECT
    typedef QSettings Superclass;

public:
    /**
     * @brief Destructor
     */
    ~ecvSettingManager() override {}

    /**
     * @brief Get singleton instance
     * @param autoInit Automatically initialize if not already done
     * @return Pointer to settings manager singleton
     */
    static ecvSettingManager *TheInstance(bool autoInit = true);

    /**
     * @brief Release singleton instance
     * 
     * Clears the singleton. Typically called at application shutdown.
     */
    static void ReleaseInstance();

    /**
     * @brief Initialize settings manager with custom path
     * @param path Path to settings file
     */
    static void Init(const QString &path);
    
    /**
     * @brief Set a value in settings (static convenience method)
     * @param section Settings section/group
     * @param key Setting key
     * @param value Value to store
     */
    static void setValue(const QString &section,
                         const QString &key,
                         const QVariant &value);
    
    /**
     * @brief Remove entire settings section
     * @param section Section to remove
     */
    static void removeNode(const QString &section);
    
    /**
     * @brief Remove specific key from section
     * @param section Settings section
     * @param key Key to remove
     */
    static void removeKey(const QString &section, const QString &key);
    
    /**
     * @brief Get a value from settings (static convenience method)
     * @param section Settings section/group
     * @param key Setting key
     * @param defaultValue Default value if key doesn't exist
     * @return Stored value or default
     */
    static QVariant getValue(const QString &section,
                             const QString &key,
                             const QVariant &defaultValue = QVariant());

    /**
     * @brief Save main window state
     * 
     * Saves window geometry, dock widget positions, and toolbar states.
     * @param window Main window to save
     * @param key Storage key identifier
     */
    virtual void saveState(const QMainWindow &window, const QString &key);
    
    /**
     * @brief Save dialog state
     * 
     * Saves dialog geometry and position.
     * @param dialog Dialog to save
     * @param key Storage key identifier
     */
    virtual void saveState(const QDialog &dialog, const QString &key);

    /**
     * @brief Restore main window state
     * 
     * Restores previously saved window geometry, dock widgets, and toolbars.
     * @param key Storage key identifier
     * @param window Main window to restore
     */
    virtual void restoreState(const QString &key, QMainWindow &window);
    
    /**
     * @brief Restore dialog state
     * 
     * Restores previously saved dialog geometry.
     * @param key Storage key identifier
     * @param dialog Dialog to restore
     */
    virtual void restoreState(const QString &key, QDialog &dialog);

    /**
     * @brief Emit modified signal
     * 
     * Manually trigger the modified() signal to notify listeners
     * that settings have changed.
     */
    virtual void alertSettingsModified();

    /**
     * @brief Create settings backup file
     * 
     * Creates a backup copy of the current settings file.
     * @param filename Backup filename (auto-generated if empty)
     * @return Backup filename on success, empty string on failure
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

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#if defined(QT_CONFIG) && QT_CONFIG(textcodec)
    void setIniCodec(QTextCodec *codec);
    void setIniCodec(const char *codecName);
    QTextCodec *iniCoxdec() const;
#endif
#endif
protected:
    /**
     * @brief Sanity check dock widget position
     * 
     * Ensures dock widgets are within visible viewport when restoring state.
     * Prevents dock widgets from being positioned off-screen.
     * @param docke_widget Dock widget to check
     */
    virtual void sanityCheckDock(QDockWidget *docke_widget);

private:
    /**
     * @brief Default constructor (private for singleton)
     * 
     * Constructor is private to enforce singleton pattern.
     */
    ecvSettingManager() {}

    QSharedPointer<QSettings> m_iniFile;  ///< Underlying settings file handle

signals:
    /**
     * @brief Signal emitted when settings are modified
     * 
     * Listeners can connect to this signal to be notified of setting changes.
     */
    void modified();
};
