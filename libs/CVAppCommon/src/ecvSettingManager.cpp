// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvSettingManager.h"

// CV_CORE_LIB
#include <CVPlatform.h>

// ECV_DB_LIB
#include <ecvSingleton.h>

// QT
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#include <QDesktopWidget>
#endif
#include <QDialog>
#include <QDockWidget>
#include <QFile>
#include <QGuiApplication>
#include <QMainWindow>
#include <QScreen>

/***************
 *** Globals ***
 ***************/

// unique console instance
static ecvSingleton<ecvSettingManager> s_manager;

ecvSettingManager* ecvSettingManager::TheInstance(bool autoInit /*=true*/) {
    if (!s_manager.instance && autoInit) {
        s_manager.instance = new ecvSettingManager();
    }

    return s_manager.instance;
}

void ecvSettingManager::ReleaseInstance() { s_manager.release(); }

void ecvSettingManager::alertSettingsModified() { emit this->modified(); }

//-----------------------------------------------------------------------------
void ecvSettingManager::saveState(const QDialog& dialog, const QString& key) {
    this->beginGroup(key);
    this->setValue("Position", dialog.pos());
    this->setValue("Size", dialog.size());
    // let's add a PID to avoid restoring dialog position across different
    // sessions. This avoids issues reported in #18163.
    this->setValue("PID", QCoreApplication::applicationPid());
    this->endGroup();
}

//-----------------------------------------------------------------------------
void ecvSettingManager::restoreState(const QString& key, QDialog& dialog) {
    this->beginGroup(key);

    if (this->contains("Size")) {
        dialog.resize(this->value("Size").toSize());
    }

    // restore position only if it is the same process.
    if (this->value("PID").value<qint64>() ==
                QCoreApplication::applicationPid() &&
        this->contains("Position")) {
        dialog.move(this->value("Position").toPoint());
    }
    this->endGroup();
}

//-----------------------------------------------------------------------------
void ecvSettingManager::saveState(const QMainWindow& window,
                                  const QString& key) {
    this->beginGroup(key);
    this->setValue("Size", window.size());
    this->setValue("Layout", window.saveState());
    this->endGroup();
}

//-----------------------------------------------------------------------------
void ecvSettingManager::restoreState(const QString& key, QMainWindow& window) {
    this->beginGroup(key);

    if (this->contains("Size")) {
        window.resize(this->value("Size").toSize());
    }

    if (this->contains("Layout")) {
        window.restoreState(this->value("Layout").toByteArray());

        QList<QDockWidget*> dockWidgets = window.findChildren<QDockWidget*>();
        foreach (QDockWidget* dock_widget, dockWidgets) {
            if (dock_widget->isFloating() == true) {
                sanityCheckDock(dock_widget);
            }
        }
    }

    this->endGroup();
}

//-----------------------------------------------------------------------------
void ecvSettingManager::sanityCheckDock(QDockWidget* dock_widget) {
    if (nullptr == dock_widget) {
        return;
    }
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QDesktopWidget desktop;
#endif

    QPoint dockTopLeft = dock_widget->pos();
    QRect dockRect(dockTopLeft, dock_widget->size());

    QRect geometry = QRect(dockTopLeft, dock_widget->frameSize());
    int titleBarHeight = geometry.height() - dockRect.height();

    // Qt5/Qt6 Compatibility: QDesktopWidget removed in Qt6, use QScreen instead
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    QScreen* screen = dock_widget->screen();
    if (!screen) {
        screen = QGuiApplication::primaryScreen();
    }
    QRect screenRect = screen ? screen->availableGeometry() : QRect();
#else
    QRect screenRect = desktop.availableGeometry(dock_widget);
#endif
    QRect desktopRect =
            QGuiApplication::primaryScreen()
                    ->availableGeometry();  // Should give us the entire Desktop
                                            // geometry
    // Ensure the top left corner of the window is on the screen
    if (!screenRect.contains(dockTopLeft)) {
        // Are we High?
        if (dockTopLeft.y() < screenRect.y()) {
            dock_widget->move(dockRect.x(), screenRect.y());
            dockTopLeft = dock_widget->pos();
            dockRect = QRect(dockTopLeft, dock_widget->frameSize());
        }
        // Are we low
        if (dockTopLeft.y() > screenRect.y() + screenRect.height()) {
            dock_widget->move(dockRect.x(),
                              screenRect.y() + screenRect.height() - 20);
            dockTopLeft = dock_widget->pos();
            dockRect = QRect(dockTopLeft, dock_widget->frameSize());
        }
        // Are we left
        if (dockTopLeft.x() < screenRect.x()) {
            dock_widget->move(screenRect.x(), dockRect.y());
            dockTopLeft = dock_widget->pos();
            dockRect = QRect(dockTopLeft, dock_widget->frameSize());
        }
        // Are we right
        if (dockTopLeft.x() > screenRect.x() + screenRect.width()) {
            dock_widget->move(
                    screenRect.x() + screenRect.width() - dockRect.width(),
                    dockRect.y());
            dockTopLeft = dock_widget->pos();
            dockRect = QRect(dockTopLeft, dock_widget->frameSize());
        }

        dockTopLeft = dock_widget->pos();
        dockRect = QRect(dockTopLeft, dock_widget->frameSize());
    }

    if (!desktopRect.contains(dockRect)) {
        // Are we too wide
        if (dockRect.x() + dockRect.width() >
            screenRect.x() + screenRect.width()) {
            if (screenRect.x() + screenRect.width() - dockRect.width() >
                screenRect.x()) {
                // Move dock side to side
                dockRect.setX(screenRect.x() + screenRect.width() -
                              dockRect.width());
                dock_widget->move(dockRect.x(), dockRect.y());
                dockTopLeft = dock_widget->pos();
                dockRect = QRect(dockTopLeft, dock_widget->frameSize());
            } else {
                // Move dock side to side + resize to fit
                dockRect.setX(screenRect.x() + screenRect.width() -
                              dockRect.width());
                dockRect.setWidth(screenRect.width());
                dock_widget->resize(dockRect.width(), dockRect.height());
                dock_widget->move(dockRect.x(), dockRect.y());
                dockTopLeft = dock_widget->pos();
                dockRect = QRect(dockTopLeft, dock_widget->frameSize());
            }
        }

        dockTopLeft = dock_widget->pos();
        dockRect = QRect(dockTopLeft, dock_widget->frameSize());
        // Are we too Tall
        if (dockRect.y() + dockRect.height() >
            screenRect.y() + screenRect.height()) {
            // See if we can move it more on screen so that the entire dock is
            // on screen
            if (screenRect.y() + screenRect.height() - dockRect.height() >
                screenRect.y()) {
                // Move dock up
                dockRect.setY(screenRect.y() + screenRect.height() -
                              dockRect.height());
                dock_widget->move(dockRect.x(), dockRect.y());
                dockTopLeft = dock_widget->pos();
                dockRect = QRect(dockTopLeft, dock_widget->frameSize());
            } else {
                // Move dock up + resize to fit
                dock_widget->resize(dockRect.width(),
                                    screenRect.height() - titleBarHeight);
                dock_widget->move(dockRect.x(), screenRect.y());
                dockTopLeft = dock_widget->pos();
                dockRect = QRect(dockTopLeft, dock_widget->frameSize());
            }
        }
    }
}

//-----------------------------------------------------------------------------
QString ecvSettingManager::backup(const QString& argName) {
    this->sync();

    QString fname = argName.isEmpty() ? (this->fileName() + ".bak") : argName;
    QFile::remove(fname);
    return QFile::copy(this->fileName(), fname) ? fname : QString();
}

void ecvSettingManager::Init(const QString& fileName) {
    // should be called only once!
    if (s_manager.instance) {
        assert(false);
        return;
    }

    s_manager.instance = new ecvSettingManager();

#ifdef CV_WINDOWS  // only support QSettings Writting in file in Windows now!
    QString configPath;
    configPath = QCoreApplication::applicationDirPath() + "/";
    configPath += fileName;
    s_manager.instance->m_iniFile = QSharedPointer<QSettings>(
            new QSettings(configPath, QSettings::IniFormat));
    // s_manager.instance->m_iniFile->setIniCodec(QTextCodec::codecForName("System"));
    s_manager.instance->m_iniFile->setIniCodec("utf8");  // set coding
    QFile file(configPath);
    if (false == file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        s_manager.instance->m_iniFile->beginGroup("section");
        s_manager.instance->m_iniFile->setValue("status", "false");
        s_manager.instance->m_iniFile->endGroup();
    }
#else
    Q_UNUSED(fileName)
    QSettings::setDefaultFormat(QSettings::NativeFormat);
    s_manager.instance->m_iniFile = QSharedPointer<QSettings>(new QSettings());
    // s_manager.instance->m_iniFile->setIniCodec(QTextCodec::codecForName("System"));
    s_manager.instance->m_iniFile->setIniCodec("utf8");  // set coding
#endif
}

void ecvSettingManager::setValue(const QString& section,
                                 const QString& key,
                                 const QVariant& value) {
    s_manager.instance->m_iniFile->beginGroup(section);  // set node name
    s_manager.instance->m_iniFile->setValue(
            key, value);  // set key and value according to key
    s_manager.instance->m_iniFile->endGroup();  // end group
}

void ecvSettingManager::removeNode(const QString& section) {
    s_manager.instance->m_iniFile->remove(section);
}

void ecvSettingManager::removeKey(const QString& section, const QString& key) {
    s_manager.instance->m_iniFile->beginGroup(section);
    s_manager.instance->m_iniFile->remove(key);
    s_manager.instance->m_iniFile->endGroup();
}

QVariant ecvSettingManager::getValue(const QString& section,
                                     const QString& key,
                                     const QVariant& defaultValue) {
    s_manager.instance->m_iniFile->beginGroup(section);
    QVariant result = s_manager.instance->m_iniFile->value(key, defaultValue);
    s_manager.instance->m_iniFile->endGroup();
    return result;
}

void ecvSettingManager::clear() { s_manager.instance->m_iniFile->clear(); }

void ecvSettingManager::sync() { s_manager.instance->m_iniFile->sync(); }

QSettings::Status ecvSettingManager::status() const {
    return s_manager.instance->m_iniFile->status();
}

bool ecvSettingManager::isAtomicSyncRequired() const {
    return s_manager.instance->m_iniFile->isAtomicSyncRequired();
}

void ecvSettingManager::setAtomicSyncRequired(bool enable) {
    s_manager.instance->m_iniFile->setAtomicSyncRequired(enable);
}

void ecvSettingManager::beginGroup(const QString& prefix) {
    s_manager.instance->m_iniFile->beginGroup(prefix);
}

void ecvSettingManager::endGroup() {
    s_manager.instance->m_iniFile->endGroup();
}

QString ecvSettingManager::group() const {
    return s_manager.instance->m_iniFile->group();
}

int ecvSettingManager::beginReadArray(const QString& prefix) {
    return s_manager.instance->m_iniFile->beginReadArray(prefix);
}

void ecvSettingManager::beginWriteArray(const QString& prefix, int size) {
    s_manager.instance->m_iniFile->beginWriteArray(prefix, size);
}

void ecvSettingManager::endArray() {
    s_manager.instance->m_iniFile->endArray();
}

void ecvSettingManager::setArrayIndex(int i) {
    s_manager.instance->m_iniFile->setArrayIndex(i);
}

QStringList ecvSettingManager::allKeys() const {
    return s_manager.instance->m_iniFile->allKeys();
}

QStringList ecvSettingManager::childKeys() const {
    return s_manager.instance->m_iniFile->childKeys();
}

QStringList ecvSettingManager::childGroups() const {
    return s_manager.instance->m_iniFile->childGroups();
}

bool ecvSettingManager::isWritable() const {
    return s_manager.instance->m_iniFile->isWritable();
}

void ecvSettingManager::setValue(const QString& key, const QVariant& value) {
    s_manager.instance->m_iniFile->setValue(key, value);
}

QVariant ecvSettingManager::value(const QString& key,
                                  const QVariant& defaultValue) const {
    return s_manager.instance->m_iniFile->value(key, defaultValue);
}

void ecvSettingManager::remove(const QString& key) {
    s_manager.instance->m_iniFile->remove(key);
}

bool ecvSettingManager::contains(const QString& key) const {
    return s_manager.instance->m_iniFile->contains(key);
}

void ecvSettingManager::setFallbacksEnabled(bool b) {
    s_manager.instance->m_iniFile->setFallbacksEnabled(b);
}

bool ecvSettingManager::fallbacksEnabled() const {
    return s_manager.instance->m_iniFile->fallbacksEnabled();
}

QString ecvSettingManager::fileName() const {
    return s_manager.instance->m_iniFile->fileName();
}

QSettings::Format ecvSettingManager::format() const {
    return s_manager.instance->m_iniFile->format();
}

QSettings::Scope ecvSettingManager::scope() const {
    return s_manager.instance->m_iniFile->scope();
}

QString ecvSettingManager::organizationName() const {
    return s_manager.instance->m_iniFile->organizationName();
}

QString ecvSettingManager::applicationName() const {
    return s_manager.instance->m_iniFile->applicationName();
}

void ecvSettingManager::setIniCodec(QTextCodec* codec) {
    s_manager.instance->m_iniFile->setIniCodec(codec);
}

void ecvSettingManager::setIniCodec(const char* codecName) {
    s_manager.instance->m_iniFile->setIniCodec(codecName);
}

QTextCodec* ecvSettingManager::iniCoxdec() const {
    return s_manager.instance->m_iniFile->iniCodec();
}
