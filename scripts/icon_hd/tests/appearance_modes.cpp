// SPDX-License-Identifier: MIT
// Real application-owner regression; launched by verify_appearance_modes.py.
#include <QApplication>
#include <QComboBox>
#include <QDebug>
#include <QDialog>
#include <QFile>
#include <QFocusFrame>
#include <QPointer>
#include <QPushButton>
#include <QResource>
#include <QSettings>
#include <QStandardItemModel>
#include <QStyleFactory>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QTest>
#include <QTreeView>

#include "ecvApplicationBase.h"
#include "ecvDisplayOptionsDlg.h"
#include "ecvSettingManager.h"

#define REQUIRE(x)                                        \
    do {                                                  \
        if (!(x)) {                                       \
            qCritical() << "FAIL line" << __LINE__ << #x; \
            return 2;                                     \
        }                                                 \
    } while (0)
int main(int argc, char **argv) {
    QTemporaryDir config;
    QString configPath =
            argc == 3 ? QString::fromLocal8Bit(argv[1]) + "/startup-config"
                      : config.path();
    QSettings::setPath(QSettings::NativeFormat, QSettings::UserScope,
                       configPath);
    ecvApplicationBase app(argc, argv, false,
                           QStringLiteral("AppearanceProbe"));
    app.setOrganizationName("ACloudViewerIsolatedProbe");
    app.setApplicationName("NativeCustom");
    REQUIRE(argc == 2 || argc == 3);
    QString out = argv[1];
    ecvSettingManager::Init(QStringLiteral("appearance-probe.ini"));
    if (argc == 3) {
        QString phase = argv[2];
        if (phase == "save-dark") {
            REQUIRE(app.setAppStyle("Custom"));
            REQUIRE(app.setCustomTheme("QDarkStyleSheet::Dark"));
            qInfo() << "PASS saved Custom Dark in isolated startup settings";
        } else {
            REQUIRE(app.restoreAppearance());
            REQUIRE(app.isCustomAppearance());
            REQUIRE(app.appearance().themeId == "QDarkStyleSheet::Dark");
            REQUIRE(app.styleSheet().isEmpty());
            REQUIRE(QString(app.style()->metaObject()->className())
                            .contains("QlementineStyle"));
            qInfo() << "PASS separate process restored Custom Dark once";
        }
        ecvSettingManager::ReleaseInstance();
        return 0;
    }
    REQUIRE(app.restoreAppearance());
    REQUIRE(!app.isCustomAppearance());
    REQUIRE(app.styleSheet().isEmpty());
    // First-run/default style is a real restorable state, never an index -1.
    {
        const auto savedDefault =
                ecvSettingManager::getValue("Appearance", "state");
        ccDisplayOptionsDlg dialog(nullptr);
        auto *combo = dialog.findChild<QComboBox *>("appStyleComboBox");
        REQUIRE(combo && combo->currentData().toString().isEmpty());
        combo->setCurrentIndex(combo->findData("Custom"));
        dialog.show();
        app.processEvents();
        auto *apply = dialog.findChild<QPushButton *>("applyButton");
        auto *cancel = dialog.findChild<QPushButton *>("cancelButton");
        REQUIRE(apply && cancel);
        QTest::mouseClick(apply, Qt::LeftButton);
        app.processEvents();
        REQUIRE(app.isCustomAppearance());
        QTest::mouseClick(cancel, Qt::LeftButton);
        app.processEvents();
        REQUIRE(app.appearance().styleKey.isEmpty());
        REQUIRE(app.styleSheet().isEmpty());
        REQUIRE(ecvSettingManager::getValue("Appearance", "state") ==
                savedDefault);
    }
    REQUIRE(app.setAppStyle("Fusion"));
    REQUIRE(app.styleSheet().isEmpty());
    REQUIRE(!app.property("ecvCustomAppearance").toBool());
    const auto native = app.appearance();
    const auto nativeFont = app.font();
    const auto savedNative = ecvSettingManager::getValue("Appearance", "state");
    REQUIRE(!app.setCustomTheme(":/qss/nord.css"));
    REQUIRE(app.styleSheet().isEmpty());
    REQUIRE(ecvSettingManager::getValue("Appearance", "state") == savedNative);
    REQUIRE(app.setAppStyle("Custom"));
    REQUIRE(app.isCustomAppearance());
    REQUIRE(app.styleSheet().isEmpty());
    REQUIRE(QString(app.style()->metaObject()->className())
                    .contains("QlementineStyle"));
    REQUIRE(app.setCustomTheme(":/qss/nord.css"));
    if (nativeFont.pointSizeF() > 0) REQUIRE(app.font().pointSizeF() > 0);
    // Application changes made while Custom is active must outlive unpolish.
    QPushButton dynamicConstraints("Application-owned constraints");
    dynamicConstraints.setMinimumWidth(10);
    dynamicConstraints.ensurePolished();
    dynamicConstraints.setMinimumWidth(222);
    dynamicConstraints.setFocusPolicy(Qt::StrongFocus);
    REQUIRE(app.applyAppearance(native, false));
    REQUIRE(dynamicConstraints.minimumWidth() == 222);
    REQUIRE(dynamicConstraints.focusPolicy() == Qt::StrongFocus);
    REQUIRE(app.setAppStyle("Custom"));
    REQUIRE(app.setCustomTheme(":/qss/nord.css"));
    const auto nord = app.appearance();
    const auto savedNord = ecvSettingManager::getValue("Appearance", "state");
    REQUIRE(!app.setCustomTheme(":/qss/does-not-exist.css"));
    REQUIRE(ecvSettingManager::getValue("Appearance", "state") == savedNord);
    REQUIRE(app.appearance().themeId == nord.themeId);
    REQUIRE(!app.setAppStyle("not-a-Qt-style"));
    REQUIRE(app.appearance().styleKey == "Custom");
    QWidget localOverride;
    QPalette localColors;
    localColors.setColor(QPalette::Text, Qt::magenta);
    localOverride.setPalette(localColors);
    REQUIRE(app.applyAppearance(native, false));
    REQUIRE(localOverride.palette().color(QPalette::Text) ==
            QColor(Qt::magenta));
    REQUIRE(app.styleSheet().isEmpty());
    REQUIRE(ecvSettingManager::getValue("Appearance", "state") == savedNord);
    REQUIRE(app.restoreAppearance());
    REQUIRE(app.appearance().styleKey == "Custom");
    REQUIRE(app.appearance().themeId == nord.themeId);
    // Construct the real Display Options dialog and trigger actual button
    // events.
    for (const QString &closeMethod :
         {QStringLiteral("cancel"), QStringLiteral("escape"),
          QStringLiteral("close")}) {
        ccDisplayOptionsDlg dialog(nullptr);
        auto *combo = dialog.findChild<QComboBox *>("appStyleComboBox");
        REQUIRE(combo);
        REQUIRE(combo->findData("Custom") >= 0);
        REQUIRE(combo->currentData().toString() == "Custom");
        auto *tabs = dialog.findChild<QTabWidget *>();
        REQUIRE(tabs);
        tabs->setCurrentIndex(tabs->count() - 1);
        dialog.show();
        app.processEvents();
        if (closeMethod == "cancel")
            dialog.grab().save(out + "/display-options-custom.png");
        combo->setCurrentIndex(combo->findData("Fusion"));
        dialog.show();
        app.processEvents();
        auto *apply = dialog.findChild<QPushButton *>("applyButton");
        REQUIRE(apply);
        dialog.activateWindow();
        apply->setFocus(Qt::TabFocusReason);
        QTest::qWait(50);
        REQUIRE(apply->hasFocus());
        QTest::mouseClick(apply, Qt::LeftButton);
        app.processEvents();
        REQUIRE(!app.isCustomAppearance());
        REQUIRE(app.styleSheet().isEmpty());
        REQUIRE(ecvSettingManager::getValue("Appearance", "state") ==
                savedNord);
        auto *opacity = dialog.findChild<QWidget *>("bbOpacityDoubleSpinBox");
        REQUIRE(opacity);
        REQUIRE(opacity->palette().color(QPalette::Disabled, QPalette::Base) ==
                app.palette().color(QPalette::Disabled, QPalette::Base));
        REQUIRE(opacity->palette().color(QPalette::Disabled, QPalette::Text) ==
                app.palette().color(QPalette::Disabled, QPalette::Text));
        if (closeMethod == "cancel")
            dialog.grab().save(out + "/display-options-fusion.png");
        if (closeMethod == "cancel") {
            auto *button = dialog.findChild<QPushButton *>("cancelButton");
            REQUIRE(button);
            QTest::mouseClick(button, Qt::LeftButton);
        } else if (closeMethod == "escape")
            QTest::keyClick(&dialog, Qt::Key_Escape);
        else
            dialog.close();
        app.processEvents();
        REQUIRE(app.isCustomAppearance());
        REQUIRE(app.appearance().themeId == nord.themeId);
        REQUIRE(ecvSettingManager::getValue("Appearance", "state") ==
                savedNord);
    }
    for (int iteration = 0; iteration < 5; ++iteration) {
        REQUIRE(app.applyAppearance(nord, false));
        ccDisplayOptionsDlg dialog(nullptr);
        auto *combo = dialog.findChild<QComboBox *>("appStyleComboBox");
        REQUIRE(combo);
        combo->setCurrentIndex(combo->findData("Fusion"));
        dialog.show();
        app.processEvents();
        auto *ok = dialog.findChild<QPushButton *>("okButton");
        REQUIRE(ok);
        dialog.activateWindow();
        ok->setFocus(Qt::TabFocusReason);
        QTest::qWait(50);
        REQUIRE(ok->hasFocus());
        QList<QPointer<QFocusFrame>> focusFrames;
        for (QWidget *widget : QApplication::allWidgets()) {
            if (auto *frame = qobject_cast<QFocusFrame *>(widget)) {
                if (frame->widget() == ok) focusFrames.append(frame);
            }
        }
        REQUIRE(!focusFrames.isEmpty());
        QTest::mouseClick(ok, Qt::LeftButton);
        app.processEvents();
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
        for (const auto &frame : focusFrames) REQUIRE(frame.isNull());
        REQUIRE(app.appearance().styleKey.compare("Fusion",
                                                  Qt::CaseInsensitive) == 0);
        REQUIRE(app.styleSheet().isEmpty());
        REQUIRE(app.restoreAppearance());
        REQUIRE(!app.isCustomAppearance());
    }
    // Existing native selections win once over legacy menu CSS during
    // migration.
    ecvSettingManager::removeNode("Appearance");
    ecvSettingManager::setValue("AppStyle", "style", "Fusion");
    ecvSettingManager::setValue("ThemeSettings", "CurrentTheme",
                                ":/qss/nord.css");
    REQUIRE(app.restoreAppearance());
    REQUIRE(!app.isCustomAppearance());
    REQUIRE(app.styleSheet().isEmpty());
    ecvSettingManager::removeNode("Appearance");
    ecvSettingManager::setValue("AppStyle", "style", "QDarkStyleSheet::Dark");
    REQUIRE(app.restoreAppearance());
    REQUIRE(app.isCustomAppearance());
    REQUIRE(app.appearance().themeId == "QDarkStyleSheet::Dark");
    if (QStyleFactory::keys().contains("Windows", Qt::CaseInsensitive)) {
        REQUIRE(app.setAppStyle("Windows"));
        REQUIRE(app.styleSheet().isEmpty());
        REQUIRE(app.style()->objectName().compare("Windows",
                                                  Qt::CaseInsensitive) == 0);
    }
    qInfo() << "PASS native/custom gate, single persistent state, invalid "
               "input unchanged, real Display Options "
               "Apply/Cancel/Escape/Close/OK, legacy migration, Fusion/Windows "
               "restored";
    ecvSettingManager::ReleaseInstance();
    return 0;
}
