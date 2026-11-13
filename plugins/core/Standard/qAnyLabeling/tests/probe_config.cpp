// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <aicore/rfdetr_capi.h>
#include <aicore/yolo_capi.h>
#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileInfo>
#include <QFrame>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPluginLoader>
#include <QSettings>
#include <QTemporaryDir>
#include <cstdio>

#include "AnnotationClassifierPrompt.h"
#include "AnnotationConfig.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    if (argc < 2 || argc > 4) return 2;
    QTemporaryDir settingsDir;
    if (!settingsDir.isValid()) return 1;
    QCoreApplication::setOrganizationName(QStringLiteral("ACloudViewerTests"));
    QCoreApplication::setApplicationName(
            QStringLiteral("qAnyLabelingDefaults"));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       settingsDir.path());
    QPluginLoader loader(
            QFileInfo(QString::fromLocal8Bit(argv[1])).absoluteFilePath());
    auto* plugin = qobject_cast<ccStdPluginInterface*>(loader.instance());
    if (!plugin) {
        std::fprintf(stderr, "%s\n", qPrintable(loader.errorString()));
        return 1;
    }
    plugin->getActions().first()->trigger();
    QApplication::processEvents();
    QMainWindow* window = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets())
        if (widget->objectName() == QStringLiteral("qAnyLabelingWorkspace"))
            window = qobject_cast<QMainWindow*>(widget);
    if (!window) return 1;
    QJsonObject report, shortcuts, checks;
    for (QAction* action : window->findChildren<QAction*>()) {
        const QString id = action->property("qAnyShortcutId").toString();
        if (id.isEmpty()) {
            const QMap<QString, QString> configActions = {
                    {QStringLiteral("Save Automatically"),
                     QStringLiteral("auto_save")},
                    {QStringLiteral("Save With Image Data"),
                     QStringLiteral("store_data")},
                    {QStringLiteral("Keep Previous Brightness"),
                     QStringLiteral("keep_prev_brightness")},
                    {QStringLiteral("Keep Previous Contrast"),
                     QStringLiteral("keep_prev_contrast")},
                    {QStringLiteral("Keep Previous Scale"),
                     QStringLiteral("keep_prev_scale")}};
            if (configActions.contains(action->text()))
                checks.insert(configActions.value(action->text()),
                              action->isChecked());
            continue;
        }
        QJsonArray sequences;
        for (const QKeySequence& key : action->shortcuts())
            sequences.append(key.toString(QKeySequence::PortableText));
        shortcuts.insert(id, sequences);
        if (action->isCheckable()) checks.insert(id, action->isChecked());
    }
    auto* model = window->findChild<QComboBox*>(
            QStringLiteral("AnnotationModelFamily"));
    auto* confidence = window->findChild<QDoubleSpinBox*>(
            QStringLiteral("AnnotationConfidence"));
    auto* iou =
            window->findChild<QDoubleSpinBox*>(QStringLiteral("AnnotationIou"));
    if (!model || !confidence || !iou) return 1;
    QJsonObject models;
    for (int i = 0; i < model->count(); ++i) {
        model->setCurrentIndex(i);
        QJsonObject defaults;
        defaults.insert(QStringLiteral("confidence"), confidence->value());
        if (!iou->isHidden())
            defaults.insert(QStringLiteral("iou"), iou->value());
        models.insert(model->itemText(i), defaults);
    }
    QJsonObject docks;
    for (QDockWidget* dock : window->findChildren<QDockWidget*>()) {
        QJsonObject state;
        state.insert(QStringLiteral("show"), !dock->isHidden());
        state.insert(QStringLiteral("closable"),
                     bool(dock->features() & QDockWidget::DockWidgetClosable));
        state.insert(QStringLiteral("movable"),
                     bool(dock->features() & QDockWidget::DockWidgetMovable));
        state.insert(QStringLiteral("floatable"),
                     bool(dock->features() & QDockWidget::DockWidgetFloatable));
        docks.insert(dock->objectName(), state);
    }
    report.insert(QStringLiteral("defaults"), annotationConfig::defaults());
    report.insert(QStringLiteral("shortcuts"), shortcuts);
    report.insert(QStringLiteral("checks"), checks);
    report.insert(QStringLiteral("models"), models);
    report.insert(QStringLiteral("docks"), docks);
    QJsonArray classifierPrompts;
    for (const QStringList& labels :
         {QStringList{},
          QStringList{QStringLiteral("cat"), QStringLiteral("dog")},
          QStringList{QString::fromUtf8("类别甲"), QStringLiteral("a\"b")}}) {
        for (const bool singleLabel : {true, false})
            classifierPrompts.append(QJsonObject{
                    {"labels", QJsonArray::fromStringList(labels)},
                    {"single_label", singleLabel},
                    {"prompt",
                     annotationClassifierPrompt(labels, singleLabel)}});
    }
    report.insert(QStringLiteral("classifier_prompts"), classifierPrompts);
    report.insert(QStringLiteral("qt_version"),
                  QString::fromLatin1(qVersion()));
    report.insert(QStringLiteral("window_font"),
                  QJsonObject{{"family", window->font().family()},
                              {"point_size", window->font().pointSizeF()},
                              {"pixel_size", window->font().pixelSize()},
                              {"logical_dpi", window->logicalDpiY()}});
    auto* dark =
            window->findChild<QAction*>(QStringLiteral("AnnotationTheme_dark"));
    auto* light = window->findChild<QAction*>(
            QStringLiteral("AnnotationTheme_light"));
    if (!dark || !light) return 1;
    auto panelStyles = [&]() {
        QJsonObject styles;
        auto* dock =
                window->findChild<QDockWidget*>(QStringLiteral("Description"));
        auto* panel = window->findChild<QFrame*>(
                QStringLiteral("AnnotationSidebarSection"));
        auto* description = window->findChild<QPlainTextEdit*>(
                QStringLiteral("AnnotationDescription"));
        auto* checkbox = window->findChild<QCheckBox*>(
                QStringLiteral("AnnotationDescriptionVisibility"));
        auto* canvas =
                window->findChild<QWidget*>(QStringLiteral("AnnotationCanvas"));
        if (!dock || !panel || !description || !checkbox || !canvas)
            return styles;
        styles.insert("dock", dock->styleSheet());
        styles.insert("panel", panel->styleSheet());
        styles.insert("description", description->styleSheet());
        styles.insert("checkbox", checkbox->styleSheet());
        QApplication::processEvents();
        const auto pixels = canvas->grab().toImage();
        styles.insert("canvas_background",
                      pixels.pixelColor(pixels.width() / 2, pixels.height() / 2)
                              .name());
        return styles;
    };
    dark->trigger();
    const auto darkPanels = panelStyles();
    if (argc == 4) {
        model->setCurrentIndex(0);
        window->resize(1280, 800);
        QApplication::processEvents();
        if (!window->grab().save(QString::fromLocal8Bit(argv[3]))) return 1;
    }
    QJsonObject colors;
    const QPalette palette = window->palette();
    const QMap<QString, QPalette::ColorRole> roles = {
            {"Window", QPalette::Window},
            {"WindowText", QPalette::WindowText},
            {"Base", QPalette::Base},
            {"AlternateBase", QPalette::AlternateBase},
            {"Text", QPalette::Text},
            {"BrightText", QPalette::BrightText},
            {"Button", QPalette::Button},
            {"ButtonText", QPalette::ButtonText},
            {"Highlight", QPalette::Highlight},
            {"HighlightedText", QPalette::HighlightedText},
            {"ToolTipBase", QPalette::ToolTipBase},
            {"ToolTipText", QPalette::ToolTipText},
            {"PlaceholderText", QPalette::PlaceholderText},
            {"Mid", QPalette::Mid},
            {"Dark", QPalette::Dark},
            {"Shadow", QPalette::Shadow}};
    const QMap<QString, QPalette::ColorGroup> groups = {
            {"Active", QPalette::Active},
            {"Inactive", QPalette::Inactive},
            {"Disabled", QPalette::Disabled}};
    for (auto group = groups.begin(); group != groups.end(); ++group)
        for (auto role = roles.begin(); role != roles.end(); ++role)
            colors.insert(group.key() + '/' + role.key(),
                          palette.color(group.value(), role.value()).name());
    report.insert(QStringLiteral("dark_palette"), colors);
    report.insert(QStringLiteral("dark_stylesheet_sha256"),
                  QString::fromLatin1(QCryptographicHash::hash(
                                              window->styleSheet().toUtf8(),
                                              QCryptographicHash::Sha256)
                                              .toHex()));
    light->trigger();
    report.insert(QStringLiteral("panel_styles"),
                  QJsonObject{{"dark", darkPanels}, {"light", panelStyles()}});
    report.insert(QStringLiteral("light_stylesheet_empty"),
                  window->styleSheet().isEmpty());
    if (shortcuts.size() < 80 || models.isEmpty() ||
        !checks.value(QStringLiteral("auto_save")).toBool() ||
        checks.value(QStringLiteral("store_data")).toBool() ||
        !checks.value(QStringLiteral("show_labels")).toBool()) {
        std::fprintf(stderr,
                     "Cold-start workspace configuration is incomplete\n");
        return 1;
    }
    const QByteArray data = QJsonDocument(report).toJson();
    if (argc >= 3) {
        QFile file(QString::fromLocal8Bit(argv[2]));
        if (!file.open(QIODevice::WriteOnly) || file.write(data) != data.size())
            return 1;
    }
    std::printf("Default probe: %d shortcuts, %d model families, %d docks\n",
                shortcuts.size(), models.size(), docks.size());
    window->close();
    return 0;
}
