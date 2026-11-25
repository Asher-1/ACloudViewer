// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Shortcuts/ecvKeySequences.h>
#include <Shortcuts/ecvModalShortcut.h>
#include <Shortcuts/ecvShortcutDecorator.h>
#include <VTKExtensions/Widgets/VtkShortcutRegistry.h>
#include <gtest/gtest.h>

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDebug>
#include <QFile>
#include <QFrame>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeySequenceEdit>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QTableView>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QXmlStreamReader>

#include "ecvPythonCodeEditor.h"
#include "ecvPythonView.h"
#include "ecvShortcutDialog.h"
#include "test_utility/QtEvents.h"

using cloudviewer::tests::pressSequence;
using cloudviewer::tests::SignalCounter;
using cloudviewer::tests::waitForWindowActive;

class ShortcutTests : public ::testing::Test {
protected:
    QTemporaryDir settings;
    ecvKeySequences& registry = ecvKeySequences::instance();
    void bind(QAction& action,
              QWidget& widget,
              const QString& id,
              const QString& key,
              Qt::ShortcutContext context = Qt::WindowShortcut) {
        action.setObjectName(id);
        action.setShortcut(QKeySequence(key));
        action.setShortcutContext(context);
        widget.addAction(&action);
        ASSERT_TRUE(registry.registerCommand(id, &action, "Test"));
    }
    void SetUp() override {
        QCoreApplication::setOrganizationName("ACloudViewerShortcutTests");
        QCoreApplication::setApplicationName("IsolatedSettings");
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                           settings.path());
        QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope,
                           settings.path());
        QSettings checker;
        checker.setFallbacksEnabled(false);
        ASSERT_TRUE(settings.isValid());
        ASSERT_TRUE(checker.fileName().startsWith(settings.path() + "/"));
    }
};

TEST_F(ShortcutTests, pluginIdentityAndFeatureFilters) {
    QWidget workspace, otherWorkspace;
    QAction file("Open", &workspace), canvas("Draw", &workspace);
    QAction other("Open", &otherWorkspace);
    workspace.addActions({&file, &canvas});
    otherWorkspace.addAction(&other);
    const QString plugin = "tests.plugin.filters";
    ASSERT_TRUE(registry.registerCommand("existing.plugin.open", &file, "Old"));
    ASSERT_TRUE(registry.registerPluginCommand(plugin, "open", &file, "File",
                                               "Annotation"));
    ASSERT_TRUE(registry.registerPluginCommand(plugin, "draw", &canvas,
                                               "Canvas", "Annotation"));
    ASSERT_TRUE(registry.registerPluginCommand("tests.plugin.other", "open",
                                               &other, "File", "Other"));
    EXPECT_EQ(file.property("shortcutCommandId").toString(),
              "existing.plugin.open");
    ecvShortcutDialog dialog({});
    auto* providers = dialog.findChild<QComboBox*>("shortcutPluginFilter");
    auto* categories = dialog.findChild<QComboBox*>("shortcutCategoryFilter");
    auto* table = dialog.findChild<QTableWidget*>();
    ASSERT_NE(providers, nullptr);
    ASSERT_NE(categories, nullptr);
    ASSERT_NE(table, nullptr);
    dialog.selectPlugin(plugin);
    dialog.show();
    dialog.activateWindow();
    ASSERT_TRUE(waitForWindowActive(&dialog));
    EXPECT_EQ(providers->currentData().toString(), plugin);
    EXPECT_EQ(categories->count(), 3);
    int visible = 0;
    for (int row = 0; row < table->rowCount(); ++row)
        visible += !table->isRowHidden(row);
    EXPECT_EQ(visible, 2);
    categories->setCurrentIndex(categories->findData("Canvas"));
    visible = 0;
    for (int row = 0; row < table->rowCount(); ++row)
        visible += !table->isRowHidden(row);
    EXPECT_EQ(visible, 1);
    // A late command updates the selected plugin's category list, without
    // dropping either filter or showing a different plugin's commands.
    QAction history("Undo", &workspace);
    workspace.addAction(&history);
    ASSERT_TRUE(registry.registerPluginCommand(plugin, "undo", &history,
                                               "History", "Annotation"));
    EXPECT_EQ(providers->currentData().toString(), plugin);
    EXPECT_EQ(categories->currentData().toString(), "Canvas");
    EXPECT_EQ(categories->count(), 4);
    EXPECT_FALSE(registry.registerPluginCommand("different.owner", "draw",
                                                &canvas, "Canvas"));
}

TEST_F(ShortcutTests, pluginEditorScopesShadowOnlyTheirOwnWorkspace) {
    QWidget workspace;
    QLineEdit editor(&workspace), outside(&workspace);
    editor.setGeometry(0, 0, 120, 25);
    outside.setGeometry(0, 35, 120, 25);
    QAction save("Workspace save", &workspace),
            blockSave("Block save", &editor);
    workspace.addAction(&save);
    editor.addAction(&blockSave);
    const QKeySequence key("Ctrl+Alt+8");
    save.setShortcut(key);
    blockSave.setShortcut(key);
    blockSave.setShortcutContext(Qt::WidgetWithChildrenShortcut);
    const QString owner = "tests.plugin.scopes";
    ASSERT_TRUE(registry.registerPluginCommand(owner, "a-workspace-save", &save,
                                               "File"));
    ASSERT_TRUE(registry.registerPluginCommand(owner, "z-block-save",
                                               &blockSave, "Editor"));
    ASSERT_TRUE(registry.conflicts(registry.bindings()).isEmpty());
    SignalCounter saved(&save, &QAction::triggered),
            blockSaved(&blockSave, &QAction::triggered);
    workspace.show();
    workspace.activateWindow();
    editor.setFocus();
    ASSERT_TRUE(waitForWindowActive(&workspace));
    ASSERT_EQ(QApplication::focusWidget(), &editor);
    pressSequence(&editor, key);
    EXPECT_EQ(saved.count(), 0);
    EXPECT_EQ(blockSaved.count(), 1);
    outside.setFocus();
    QApplication::processEvents();
    pressSequence(&outside, key);
    EXPECT_EQ(saved.count(), 1);
    EXPECT_EQ(blockSaved.count(), 1);
    editor.setFocus();
    blockSave.setEnabled(false);
    QApplication::processEvents();
    pressSequence(&editor, key);
    EXPECT_EQ(saved.count(), 2);
    EXPECT_EQ(blockSaved.count(), 1);
    blockSave.setEnabled(true);
    auto candidate = registry.bindings();
    const QString blockId = blockSave.property("shortcutCommandId").toString();
    const QKeySequence replacement("Ctrl+Alt+9");
    candidate[blockId] = {replacement};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    pressSequence(&editor, key);
    pressSequence(&editor, replacement);
    EXPECT_EQ(saved.count(), 3);
    EXPECT_EQ(blockSaved.count(), 2);
    // Another plugin does not acquire the same shadowing exemption.
    QAction foreign("Foreign editor", &editor);
    editor.addAction(&foreign);
    foreign.setShortcutContext(Qt::WidgetWithChildrenShortcut);
    ASSERT_TRUE(registry.registerPluginCommand("tests.plugin.foreign", "save",
                                               &foreign, "Editor"));
    candidate = registry.bindings();
    candidate[foreign.property("shortcutCommandId").toString()] = {key};
    EXPECT_FALSE(registry.conflicts(candidate).isEmpty());
}

TEST_F(ShortcutTests,
       pluginLegacyMigrationUsesCentralTransactionAndPrecedence) {
    QWidget workspace;
    QAction save("Save", &workspace), open("Open", &workspace);
    workspace.addActions({&save, &open});
    save.setShortcut(QKeySequence("Ctrl+Alt+1"));
    open.setShortcut(QKeySequence("Ctrl+Alt+2"));
    const QString owner = "tests.plugin.migration";
    ASSERT_TRUE(registry.registerPluginCommand(owner, "save", &save, "File"));
    ASSERT_TRUE(registry.registerPluginCommand(owner, "open", &open, "File"));
    const QString saveId = save.property("shortcutCommandId").toString();
    const QString openId = open.property("shortcutCommandId").toString();
    QSettings legacy;
    legacy.setValue("OldPlugin/shortcuts/save",
                    QStringList{"Ctrl+Alt+3", "Ctrl+Alt+4"});
    legacy.sync();
    QStringList errors;
    ASSERT_TRUE(registry.migrateLegacyBindings("OldPlugin/shortcuts",
                                               {{"save", saveId}}, &errors))
            << errors.join("; ").toStdString();
    EXPECT_EQ(save.shortcuts(),
              (QList<QKeySequence>{QKeySequence("Ctrl+Alt+3"),
                                   QKeySequence("Ctrl+Alt+4")}));
    EXPECT_EQ(registry.defaultBindings().value(saveId),
              QList<QKeySequence>{QKeySequence("Ctrl+Alt+1")});
    const auto committed = registry.bindings();
    const auto stored =
            QSettings().value("Shortcuts/commands-v2").toByteArray();
    legacy.setValue("OldPlugin/shortcuts/save", "Ctrl+Alt+5");
    ASSERT_TRUE(registry.migrateLegacyBindings("OldPlugin/shortcuts",
                                               {{"save", saveId}}, &errors));
    EXPECT_EQ(registry.bindings(), committed);
    EXPECT_EQ(QSettings().value("Shortcuts/commands-v2").toByteArray(), stored);
    // A fresh command without a central override rejects an entire malformed
    // legacy batch; it must not apply the valid member or rewrite either store.
    QAction fresh("Fresh", &workspace);
    workspace.addAction(&fresh);
    ASSERT_TRUE(registry.registerPluginCommand(owner, "fresh", &fresh, "File"));
    legacy.setValue("OldPlugin/shortcuts/fresh", "Ctrl+Alt+6");
    legacy.setValue("OldPlugin/shortcuts/alias", "Ctrl+Alt+7");
    const QString freshId = fresh.property("shortcutCommandId").toString();
    const auto beforeFailure = registry.bindings();
    EXPECT_FALSE(registry.migrateLegacyBindings(
            "OldPlugin/shortcuts", {{"fresh", freshId}, {"alias", freshId}},
            &errors));
    EXPECT_EQ(registry.bindings(), beforeFailure);
    EXPECT_EQ(QSettings().value("Shortcuts/commands-v2").toByteArray(), stored);
    EXPECT_EQ(legacy.value("OldPlugin/shortcuts/fresh").toString(),
              "Ctrl+Alt+6");
}

TEST_F(ShortcutTests, factoryUiAndVtkDefaults) {
    QWidget window, render(&window);
    QFile file(QStringLiteral(SHORTCUT_SOURCE_ROOT
                              "/app/ui_templates/MainWindow.ui"));
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    QXmlStreamReader xml(&file);
    QString name;
    bool shortcut = false;
    bool shortcutContext = false;
    while (!xml.atEnd()) {
        xml.readNext();
        if (xml.isStartElement() && xml.name() == QStringLiteral("action"))
            name = xml.attributes().value("name").toString();
        else if (xml.isStartElement() &&
                 xml.name() == QStringLiteral("property")) {
            const auto property = xml.attributes().value("name");
            shortcut =
                    !name.isEmpty() && property == QStringLiteral("shortcut");
            shortcutContext = (name == "actionOpen" || name == "actionSave") &&
                              property == QStringLiteral("shortcutContext");
        } else if (xml.isStartElement() &&
                   xml.name() == QStringLiteral("enum") && shortcutContext) {
            EXPECT_EQ(xml.readElementText(),
                      QString("Qt::ShortcutContext::WindowShortcut"));
        } else if (xml.isStartElement() &&
                   xml.name() == QStringLiteral("string") && shortcut) {
            const auto key = xml.readElementText();
            auto* action = new QAction(name, &window);
            bind(*action, window, "factory." + name, key);
            shortcut = false;
        } else if (xml.isEndElement() && xml.name() == QStringLiteral("action"))
            name.clear();
    }
    ASSERT_TRUE(!xml.hasError());
    for (const auto& def : vtkDefaultShortcuts()) {
        auto* action = new QAction(def.label, &render);
        bind(*action, render, "factory." + def.id,
             def.defaultKey.toString(QKeySequence::PortableText),
             Qt::WidgetShortcut);
    }
    const QPair<const char*, const char*> runtimeDefaults[] = {
            {"PreviousTab", "Ctrl+PgUp"},
            {"NextTab", "Ctrl+PgDown"},
            {"NewTab", "Ctrl+Shift+T"},
            {"CloseTab", "Ctrl+W"},
            {"LayoutFullscreen", "F11"},
            {"ViewFullscreen", "Ctrl+F11"},
            {"Decorations", "Ctrl+D"},
            {"Undo", "Ctrl+Z"},
            {"Redo", "Ctrl+Shift+Z"},
            {"CellsOn", "S"},
            {"PointsOn", "D"},
            {"CellsThrough", "F"},
            {"PointsThrough", "G"},
            {"Blocks", "B"},
            {"BlocksThrough", "N"},
            {"Grow", "+"},
            {"Shrink", "-"},
            {"CancelTool", "Esc"}};
    for (const auto& entry : runtimeDefaults) {
        auto* action = new QAction(entry.first, &window);
        bind(*action, window, "factory.runtime." + QString(entry.first),
             entry.second);
    }
    const auto conflicts = registry.conflicts(registry.defaultBindings());
    for (const auto& conflict : conflicts)
        qWarning() << conflict.first << conflict.second << conflict.sequence;
    ASSERT_TRUE(conflicts.isEmpty()) << "Factory UI/VTK commands must not "
                                        "start with overlapping bindings";
}
TEST_F(ShortcutTests, prefixSecondaryAndAtomicSwap) {
    QWidget window;
    QAction a("A", &window), b("B", &window);
    bind(a, window, "test.a", "Ctrl+K");
    bind(b, window, "test.b", "Ctrl+L");
    const auto initial = registry.bindings();
    auto candidate = initial;
    candidate["test.b"] = {QKeySequence("Ctrl+K, Ctrl+C")};
    EXPECT_EQ(registry.conflicts(candidate).size(), 1);
    QStringList errors;
    ASSERT_TRUE(!registry.applyBindings(candidate, &errors, false));
    EXPECT_EQ(registry.bindings(), initial);
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+L"));
    candidate = initial;
    candidate["test.a"] = {QKeySequence("Ctrl+1"), QKeySequence("Ctrl+L")};
    EXPECT_EQ(registry.conflicts(candidate).size(), 1);
    candidate["test.a"] = {QKeySequence("Ctrl+K, Ctrl+C")};
    candidate["test.b"] = {QKeySequence("Ctrl+K, Ctrl+D")};
    ASSERT_TRUE(registry.conflicts(candidate).isEmpty());
    candidate["test.a"] = initial.value("test.b");
    candidate["test.b"] = initial.value("test.a");
    ASSERT_TRUE(registry.applyBindings(candidate, &errors));
    EXPECT_EQ(a.shortcut(), QKeySequence("Ctrl+L"));
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+K"));
    EXPECT_EQ(registry.defaultBindings().value("test.a"),
              initial.value("test.a"));
}
TEST_F(ShortcutTests, invalidSequencesAndOwnPrefixes) {
    EXPECT_FALSE(registry.isValidSequence(QKeySequence(Qt::CTRL)));
    EXPECT_FALSE(registry.isValidSequence(QKeySequence(Qt::Key_unknown)));
    EXPECT_TRUE(registry.isValidSequence(QKeySequence("Ctrl+K, Ctrl+C")));
    QWidget window;
    QAction action("Validation", &window);
    bind(action, window, "validation.command", "Ctrl+Alt+1");
    auto candidate = registry.bindings();
    const auto initial = candidate;
    candidate["validation.command"] = {QKeySequence("not-a-real-key")};
    ASSERT_TRUE(!registry.applyBindings(candidate, nullptr, false));
    candidate["validation.command"] = {QKeySequence(Qt::CTRL)};
    ASSERT_TRUE(!registry.applyBindings(candidate, nullptr, false));
    candidate["validation.command"] = {QKeySequence("Ctrl+K"),
                                       QKeySequence("Ctrl+K, Ctrl+C")};
    EXPECT_EQ(registry.conflicts(candidate).size(), 1);
    ASSERT_TRUE(!registry.applyBindings(candidate, nullptr, false));
    EXPECT_EQ(registry.bindings(), initial);
    action.setShortcut(QKeySequence("Ctrl+Alt+4"));
    EXPECT_EQ(action.shortcut(), QKeySequence("Ctrl+Alt+1"));
    EXPECT_EQ(registry.bindings(), initial);
}
TEST_F(ShortcutTests, contextAndInstances) {
    QWidget window, otherWindow;
    QWidget left(&window), right(&window);
    QAction a("A", &left), b("B", &right), c("C", &otherWindow);
    bind(a, left, "context.a", "Ctrl+Q", Qt::WidgetShortcut);
    bind(b, right, "context.b", "Ctrl+Q", Qt::WidgetShortcut);
    bind(c, otherWindow, "context.c", "Ctrl+Q");
    ASSERT_TRUE(registry.conflicts(registry.bindings()).isEmpty());
    a.setShortcutContext(Qt::WindowShortcut);
    EXPECT_EQ(registry.conflicts(registry.bindings()).size(), 1);
    a.setShortcutContext(Qt::ApplicationShortcut);
    EXPECT_EQ(registry.conflicts(registry.bindings()).size(), 2);
    a.setShortcutContext(Qt::WidgetShortcut);
    QAction duplicate("Duplicate", &left);
    left.addAction(&duplicate);
    duplicate.setShortcutContext(Qt::WidgetShortcut);
    ASSERT_TRUE(!registry.registerCommand("context.a", &duplicate, "Test"));
    QWidget thirdWindow;
    QAction replica("Replica", &thirdWindow);
    thirdWindow.addAction(&replica);
    replica.setShortcutContext(Qt::WidgetShortcut);
    ASSERT_TRUE(registry.registerCommand("context.a", &replica, "Test"));
    EXPECT_EQ(replica.shortcuts(), a.shortcuts());
}
TEST_F(ShortcutTests, hostFileShortcutsAndPluginEditorHaveSeparateWindows) {
    for (const auto& key : {QKeySequence("Ctrl+O"), QKeySequence("Ctrl+S")}) {
        QWidget host, editor;
        host.setFocusPolicy(Qt::StrongFocus);
        editor.setFocusPolicy(Qt::StrongFocus);
        QAction hostAction("Host file", &host),
                editorAction("Editor file", &editor);
        editor.addAction(&editorAction);
        editorAction.setShortcut(key);
        editorAction.setShortcutContext(Qt::WidgetWithChildrenShortcut);
        // The Python plugin constructs its editor before host registration.
        const QString localId = key.toString(QKeySequence::PortableText);
        ASSERT_TRUE(registry.registerPluginCommand(
                "tests.python-editor", localId, &editorAction, "Editor"));
        bind(hostAction, host, "host.file." + localId, localId);
        EXPECT_EQ(hostAction.shortcut(), key);
        EXPECT_EQ(editorAction.shortcut(), key);
        EXPECT_TRUE(registry.conflicts(registry.bindings()).isEmpty());
        SignalCounter hostHits(&hostAction, &QAction::triggered);
        SignalCounter editorHits(&editorAction, &QAction::triggered);
        host.show();
        host.activateWindow();
        host.setFocus();
        ASSERT_TRUE(waitForWindowActive(&host));
        pressSequence(&host, key);
        EXPECT_EQ(hostHits.count(), 1);
        EXPECT_EQ(editorHits.count(), 0);
        editor.show();
        editor.activateWindow();
        editor.setFocus();
        ASSERT_TRUE(waitForWindowActive(&editor));
        pressSequence(&editor, key);
        EXPECT_EQ(hostHits.count(), 1);
        EXPECT_EQ(editorHits.count(), 1);
        editor.hide();
        host.activateWindow();
        host.setFocus();
        ASSERT_TRUE(waitForWindowActive(&host));
        pressSequence(&host, key);
        EXPECT_EQ(hostHits.count(), 2);
        EXPECT_EQ(editorHits.count(), 1);
    }
}

TEST_F(ShortcutTests, rebindDispatchChordMetaAndInputFocus) {
    QWidget window;
    QLineEdit input(&window);
    input.setGeometry(0, 0, 150, 30);
    window.setFocusPolicy(Qt::StrongFocus);
    QAction command("Command", &window);
    bind(command, window, "dispatch.command", "Ctrl+Alt+7");
    SignalCounter triggered(&command, &QAction::triggered);
    window.show();
    window.activateWindow();
    window.setFocus();
    ASSERT_TRUE(waitForWindowActive(&window));
    pressSequence(&window, QKeySequence("Ctrl+Alt+7"));
    EXPECT_EQ(triggered.count(), 1);
    auto candidate = registry.bindings();
    candidate["dispatch.command"] = {QKeySequence("Ctrl+K, Ctrl+C"),
                                     QKeySequence("Meta+J")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    pressSequence(&window, QKeySequence("Ctrl+Alt+7"));
    EXPECT_EQ(triggered.count(), 1);
    pressSequence(&window, QKeySequence("Ctrl+K, Ctrl+C"));
    EXPECT_EQ(triggered.count(), 2);
    pressSequence(&window, QKeySequence("Meta+J"));
    EXPECT_EQ(triggered.count(), 3);
    command.setEnabled(false);
    pressSequence(&window, QKeySequence("Meta+J"));
    EXPECT_EQ(triggered.count(), 3);
    command.setEnabled(true);
    candidate["dispatch.command"] = {QKeySequence("S")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    input.setFocus();
    pressSequence(&input, QKeySequence(Qt::Key_S));
    EXPECT_EQ(input.text(), QString("s"));
    EXPECT_EQ(triggered.count(), 3);
}
TEST_F(ShortcutTests, perViewInstancesAndDetachedHost) {
    QWidget window, left(&window), right(&window), leftRender(&left),
            rightRender(&right);
    left.setGeometry(0, 0, 100, 100);
    right.setGeometry(110, 0, 100, 100);
    leftRender.setFocusPolicy(Qt::StrongFocus);
    rightRender.setFocusPolicy(Qt::StrongFocus);
    QAction a("View action", &left), b("View action", &right);
    bind(a, left, "view.test-command", "Ctrl+Alt+0",
         Qt::WidgetWithChildrenShortcut);
    bind(b, right, "view.test-command", "Ctrl+Alt+0",
         Qt::WidgetWithChildrenShortcut);
    SignalCounter first(&a, &QAction::triggered),
            second(&b, &QAction::triggered);
    window.show();
    window.activateWindow();
    leftRender.setFocus();
    ASSERT_TRUE(waitForWindowActive(&window));
    pressSequence(&leftRender, QKeySequence("Ctrl+Alt+0"));
    EXPECT_EQ(first.count(), 1);
    EXPECT_EQ(second.count(), 0);
    auto candidate = registry.bindings();
    candidate["view.test-command"] = {QKeySequence("Meta+Y")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    rightRender.setFocus();
    pressSequence(&rightRender, QKeySequence("Ctrl+Alt+0"));
    EXPECT_EQ(second.count(), 0);
    pressSequence(&rightRender, QKeySequence("Meta+Y"));
    EXPECT_EQ(second.count(), 1);
    EXPECT_EQ(first.count(), 1);
    QWidget host;
    host.setFocusPolicy(Qt::StrongFocus);
    QAction appCommand("Application command", &window);
    bind(appCommand, window, "application.detached", "Ctrl+F11",
         Qt::ApplicationShortcut);
    SignalCounter appTriggered(&appCommand, &QAction::triggered);
    host.show();
    host.activateWindow();
    host.setFocus();
    ASSERT_TRUE(waitForWindowActive(&host));
    pressSequence(&host, QKeySequence("Ctrl+F11"));
    EXPECT_EQ(appTriggered.count(), 1);
    candidate = registry.bindings();
    candidate["application.detached"] = {QKeySequence("Ctrl+K, Ctrl+L")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    pressSequence(&host, QKeySequence("Ctrl+F11"));
    EXPECT_EQ(appTriggered.count(), 1);
    pressSequence(&host, QKeySequence("Ctrl+K, Ctrl+L"));
    EXPECT_EQ(appTriggered.count(), 2);
}
TEST_F(ShortcutTests,
       scopeTransitionsSuspendAndRestoreWithoutChangingSettings) {
    QWidget firstHost, secondHost, first(&firstHost), second(&secondHost),
            render(&second);
    render.setFocusPolicy(Qt::StrongFocus);
    QAction a("First", &first), b("Second", &second);
    bind(a, first, "transition.a", "Ctrl+Alt+8");
    bind(b, second, "transition.b", "Ctrl+Alt+8");
    SignalCounter firstHits(&a, &QAction::triggered),
            secondHits(&b, &QAction::triggered);
    const auto configured = registry.bindings();
    firstHost.show();
    secondHost.show();
    first.show();
    second.show();
    secondHost.activateWindow();
    render.setFocus();
    ASSERT_TRUE(waitForWindowActive(&secondHost));
    pressSequence(&render, QKeySequence("Ctrl+Alt+8"));
    EXPECT_EQ(secondHits.count(), 1);

    second.setParent(&firstHost);
    second.show();
    firstHost.activateWindow();
    render.setFocus();
    ASSERT_TRUE(waitForWindowActive(&firstHost));
    EXPECT_FALSE(a.shortcuts().isEmpty());
    EXPECT_TRUE(b.shortcuts().isEmpty());
    pressSequence(&render, QKeySequence("Ctrl+Alt+8"));
    EXPECT_EQ(firstHits.count(), 1);
    EXPECT_EQ(secondHits.count(), 1);
    EXPECT_EQ(registry.bindings(), configured);

    second.setParent(&secondHost);
    second.show();
    secondHost.activateWindow();
    render.setFocus();
    ASSERT_TRUE(waitForWindowActive(&secondHost));
    EXPECT_EQ(b.shortcuts(), configured.value("transition.b"));
    pressSequence(&render, QKeySequence("Ctrl+Alt+8"));
    EXPECT_EQ(secondHits.count(), 2);

    b.setShortcutContext(Qt::ApplicationShortcut);
    // An inactive window owner cannot block an application-wide command in
    // the currently focused, disjoint window. It regains priority on return.
    EXPECT_EQ(b.shortcuts(), configured.value("transition.b"));
    pressSequence(&render, QKeySequence("Ctrl+Alt+8"));
    EXPECT_EQ(secondHits.count(), 3);
    first.setFocusPolicy(Qt::StrongFocus);
    firstHost.activateWindow();
    first.setFocus();
    ASSERT_TRUE(waitForWindowActive(&firstHost));
    EXPECT_TRUE(b.shortcuts().isEmpty());
    pressSequence(&first, QKeySequence("Ctrl+Alt+8"));
    EXPECT_EQ(firstHits.count(), 2);
    b.setShortcutContext(Qt::WindowShortcut);
    EXPECT_EQ(b.shortcuts(), configured.value("transition.b"));
    second.removeAction(&b);
    first.addAction(&b);
    QCoreApplication::processEvents();
    EXPECT_TRUE(b.shortcuts().isEmpty());
    first.removeAction(&b);
    second.addAction(&b);
    QCoreApplication::processEvents();
    EXPECT_EQ(b.shortcuts(), configured.value("transition.b"));
    EXPECT_EQ(registry.bindings(), configured);
}

TEST_F(ShortcutTests, dialogDefaultsAndAlternatives) {
    QWidget window;
    QAction action("Dialog Command", &window);
    bind(action, window, "dialog.command", "Ctrl+8");
    auto candidate = registry.bindings();
    candidate["dialog.command"] = {QKeySequence("Ctrl+9"),
                                   QKeySequence("Meta+9")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    ecvShortcutEditDialog editor;
    editor.setKeySequences(candidate.value("dialog.command"));
    EXPECT_EQ(editor.keySequences(), candidate.value("dialog.command"));
    auto* list = editor.findChild<QListWidget*>();
    auto* recorder = editor.findChild<QKeySequenceEdit*>();
    recorder->setKeySequence(QKeySequence("Ctrl+0"));
    list->setCurrentRow(1);
    EXPECT_EQ(editor.keySequences().first(), QKeySequence("Ctrl+0"));
    ecvShortcutDialog dialog({&action}, &window);
    auto* table = dialog.findChild<QTableWidget*>();
    int row = -1;
    for (int i = 0; i < table->rowCount(); ++i)
        if (table->item(i, 0)->data(Qt::UserRole).toString() ==
            "dialog.command")
            row = i;
    ASSERT_TRUE(row >= 0);
    ASSERT_TRUE(table->item(row, 1)->text().contains("Meta+9"));
    table->selectRow(row);
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "onResetSelected",
                                          Qt::DirectConnection));
    EXPECT_EQ(action.shortcut(), QKeySequence("Ctrl+8"));
    dialog.show();
    QAction dynamic("Dynamic", &window);
    bind(dynamic, window, "dialog.dynamic", "Ctrl+Alt+8");
    ASSERT_TRUE(table->rowCount() >= 2);
}
TEST_F(ShortcutTests, transientToolOwnership) {
    QWidget window, left(&window), right(&window);
    left.setFocusPolicy(Qt::StrongFocus);
    right.setFocusPolicy(Qt::StrongFocus);
    QAction a("Left", &left), b("Right", &right), tool("Tool", &left);
    bind(a, left, "transient.left", "Ctrl+Alt+5", Qt::WidgetShortcut);
    bind(b, right, "transient.right", "Ctrl+Alt+5", Qt::WidgetShortcut);
    SignalCounter leftTriggered(&a, &QAction::triggered),
            rightTriggered(&b, &QAction::triggered);
    SignalCounter toolTriggered(&tool, &QAction::triggered);
    window.show();
    left.show();
    right.show();
    window.activateWindow();
    left.setFocus();
    ASSERT_TRUE(waitForWindowActive(&window));
    auto* modal =
            registry.addModalShortcut(QKeySequence("Ctrl+Alt+5"), &tool, &left);
    modal->setContextWidget(&left, Qt::WidgetShortcut);
    ASSERT_TRUE(a.shortcuts().isEmpty());
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+Alt+5"));
    EXPECT_EQ(registry.bindings().value("transient.left").first(),
              QKeySequence("Ctrl+Alt+5"));
    pressSequence(&left, QKeySequence("Ctrl+Alt+5"));
    EXPECT_EQ(toolTriggered.count(), 1);
    EXPECT_EQ(leftTriggered.count(), 0);
    right.setFocus();
    pressSequence(&right, QKeySequence("Ctrl+Alt+5"));
    EXPECT_EQ(rightTriggered.count(), 1);
    QAction global("Window Command", &window);
    bind(global, window, "transient.window", "Ctrl+Shift+7");
    SignalCounter globalTriggered(&global, &QAction::triggered);
    left.setFocus();
    auto* localOwner = registry.addModalShortcut(QKeySequence("Ctrl+Shift+7"),
                                                 &tool, &left);
    localOwner->setContextWidget(&left, Qt::WidgetShortcut);
    ASSERT_TRUE(global.shortcuts().isEmpty());
    right.setFocus();
    EXPECT_EQ(global.shortcut(), QKeySequence("Ctrl+Shift+7"));
    pressSequence(&right, QKeySequence("Ctrl+Shift+7"));
    EXPECT_EQ(globalTriggered.count(), 1);
    delete localOwner;
    auto candidate = registry.bindings();
    candidate["transient.left"] = {QKeySequence("Ctrl+Alt+6")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    left.setFocus();
    pressSequence(&left, QKeySequence("Ctrl+Alt+6"));
    EXPECT_EQ(leftTriggered.count(), 1);
    modal->setEnabled(false);
    EXPECT_EQ(a.shortcut(), QKeySequence("Ctrl+Alt+6"));
    modal->setEnabled(true, false);
    delete modal;
    EXPECT_EQ(a.shortcut(), QKeySequence("Ctrl+Alt+6"));
}
TEST_F(ShortcutTests, dynamicRegistrationAndLifetime) {
    QSettings().remove("Shortcuts/commands-v2");
    ASSERT_TRUE(registry.restoreBindings());
    QWidget window, otherWindow;
    QAction a("A", &window), b("B", &window);
    bind(a, window, "dynamic.a", "Ctrl+Alt+2");
    bind(b, window, "dynamic.b", "Ctrl+Alt+2");
    ASSERT_TRUE(b.shortcuts().isEmpty());
    ASSERT_TRUE(registry.bindings().value("dynamic.b").isEmpty());
    EXPECT_EQ(registry.defaultBindings().value("dynamic.b").first(),
              QKeySequence("Ctrl+Alt+2"));
    auto candidate = registry.bindings();
    candidate["dynamic.b"] = {QKeySequence("Ctrl+Alt+3")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    auto* replica = new QAction("Replica", &otherWindow);
    bind(*replica, otherWindow, "dynamic.b", "Ctrl+Alt+2");
    EXPECT_EQ(replica->shortcut(), QKeySequence("Ctrl+Alt+3"));
    QObject::connect(&b, &QAction::changed, &window, [&replica]() {
        delete replica;
        replica = nullptr;
    });
    candidate["dynamic.b"] = {QKeySequence("Ctrl+Alt+4")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    ASSERT_TRUE(!replica);
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+Alt+4"));
}
TEST_F(ShortcutTests, failedPersistenceAndLegacyMigration) {
    QWidget window;
    QAction action("&Legacy Plugin", &window);
    action.setObjectName("oldPluginAction");
    action.setShortcut(QKeySequence("Ctrl+Alt+8"));
    window.addAction(&action);
    ASSERT_TRUE(registry.registerCommand("plugin/test-owner/command", &action,
                                         "Plugins"));
    QSettings().remove("Shortcuts/commands-v2");
    QSettings().setValue("Shortcuts/oldPluginAction",
                         QKeySequence("Ctrl+Alt+7"));
    ASSERT_TRUE(registry.restoreBindings());
    EXPECT_EQ(action.shortcut(), QKeySequence("Ctrl+Alt+7"));
    const auto before = registry.bindings();
    auto candidate = before;
    candidate["plugin/test-owner/command"] = {QKeySequence("Ctrl+Alt+9")};
    QFile blocker(settings.path() + "/BlockedShortcutSettings");
    ASSERT_TRUE(blocker.open(QIODevice::WriteOnly));
    blocker.close();
    const auto originalOrganization = QCoreApplication::organizationName();
    QCoreApplication::setOrganizationName("BlockedShortcutSettings");
    QStringList errors;
    const bool applied = registry.applyBindings(candidate, &errors);
    const bool pendingWrite = QSettings().contains("Shortcuts/commands-v2");
    QCoreApplication::setOrganizationName(originalOrganization);
    ASSERT_TRUE(!applied);
    ASSERT_TRUE(!errors.isEmpty());
    ASSERT_TRUE(!pendingWrite);
    EXPECT_EQ(registry.bindings(), before);
    EXPECT_EQ(action.shortcut(), QKeySequence("Ctrl+Alt+7"));
}
TEST_F(ShortcutTests, inactiveCommandsAndLocalControls) {
    QWidget window;
    auto* closed = new QAction("Closed Tool", &window);
    bind(*closed, window, "lifetime.closed", "Ctrl+Alt+0");
    delete closed;
    ecvShortcutDialog dialog({}, &window);
    auto* rows = dialog.findChild<QTableWidget*>();
    int row = -1;
    for (int i = 0; i < rows->rowCount(); ++i)
        if (rows->item(i, 0)->data(Qt::UserRole).toString() ==
            "lifetime.closed")
            row = i;
    ASSERT_TRUE(row >= 0);
    auto candidate = registry.bindings();
    candidate["lifetime.closed"] = {QKeySequence("Ctrl+Alt+1")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    QAction reopened("Reopened Tool", &window);
    bind(reopened, window, "lifetime.closed", "Ctrl+Alt+2");
    EXPECT_EQ(reopened.shortcut(), QKeySequence("Ctrl+Alt+1"));
    EXPECT_EQ(registry.defaultBindings().value("lifetime.closed").first(),
              QKeySequence("Ctrl+Alt+0"));

    QWidget python(&window), spreadsheet(&window);
    python.setGeometry(0, 0, 200, 100);
    spreadsheet.setGeometry(210, 0, 200, 100);
    ecvPythonCodeEditor editor(&python);
    QPushButton run("Run", &python);
    QAction runAction("Run", &python);
    bind(runAction, python, "test.python.run", "Ctrl+Return",
         Qt::WidgetWithChildrenShortcut);
    QObject::connect(&runAction, &QAction::triggered, &run,
                     &QPushButton::click);
    SignalCounter executed(&run, &QPushButton::clicked);
    QTableView table(&spreadsheet);
    QAction copy("Copy", &table);
    bind(copy, table, "test.spreadsheet.copy", "Ctrl+C",
         Qt::WidgetWithChildrenShortcut);
    SignalCounter copied(&copy, &QAction::triggered);
    QLineEdit input(&window);
    window.show();
    window.activateWindow();
    editor.setFocus();
    ASSERT_TRUE(waitForWindowActive(&window));
    pressSequence(&editor, QKeySequence("Ctrl+Return"));
    EXPECT_EQ(executed.count(), 1);
    candidate = registry.bindings();
    candidate["test.python.run"] = {QKeySequence("Ctrl+K, Ctrl+P")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    pressSequence(&editor, QKeySequence("Ctrl+Return"));
    EXPECT_EQ(executed.count(), 1);
    pressSequence(&editor, QKeySequence("Ctrl+K, Ctrl+P"));
    EXPECT_EQ(executed.count(), 2);
    table.setFocus();
    pressSequence(&table, QKeySequence("Ctrl+C"));
    EXPECT_EQ(copied.count(), 1);
    input.setText("native editing");
    input.selectAll();
    input.setFocus();
    pressSequence(&input, QKeySequence("Ctrl+C"));
    EXPECT_EQ(copied.count(), 1);
    EXPECT_EQ(QApplication::clipboard()->text(), QString("native editing"));
}
TEST_F(ShortcutTests, importTransactionsAndPendingOwners) {
    QWidget window;
    QAction a("&Old Import A", &window), b("Old Import B", &window);
    bind(a, window, "import.a", "Ctrl+Alt+3");
    bind(b, window, "import.b", "Ctrl+Alt+4");
    SignalCounter changed(&registry, &ecvKeySequences::commandsChanged);
    QStringList errors, missing;
    ASSERT_TRUE(registry.importBindings(
            R"({"version":2,"bindings":{"import.a":["Ctrl+Alt+4"],"import.b":["Ctrl+Alt+3"]}})",
            &errors));
    EXPECT_EQ(changed.count(), 1);
    EXPECT_EQ(a.shortcut(), QKeySequence("Ctrl+Alt+4"));
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+Alt+3"));
    const auto before = registry.bindings();
    const auto stored = QSettings().value("Shortcuts/commands-v2");
    ASSERT_TRUE(!registry.importBindings(
            R"({"version":2,"bindings":{"import.a":["Ctrl+Alt+9"],"import.b":["Ctrl+Alt+9"]}})",
            &errors));
    EXPECT_EQ(registry.bindings(), before);
    EXPECT_EQ(QSettings().value("Shortcuts/commands-v2"), stored);
    EXPECT_EQ(changed.count(), 1);
    ASSERT_TRUE(
            !registry.importBindings(R"({"__vtk__":"malformed"})", &errors));
    ASSERT_TRUE(!registry.importBindings(
            R"({"version":2,"bindings":{"import.a":["Ctrl+Alt+8"],"plugin/absent/invalid":["not-a-key"]}})",
            &errors));
    EXPECT_EQ(registry.bindings(), before);
    ASSERT_TRUE(registry.importBindings(R"({"Old Import A":"Ctrl+Alt+7"})",
                                        &errors));
    EXPECT_EQ(a.shortcut(), QKeySequence("Ctrl+Alt+7"));
    const auto afterLegacy = registry.bindings();
    ASSERT_TRUE(!registry.importBindings(
            R"({"Old Import A":"Ctrl+Alt+8","import.a":"Ctrl+Alt+9"})",
            &errors));
    EXPECT_EQ(registry.bindings(), afterLegacy);
    ASSERT_TRUE(!registry.importBindings(
            R"({"Unknown Legacy Command":"Ctrl+Alt+9"})", &errors));
    ASSERT_TRUE(registry.importBindings(
            R"({"version":2,"bindings":{"plugin/absent/run":["Meta+Shift+9"]}})",
            &errors, &missing));
    EXPECT_EQ(missing, QStringList{"plugin/absent/run"});
    const auto exported =
            QJsonDocument::fromJson(registry.exportBindings()).object();
    ASSERT_TRUE(exported.value("bindings")
                        .toObject()
                        .contains("plugin/absent/run"));
    QAction loaded("Loaded later", &window);
    bind(loaded, window, "plugin/absent/run", "Ctrl+Alt+8");
    EXPECT_EQ(loaded.shortcut(), QKeySequence("Meta+Shift+9"));
    ASSERT_TRUE(registry.importBindings(
            R"({"version":2,"bindings":{"plugin/absent/collision":["Ctrl+Alt+3"]}})",
            &errors, &missing));
    QAction collision("Collision", &window);
    bind(collision, window, "plugin/absent/collision", "Ctrl+Alt+8");
    ASSERT_TRUE(collision.shortcuts().isEmpty());
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+Alt+3"));
}
TEST_F(ShortcutTests, settingsRoundTripAndInvalidRestore) {
    QWidget window;
    QAction action("Settings", &window);
    bind(action, window, "settings.command", "Ctrl+Alt+9");
    auto candidate = registry.bindings();
    candidate["settings.command"] = {};
    ASSERT_TRUE(registry.applyBindings(candidate));
    ASSERT_TRUE(action.shortcuts().isEmpty());
    candidate["settings.command"] = {QKeySequence("Ctrl+Alt+0")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    ASSERT_TRUE(registry.restoreBindings());
    ASSERT_TRUE(action.shortcuts().isEmpty());
    const auto before = registry.bindings();
    QSettings().setValue("Shortcuts/commands-v2", QByteArray("{broken"));
    ASSERT_TRUE(!registry.restoreBindings());
    EXPECT_EQ(registry.bindings(), before);
}
TEST_F(ShortcutTests, ModalContextsAndPrefixes) {
    QWidget window, left(&window), right(&window);
    left.setFocusPolicy(Qt::StrongFocus);
    right.setFocusPolicy(Qt::StrongFocus);
    window.show();
    left.show();
    right.show();
    window.activateWindow();
    left.setFocus();
    ASSERT_TRUE(waitForWindowActive(&window));
    auto* first = registry.addModalShortcut(QKeySequence("Ctrl+K"), nullptr,
                                            &left, Qt::WidgetShortcut);
    first->setContextWidget(&left, Qt::WidgetShortcut);
    auto* disjoint = registry.addModalShortcut(QKeySequence("Ctrl+K"), nullptr,
                                               &right, Qt::WidgetShortcut);
    disjoint->setContextWidget(&right, Qt::WidgetShortcut);
    first->setEnabled(true, false);
    EXPECT_TRUE(first->isEnabled());
    EXPECT_TRUE(disjoint->isEnabled());
    auto* prefix = registry.addModalShortcut(
            QKeySequence("Ctrl+K, Ctrl+C"), nullptr, &left, Qt::WidgetShortcut);
    prefix->setContextWidget(&left, Qt::WidgetShortcut);
    EXPECT_TRUE(prefix->isEnabled());
    EXPECT_FALSE(first->isEnabled());
    EXPECT_TRUE(disjoint->isEnabled());
    delete prefix;
    delete disjoint;
    delete first;
}

TEST_F(ShortcutTests, unavailableOwnersDoNotBlockAndDialogTracksScopes) {
    QWidget firstHost, secondHost, first(&firstHost), second(&secondHost);
    QAction a("First", &first), b("Second", &second);
    bind(a, first, "eligible.a", "Ctrl+Alt+8");
    bind(b, second, "eligible.b", "Ctrl+Alt+8");
    firstHost.show();
    secondHost.show();
    ecvShortcutDialog dialog({&a, &b});
    dialog.show();
    auto* table = dialog.findChild<QTableWidget*>();
    auto rowFor = [&](const QString& id) {
        for (int i = 0; i < table->rowCount(); ++i)
            if (table->item(i, 0)->data(Qt::UserRole).toString() == id)
                return i;
        return -1;
    };
    second.setParent(&firstHost);
    second.show();
    ASSERT_TRUE(b.shortcuts().isEmpty());
    EXPECT_TRUE(
            table->item(rowFor("eligible.b"), 2)->text().contains("Suspended"));
    bool conflictShown = false;
    for (auto* label : dialog.findChildren<QLabel*>())
        conflictShown |=
                label->text().contains("1 conflicting command binding");
    EXPECT_TRUE(conflictShown);
    const auto configured = registry.bindings();
    a.setEnabled(false);
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+Alt+8"));
    EXPECT_EQ(a.shortcuts(), configured.value("eligible.a"));
    EXPECT_TRUE(registry.effectiveBindings("eligible.a").isEmpty());
    a.setEnabled(true);
    EXPECT_TRUE(b.shortcuts().isEmpty());
    a.setVisible(false);
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+Alt+8"));
    a.setVisible(true);
    EXPECT_TRUE(b.shortcuts().isEmpty());
    first.setEnabled(false);
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+Alt+8"));
    first.setEnabled(true);
    first.hide();
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+Alt+8"));
    first.show();
    EXPECT_TRUE(b.shortcuts().isEmpty());
    second.setParent(&secondHost);
    second.show();
    EXPECT_EQ(b.shortcut(), QKeySequence("Ctrl+Alt+8"));
    EXPECT_FALSE(
            table->item(rowFor("eligible.b"), 2)->text().contains("Suspended"));
    EXPECT_EQ(registry.bindings(), configured);
    // Availability must notify the settings view even with no competing owner
    // and no change to the disabled QAction's configured Qt keys.
    SignalCounter updates(&registry, &ecvKeySequences::commandsChanged);
    a.setEnabled(false);
    EXPECT_EQ(updates.count(), 1);
    EXPECT_TRUE(registry.effectiveBindings("eligible.a").isEmpty());
    a.setEnabled(true);
    EXPECT_EQ(updates.count(), 2);
}

TEST_F(ShortcutTests, modalRequestsRestoreInOrderAndCanReclaimPriority) {
    QFrame window;
    window.show();
    window.activateWindow();
    ASSERT_TRUE(waitForWindowActive(&window));
    ecvShortcutDecorator decoration(&window);
    const QKeySequence key("Ctrl+Alt+7");
    auto* first = registry.addModalShortcut(key, nullptr, &window);
    decoration.addShortcut(first);
    SignalCounter firstHits(first, &ecvModalShortcut::activated);
    SignalCounter firstEnabled(first, &ecvModalShortcut::enabled);
    auto* stopped = registry.addModalShortcut(key, nullptr, &window);
    stopped->setEnabled(false, false);
    auto* newest = registry.addModalShortcut(key, nullptr, &window);
    SignalCounter newestHits(newest, &ecvModalShortcut::activated);
    EXPECT_FALSE(first->isEnabled());
    EXPECT_TRUE(first->isRequestedEnabled());
    first->setEnabled(true, false);
    EXPECT_TRUE(first->isEnabled());
    EXPECT_FALSE(newest->isEnabled());
    EXPECT_TRUE(newest->isRequestedEnabled());
    pressSequence(&window, key);
    EXPECT_EQ(firstHits.count(), 1);
    EXPECT_EQ(newestHits.count(), 0);
    EXPECT_EQ(firstEnabled.count(),
              2);  // release stopped, then explicit reclaim
    first->setEnabled(false, false);
    EXPECT_TRUE(newest->isEnabled());
    pressSequence(&window, key);
    EXPECT_EQ(firstHits.count(), 1);
    EXPECT_EQ(newestHits.count(), 1);
    first->setEnabled(true, false);
    newest->setEnabled(true, false);
    delete newest;
    EXPECT_TRUE(first->isEnabled());
    pressSequence(&window, key);
    EXPECT_EQ(firstHits.count(), 2);
    EXPECT_FALSE(stopped->isEnabled());
    EXPECT_FALSE(stopped->isRequestedEnabled());
    first->setContextWidget(nullptr);
    first->setEnabled(false, false);
    first->setContextWidget(&window);
    EXPECT_FALSE(first->isEnabled());
    first->setEnabled(true, false);
    EXPECT_TRUE(first->isEnabled());
    window.hide();
    EXPECT_FALSE(first->isEnabled());
    EXPECT_TRUE(first->isRequestedEnabled());
    window.show();
    EXPECT_TRUE(first->isEnabled());
    delete stopped;
    delete first;
}

TEST_F(ShortcutTests, modalScopeMergeAndPrefixRestoreWithoutKeyEvent) {
    QWidget firstHost, secondHost, first(&firstHost), second(&secondHost);
    firstHost.show();
    secondHost.show();
    auto* prefix = registry.addModalShortcut(QKeySequence("Ctrl+Alt+9"),
                                             nullptr, &first);
    auto* chord = registry.addModalShortcut(QKeySequence("Ctrl+Alt+9, Ctrl+C"),
                                            nullptr, &second);
    EXPECT_TRUE(prefix->isEnabled());
    EXPECT_TRUE(chord->isEnabled());
    SignalCounter prefixHits(prefix, &ecvModalShortcut::activated);
    SignalCounter chordHits(chord, &ecvModalShortcut::activated);
    second.setParent(&firstHost);
    second.show();
    EXPECT_FALSE(prefix->isEnabled());
    EXPECT_TRUE(chord->isEnabled());
    EXPECT_TRUE(prefix->isRequestedEnabled());
    firstHost.activateWindow();
    ASSERT_TRUE(waitForWindowActive(&firstHost));
    pressSequence(&second, QKeySequence("Ctrl+Alt+9, Ctrl+C"));
    EXPECT_EQ(chordHits.count(), 1);
    EXPECT_EQ(prefixHits.count(), 0);
    second.setParent(&secondHost);
    second.show();
    EXPECT_TRUE(prefix->isEnabled());
    EXPECT_TRUE(chord->isEnabled());
    chord->setContextWidget(&first);
    EXPECT_FALSE(prefix->isEnabled());
    delete chord;
    EXPECT_TRUE(prefix->isEnabled());
    firstHost.activateWindow();
    ASSERT_TRUE(waitForWindowActive(&firstHost));
    pressSequence(&first, QKeySequence("Ctrl+Alt+9"));
    EXPECT_EQ(prefixHits.count(), 1);
    delete prefix;
}

TEST_F(ShortcutTests, mixedModalContextsFollowActualFocus) {
    QWidget window;
    QLabel left(&window), right(&window);
    left.setFocusPolicy(Qt::StrongFocus);
    right.setFocusPolicy(Qt::StrongFocus);
    window.show();
    window.activateWindow();
    left.setFocus();
    ASSERT_TRUE(waitForWindowActive(&window));
    const QKeySequence key("Ctrl+Alt+5");
    auto* broad = registry.addModalShortcut(key, nullptr, &window);
    auto* narrow =
            registry.addModalShortcut(key, nullptr, &left, Qt::WidgetShortcut);
    SignalCounter broadHits(broad, &ecvModalShortcut::activated);
    SignalCounter narrowHits(narrow, &ecvModalShortcut::activated);
    pressSequence(&left, key);
    EXPECT_EQ(narrowHits.count(), 1);
    EXPECT_EQ(broadHits.count(), 0);
    right.setFocus();
    QCoreApplication::processEvents();
    EXPECT_TRUE(broad->isEnabled());
    pressSequence(&right, key);
    EXPECT_EQ(broadHits.count(), 1);
    EXPECT_EQ(narrowHits.count(), 1);
    left.setFocus();
    QCoreApplication::processEvents();
    EXPECT_FALSE(broad->isEnabled());
    pressSequence(&left, key);
    EXPECT_EQ(narrowHits.count(), 2);
    delete narrow;
    delete broad;

    auto* application = registry.addModalShortcut(key, nullptr, &window,
                                                  Qt::ApplicationShortcut);
    auto* local = registry.addModalShortcut(key, nullptr, &window,
                                            Qt::WindowShortcut);
    SignalCounter applicationHits(application, &ecvModalShortcut::activated);
    SignalCounter localHits(local, &ecvModalShortcut::activated);
    pressSequence(&left, key);
    EXPECT_EQ(localHits.count(), 1);
    EXPECT_EQ(applicationHits.count(), 0);
    QWidget otherWindow;
    otherWindow.setFocusPolicy(Qt::StrongFocus);
    otherWindow.show();
    otherWindow.activateWindow();
    otherWindow.setFocus();
    ASSERT_TRUE(waitForWindowActive(&otherWindow));
    pressSequence(&otherWindow, key);
    EXPECT_EQ(applicationHits.count(), 1);
    EXPECT_EQ(localHits.count(), 1);
    window.activateWindow();
    left.setFocus();
    ASSERT_TRUE(waitForWindowActive(&window));
    pressSequence(&left, key);
    EXPECT_EQ(localHits.count(), 2);
    EXPECT_EQ(applicationHits.count(), 1);
    delete local;
    delete application;
}

TEST_F(ShortcutTests, hintsPreserveProseAndTrackAlternativesAndClearing) {
    QWidget window;
    QAction action("Select", &window);
    action.setToolTip("Select Some Samples");
    bind(action, window, "hint.command", "S");
    auto candidate = registry.bindings();
    candidate["hint.command"] = {QKeySequence("Ctrl+Alt+6"),
                                 QKeySequence("Meta+6")};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    EXPECT_TRUE(action.toolTip().startsWith("Select Some Samples\n"))
            << action.toolTip().toStdString();
    EXPECT_TRUE(action.toolTip().contains("Meta+6"));
    EXPECT_TRUE(action.toolTip().contains("Ctrl+Alt+6"));
    action.setToolTip("Changed description");
    EXPECT_TRUE(action.toolTip().startsWith("Changed description\n"));
    candidate["hint.command"] = {};
    ASSERT_TRUE(registry.applyBindings(candidate, nullptr, false));
    EXPECT_TRUE(action.toolTip().contains("Unbound"));
    EXPECT_FALSE(action.toolTip().contains("Ctrl+Alt+6"));
    EXPECT_TRUE(action.toolTip().startsWith("Changed description\n"));
}

TEST_F(ShortcutTests, rejectedEditorDraftSurvivesConflictAndFailedSave) {
    QWidget window;
    QAction action("Edit", &window), other("Conflict", &window);
    bind(action, window, "draft.a", "Ctrl+Alt+1");
    bind(other, window, "draft.b", "Ctrl+Alt+2");
    ecvShortcutDialog dialog({&action, &other});
    dialog.show();
    auto* table = dialog.findChild<QTableWidget*>();
    auto* editor = dialog.findChild<ecvShortcutEditDialog*>();
    ASSERT_NE(editor, nullptr);
    QTableWidgetItem* item = nullptr;
    for (int row = 0; row < table->rowCount(); ++row)
        if (table->item(row, 0)->data(Qt::UserRole).toString() == "draft.a")
            item = table->item(row, 0);
    ASSERT_NE(item, nullptr);
    QFile blocker(settings.path() + "/BlockedDraftSettings");
    ASSERT_TRUE(blocker.open(QIODevice::WriteOnly));
    blocker.close();
    const QString organization = QCoreApplication::organizationName();
    const auto initial = registry.bindings();
    const QList<QKeySequence> conflict = {QKeySequence("Ctrl+Alt+2"),
                                          QKeySequence("Meta+1")};
    const QList<QKeySequence> valid = {QKeySequence("Ctrl+Alt+3"),
                                       QKeySequence("Meta+1")};
    auto dismissWarning = [editor]() {
        QTimer::singleShot(0, editor, [editor]() {
            auto* warning = editor->findChild<QMessageBox*>();
            EXPECT_NE(warning, nullptr);
            if (warning) warning->accept();
        });
    };
    QTimer::singleShot(0, editor, [&]() {
        editor->setKeySequences(conflict);
        dismissWarning();
        editor->accept();
        EXPECT_TRUE(editor->isVisible());
        EXPECT_EQ(editor->keySequences(), conflict);
        EXPECT_EQ(registry.bindings(), initial);
        editor->setKeySequences(valid);
        QCoreApplication::setOrganizationName("BlockedDraftSettings");
        dismissWarning();
        editor->accept();
        QCoreApplication::setOrganizationName(organization);
        EXPECT_TRUE(editor->isVisible());
        EXPECT_EQ(editor->keySequences(), valid);
        EXPECT_EQ(registry.bindings(), initial);
        editor->accept();
    });
    table->itemDoubleClicked(item);
    EXPECT_FALSE(editor->isVisible());
    EXPECT_EQ(registry.bindings().value("draft.a"), valid);
    EXPECT_EQ(action.shortcuts(), valid);
}

TEST_F(ShortcutTests, PythonViewPreservesNormalEditing) {
    ecvPythonView view;
    auto* editor = view.findChild<ecvPythonCodeEditor*>();
    ASSERT_NE(editor, nullptr);
    editor->clear();
    view.show();
    view.activateWindow();
    editor->setFocus();
    ASSERT_TRUE(waitForWindowActive(&view));
    pressSequence(editor, QKeySequence(Qt::Key_X));
    EXPECT_EQ(editor->toPlainText(), QString("x"));
    pressSequence(editor, QKeySequence(Qt::Key_Backspace));
    EXPECT_TRUE(editor->toPlainText().isEmpty());
    editor->setPlainText("copy text");
    editor->selectAll();
    pressSequence(editor, QKeySequence::Copy);
    EXPECT_EQ(QApplication::clipboard()->text(), QString("copy text"));
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    // Set both scopes before constructing the registry or any test fixture.
    QTemporaryDir isolated;
    if (!isolated.isValid()) return 1;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       isolated.path());
    QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope,
                       isolated.path());
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
