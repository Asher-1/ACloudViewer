// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Use the application's Qt discovery contract across the plugin ABI boundary.
#include <QAction>
#include <QApplication>
#include <QPluginLoader>
#include <QSettings>
#include <QShortcutEvent>
#include <QTemporaryDir>
#include <ecvStdPluginInterface.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <memory>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#pragma push_macro("slots")
#undef slots
#include <Python.h>
#pragma pop_macro("slots")

namespace
{
QString pluginPath;
bool preloadRequested = false;
QString preloadError;

class PythonPluginLoading : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        ASSERT_TRUE(preloadError.isEmpty()) << qPrintable(preloadError);
        ASSERT_FALSE(pluginPath.isEmpty()) << "Pass the built plugin module path";
        loader = std::make_unique<QPluginLoader>(pluginPath);
#if !defined(_WIN32)
        // Preserve the underlying dependency diagnostic; Qt may collapse it to
        // 'Unknown error'. This probe does not replace Qt interface discovery.
        void *probe = dlopen(pluginPath.toLocal8Bit().constData(), RTLD_NOW);
        if (!probe)
            std::fprintf(stderr, "dyld: %s\n", dlerror());
        else
            dlclose(probe);
#endif
        object = loader->instance();
        ASSERT_NE(object, nullptr) << qPrintable(loader->errorString());
        plugin = qobject_cast<ccPluginInterface *>(object);
        standard = qobject_cast<ccStdPluginInterface *>(object);
        ASSERT_NE(plugin, nullptr) << qPrintable(loader->errorString());
        ASSERT_NE(standard, nullptr) << qPrintable(loader->errorString());
    }
    void TearDown() override
    {
        // Stop before the loader is destroyed even when an assertion aborts
        // the scenario; keep the production plugin/interpreter teardown path.
        if (standard)
            standard->stop();
        loader.reset();
    }
    std::unique_ptr<QPluginLoader> loader;
    QObject *object = nullptr;
    ccPluginInterface *plugin = nullptr;
    ccStdPluginInterface *standard = nullptr;
};

TEST_F(PythonPluginLoading, DiscoversStableActionsAndImportsProductionModules)
{
    const auto actions = standard->getActions();
    ASSERT_FALSE(actions.isEmpty());
    EXPECT_EQ(actions, standard->getActions());
    for (const auto *action : actions)
    {
        ASSERT_NE(action, nullptr);
        EXPECT_FALSE(action->text().isEmpty());
    }
    ASSERT_TRUE(Py_IsInitialized());
    EXPECT_EQ(PyRun_SimpleString("import sys, pycc, cccorelib\n"
                                 "assert all('\\r' not in p and '\\n' not in p for p in sys.path)\n"
                                 "print('Embedded Python prefix:', sys.prefix, flush=True)\n"
                                 "print('Embedded Python paths:', sys.path, flush=True)\n"
                                 "cloud = pycc.ccPointCloud('plugin loading')\n"
                                 "field = cloud.getScalarField(cloud.addScalarField('values'))\n"
                                 "assert field.getHistogram() == []\n"),
              0);
#if defined(__linux__) && defined(RTLD_DEEPBIND)
    if (preloadRequested)
    {
        EXPECT_EQ(PyRun_SimpleString("import os, sys, readline\n"
                                     "assert not (sys.getdlopenflags() & os.RTLD_DEEPBIND)\n"
                                     "readline.clear_history()\n"
                                     "readline.add_history('embedded readline')\n"
                                     "assert readline.get_history_item(1) == 'embedded readline'\n"
                                     "readline.remove_history_item(0)\n"
                                     "assert readline.get_current_history_length() == 0\n"),
                  0);
    }
#endif
}

TEST_F(PythonPluginLoading, HiddenEditorDoesNotInterceptForeignAmbiguousShortcuts)
{
    standard->getActions();
    auto *showEditor = object->findChild<QAction *>(QStringLiteral("showEditor"));
    ASSERT_NE(showEditor, nullptr);
    QWidget *editor = nullptr;
    for (auto *window : QApplication::topLevelWidgets())
    {
        if (qstrcmp(window->metaObject()->className(), "PythonEditor") == 0)
        {
            editor = window;
            break;
        }
    }
    ASSERT_NE(editor, nullptr);
    ASSERT_FALSE(editor->isVisible());
    auto *newFile = editor->findChild<QAction *>(QStringLiteral("actionNew"));
    ASSERT_NE(newFile, nullptr);
    ASSERT_FALSE(newFile->shortcut().isEmpty());
    EXPECT_EQ(newFile->shortcutContext(), Qt::WidgetWithChildrenShortcut);

    QObject receiver;
    int editorTriggers = 0;
    QObject::connect(newFile, &QAction::triggered, &receiver, [&]() { ++editorTriggers; });
    QShortcutEvent foreignShortcut(newFile->shortcut(), 1, true);
    QApplication::sendEvent(&receiver, &foreignShortcut);
    EXPECT_EQ(editorTriggers, 0);

    // Keep normal action wiring and the persistent editor's close/reopen path.
    showEditor->trigger();
    ASSERT_TRUE(editor->isVisible());
    QShortcutEvent visibleForeignShortcut(newFile->shortcut(), 2, true);
    QApplication::sendEvent(&receiver, &visibleForeignShortcut);
    EXPECT_EQ(editorTriggers, 0);
    newFile->trigger();
    EXPECT_EQ(editorTriggers, 1);
    EXPECT_TRUE(editor->close());
    EXPECT_FALSE(editor->isVisible());
    showEditor->trigger();
    EXPECT_TRUE(editor->isVisible());
    newFile->trigger();
    EXPECT_EQ(editorTriggers, 2);
    EXPECT_TRUE(editor->close());
}
} // namespace

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    QTemporaryDir settings;
    if (!settings.isValid())
        return 1;
    qputenv("XDG_CONFIG_HOME", settings.path().toUtf8());
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settings.path());
    QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope, settings.path());
    if (argc == 2 || argc == 3)
        pluginPath = QString::fromLocal8Bit(argv[1]);
    else
        preloadError = QStringLiteral("Expected plugin module and optional libedit path");
    preloadRequested = argc == 3;
#if defined(__linux__)
    // Preload before QApplication can initialize Mesa/LLVM, preserving the
    // original readline-isolation reproduction sequence and process lifetime.
    if (preloadRequested && !dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL))
        preloadError = QString::fromLocal8Bit(dlerror());
#endif
    QApplication app(argc, argv);
    return RUN_ALL_TESTS();
}
