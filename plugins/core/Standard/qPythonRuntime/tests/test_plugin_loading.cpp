// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Use the application manager's Qt discovery path, not a manual interface pointer cast.
#include <QAction>
#include <QApplication>
#include <QPluginLoader>
#include <QTemporaryDir>
#include <cstdio>
#include <ecvStdPluginInterface.h>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#pragma push_macro("slots")
#undef slots
#include <Python.h>
#pragma pop_macro("slots")

int main(int argc, char **argv)
{
    QTemporaryDir settings;
    if (!settings.isValid() || (argc != 2 && argc != 3))
        return 1;
    qputenv("XDG_CONFIG_HOME", settings.path().toUtf8());
#if defined(__linux__)
    // Mesa/LLVM can expose libedit globally before the embedded interpreter starts.
    // Keep it loaded to reproduce its symbol collision with GNU readline.
    if (argc == 3 && !dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL))
    {
        std::fprintf(stderr, "FAIL: libedit preload: %s\n", dlerror());
        return 1;
    }
#endif
    QApplication app(argc, argv);
    QPluginLoader loader(QString::fromLocal8Bit(argv[1]));
#if !defined(_WIN32)
    // QPluginLoader collapses a failed dependency load into "Unknown error".
    // Resolve the Mach-O/ELF directly first so the real dyld message (missing
    // library, undefined symbol, rpath gap) reaches the CI log verbatim.
    void *probe = dlopen(argv[1], RTLD_NOW);
    if (!probe)
        std::fprintf(stderr, "dyld: %s\n", dlerror());
    else
        dlclose(probe);
#endif
    QObject *object = loader.instance();
    // Cross-module casts must follow the application's discovery contract:
    // Q_DECLARE_INTERFACE turns qobject_cast into a string-comparison against
    // the plugin's qt_metacast, which is stable across the dylib boundary.
    // A C++ RTTI dynamic_cast into the interface compares typeinfo pointers
    // that static builds duplicate per module under -fvisibility=hidden (the
    // interface library is linked statically into both this test and the
    // plugin), so dyld never coalesces them and the cast returns null on
    // macOS/Windows static lanes even though the plugin is fully loadable.
    auto *plugin = qobject_cast<ccPluginInterface *>(object);
    auto *standard = qobject_cast<ccStdPluginInterface *>(object);
    if (!plugin || !standard)
    {
        std::fprintf(
            stderr, "FAIL: Qt plugin interface discovery: %s\n", qPrintable(loader.errorString()));
        return 1;
    }
    const auto actions = standard->getActions();
    if (actions.isEmpty() || actions != standard->getActions())
        return 1;
    for (const auto *action : actions)
        if (!action || action->text().isEmpty())
            return 1;
    if (!Py_IsInitialized())
        return 1;
    // Import the registered production modules and exercise the formerly broken value boundary.
    const int result =
        PyRun_SimpleString("import sys, pycc, cccorelib\n"
                           "assert all('\\r' not in p and '\\n' not in p for p in sys.path)\n"
                           "print('Embedded Python prefix:', sys.prefix, flush=True)\n"
                           "print('Embedded Python paths:', sys.path, flush=True)\n"
                           "cloud = pycc.ccPointCloud('plugin loading')\n"
                           "field = cloud.getScalarField(cloud.addScalarField('values'))\n"
                           "assert field.getHistogram() == []\n");
    if (result != 0)
        return 1;
#if defined(__linux__) && defined(RTLD_DEEPBIND)
    if (argc == 3 &&
        PyRun_SimpleString("import os, sys, readline\n"
                           "assert not (sys.getdlopenflags() & os.RTLD_DEEPBIND)\n"
                           "readline.clear_history()\n"
                           "readline.add_history('embedded readline')\n"
                           "assert readline.get_history_item(1) == 'embedded readline'\n"
                           "readline.remove_history_item(0)\n"
                           "assert readline.get_current_history_length() == 0\n") != 0)
        return 1;
#endif
    standard->stop();
    std::puts("PASS: Qt interface discovery, repeatable actions and production module import");
    return 0;
}
