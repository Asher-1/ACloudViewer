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

#pragma push_macro("slots")
#undef slots
#include <Python.h>
#pragma pop_macro("slots")

int main(int argc, char **argv)
{
    QTemporaryDir settings;
    if (!settings.isValid() || argc != 2)
        return 1;
    qputenv("XDG_CONFIG_HOME", settings.path().toUtf8());
    QApplication app(argc, argv);
    QPluginLoader loader(QString::fromLocal8Bit(argv[1]));
    QObject *object = loader.instance();
    auto *plugin = qobject_cast<ccPluginInterface *>(object);
    auto *standard = dynamic_cast<ccStdPluginInterface *>(plugin);
    if (!standard)
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
        PyRun_SimpleString("import pycc, cccorelib\n"
                           "cloud = pycc.ccPointCloud('plugin loading')\n"
                           "field = cloud.getScalarField(cloud.addScalarField('values'))\n"
                           "assert field.getHistogram() == []\n");
    if (result != 0)
        return 1;
    standard->stop();
    std::puts("PASS: Qt interface discovery, repeatable actions and production module import");
    return 0;
}
