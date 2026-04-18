// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericGLDisplay.h"

#include <QMap>
#include <QMutex>
#include <QMutexLocker>

namespace {
QMutex s_registryMutex;
QMap<QWidget*, ecvGenericGLDisplay*> s_displayRegistry;
}  // namespace

ecvGenericGLDisplay* ecvGenericGLDisplay::FromWidget(QWidget* widget) {
    QMutexLocker lock(&s_registryMutex);
    auto it = s_displayRegistry.find(widget);
    return (it != s_displayRegistry.end()) ? it.value() : nullptr;
}

void ecvGenericGLDisplay::RegisterGLDisplay(QWidget* widget,
                                            ecvGenericGLDisplay* display) {
    QMutexLocker lock(&s_registryMutex);
    s_displayRegistry.insert(widget, display);
}

void ecvGenericGLDisplay::UnregisterGLDisplay(QWidget* widget) {
    QMutexLocker lock(&s_registryMutex);
    s_displayRegistry.remove(widget);
}
