// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file signalledrunable.h
/// @brief QRunnable with Qt signals (emits finished() when done).

#include <QObject>
#include <QRunnable>

#include "qVTK.h"

namespace VtkUtils {

/// @class SignalledRunnable
/// @brief QRunnable that can emit Qt signals; subclasses implement run() and
/// emit finished().
class QVTK_ENGINE_LIB_API SignalledRunnable : public QObject, public QRunnable {
    Q_OBJECT
public:
    SignalledRunnable();

signals:
    void finished();
};

}  // namespace VtkUtils
