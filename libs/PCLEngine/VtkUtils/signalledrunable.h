// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef SIGNALLEDRUNNABLE_H
#define SIGNALLEDRUNNABLE_H

#include <QObject>
#include <QRunnable>

#include "../qPCL.h"

namespace VtkUtils {

class QPCL_ENGINE_LIB_API SignalledRunnable : public QObject, public QRunnable {
    Q_OBJECT
public:
    SignalledRunnable();

signals:
    void finished();
};

}  // namespace VtkUtils
#endif  // SIGNALLEDRUNABLE_H
