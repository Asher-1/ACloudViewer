// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QObject>

#include "qPCL.h"

namespace VtkUtils {

class QPCL_ENGINE_LIB_API SignalBlocker {
public:
    explicit SignalBlocker(QObject* object = nullptr);
    ~SignalBlocker();

    void addObject(QObject* object);

private:
    QObjectList m_objectList;
};

}  // namespace VtkUtils
