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

class QPCL_ENGINE_LIB_API MovePointInteractorStyle : public QObject {
    Q_OBJECT
public:
    explicit MovePointInteractorStyle(QObject *parent = 0);

signals:

public slots:
};

}  // namespace VtkUtils
