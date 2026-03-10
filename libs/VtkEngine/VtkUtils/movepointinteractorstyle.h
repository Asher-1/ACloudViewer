// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file movepointinteractorstyle.h
/// @brief VTK interactor style for moving points (placeholder).

#include <QObject>

#include "qVTK.h"

namespace VtkUtils {

/// @class MovePointInteractorStyle
/// @brief Interactor style for point manipulation (QObject wrapper).
class QVTK_ENGINE_LIB_API MovePointInteractorStyle : public QObject {
    Q_OBJECT
public:
    explicit MovePointInteractorStyle(QObject *parent = 0);

signals:

public slots:
};

}  // namespace VtkUtils
