// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file signalblocker.h
/// @brief RAII helper to block Qt signals during object lifetime.

#include <QObject>

#include "qVTK.h"

namespace VtkUtils {

/// @class SignalBlocker
/// @brief Blocks signals on QObjects in constructor, restores in destructor.
class QVTK_ENGINE_LIB_API SignalBlocker {
public:
    /// @param object Object to block (optional; can add more via addObject)
    explicit SignalBlocker(QObject* object = nullptr);
    ~SignalBlocker();

    /// @param object Additional object to block
    void addObject(QObject* object);

private:
    QObjectList m_objectList;
};

}  // namespace VtkUtils
