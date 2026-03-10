// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "signalblocker.h"

namespace VtkUtils {

SignalBlocker::SignalBlocker(QObject* object) { addObject(object); }

void SignalBlocker::addObject(QObject* object) {
    if (object) {
        object->blockSignals(true);
        m_objectList.append(object);
    }
}

SignalBlocker::~SignalBlocker() {
    foreach (QObject* obj, m_objectList) obj->blockSignals(false);
}

}  // namespace VtkUtils
