// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "signalledrunable.h"
#include "utils.h"
#include "vector4f.h"

namespace VtkUtils {
class TableModel;
class QPCL_ENGINE_LIB_API ModelToVectorsConverter : public SignalledRunnable {
    Q_OBJECT
public:
    ModelToVectorsConverter(TableModel* model);

    void run();

    QList<Vector4F> vectors() const;

private:
    QList<Vector4F> m_vectors;
    TableModel* m_model = nullptr;
};

}  // namespace VtkUtils
