// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"
#include "signalledrunable.h"

class vtkTable;
namespace VtkUtils {
class TableModel;
class QPCL_ENGINE_LIB_API ModelToVtkTableConverter : public SignalledRunnable {
    Q_OBJECT
public:
    explicit ModelToVtkTableConverter(TableModel* model);

    void setLabels(const QStringList& labels);
    QStringList labels() const;

    void run();

    vtkTable* table() const;

private:
    TableModel* m_model = nullptr;
    vtkTable* m_table = nullptr;
    QStringList m_labels;
};

}  // namespace VtkUtils
