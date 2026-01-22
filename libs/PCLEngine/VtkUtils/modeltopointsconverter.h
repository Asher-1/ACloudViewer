// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QList>
#include <QVector>

#include "point3f.h"
#include "signalledrunable.h"
#include "tablemodel.h"
#include "utils.h"

namespace VtkUtils {

class QPCL_ENGINE_LIB_API ModelToPointsConverter : public SignalledRunnable {
    Q_OBJECT
public:
    explicit ModelToPointsConverter(TableModel* model);

    QList<Point3F> points() const;
    QVector<Tuple3ui> vertices() const;

    void run();

private:
    TableModel* m_model = nullptr;
    QList<Point3F> m_points;
};

}  // namespace VtkUtils
