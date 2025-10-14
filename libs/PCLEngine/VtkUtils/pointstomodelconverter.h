// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef POINTSTOMODELCONVERTER_H
#define POINTSTOMODELCONVERTER_H

#include <QObject>
#include <QRunnable>

#include "point3f.h"
#include "qPCL.h"
#include "signalledrunable.h"
#include "utils.h"

namespace VtkUtils {

class TableModel;
class QPCL_ENGINE_LIB_API PointsToModelConverter : public SignalledRunnable {
    Q_OBJECT
public:
    PointsToModelConverter(const QList<Point3F>& points, TableModel* model);

    void run();

private:
    QList<Point3F> m_points;
    TableModel* m_model = nullptr;
};

}  // namespace VtkUtils

#endif  // POINTSTOMODELCONVERTER_H
