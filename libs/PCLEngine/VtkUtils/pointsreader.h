// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QRunnable>

#include "point3f.h"
#include "qPCL.h"
#include "signalledrunable.h"
#include "utils.h"

namespace VtkUtils {

class QPCL_ENGINE_LIB_API PointsReader : public SignalledRunnable {
    Q_OBJECT
public:
    explicit PointsReader(const QString& file);

    void run();

    const QList<Point3F>& points() const;

private:
    QString m_file;
    QList<Point3F> m_points;
};

}  // namespace VtkUtils
