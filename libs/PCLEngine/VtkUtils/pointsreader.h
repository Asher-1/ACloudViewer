#ifndef POINTSREADER_H
#define POINTSREADER_H

#include "qPCL.h"

#include <QRunnable>
#include "signalledrunable.h"
#include "utils.h"
#include "point3f.h"

namespace VtkUtils
{

class QPCL_ENGINE_LIB_API PointsReader : public SignalledRunnable
{
    Q_OBJECT
public:
    explicit PointsReader(const QString& file);

    void run();

    const QList<Point3F>& points() const;

private:
    QString m_file;
    QList<Point3F> m_points;
};

} // namespace VtkUtils
#endif // POINTSREADER_H
