#ifndef POINTSTOMODELCONVERTER_H
#define POINTSTOMODELCONVERTER_H

#include "qPCL.h"

#include <QObject>
#include <QRunnable>
#include "utils.h"
#include "point3f.h"
#include "signalledrunable.h"

namespace VtkUtils
{

class TableModel;
class QPCL_ENGINE_LIB_API PointsToModelConverter : public SignalledRunnable
{
    Q_OBJECT
public:
    PointsToModelConverter(const QList<Point3F>& points, TableModel* model);

    void run();

private:
    QList<Point3F> m_points;
    TableModel* m_model = nullptr;
};

} // namespace VtkUtils

#endif // POINTSTOMODELCONVERTER_H
