#ifndef MODELTOPOINTSCONVERTER_H
#define MODELTOPOINTSCONVERTER_H

#include "signalledrunable.h"
#include "utils.h"
#include "tablemodel.h"
#include "point3f.h"

namespace VtkUtils
{

class QPCL_ENGINE_LIB_API ModelToPointsConverter : public SignalledRunnable
{
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

} // namespace Utils

#endif // MODELTOPOINTSCONVERTER_H
