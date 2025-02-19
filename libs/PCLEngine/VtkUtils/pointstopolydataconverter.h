#ifndef POINTSTOPOLYDATACONVERTER_H
#define POINTSTOPOLYDATACONVERTER_H

#include "../qPCL.h"

// CV_CORE_LIB
#include <CVGeom.h>

#include "point3f.h"
#include "signalledrunable.h"

#include <vtkSmartPointer.h>

class vtkPolyData;
namespace VtkUtils
{
class QPCL_ENGINE_LIB_API PointsToPolyDataConverter : public SignalledRunnable
{
    Q_OBJECT
public:
    explicit PointsToPolyDataConverter(const QVector<Point3F>& points, const QVector<Tuple3ui>& vertices = QVector<Tuple3ui>());

    void run();

    vtkPolyData* polyData() const;

private:
    QVector<Point3F> m_points;
    QVector<Tuple3ui> m_vertices;
    vtkSmartPointer<vtkPolyData> m_polyData;
};

} // namespace VtkUtils
#endif // POINTSTOPOLYDATACONVERTER_H
