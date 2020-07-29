#ifndef CONTOURWIDGETOBSERVER_H
#define CONTOURWIDGETOBSERVER_H

#include "abstractwidgetobserver.h"

#include <vtkSmartPointer.h>

class vtkPolyData;
namespace VtkUtils
{

class QPCL_ENGINE_LIB_API ContourWidgetObserver : public AbstractWidgetObserver
{
    Q_OBJECT
public:
    explicit ContourWidgetObserver(QObject* parent = nullptr);

signals:
    void dataChanged(vtkPolyData* data);

protected:
    void Execute(vtkObject *caller, unsigned long eventId, void* callData);

    vtkSmartPointer<vtkPolyData> m_polyData;
};

} // namespace VtkUtils
#endif // CONTOURWIDGETOBSERVER_H
