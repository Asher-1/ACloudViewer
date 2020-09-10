#ifndef POINTWIDGETOBSERVER_H
#define POINTWIDGETOBSERVER_H

#include "abstractwidgetobserver.h"

namespace VtkUtils
{

class QPCL_ENGINE_LIB_API PointWidgetObserver : public AbstractWidgetObserver
{
    Q_OBJECT
public:
    explicit PointWidgetObserver(QObject* parent = nullptr);

signals:
    void positionChanged(double* position);

protected:
    void Execute(vtkObject *caller, unsigned long eventId, void *callData);
};

} // namespace VtkUtils
#endif // POINTWIDGETOBSERVER_H
