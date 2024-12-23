#ifndef SLIDERWIDGETOBSERVER_H
#define SLIDERWIDGETOBSERVER_H

#include "abstractwidgetobserver.h"

namespace VtkUtils
{

class QPCL_ENGINE_LIB_API SliderWidgetObserver : public AbstractWidgetObserver
{
    Q_OBJECT
public:
    explicit SliderWidgetObserver(QObject* parent = nullptr);

signals:
    void valueChanged(double value);

protected:
    void Execute(vtkObject *caller, unsigned long eventId, void* callData);
};

} // namespace VtkUtils
#endif // SLIDERWIDGETOBSERVER_H
