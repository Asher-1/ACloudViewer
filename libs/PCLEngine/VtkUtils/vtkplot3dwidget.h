#ifndef VTKPLOT3DWIDGET_H
#define VTKPLOT3DWIDGET_H

#include "vtkplotwidget.h"
#include "qPCL.h"

namespace VtkUtils
{
class VtkPlot3DWidgetPrivate;
class QPCL_ENGINE_LIB_API VtkPlot3DWidget : public VtkPlotWidget
{
    Q_OBJECT
public:
    explicit VtkPlot3DWidget(QWidget* parent = nullptr);
    ~VtkPlot3DWidget();

    vtkContextItem* chart() const;

private:
    VtkPlot3DWidgetPrivate* d_ptr;
    Q_DISABLE_COPY(VtkPlot3DWidget)
};

} // namespace VtkUtils
#endif // VTKPLOT3DWIDGET_H
