#ifndef VTKPLOTWIDGET_H
#define VTKPLOTWIDGET_H

//#define vtkRenderingCore_AUTOINIT 3(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL) // opengl2
//#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL) // opengl2

//#include <vtkAutoInit.h>

//VTK_MODULE_INIT(vtkRenderingOpenGL) // opengl2
//VTK_MODULE_INIT(vtkInteractionStyle)
//VTK_MODULE_INIT(vtkRenderingContextOpenGL) // opengl2

//#include <vtkAutoInit.h> 
//VTK_MODULE_INIT(vtkRenderingFreeType)
//VTK_MODULE_INIT(vtkRenderingOpenGL2);
//VTK_MODULE_INIT(vtkInteractionStyle);

#include <QVTKOpenGLNativeWidget.h>
#include "qPCL.h"

class vtkChartXY;
class vtkContextItem;
class vtkContextView;
namespace VtkUtils
{

class VtkPlotWidgetPrivate;
class VtkPlotWidget : public QVTKOpenGLNativeWidget
{
    Q_OBJECT
public:
    explicit VtkPlotWidget(QWidget* parent = nullptr);
    virtual ~VtkPlotWidget();

    virtual vtkContextItem* chart() const = 0;

    vtkContextView* contextView() const;
    vtkRenderWindow* GetRenderWindow() {return this->renderWindow(); }

protected:
    void init();

private:
    VtkPlotWidgetPrivate* d_ptr;
    Q_DISABLE_COPY(VtkPlotWidget)
};

} // namespace VtkUtils
#endif // VTKPLOTWIDGET_H
