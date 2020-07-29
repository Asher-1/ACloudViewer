#ifndef VTKWIDGET_H
#define VTKWIDGET_H

//#define vtkRenderingCore_AUTOINIT 3(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL) // ogl2
//#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL)

#include <vtkAutoInit.h>
#include <vtkAutoInit.h>

//VTK_MODULE_INIT(vtkRenderingOpenGL) // ogl2
//VTK_MODULE_INIT(vtkInteractionStyle)

#include <QVTKWidget.h>
#include <vtkSmartPointer.h>

#include "../qPCL.h"

class vtkActor;
class vtkProp;
class vtkLODActor;
class vtkDataSet;
namespace VtkUtils
{

class VtkWidgetPrivate;
class QPCL_ENGINE_LIB_API VtkWidget : public QVTKWidget
{
	Q_OBJECT
public:
	explicit VtkWidget(QWidget* parent = nullptr);
	virtual ~VtkWidget();

	void setMultiViewports(bool multi = true);
	bool multiViewports() const;

    void createActorFromVTKDataSet(
		const vtkSmartPointer<vtkDataSet> &data,
		vtkSmartPointer<vtkLODActor> &actor, 
		bool use_scalars = true);

	void addActor(vtkProp* actor, const QColor& clr = Qt::black);
	void addViewProp(vtkProp* prop);
	QList<vtkProp*> actors() const;

	void setActorsVisible(bool visible);
	void setActorVisible(vtkProp* actor, bool visible);
	bool actorVisible(vtkProp* actor);

	void setBackgroundColor(const QColor& clr);
	void setBackgroundColor();
	QColor backgroundColor() const;

	vtkRenderer* defaultRenderer();
	bool defaultRendererTaken() const;

	void showOrientationMarker(bool show = true);

protected:
	void setBounds(double* bounds);

	double xMin() const;
	double xMax() const;
	double yMin() const;
	double yMax() const;
	double zMin() const;
	double zMax() const;

private:
	VtkWidgetPrivate* d_ptr;
	Q_DISABLE_COPY(VtkWidget)
};

} // namespace VtkUtils
#endif // VTKWIDGET_H
