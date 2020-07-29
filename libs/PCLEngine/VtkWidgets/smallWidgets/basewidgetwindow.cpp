#include "basewidgetwindow.h"
#include "ui_basewidgetwindow.h"

#include <VtkUtils/vtkwidget.h>
#include <VtkUtils/vtkutils.h>

#include <vtkActor.h>
#include <vtkConeSource.h>
#include <vtkRenderer.h>
#include <vtkPolyDataMapper.h>

// ECV_DB_LIB
#include <ecvHObject.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

BaseWidgetWindow::BaseWidgetWindow(QWidget *parent) :
    QWidget(parent),
    m_ui(new Ui::BaseWidgetWindow)
{
}

BaseWidgetWindow::~BaseWidgetWindow()
{
    delete m_ui;
}

bool BaseWidgetWindow::setInput(const ccHObject * obj)
{
	return false;
}

ccHObject * BaseWidgetWindow::getOutput() const
{
	return nullptr;
}

void BaseWidgetWindow::createUi()
{
    m_vtkWidget = new VtkUtils::VtkWidget(this);
    m_ui->previewLayout->addWidget(m_vtkWidget);

    m_theActor = vtkSmartPointer<vtkActor>::Take(VtkUtils::createSourceActor<vtkConeSource>());
    m_vtkWidget->defaultRenderer()->AddActor(m_theActor);
    m_theActor->SetVisibility(0);
}

