#include "thresholdwindow.h"

#include <VtkUtils/vtkutils.h>
#include <VtkUtils/vtkwidget.h>

#include <vtkThreshold.h>
#include <vtkLookupTable.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkLODActor.h>

#include <QDebug>

ThresholdWindow::ThresholdWindow(QWidget* parent) : IsosurfaceWindow(parent)
{
    setWindowTitle(tr("Threshold"));
}

ThresholdWindow::~ThresholdWindow()
{

}

void ThresholdWindow::apply()
{
    if (!m_dataObject) {
        qDebug() << "Threshold::apply: null data object.";
        return;
    }

    VTK_CREATE(vtkThreshold, thresholdFilter);
    thresholdFilter->SetInputData(m_dataObject);
    thresholdFilter->SetAllScalars(false);
    thresholdFilter->SetLowerThreshold(m_minScalar);
    thresholdFilter->SetUpperThreshold(m_maxScalar);
    thresholdFilter->Update();

    vtkSmartPointer<vtkLookupTable> lut = createLookupTable(m_minScalar, m_maxScalar);
    lut->SetNumberOfColors(m_numOfContours);
    lut->Build();

    VTK_CREATE(vtkDataSetSurfaceFilter, dssFilter);
    dssFilter->SetInputConnection(thresholdFilter->GetOutputPort());

	setResultData(dssFilter->GetOutput());

    VTK_CREATE(vtkPolyDataMapper, mapper);
    mapper->SetLookupTable(lut);
    mapper->SetInputConnection(dssFilter->GetOutputPort());
    mapper->ScalarVisibilityOn();

    VtkUtils::vtkInitOnce(m_filterActor);
    m_filterActor->SetMapper(mapper);

    m_vtkWidget->defaultRenderer()->AddActor(m_filterActor);

    applyDisplayEffect();
}
