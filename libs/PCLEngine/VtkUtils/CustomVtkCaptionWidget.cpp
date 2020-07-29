#include "CustomVtkCaptionWidget.h"

#include <vtkRenderer.h>
#include <vtkDoubleArray.h>
#include <vtkHandleWidget.h>

vtkStandardNewMacro(CustomVtkCaptionWidget);

void CustomVtkCaptionWidget::SetHandleEnabled(bool state)
{
	this->HandleWidget->SetEnabled(state);
	state ? this->HandleWidget->ProcessEventsOn() : 
		this->HandleWidget->ProcessEventsOff();
}


