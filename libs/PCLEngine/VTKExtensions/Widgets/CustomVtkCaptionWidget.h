#ifndef QVTK_CUSTOM_VTKCAPTIONWIDGET_HEADER
#define QVTK_CUSTOM_VTKCAPTIONWIDGET_HEADER

#include "qPCL.h"
/**
 * @brief The CustomVtkCaptionWidget class
 * CustomVtkCaptionWidget
 */

#include <vtkCaptionWidget.h>

class QPCL_ENGINE_LIB_API CustomVtkCaptionWidget : public vtkCaptionWidget {
public:
	static CustomVtkCaptionWidget *New();

	vtkTypeMacro(CustomVtkCaptionWidget, vtkCaptionWidget);


	void SetHandleEnabled(bool state);

};
#endif // QVTK_CUSTOM_VTKCAPTIONWIDGET_HEADER
