#ifndef VTK_WIDGETS_FACTORY_H
#define VTK_WIDGETS_FACTORY_H

#include "../qPCL.h"
#include <ecvWidgetsInterface.h>


class QPCL_ENGINE_LIB_API VtkWidgetsFactory
{
public:
	VtkWidgetsFactory() = default;
	~VtkWidgetsFactory() = default;

	static DBLib::ecvWidgetsInterface::Shared GetFilterWidgetInterface();
	static DBLib::ecvWidgetsInterface::Shared GetSmallWidgetsInterface();
	static DBLib::ecvWidgetsInterface::Shared GetSurfaceWidgetsInterface();

private:

};

#endif // VTK_WIDGETS_FACTORY_H
