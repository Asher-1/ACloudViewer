//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER PLUGIN: qPCL                         #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                         COPYRIGHT: DAHAI LU                         #
//#                                                                        #
//##########################################################################
//
#ifndef MLSSMOOTHINGUPSAMPLING_H
#define MLSSMOOTHINGUPSAMPLING_H

#include "BasePclModule.h"

class MLSDialog;

namespace PCLModules
{
	struct MLSParameters;
}

class MLSSmoothingUpsampling : public BasePclModule
{
public:
	MLSSmoothingUpsampling();
	virtual ~MLSSmoothingUpsampling();

protected:

	//inherited from BasePclModule
	int openInputDialog();
	int compute();
	void getParametersFromDialog();

	MLSDialog* m_dialog;
	PCLModules::MLSParameters * m_parameters; //We directly store all the parameters here
};

#endif // MLSSMOOTHINGUPSAMPLING_H
