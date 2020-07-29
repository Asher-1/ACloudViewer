//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
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
//#                 COPYRIGHT: Asher                                       #
//#                                                                        #
//##########################################################################

#include "ecvOptions.h"
#include "ecvSettingManager.h"


//eCV_db
#include <ecvSingleton.h>

//! Unique instance of ecvOptions
static ecvSingleton<ecvOptions> s_options;

ecvOptions& ecvOptions::InstanceNonConst()
{
	if (!s_options.instance)
	{
		s_options.instance = new ecvOptions();
		s_options.instance->fromPersistentSettings();
	}

	return *s_options.instance;
}

void ecvOptions::ReleaseInstance()
{
	s_options.release();
}

void ecvOptions::Set(const ecvOptions& params)
{
	InstanceNonConst() = params;
}

ecvOptions::ecvOptions()
{
	reset();
}

void ecvOptions::reset()
{
	normalsDisplayedByDefault = false;
	useNativeDialogs = true;
}

void ecvOptions::fromPersistentSettings()
{
	normalsDisplayedByDefault = 
		ecvSettingManager::getValue("Options", "normalsDisplayedByDefault", false).toBool();	
	useNativeDialogs = 
		ecvSettingManager::getValue("Options", "useNativeDialogs", true).toBool();
}

void ecvOptions::toPersistentSettings() const
{
	ecvSettingManager::setValue("Options", "normalsDisplayedByDefault", normalsDisplayedByDefault);
	ecvSettingManager::setValue("Options", "useNativeDialogs", useNativeDialogs);
}
