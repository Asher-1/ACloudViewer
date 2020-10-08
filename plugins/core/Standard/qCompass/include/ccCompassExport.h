//##########################################################################
//#                                                                        #
//#                    EROWCLOUDVIEWER PLUGIN: ccCompass                   #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                     COPYRIGHT: Sam Thiele  2017                        #
//#                                                                        #
//##########################################################################

#ifndef CCCOMPASSEXPORT_H
#define CCCOMPASSEXPORT_H

class QString;

class ecvMainAppInterface;


namespace ccCompassExport
{
	void saveCSV( ecvMainAppInterface *app, const QString &filename );
	void saveSVG( ecvMainAppInterface *app, const QString &filename, float zoom );
	void saveXML( ecvMainAppInterface *app, const QString &filename );
};

#endif // CCCOMPASSEXPORT_H
