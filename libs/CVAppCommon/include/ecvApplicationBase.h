#ifndef ECV_APPLICATION_BASE_H
#define ECV_APPLICATION_BASE_H

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
//#          COPYRIGHT: ACloudViewer project                            #
//#                                                                        #
//##########################################################################

#include "CVAppCommon.h"

//Qt
#include <QApplication>

//! Mimic Qt's qApp for easy access to the application instance
#define ecvApp (static_cast<ecvApplicationBase *>( QCoreApplication::instance() ))

class CVAPPCOMMON_LIB_API ecvApplicationBase : public QApplication
{
public:
    //! This must be called before instantiating the application class so it
    //! can setup OpenGL first.
    static void	InitOpenGL();
	
	ecvApplicationBase( int &argc, char **argv, bool isCommandLine, const QString &version );
	
	bool isCommandLine() const { return c_CommandLine; }

	QString versionStr() const;
	QString versionLongStr( bool includeOS ) const;
	
	const QString &translationPath() const;
	
private:
	void setupPaths();
		
	const QString c_VersionStr;
	
	QString	m_ShaderPath;
	QString	m_TranslationPath;
	QStringList m_PluginPaths;

	const bool c_CommandLine;
};

#endif // ECV_APPLICATION_BASE_H
