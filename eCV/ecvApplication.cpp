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
//#          COPYRIGHT: CLOUDVIEWER  project                               #
//#                                                                        #
//##########################################################################

#include <QtGlobal>

#ifdef Q_OS_MAC
#include <QFileOpenEvent>
#endif

// qCC_io
#include "FileIO.h"

#include "ecvApplication.h"
#include "MainWindow.h"

ecvApplication::ecvApplication(int &argc, char **argv, bool isCommandLine)
    : ecvApplicationBase( argc, argv, isCommandLine, QStringLiteral( "3.8.0 (Asher)" ) )
{
	setApplicationName( "ErowCloudViewer" );
	
	FileIO::setWriterInfo( applicationName(), versionStr() );
}

bool ecvApplication::event(QEvent *inEvent)
{
#ifdef Q_OS_MAC
	switch ( inEvent->type() )
	{
		case QEvent::FileOpen:
		{
			MainWindow* mainWindow = MainWindow::TheInstance();
			
			if ( mainWindow == nullptr )
			{
				return false;
			}
			
			mainWindow->addToDB( QStringList(static_cast<QFileOpenEvent *>(inEvent)->file()) );
			return true;
		}
			
		default:
			break;
	}
#endif
	
	return ecvApplicationBase::event( inEvent );
}
