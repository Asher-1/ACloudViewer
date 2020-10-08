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
//#          COPYRIGHT: CloudCompare project                               #
//#                                                                        #
//##########################################################################

#include "qCoreIO.h"

#include "MAFilter.h"
#include "PDMSFilter.h"
#include "MascaretFilter.h"
#include "SimpleBinFilter.h"
#include "HeightProfileFilter.h"

qCoreIO::qCoreIO( QObject *parent ) :
	QObject( parent ),
	ccIOPluginInterface( ":/CC/plugin/CoreIO/info.json" )
{
}

void qCoreIO::registerCommands( ccCommandLineInterface *inCmdLine )
{
	Q_UNUSED( inCmdLine );
}

ccIOPluginInterface::FilterList qCoreIO::getFilters()
{
	return {
		FileIOFilter::Shared( new SimpleBinFilter ),
		FileIOFilter::Shared( new PDMSFilter ),
		FileIOFilter::Shared( new MAFilter ),
		FileIOFilter::Shared( new MascaretFilter ),
		FileIOFilter::Shared( new HeightProfileFilter ),
	};
}
