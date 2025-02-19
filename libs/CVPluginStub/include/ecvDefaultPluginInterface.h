//##########################################################################
//#                                                                        #
//#                            CLOUDVIEWER                                 #
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
//#          COPYRIGHT: ACloudViewer project                            #
//#                                                                        #
//##########################################################################

#ifndef ECV_DEFAULT_PLUGIN_INTERFACE_HEADER
#define ECV_DEFAULT_PLUGIN_INTERFACE_HEADER

#include <QString>

#include "ecvPluginInterface.h"


class ccDefaultPluginData;


class ccDefaultPluginInterface : public ccPluginInterface
{
public:
	virtual ~ccDefaultPluginInterface();
	
	virtual bool isCore() const override;

	virtual QString getName() const override;
	virtual QString getDescription() const override;
	
	virtual QIcon getIcon() const override;
	
	virtual ReferenceList getReferences() const override;
	virtual ContactList getAuthors() const override;
	virtual ContactList getMaintainers() const override;
	
protected:
	ccDefaultPluginInterface( const QString &resourcePath = QString() );
	
private:
	void setIID( const QString& iid ) override;
	const QString& IID() const override;

	ccDefaultPluginData	*m_data;
};

#endif // ECV_DEFAULT_PLUGIN_INTERFACE_HEADER
