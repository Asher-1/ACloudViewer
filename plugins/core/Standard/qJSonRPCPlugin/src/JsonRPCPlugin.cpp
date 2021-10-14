//##########################################################################
//#                                                                        #
//#                CLOUDCOMPARE PLUGIN: JsonRPCPlugin                      #
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
//#                             COPYRIGHT: XXX                             #
//#                                                                        #
//##########################################################################

//Qt
#include <QtGui>

//local
#include "JsonRPCPlugin.h"

// ECV_DB_LIB
#include <ecvGenericPointCloud.h>
#include <ecvDisplayTools.h>

// ECV_IO_LIB
#include <FileIOFilter.h>

//ECV plugins
#include <ecvMainAppInterface.h>

JsonRPCPlugin::JsonRPCPlugin( QObject *parent )
	: QObject( parent )
	, ccStdPluginInterface( ":/CC/plugin/JsonRPCPlugin/info.json" )
{
    qDebug() << "JsonRPCPlugin::JsonRPCPlugin";

    connect(&rpc_server, &JsonRPCServer::execute, this, &JsonRPCPlugin::execute);
}

QList<QAction *> JsonRPCPlugin::getActions()
{
    qDebug() << "JsonRPCPlugin::getActions";

	// default action (if it has not been already created, this is the moment to do it)
	if ( !m_action )
	{
		m_action = new QAction( getName(), this );
		m_action->setToolTip( getDescription() );
        m_action->setIcon( getIcon() );
        m_action->setCheckable(true);
        m_action->setChecked(false);
		m_action->setEnabled(true);

		// Connect appropriate signal
        connect( m_action, &QAction::triggered, this, &JsonRPCPlugin::triggered);
	}

    return { m_action };
}

void JsonRPCPlugin::triggered(bool checked)
{
    qDebug() << "JsonRPCPlugin::triggered " << checked;

    if (checked)
	{
        rpc_server.listen(6001);
    }
	else
	{
        rpc_server.close();
    }
}

JsonRPCResult JsonRPCPlugin::execute(QString method, QMap<QString, QVariant> params)
{
    qDebug() << method << params;
    if (m_app == nullptr)
	{
        return JsonRPCResult();
    }

    JsonRPCResult result;
    bool need_redraw = false;
    if (method == "open")
	{
        QString filename = params["filename"].toString();
        // code copied from MainWindow::addToDB()
        //to use the same 'global shift' for multiple files
		CCVector3d loadCoordinatesShift(0, 0, 0);
        bool loadCoordinatesTransEnabled = false;

        FileIOFilter::LoadParameters parameters;
        {
            parameters.alwaysDisplayLoadDialog = true;
            parameters.shiftHandlingMode = ecvGlobalShiftManager::DIALOG_IF_NECESSARY;
            parameters.coordinatesShift = &loadCoordinatesShift;
            parameters.coordinatesShiftEnabled = &loadCoordinatesTransEnabled;
            parameters.parentWidget = m_app->getActiveWindow();
        }

        if(params.contains("silent"))
		{
            parameters.alwaysDisplayLoadDialog = false;
        }
        CC_FILE_ERROR res = CC_FERR_NO_ERROR;
        ccHObject* newGroup = FileIOFilter::LoadFromFile(filename, parameters, res, params["filter"].toString());

        if (newGroup)
		{
            //disable the normals on all loaded clouds!
            ccHObject::Container clouds;
            newGroup->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD);
            for (ccHObject* cloud : clouds)
			{
                if (cloud) 
				{
                    static_cast<ccGenericPointCloud*>(cloud)->showNormals(false);
                }
            }

            // apply matrix if possible
            QList<QVariant> transformation = params["transformation"].toList();
            if (transformation.size() == 4*4)
			{
                std::vector<double> values(4*4);
                bool success;
				for (unsigned i = 0; i < 4 * 4; ++i)
				{
                    double d = transformation[i].toDouble(&success);
                    qDebug() << transformation[i].toString() << transformation[i].toDouble();
                    if (!success)
					{
						break;
                    }
					values[((i % 4) * 4) + (i / 4)] = d;
                }
                if (success)
				{
                    ccGLMatrix mat = ccGLMatrix(values.data());
                    qDebug() << "apply matrix: " << mat.toString();
                    newGroup->setGLTransformation(mat);
                    newGroup->applyGLTransformation_recursive();
                }
            }

            m_app->addToDB(newGroup);
            need_redraw = true;
            result = JsonRPCResult::success(0);
        }
		else
		{
            result = JsonRPCResult::error(1, "cancelled by user");
        }
    }
	else if (method == "clear")
	{

        // remove everything below root
        auto root = m_app->dbRootObject();
        ccHObject* child;
        while ((child = root->getChild(0)) != nullptr)
		{
            m_app->removeFromDB(child, true);
        }
        need_redraw = true;
        result = JsonRPCResult::success(0);
    }
    
	// redraw
    if (need_redraw)
	{
        QWidget* win = m_app->getActiveWindow();
        if (win)
        {
            ecvDisplayTools::RedrawDisplay();
        }
    }

    return result;
}
