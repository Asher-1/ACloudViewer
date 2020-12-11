//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER  PLUGIN: qPCL                        #
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

#include "BasePclModule.h"

// CV_CORE_LIB
#include <CVPlatform.h>

// ECV_DB_LIB
#include <ecvPointCloud.h>
#include <ecvHObjectCaster.h>

// eCV
#include <ecvMainAppInterface.h>

//Qt
#include <QAction>
#include <QFuture>
#include <QApplication>
#include <QProgressDialog>
#include <qtconcurrentrun.h>

//system
#if defined(CV_WINDOWS)
#include "windows.h"
#else
#include <time.h>
#include <unistd.h>
#endif

BasePclModule::BasePclModule(PclModuleDescription desc, ccPluginInterface * parent_plugin)
    : m_action(nullptr)
	, m_desc(desc)
	, m_show_progress(true)
{
	initAction();
	m_parent_plugin = parent_plugin;
}

void BasePclModule::initAction()
{
	if (m_action)
		return;

	m_action = new QAction(getIcon(), getEntryName(), this);
	m_action->setStatusTip(getStatusTip());
	//connect this action
	connect(m_action, SIGNAL(triggered()), this, SLOT(performAction()));
}

void BasePclModule::throwError(int errCode)
{
	QString errMsg = getErrorMessage(errCode);

	//DGM: libraries shouldn't issue message themselves! The information should be sent to the plugin!
	emit newErrorMessage(errMsg);
}


void BasePclModule::updateSelectedEntities(const ccHObject::Container& selectedEntities)
{
	m_selected = selectedEntities;

	if (m_action)
		m_action->setEnabled(checkSelected() == 1);
}

int BasePclModule::performAction()
{
	//check if selected entities are good
	int check_result = checkSelected();
	if (check_result != 1)
	{
		throwError(check_result);
		return check_result;
	}

	//if dialog is needed open the dialog
	int dialog_result = openInputDialog();
	if (dialog_result < 1)
	{
		if (dialog_result < 0)
			throwError(dialog_result);
		else
			dialog_result = 1; //the operation is canceled by the user, no need to throw an error!
		return dialog_result;
	}

	//get the parameters from the dialog
	getParametersFromDialog();

	//are the given parameters ok?
	int par_status = checkParameters();
	if (par_status != 1)
	{
		throwError(par_status);
		return par_status;
	}

	//if so go ahead with start()
	int start_status = start();
	if (start_status != 1)
	{
		throwError(start_status);
		return start_status;
	}

	//if we have an output dialog is time to show it
	int out_dialog_result = openOutputDialog();
	if (out_dialog_result < 1)
	{
		if (out_dialog_result < 0)
			throwError(out_dialog_result);
		else
			out_dialog_result = 1; // the operation is canceled by the user, no need to throw an error!
		return out_dialog_result; // maybe some module could ask the user if he wants to ac
	}

	return 1;
}

int BasePclModule::checkSelected()
{
	//In most of the cases we just need 1 CC_POINT_CLOUD
	if (m_selected.empty())
		return -11;

	if (m_selected.size() != 1)
		return -12;

	if (!m_selected[0]->isA(CV_TYPES::POINT_CLOUD) )
		return -13;

	return 1;
}

static BasePclModule* s_module = 0;
static int s_computeStatus = 0;
static bool s_computing = false;
static void doCompute()
{
	if (!s_module)
	{
		s_computeStatus = -1;
		return;
	}

	s_computeStatus = s_module->compute();
}

int BasePclModule::start()
{
	if (s_computing)
	{
		throwError(-32);
		return -1;
	}

	QProgressDialog progressCb(tr("Operation in progress"), QString(), 0, 0);

	if (m_show_progress)
	{
		progressCb.setWindowTitle(getModuleName());
		progressCb.show();
		QApplication::processEvents();
	}

	s_computeStatus = -1;
	s_module = this;
	s_computing = true;
	int progress = 0;

	QFuture<void> future = QtConcurrent::run(doCompute);
	while (!future.isFinished())
	{
#if defined(CV_WINDOWS)
		::Sleep(500);
#else
		usleep(500 * 1000);
#endif
		if (m_show_progress)
			progressCb.setValue(++progress);
	}
	
	int is_ok = s_computeStatus;
	s_module = 0;
	s_computing = false;

	if (m_show_progress)
	{
		progressCb.close();
		QApplication::processEvents();
	}

	if (is_ok < 0)
	{
		throwError(is_ok);
		//return -1;
	}

	return 1;
}

QString BasePclModule::getModuleName() const
{
	return m_desc.m_module_name;
}

QString BasePclModule::getEntryName() const
{
	return m_desc.m_entry_name;
}

QString BasePclModule::getStatusTip() const
{
	return m_desc.m_status_tip;
}

QIcon BasePclModule::getIcon() const
{
	return m_desc.m_icon;
}

QAction* BasePclModule::getAction()
{
	return m_action;
}

QString BasePclModule::getErrorMessage(int errorCode)
{
	QString errorMsg;
	switch(errorCode)
	{
	//ERRORS RELATED TO SELECTION
	case -11:
		errorMsg=QString(tr("No entity selected in tree."));
		break;

	case -12:
		errorMsg=QString(tr("Too many entities selected."));
		break;

	case -13:
		errorMsg=QString(tr("Wrong type of entity selected"));
		break;

	//ERRORS RELATED TO DIALOG
	case -21:
		errorMsg=QString(tr("Dialog was not correctly filled in"));
		break;

	//ERRORS RELATED TO COMPUTATION
	case -31:
		errorMsg=QString(tr("Errors while computing"));
		break;
	case -32:
		errorMsg=QString(tr("Thread already in use!"));
		break;

	// DEFAULT
	default:
		errorMsg= tr("Undefined error in %1 module").arg(getModuleName());
		break;
	}

	return errorMsg;
}

ccPointCloud* BasePclModule::getSelectedEntityAsCCPointCloud() const
{
	//does we have any selected entity?
	if (m_selected.size() == 0)
		return nullptr;

	ccHObject* entity = m_selected.at(0);
	if (!entity->isA(CV_TYPES::POINT_CLOUD))
		return nullptr;

	return ccHObjectCaster::ToPointCloud(entity);
}

ccHObject *BasePclModule::getSelectedEntityAsCCHObject() const
{
	//does we have any selected entity?
	if (m_selected.size() == 0)
		return nullptr;

	return m_selected.at(0);
}

ccHObject::Container BasePclModule::getSelectedThatHaveMetaData(const QString key) const
{
	ccHObject::Container new_sel;

	for (size_t i = 0; i < m_selected.size(); ++i)
	{
		ccHObject * obj = m_selected.at(i);
		if (obj->hasMetaData(key))
			new_sel.push_back(obj);
	}

	return new_sel;
}

void BasePclModule::getAllEntitiesOfType(CV_CLASS_ENUM type, ccHObject::Container& entities)
{
	if (!m_app || !m_app->dbRootObject())
		return;

	m_app->dbRootObject()->filterChildren(entities,true,type);
}

void BasePclModule::getAllEntitiesThatHaveMetaData(QString key, ccHObject::Container &entities)
{
	entities.clear(); //better be sure
	ccHObject::Container tempContainer;
	getAllEntitiesOfType(CV_TYPES::HIERARCHY_OBJECT, tempContainer);

	for (ccHObject::Container::const_iterator it = tempContainer.begin(); it != tempContainer.end(); ++it )
	{
		if ((*it)->hasMetaData(key))
			entities.push_back(*it);
	}
}

void BasePclModule::getSelectedEntitiesThatAreCCPointCloud(ccHObject::Container & entities)
{
	ccHObject::Container selected = m_selected;
	for (size_t i = 0 ; i < selected.size(); ++i)
	{
		ccHObject * this_obj = selected[i];
		if (this_obj->isA(CV_TYPES::POINT_CLOUD))
		{
			entities.push_back(this_obj);
		}
	}
}

void BasePclModule::getSelectedEntitiesThatAre(CV_CLASS_ENUM  kind, ccHObject::Container & entities)
{
	ccHObject::Container selected = m_selected;
	for (size_t i = 0 ; i < selected.size(); ++i)
	{
		ccHObject * this_obj = selected[i];
		if (this_obj->isA(kind))
		{
			entities.push_back(this_obj);
		}
	}
}

int BasePclModule::hasSelectedScalarField(std::string field_name)
{
	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (!cloud)
		return -1;

	int result = cloud->getScalarFieldIndexByName(field_name.c_str());

	return (result >= 0 ? 1 : 0);
}

int BasePclModule::hasSelectedScalarField()
{
	if (isFirstSelectedCcPointCloud() != 1)
		return -1;

	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (!cloud)
		return -1;

	return (cloud->hasScalarFields() ? 1 : 0);
}

std::vector<std::string> BasePclModule::getSelectedAvailableScalarFields()
{
	std::vector<std::string> field_names;

	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (!cloud)
		return field_names;

	unsigned n_fields = cloud->getNumberOfScalarFields();
	field_names.reserve(n_fields);
	for (unsigned i = 0; i < n_fields; i++)
		field_names.push_back(cloud->getScalarFieldName(i));

	return field_names;
}

int BasePclModule::isFirstSelectedCcPointCloud()
{
	if (!m_selected.empty() && m_selected.at(0)->isA(CV_TYPES::POINT_CLOUD))
		return 1;

	return -1;
}

int BasePclModule::hasSelectedRGB()
{
	if (isFirstSelectedCcPointCloud() != 1)
		return -1;
	//get the cloud

	ccPointCloud * cloud;
	cloud = getSelectedEntityAsCCPointCloud();

	return cloud->hasColors();
}
