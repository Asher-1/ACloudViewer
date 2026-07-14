// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qBroom.h"
#include "qBroomDlg.h"
#include "qBroomDisclaimerDialog.h"

//Qt
#include <QtGui>

//qCC_db
#include <ecvPointCloud.h>

//system
#include <assert.h>

qBroom::qBroom(QObject* parent)
	: QObject( parent )
	, ccStdPluginInterface( ":/CC/plugin/qBroom/info.json" )
	, m_action( nullptr )
{
}

QList<QAction *> qBroom::getActions()
{
	//default action
	if (!m_action)
	{
		m_action = new QAction(getName(),this);
		m_action->setToolTip(getDescription());
		m_action->setIcon(getIcon());
		//connect signal
		connect(m_action, &QAction::triggered, this, &qBroom::doAction);
	}

	return QList<QAction *>{ m_action };
}

void qBroom::onNewSelection(const ccHObject::Container& selectedEntities)
{
	if (m_action)
	{
		//a single point cloud must be selected
		m_action->setEnabled(selectedEntities.size() == 1 && selectedEntities.front()->isA(CV_TYPES::POINT_CLOUD));
	}
}

void qBroom::doAction()
{
	if (!m_app)
	{
		assert(false);
		return;
	}

	//disclaimer accepted?
	if (!ShowDisclaimer(m_app))
	{
		return;
	}

	const ccHObject::Container& selectedEntities = m_app->getSelectedEntities();

	if ( !m_app->haveOneSelection() || !selectedEntities.front()->isA(CV_TYPES::POINT_CLOUD))
	{
		m_app->dispToConsole("Select one cloud!", ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}

	ccPointCloud* cloud = static_cast<ccPointCloud*>(selectedEntities.front());

	qBroomDlg broomDlg(m_app);

	// Show the dialog first so Qt lays out the VTK widget and
	// QOpenGLWidget::initializeGL() runs before we feed it geometry.
	broomDlg.show();
	QCoreApplication::processEvents();

	//automatically deselect the input cloud
	m_app->setSelectedInDB(cloud, false);

	if (broomDlg.setCloud(cloud))
	{
		broomDlg.exec();
	}

	//currently selected entities appearance may have changed!
	m_app->refreshAll();
}
