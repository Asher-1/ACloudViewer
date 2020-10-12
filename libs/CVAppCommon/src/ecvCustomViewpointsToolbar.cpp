/*=========================================================================

   Program: ParaView
   Module:    ecvCustomViewpointsToolbar.cxx

   Copyright (c) 2005,2006 Sandia Corporation, Kitware Inc.
   All rights reserved.

   ParaView is a free software; you can redistribute it and/or modify it
   under the terms of the ParaView license version 1.2.

   See License_v1.2.txt for the full ParaView license.
   A copy of this license can be obtained by contacting
   Kitware Inc.
   28 Corporate Drive
   Clifton Park, NY 12065
   USA

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

========================================================================*/
#include "ecvCustomViewpointsToolbar.h"

// LOCAL
#include "ecvCameraParamEditDlg.h"
#include "ecvCustomViewpointButtonDlg.h"
#include "ecvSettingManager.h"

// ECV_DB_LIB
#include <ecvDisplayTools.h>

#ifdef ECV_PCL_ENGINE_LIBRARY_BUILD
#include <../PCLEngine/Tools/EditCameraTool.h>
#endif

// QT
#include <QPainter>
#include <QApplication>

//-----------------------------------------------------------------------------
void ecvCustomViewpointsToolbar::constructor()
{
	// Create base pixmap
	this->BasePixmap.fill(QColor(0, 0, 0, 0));
	QPainter pixPaint(&this->BasePixmap);
	pixPaint.drawPixmap(0, 0, 48, 48, QPixmap(":/Resources/images/svg/pqCamera.png"));

	// Create plus pixmap
	this->PlusPixmap = this->BasePixmap.copy();
	QPainter pixWithPlusPaint(&this->PlusPixmap);
	pixWithPlusPaint.drawPixmap(32, 32, 32, 32, QPixmap(":/Resources/images/svg/pqPlus.png"));
	this->ConfigPixmap = this->BasePixmap.copy();
	QPainter pixWithConfigPaint(&this->ConfigPixmap);
	pixWithConfigPaint.drawPixmap(32, 32, 32, 32, QPixmap(":/Resources/images/svg/pqWrench.png"));

	this->PlusAction = this->ConfigAction = nullptr;

	this->setWindowTitle(tr("Custom Viewpoints Toolbar"));
	this->updateCustomViewpointActions();
	this->connect(ecvDisplayTools::TheInstance(), SIGNAL(cameraParamChanged()), SLOT(updateEnabledState()));
	ecvSettingManager* settings = ecvSettingManager::TheInstance();
	this->connect(settings, SIGNAL(modified()), SLOT(updateCustomViewpointActions()));
}

//-----------------------------------------------------------------------------
void ecvCustomViewpointsToolbar::updateEnabledState()
{
	this->setEnabled(ecvDisplayTools::GetCurrentScreen() != nullptr);
	auto actions = this->actions();
	actions[actions.size() - 2]->setEnabled(
		actions.size() < ecvCustomViewpointButtonDlg::MAXIMUM_NUMBER_OF_ITEMS + 2);
}

//-----------------------------------------------------------------------------
void ecvCustomViewpointsToolbar::updateCustomViewpointActions()
{
	// Recover tooltips from settings
	QStringList tooltips = ecvCameraParamEditDlg::CustomViewpointToolTips();

	if (!this->ConfigAction)
	{
		this->ConfigAction = this->addAction(QIcon(this->ConfigPixmap),
			tr("Configure custom viewpoints"), this, SLOT(ConfigureCustomViewpoints()));
		this->ConfigAction->setObjectName("ConfigAction");
	}

	if (!this->PlusAction)
	{
		this->PlusAction =
			this->addAction(QIcon(this->PlusPixmap), tr("Add current viewpoint as custom viewpoint"),
				this, SLOT(addCurrentViewpointToCustomViewpoints()));
		this->PlusAction->setObjectName("PlusAction");
	}

	// Remove unused actions
	for (int cc = this->ViewpointActions.size(); cc > tooltips.size(); cc--)
	{
		this->removeAction(this->ViewpointActions[cc - 1]);
	}
	if (this->ViewpointActions.size() > tooltips.size())
	{
		this->ViewpointActions.resize(tooltips.size());
	}

	// add / change actions for custom views.
	for (int cc = 0; cc < tooltips.size(); cc++)
	{
		QPixmap pixmap;
		if (this->ViewpointActions.size() > cc)
		{
			this->ViewpointActions[cc]->setToolTip(tooltips[cc]);
		}
		else
		{
			// action does not exist yet, create it
			pixmap = this->BasePixmap.copy();
			QPainter pixWithNumberPaint(&pixmap);
			QFont font = pixWithNumberPaint.font();
			font.setPixelSize(24);
			pixWithNumberPaint.setFont(font);
			pixWithNumberPaint.setPen(QApplication::palette().windowText().color());
			pixWithNumberPaint.drawText(
				pixmap.rect(), Qt::AlignRight | Qt::AlignBottom, QString::number(cc + 1));

			QAction* action = this->addAction(QIcon(pixmap), "", this, SLOT(ApplyCustomViewpoint()));
			action->setObjectName(QString("ViewpointAction%1").arg(cc));
			action->setToolTip(tooltips[cc]);
			action->setData(cc);
			this->ViewpointActions.push_back(action);
		}
	}

	this->updateEnabledState();
}

//-----------------------------------------------------------------------------
void ecvCustomViewpointsToolbar::ConfigureCustomViewpoints()
{
	ecvCameraParamEditDlg::ConfigureCustomViewpoints(this);
}

//-----------------------------------------------------------------------------
void ecvCustomViewpointsToolbar::SetToCurrentViewpoint()
{
	if (ecvDisplayTools::GetCurrentScreen())
	{
		int customViewpointIndex;
		QAction* action = qobject_cast<QAction*>(this->sender());
		if (!action)
		{
			return;
		}

		customViewpointIndex = action->data().toInt();
		if (ecvCameraParamEditDlg::SetToCurrentViewpoint(customViewpointIndex))
		{
			ecvDisplayTools::UpdateScreen();
		}
	}
}

//-----------------------------------------------------------------------------
void ecvCustomViewpointsToolbar::DeleteCustomViewpoint()
{
	if (ecvDisplayTools::GetCurrentScreen())
	{
		int customViewpointIndex;
		QAction* action = qobject_cast<QAction*>(this->sender());
		if (!action)
		{
			return;
		}

		customViewpointIndex = action->data().toInt();
		if (ecvCameraParamEditDlg::DeleteCustomViewpoint(customViewpointIndex))
		{
			ecvDisplayTools::UpdateScreen();
		}
	}
}

//-----------------------------------------------------------------------------
void ecvCustomViewpointsToolbar::addCurrentViewpointToCustomViewpoints()
{

#ifdef ECV_PCL_ENGINE_LIBRARY_BUILD // update camera parameters in case
	EditCameraTool::UpdateCameraInfo();
#endif

	ecvCameraParamEditDlg::AddCurrentViewpointToCustomViewpoints();
}

//-----------------------------------------------------------------------------
void ecvCustomViewpointsToolbar::ApplyCustomViewpoint()
{
#ifdef ECV_PCL_ENGINE_LIBRARY_BUILD
	if (ecvDisplayTools::GetCurrentScreen())
	{
		int customViewpointIndex;
		QAction* action = qobject_cast<QAction*>(this->sender());
		if (!action)
		{
			return;
		}

		customViewpointIndex = action->data().toInt();
		if (ecvCameraParamEditDlg::ApplyCustomViewpoint(customViewpointIndex))
		{
			EditCameraTool::UpdateCamera();
		}
	}
#else
	CVLog::Warning("[ecvCustomViewpointsToolbar::ApplyCustomViewpoint] please use pcl as backend and then try again!");
#endif

}
