//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                              #
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
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#include "ecvProgressDialog.h"

//Qt
#include <QCoreApplication>
#include <QPushButton>
#include <QProgressBar>

ecvProgressDialog::ecvProgressDialog(bool showCancelButton,
									QWidget *parent/*=0*/ )
	: QProgressDialog(parent)
	, m_currentValue(0)
	, m_lastRefreshValue(-1)
{
	setAutoClose(true);

	resize(400, 200);
	setRange(0, 100);
	setMinimumWidth(400);

	QPushButton* cancelButton = 0;
	if (showCancelButton)
	{
		cancelButton = new QPushButton("Cancel");
		cancelButton->setDefault(false);
		cancelButton->setFocusPolicy(Qt::NoFocus);
	}
	setCancelButton(cancelButton);

	connect(this, &ecvProgressDialog::scheduleRefresh, this, &ecvProgressDialog::refresh, Qt::QueuedConnection); //can't use DirectConnection here!
}

void ecvProgressDialog::refresh()
{
	int value = m_currentValue;
	if (m_lastRefreshValue != value)
	{
		m_lastRefreshValue = value;
		setValue(value); //See Qt doc: if the progress dialog is modal, setValue() calls QApplication::processEvents()
	}
}

void ecvProgressDialog::update(float percent)
{
	//thread-safe
	int value = static_cast<int>(percent);
	if (value != m_currentValue)
	{
		m_currentValue = value;
		emit scheduleRefresh();
		QCoreApplication::processEvents(); //we let the main thread breath (so that the call to 'refresh' can be performed)
	}
}

void ecvProgressDialog::setMethodTitle(QString methodTitle)
{
	setWindowTitle(methodTitle);
}

void ecvProgressDialog::setInfo(QString infoStr)
{
	setLabelText(infoStr);
	if (isVisible())
	{
		QProgressDialog::update();
		QCoreApplication::processEvents();
	}
}

void ecvProgressDialog::start()
{
	m_lastRefreshValue = -1;
	show();
	QCoreApplication::processEvents();
}

void ecvProgressDialog::stop()
{
	hide();
	QCoreApplication::processEvents();
}
