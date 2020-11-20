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
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

//Local
#include "ecvConsole.h"
#include "ecvHead.h"
#include "MainWindow.h"
#include <CommonSettings.h>
#include "ecvSettingManager.h"
#include "ecvPersistentSettings.h"

// ECV_DB_LIB
#include <ecvSingleton.h>

//Qt
#include <QApplication>
#include <QClipboard>
#include <QColor>
#include <QKeyEvent>
#include <QMessageBox>
#include <QTextStream>
#include <QThread>
#include <QTime>

// system
#include <cassert>
#ifdef QT_DEBUG
#include <iostream>
#endif

/***************
 *** Globals ***
 ***************/

//unique console instance
static ecvSingleton<ecvConsole> s_console;

bool ecvConsole::s_redirectToStdOut = false;
bool ecvConsole::s_showQtMessagesInConsole = false;


// ecvCustomQListWidget
ecvCustomQListWidget::ecvCustomQListWidget(QWidget *parent)
	: QListWidget(parent)
{
}

void ecvCustomQListWidget::keyPressEvent(QKeyEvent *event)
{
	if (event->matches(QKeySequence::Copy))
	{
		int itemsCount = count();
		QStringList strings;
		for (int i = 0; i < itemsCount; ++i)
		{
			if (item(i)->isSelected())
			{
				strings << item(i)->text();
			}
		}
		
		QApplication::clipboard()->setText(strings.join("\n"));
	}
	else
	{
		QListWidget::keyPressEvent(event);
	}
}


// ecvConsole
ecvConsole* ecvConsole::TheInstance(bool autoInit/*=true*/)
{
	if (!s_console.instance && autoInit)
	{
		s_console.instance = new ecvConsole;
		CVLog::RegisterInstance(s_console.instance);
	}

	return s_console.instance;
}

void ecvConsole::ReleaseInstance(bool flush/*=true*/)
{
	if (flush && s_console.instance)
	{
		//DGM: just in case some messages are still in the queue
		s_console.instance->refresh();
	}
	CVLog::RegisterInstance(nullptr);
	s_console.release();
}

ecvConsole::ecvConsole()
	: m_textDisplay(nullptr)
	, m_parentWidget(nullptr)
	, m_parentWindow(nullptr)
	, m_logStream(nullptr)
{
}

ecvConsole::~ecvConsole()
{
	setLogFile(QString()); //to close/delete any active stream
}

void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
#ifndef QT_DEBUG
	if (!ecvConsole::QtMessagesEnabled())
	{
		return;
	}

	if (type == QtDebugMsg)
	{
		return;
	}
#endif

	QString message = QString("[%1] ").arg(context.function) + msg; // QString("%1 (%1:%1, %1)").arg(msg).arg(context.file).arg(context.line).arg(context.function);

	//in this function, you can write the message to any stream!
	switch (type)
	{
	case QtDebugMsg:
		CVLog::PrintDebug(msg);
		break;
	case QtWarningMsg:
		message.prepend("[Qt WARNING] ");
		CVLog::Warning(message);
		break;
	case QtCriticalMsg:
		message.prepend("[Qt CRITICAL] ");
		CVLog::Warning(message);
		break;
	case QtFatalMsg:
		message.prepend("[Qt FATAL] ");
		CVLog::Warning(message);
		break;
	case QtInfoMsg:
		message.prepend("[Qt INFO] ");
		CVLog::Warning(message);
		break;
	}
	
#ifdef QT_DEBUG
	// Also send the message to the console so we can look at the output when CC has quit
	//	(in Qt Creator's Application Output for example)
	switch (type)
	{
		case QtDebugMsg:
		case QtWarningMsg:
		case QtInfoMsg:
			std::cout << message.toStdString() << std::endl;
			break;
			
		case QtCriticalMsg:
		case QtFatalMsg:
			std::cerr << message.toStdString() << std::endl;
			break;
	}
	
#endif
}

void ecvConsole::EnableQtMessages(bool state)
{
	s_showQtMessagesInConsole = state;

	// persistent settings
	ecvSettingManager::setValue(ecvPS::Console(), "QtMessagesEnabled", s_showQtMessagesInConsole);
}

void ecvConsole::Init(	QListWidget* textDisplay/*=0*/,
						QWidget* parentWidget/*=0*/,
						MainWindow* parentWindow/*=0*/,
						bool redirectToStdOut/*=false*/)
{
	//should be called only once!
	if (s_console.instance)
	{
		assert(false);
		return;
	}
	
	s_console.instance = new ecvConsole;
	s_console.instance->m_textDisplay = textDisplay;
	s_console.instance->m_parentWidget = parentWidget;
	s_console.instance->m_parentWindow = parentWindow;
	s_redirectToStdOut = redirectToStdOut;

	//auto-start
	if (textDisplay)
	{
		//load from persistent settings
		s_showQtMessagesInConsole = ecvSettingManager::getValue(ecvPS::Console(), 
			"QtMessagesEnabled", false).toBool();

#ifdef CV_WINDOWS // only support Log file in Windows now!
		// set log file.
		s_console.instance->setLogFile(Settings::LOGFILE);
#endif

		//install : set the callback for Qt messages
		qInstallMessageHandler(myMessageOutput);

		s_console.instance->setAutoRefresh(true);
	}

	CVLog::RegisterInstance(s_console.instance);
}

void ecvConsole::setAutoRefresh(bool state)
{
	if (state)
	{
		connect(&m_timer, &QTimer::timeout, this, &ecvConsole::refresh);
		m_timer.start(1000);
	}
	else
	{
		m_timer.stop();
		disconnect(&m_timer, &QTimer::timeout, this, &ecvConsole::refresh);
	}
}

void ecvConsole::refresh()
{
	m_mutex.lock();
	
	if ((m_textDisplay || m_logStream) && !m_queue.isEmpty())
	{
		for (QVector<ConsoleItemType>::const_iterator it=m_queue.constBegin(); it!=m_queue.constEnd(); ++it)
		{
			 //it->second = message severity
			bool debugMessage = (it->second & LOG_DEBUG);
#ifndef QT_DEBUG
			//skip debug message in release mode
			if (debugMessage)
				continue;
#endif

			//destination: log file
			if (m_logStream)
			{
				*m_logStream << it->first << endl;
			}

			//destination: console widget
			if (m_textDisplay)
			{
				//it->first = message text
				QListWidgetItem* item = new QListWidgetItem(it->first);

				//set color based on the message severity
				//Error
				if (it->second & LOG_ERROR)
				{
					item->setForeground(Qt::red);
				}
				//Warning
				else if (it->second & LOG_WARNING)
				{
					item->setForeground(Qt::darkRed);
					//we also force the console visibility if a warning message arrives!
					if (m_parentWindow)
						m_parentWindow->forceConsoleDisplay();
				}
				//Standard
				else
				{
#ifdef QT_DEBUG
					if (debugMessage)
						item->setForeground(Qt::blue);
					else
#endif
						item->setForeground(Qt::black);
				}

				m_textDisplay->addItem(item);
			}
		}

		if (m_logStream)
			m_logFile.flush();

		if (m_textDisplay)
			m_textDisplay->scrollToBottom();
	}

	m_queue.clear();

	m_mutex.unlock();
}

void ecvConsole::logMessage(const QString& message, int level)
{
#ifndef QT_DEBUG
	//skip debug messages in release mode
	if (level & LOG_DEBUG)
	{
		return;
	}
#endif

	//QString line = __LINE__;
	//QString filename = __FILE__;
	//QString functionname = __FUNCTION__;

	QString formatedMessage = QStringLiteral("[") + DATETIME + QStringLiteral("] ") + message;

	if (s_redirectToStdOut)
	{
		printf("%s\n", qPrintable(message));
	}
	if (m_textDisplay || m_logStream)
	{
		m_mutex.lock();
		m_queue.push_back(ConsoleItemType(formatedMessage,level));
		m_mutex.unlock();
	}
#ifdef QT_DEBUG
	else
	{
		//Error
		if (level & LOG_ERROR)
		{
			if (level & LOG_DEBUG)
				printf("ERR-DBG: ");
			else
				printf("ERR: ");
		}
		//Warning
		else if (level & LOG_WARNING)
		{
			if (level & LOG_DEBUG)
				printf("WARN-DBG: ");
			else
				printf("WARN: ");
		}
		//Standard
		else
		{
			if (level & LOG_DEBUG)
				printf("MSG-DBG: ");
			else
				printf("MSG: ");
		}
		printf(" %s\n",qPrintable(formatedMessage));
	}
#endif

	//we display the error messages in a popup dialog
	if (	(level & LOG_ERROR)
		&&	qApp
		&&	m_parentWidget
		&&	QThread::currentThread() == qApp->thread()
		)
	{
		QMessageBox::warning(m_parentWidget, "Error", message);
	}
}

bool ecvConsole::setLogFile(const QString& filename)
{
	//close previous stream (if any)
	if (m_logStream)
	{
		m_mutex.lock();
		delete m_logStream;
		m_logStream = nullptr;
		m_mutex.unlock();

		if (m_logFile.isOpen())
		{
			m_logFile.close();
		}
	}
	
	if (!filename.isEmpty())
	{
		QString logPath;
		logPath = QCoreApplication::applicationDirPath() + "/";
		logPath += filename;
		m_logFile.setFileName(logPath);
		if (!m_logFile.open(QFile::Text| QFile::WriteOnly | QFile::Append))
		{
			return Error(QString("[Console] Failed to open/create log file '%1'").arg(filename));
		}

		m_mutex.lock();
		m_logStream = new QTextStream(&m_logFile);
		m_mutex.unlock();
		setAutoRefresh(true);
	}

	return true;
}
