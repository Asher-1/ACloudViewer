//##########################################################################
//#                                                                        #
//#                              cloudViewer                               #
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

#include "CVLog.h"

//CV_CORE_lib
#include <CVPlatform.h>

//System
#include <cassert>
#include <vector>

#if !defined(CV_WINDOWS)
#define _vsnprintf vsnprintf
#endif

/***************
 *** Globals ***
 ***************/

//buffer for formatted string generation
static const size_t s_bufferMaxSize = 4096;
static char s_buffer[s_bufferMaxSize];

//! Message
struct Message
{
	Message(const QString& t, int f)
		: text(t)
		, flags(f)
	{}

	QString text;
	int flags;
};

//message backup system
static bool s_backupEnabled;
//backuped messages
static std::vector<Message> s_backupMessages;

//unique console instance
static CVLog* s_instance = nullptr;

CVLog* CVLog::TheInstance()
{
	return s_instance;
}

void CVLog::EnableMessageBackup(bool state)
{
	s_backupEnabled = state;
}

void CVLog::LogMessage(const QString& message, int level)
{
#ifndef QT_DEBUG
	//skip debug messages in release mode as soon as possible
	if (level & DEBUG_FLAG)
	{
		return;
	}
#endif

	if (s_instance)
	{
		s_instance->logMessage(message, level);
	}
	else if (s_backupEnabled)
	{
		try
		{
			s_backupMessages.emplace_back(message, level);
		}
		catch (const std::bad_alloc&)
		{
			//nothing to do, the message will be lost...
		}
	}
}

void CVLog::RegisterInstance(CVLog* logInstance)
{
	s_instance = logInstance;
	if (s_instance)
	{
		//if we have a valid instance, we can now flush the backuped messages
		for (const Message& message : s_backupMessages)
		{
			s_instance->logMessage(message.text, message.flags);
		}
		s_backupMessages.clear();
	}
}

//Conversion from '...' parameters to QString so as to call CVLog::logMessage
//(we get the "..." parameters as "printf" would do)
#define LOG_ARGS(flags)\
	if (s_instance || s_backupEnabled)\
	{\
		va_list args;\
		va_start(args, format);\
		_vsnprintf(s_buffer, s_bufferMaxSize, format, args);\
		va_end(args);\
		LogMessage(QString(s_buffer), flags);\
	}\

bool CVLog::Print(const char* format, ...)
{
	LOG_ARGS(LOG_STANDARD)
	return true;
}

bool CVLog::Warning(const char* format, ...)
{
	LOG_ARGS(LOG_WARNING)
	return false;
}

bool CVLog::Error(const char* format, ...)
{
	LOG_ARGS(LOG_ERROR)
	return false;
}

bool CVLog::PrintDebug(const char* format, ...)
{
#ifdef QT_DEBUG
	LOG_ARGS(LOG_STANDARD | DEBUG_FLAG)
#endif
	return true;
}

bool CVLog::WarningDebug(const char* format, ...)
{
#ifdef QT_DEBUG
	LOG_ARGS(LOG_WARNING | DEBUG_FLAG)
#endif
	return false;
}

bool CVLog::ErrorDebug(const char* format, ...)
{
#ifdef QT_DEBUG
	LOG_ARGS(LOG_ERROR | DEBUG_FLAG)
#endif
	return false;
}