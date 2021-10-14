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

#ifndef CV_LOG_HEADER
#define CV_LOG_HEADER

//Local
#include "CVCoreLib.h"

//system
#include <stdio.h>
#include <string>

//Qt
#include <QString>

//! Main log interface
/** This interface is meant to be used as a unique (static) instance.
	It should be thread safe!
**/


class CV_CORE_LIB_API CVLog
{
public:

	//! Destructor
	virtual ~CVLog() {}

	//! Returns the static and unique instance
	static CVLog* TheInstance();

	//! Registers a unique instance
	static void RegisterInstance(CVLog* logInstance);

	//! Enables the message backup system
	/** Stores the messages until a valid logging instance is registered.
	**/
	static void EnableMessageBackup(bool state);

	//! Message level
	enum MessageLevelFlags
	{
		LOG_STANDARD		= 0, /**< Standard message (Print) **/
		LOG_DEBUG	        = 1, /**< Debug only flag **/
		LOG_WARNING			= 2, /**< Warning message (Warning) **/
		LOG_ERROR			= 4, /**< Error message (Error) **/
	};

	//! Static shortcut to CVLog::logMessage
	static void LogMessage(const QString& message, int level);

	//! Generic message logging method
	/** To be implemented by child class.
		\warning MUST BE THREAD SAFE!
		\param message message
		\param level message severity (see MessageLevelFlags)
	**/
	virtual void logMessage(const QString& message, int level) = 0;

	//! Prints out a formatted message in console
	/** Works just like the 'printf' command.
		\return always return 'true'
	**/
	static bool Print(const char *format, ...);

	//! QString version of CVLog::Print
	inline static bool Print(const QString& message) { LogMessage(message, LOG_STANDARD); return true; }

	//! Same as Print, but works only in debug mode
	/** Works just like the 'printf' command.
		\return always return 'true'
	**/
	static bool PrintDebug(const char *format, ...);

	//! QString version of CVLog::PrintDebug
	inline static bool PrintDebug(const QString& message) { LogMessage(message, LOG_STANDARD | LOG_DEBUG); return true; }

	//! Prints out a formatted warning message in console
	/** Works just like the 'printf' command.
		\return always return 'false'
	**/
	static bool Warning(const char *format, ...);

	//! QString version of CVLog::Warning
	inline static bool Warning(const QString& message) { LogMessage(message, LOG_WARNING); return false; }

	//! Same as Warning, but works only in debug mode
	/** Works just like the 'printf' command.
		\return always return 'false'
	**/
	static bool WarningDebug(const char *format, ...);

	//! QString version of CVLog::WarningDebug
	inline static bool WarningDebug(const QString& message) { LogMessage(message, LOG_WARNING | LOG_DEBUG); return false; }

	//! Display an error dialog with formatted message
	/** Works just like the 'printf' command.
		\return always return 'false'
	**/
	static bool Error(const char *format, ...);

	//! QString version of 'Error'
	inline static bool Error(const QString& message) { LogMessage(message, LOG_ERROR); return false; }

	//! Same as Error, but works only in debug mode
	/** Works just like the 'printf' command.
		\return always return 'false'
	**/
	static bool ErrorDebug(const char *format, ...);

	//! QString version of CVLog::ErrorDebug
	static bool ErrorDebug(const QString& message) { LogMessage(message, LOG_ERROR | LOG_DEBUG); return false; }
};

#endif //CC_LOG_HEADER
