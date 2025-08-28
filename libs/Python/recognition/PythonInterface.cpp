// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// LOCAL
#include "recognition/PythonInterface.h"

// CV_CORE_LIB
#include <CVTools.h>

// PYTHON
//#undef slots
#include <pybind11/embed.h>

// SYSTEM
#include <string>

namespace PythonInterface
{
	static std::wstring PYTHON_HOME = L"";
	bool SetPythonHome(const wchar_t * pyHome)
	{
		if (!pyHome)
		{
			return false;
		}

		wchar_t * s = const_cast<wchar_t *>(pyHome);
		Py_SetPythonHome(s);
		return true;
	}

	bool SetPythonHome(const char * pyHome)
	{
		if (!pyHome)
		{
			return false;
		}
		PYTHON_HOME = CVTools::Char2Wchar(pyHome);

		return SetPythonHome(PYTHON_HOME.c_str());
	}

}
