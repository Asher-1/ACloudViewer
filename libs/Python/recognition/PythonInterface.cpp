// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
