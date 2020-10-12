// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#ifndef ECV_POINTS_TO_NUMPY_HEADER
#define ECV_POINTS_TO_NUMPY_HEADER

#include "eCV_python.h"

#include <CVConst.h>

#include "Utility/matrix.h"

class ccHObject;
class ccPointCloud;

namespace cloudViewer {
namespace utility {

/// Class IJsonConvertible defines the behavior of a class that can convert
/// itself to/from a json::Value.
class ECV_PYTHON_LIB_API Points2Numpy {
public:
	explicit Points2Numpy();

	void setInputCloud(const ccPointCloud* cloud);
	
	bool getOutputData(Matrix<PointCoordinateType> &out);

	void batchConvertToNumpy(const ccHObject* cloud, std::vector< Matrix<PointCoordinateType> >& numpyContainer);

private:
	//! Associated cloud
	const ccPointCloud* m_cc_cloud;

	bool m_partialVisibility;
	unsigned m_visibilityNum;
};

}  // namespace utility
}  // namespace cloudViewer

#endif ECV_POINTS_TO_NUMPY_HEADER