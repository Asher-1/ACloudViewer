// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

#pragma once

#include "eCV_python.h"
#include "Utility/ClassMap.h"

class ccHObject;
namespace cloudViewer {
namespace utility {

class ECV_PYTHON_LIB_API DeepSemanticSegmentation {
public:
    DeepSemanticSegmentation();
    ~DeepSemanticSegmentation();

public:
	inline void setEnableVotes(bool state) { m_useVotes = state; }
	inline void setEnableSampling(bool state) { m_useGridSampling = state; }
    void setInputCloud(const ccHObject* cloud);
	void compute(std::vector< std::vector<size_t> >& clusters, std::vector< ClassMap::ClusterMap > &cluster_map);

protected:
	void extract(
		const std::vector< std::vector<size_t> >& preds,
		std::vector< ClassMap::ClusterMap > &clusters);
	
private:
	bool m_batchMode;
	const ccHObject* m_container;
	bool m_useGridSampling;
	bool m_useVotes;
};

}  // namespace utility
}  // namespace cloudViewer
