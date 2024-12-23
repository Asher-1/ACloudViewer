// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                          -
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
#include "utility/PointsToNumpy.h"
#include "utility/PybindMatrix.h"
#include "utility/PythonModules.h"
#include "recognition/DeepSemanticSegmentation.h"

// pybind11
#undef slots
#include <pybind11/embed.h>

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecvHObject.h>
#include <ecvPointCloud.h>
#include <ecvHObjectCaster.h>

// QT
#include <QThread>

namespace py = pybind11;

namespace cloudViewer {
namespace utility {

DeepSemanticSegmentation::DeepSemanticSegmentation()
    : m_container(nullptr)
	, m_batchMode(false)
	, m_useGridSampling(true)
	, m_useVotes(false)
{}

DeepSemanticSegmentation::~DeepSemanticSegmentation() 
{}

void DeepSemanticSegmentation::setInputCloud(const ccHObject * cloud)
{
	if (!cloud) return;

	m_container = cloud;
	if (m_container->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		m_batchMode = false;
	}
	else
	{
		m_batchMode = true;
	}
}

void DeepSemanticSegmentation::compute(
	std::vector< std::vector<size_t> >& clusters, 
	std::vector< ClassMap::ClusterMap > &cluster_map)
{
	if (!m_container)
	{
		CVLog::Warning(
			"[DeepSemanticSegmentation::compute] failed: must setInpuCloud before compute!");
		return;
	}

	if (m_batchMode && m_container->getChildrenNumber() == 0)
	{
		CVLog::Warning(
			"[DeepSemanticSegmentation::compute] failed: input cloud is empty!");
		return;
	}

	std::vector< Matrix<PointCoordinateType> > data;
	if (m_batchMode)
	{
		Points2Numpy().batchConvertToNumpy(m_container, data);
	}
	else
	{
		Points2Numpy p2npy;
		p2npy.setInputCloud(ccHObjectCaster::ToPointCloud(const_cast<ccHObject*>(m_container)));
		Matrix<PointCoordinateType> matrixPy;
		if (!p2npy.getOutputData(matrixPy))
		{
			// ignore
			return;
		}
		data.push_back(matrixPy);
	}

	if (data.empty())
	{
		return;
	}

	// run deep network RandLANet
	{
		try
		{
			// Init Python Interpreter
			py::initialize_interpreter();

			// import randLANet module
			py::module inferencer = py::module::import(PythonModules::RandLANet);
			py::list data_list = py::cast(data);

			py::module multiplyProcessor = py::module::import("multiprocessing");
			int threadNumber = QThread::idealThreadCount();
			CVLog::Print(QString("use %1 threads for semantic segmentation!").arg(threadNumber));

			py::object pool = multiplyProcessor.attr("Pool")(threadNumber);
			// get inference result
			py::object result =
				inferencer.attr(PythonModules::RandLAInterface)(data_list, m_useGridSampling, m_useVotes, pool);
			clusters = result.cast< std::vector< std::vector<size_t> > >();
		}
		catch (py::error_already_set const &pythonErr)
		{
			// close Python Interpreter
			py::finalize_interpreter();
			CVLog::Error(pythonErr.what());
			return;
		}
		catch (...)
		{
			// close Python Interpreter
			py::finalize_interpreter();
			CVLog::Error("[py::initialize_interpreter] unexpected error occurred!");
			return;
		}
	}
	
	// close Python Interpreter
	py::finalize_interpreter();

	if (clusters.empty())
	{
		CVLog::Warning("[DeepSemanticSegmentation::compute] failed empty result!");
		return;
	}

	// convert preds to clusters
	extract(clusters, cluster_map);
}

void DeepSemanticSegmentation::extract(
	const std::vector< std::vector<size_t> >& preds, 
	std::vector< ClassMap::ClusterMap >& clusters)
{
	assert(m_container);
	for (size_t i = 0; i < preds.size(); ++i)
	{
		ClassMap::ClusterMap clusterMap;
        std::size_t index = 0;
		for (std::vector<size_t>::const_iterator it = preds[i].begin(); 
			it != preds[i].end(); ++it, ++index)
		{
			const size_t ptInd = *it;
			const std::string label = ClassMap::SemanticMap[ptInd];

			if (clusterMap.find(label) == clusterMap.end())
			{
				clusterMap[label] = std::vector<size_t>();
			}

            clusterMap[label].push_back(index);
		}

		clusters.push_back(clusterMap);
	}
}

}  // namespace utility
}  // namespace cloudViewer
