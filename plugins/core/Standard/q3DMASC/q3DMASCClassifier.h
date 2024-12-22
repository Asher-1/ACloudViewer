#pragma once

//##########################################################################
//#                                                                        #
//#                     ACLOUDVIEWER PLUGIN: q3DMASC                       #
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
//#                 COPYRIGHT: Dimitri Lague / CNRS / UEB                  #
//#                                                                        #
//##########################################################################

//Local
#include "Parameters.h"
#include "FeaturesInterface.h"

//Qt
#include <QString>

//CCLib
#include <ReferenceCloud.h>

//OpenCV
#include <opencv2/ml.hpp>

class QWidget;
class ecvMainAppInterface;
class ConfusionMatrix;
class Train3DMASCDialog;

//! 3DMASC classifier
namespace masc
{
	class Classifier
	{
	public:

		//! Default constructor
		Classifier();

		//! Train the classifier
		bool train(	const ccPointCloud* cloud,
					const RandomTreesParams& params,
					const Feature::Source::Set& featureSources,
					QString& errorMessage,
					cloudViewer::ReferenceCloud* trainSubset = nullptr,
					ecvMainAppInterface* app = nullptr,
					QWidget* parentWidget = nullptr);

		//! Classifier accuracy metrics
		struct AccuracyMetrics
		{
			unsigned sampleCount = 0;
			unsigned goodGuess = 0;
			float ratio = 0.0f;
		};

		//! Evaluates the classifier
		bool evaluate(	const Feature::Source::Set& featureSources,
						ccPointCloud* testCloud,
						AccuracyMetrics& metrics,
						QString& errorMessage,
						Train3DMASCDialog& train3DMASCDialog,
						cloudViewer::ReferenceCloud* testSubset = nullptr,
						QString outputSFName = QString(),
						QWidget* parentWidget = nullptr,
						ecvMainAppInterface* app = nullptr);

		//! Applies the classifier
		bool classify(	const Feature::Source::Set& featureSources,
						ccPointCloud* cloud,
						QString& errorMessage,
						QWidget* parentWidget = nullptr,
						ecvMainAppInterface* app = nullptr);

		//! Returns whether the classifier is valid or not
		bool isValid() const;

		//! Saves the classifier to file
		bool toFile(QString filename, QWidget* parentWidget = nullptr) const;
		//! Loads the classifier from file
		bool fromFile(QString filename, QWidget* parentWidget = nullptr);

		inline cv::Mat getVarImportance() const { return m_rtrees->getVarImportance(); }

	protected:

		//! Random trees (OpenCV)
		cv::Ptr<cv::ml::RTrees> m_rtrees;
	};

}; //namespace masc
