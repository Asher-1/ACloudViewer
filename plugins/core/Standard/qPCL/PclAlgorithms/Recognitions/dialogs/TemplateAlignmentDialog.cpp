//##########################################################################
//#                                                                        #
//#                     CLOUDVIEWER  PLUGIN: qPCL                          #
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
//#      COPYRIGHT: UEB (UNIVERSITE EUROPEENNE DE BRETAGNE) / CNRS         #
//#                                                                        #
//##########################################################################

#include "TemplateAlignmentDialog.h"

// ECV_DB_LIB
#include <ecvMainAppInterface.h>
#include <ecvPointCloud.h>

//Qt
#include <QSettings>
#include <QMainWindow>
#include <QComboBox>
#include <QPushButton>
#include <QApplication>
#include <QThread>

//system
#include <limits>

TemplateAlignmentDialog::TemplateAlignmentDialog(ecvMainAppInterface* app)
	: QDialog(app ? app->getActiveWindow() : 0)
	, Ui::TemplateAlignmentDialog()
	, m_app(app)
{
	setupUi(this);

	int maxThreadCount = QThread::idealThreadCount();
	maxThreadCountSpinBox->setRange(1, maxThreadCount);
	maxThreadCountSpinBox->setSuffix(QString("/%1").arg(maxThreadCount));
	maxThreadCountSpinBox->setValue(maxThreadCount);

	loadParamsFromPersistentSettings();

	connect(template1CloudComboBox,	SIGNAL(currentIndexChanged (int)),	this,	SLOT(onCloudChanged(int)));
	connect(template2CloudComboBox,	SIGNAL(currentIndexChanged (int)),	this,	SLOT(onCloudChanged(int)));

	onCloudChanged(0);
}

void TemplateAlignmentDialog::refreshCloudComboBox()
{
	if (m_app)
	{
		//add list of clouds to the combo-boxes
		ccHObject::Container clouds;
		if (m_app->dbRootObject())
			m_app->dbRootObject()->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD);

		unsigned cloudCount = 0;
		template1CloudComboBox->clear();
		template2CloudComboBox->clear();
		evaluationCloudComboBox->clear();
		for (size_t i = 0; i < clouds.size(); ++i)
		{
			if (clouds[i]->isA(CV_TYPES::POINT_CLOUD)) // as filterChildren only test 'isKindOf'
			{
				QString name = getEntityName(clouds[i]);
				QVariant uniqueID(clouds[i]->getUniqueID());
				template1CloudComboBox->addItem(name, uniqueID);
				template2CloudComboBox->addItem(name, uniqueID);
				evaluationCloudComboBox->addItem(name, uniqueID);
				++cloudCount;
			}
		}

		//if 3 clouds are loaded, then there's chances that the first one is the global cloud!
		template1CloudComboBox->setCurrentIndex(cloudCount > 0 ? (cloudCount > 2 ? 1 : 0) : -1);
		template2CloudComboBox->setCurrentIndex(cloudCount > 1 ? (cloudCount > 2 ? 2 : 1) : -1);

		if (cloudCount < 1 && m_app)
			m_app->dispToConsole(tr("You need at least 1 loaded clouds to perform alignment"), ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
	}
}

bool TemplateAlignmentDialog::validParameters() const
{
	int c1 = template1CloudComboBox->currentIndex();
	if (template1checkBox->isChecked())
	{
		if (c1 < 0)
		{
			return false;
		}
	}
	int c2 = template2CloudComboBox->currentIndex();
	if (template2checkBox->isChecked())
	{
		if (c2 < 0)
		{
			return false;
		}
	}


	if (template1checkBox->isChecked() && template2checkBox->isChecked())
	{
		if (c1 == c2)
			return false;
	}

	return true;
}

int TemplateAlignmentDialog::getMaxThreadCount() const
{
	return maxThreadCountSpinBox->value();
}

float TemplateAlignmentDialog::getNormalRadius() const
{
	return static_cast<float>(normalRadiusSpinBox->value());
}

float TemplateAlignmentDialog::getFeatureRadius() const
{
	return static_cast<float>(featureRadiusSpinBox->value());
}

float TemplateAlignmentDialog::getMinSampleDistance() const
{
	return static_cast<float>(minSampleDistanceSpinBox->value());
}

float TemplateAlignmentDialog::getMaxCorrespondenceDistance() const
{
	return static_cast<float>(maxCorrespondenceDistanceSpinBox->value() * maxCorrespondenceDistanceSpinBox->value());
}

int TemplateAlignmentDialog::getMaxIterations() const
{
	return maxIterationsSpinBox->value();
}

float TemplateAlignmentDialog::getVoxelGridLeafSize() const
{
	if (useVoxelGridCheckBox->isChecked())
	{
		return static_cast<float>(leafSizeSpinBox->value());
	}
	else
	{
		return -1.0f;
	}
}

bool TemplateAlignmentDialog::getScales(std::vector<float>& scales) const
{
	scales.clear();

	try
	{
		if (scalesRampRadioButton->isChecked())
		{
			double maxScale = maxScaleDoubleSpinBox->value();
			double step = stepScaleDoubleSpinBox->value();
			double minScale	= minScaleDoubleSpinBox->value();
			if (maxScale < minScale || maxScale < 0 || step < 1.0e-6)
				return false;
			unsigned stepCount = static_cast<unsigned>( floor((maxScale-minScale)/step + 1.0e-6) ) + 1;
			scales.resize(stepCount);
			for (unsigned i=0; i<stepCount; ++i)
				scales[i] = static_cast<float>(maxScale - i*step);
		}
		else if (scalesListRadioButton->isChecked())
		{
			QStringList scaleList = scalesListLineEdit->text().split(' ', QString::SkipEmptyParts);
		
			int listSize = scaleList.size();
			scales.resize(listSize);
			for (int i=0; i<listSize; ++i)
			{
				bool ok = false;
				float f;
				f = scaleList[i].toFloat(&ok);
				if (!ok)
					return false;
				scales[i] = f;
			}
		}
		else
		{
			return false;
		}
	}
	catch (const std::bad_alloc&)
	{
		return false;
	}

	return true;
}


void TemplateAlignmentDialog::onCloudChanged(int dummy)
{
	buttonBox->button(QDialogButtonBox::Ok)->setEnabled(validParameters());
}

ccPointCloud* TemplateAlignmentDialog::getTemplate1Cloud()
{
	//return the cloud currently selected in the combox box
	if (template1checkBox->isChecked())
	{
		return getCloudFromCombo(template1CloudComboBox, m_app->dbRootObject());
	}
	else
	{
		return nullptr;
	}
}

ccPointCloud* TemplateAlignmentDialog::getTemplate2Cloud()
{
	//return the cloud currently selected in the combox box
	if (template2checkBox->isChecked())
	{
		return getCloudFromCombo(template2CloudComboBox, m_app->dbRootObject());
	}
	else
	{
		return nullptr;
	}
}

ccPointCloud* TemplateAlignmentDialog::getEvaluationCloud()
{
	//return the cloud currently selected in the combox box
	return getCloudFromCombo(evaluationCloudComboBox, m_app->dbRootObject());
}

void TemplateAlignmentDialog::loadParamsFromPersistentSettings()
{
	QSettings settings("templateAlignment");
	settings.beginGroup("Align");

	//read out parameters
	//double minScale = settings.value("MinScale",minScaleDoubleSpinBox->value()).toDouble();
	//double step = settings.value("Step",stepScaleDoubleSpinBox->value()).toDouble();
	//double maxScale = settings.value("MaxScale",maxScaleDoubleSpinBox->value()).toDouble();
	//QString scalesList = settings.value("ScalesList",scalesListLineEdit->text()).toString();
	//bool scalesRampEnabled = settings.value("ScalesRampEnabled",scalesRampRadioButton->isChecked()).toBool();

	//unsigned maxPoints = settings.value("MaxPoints",maxPointsSpinBox->value()).toUInt();
	//int classifParam = settings.value("ClassifParam",paramComboBox->currentIndex()).toInt();
	//int maxThreadCount = settings.value("MaxThreadCount", maxThreadCountSpinBox->maximum()).toInt();

	////apply parameters

	//minScaleDoubleSpinBox->setValue(minScale);
	//stepScaleDoubleSpinBox->setValue(step);
	//maxScaleDoubleSpinBox->setValue(maxScale);
	//scalesListLineEdit->setText(scalesList);
	//if (scalesRampEnabled)
	//	scalesRampRadioButton->setChecked(true);
	//else
	//	scalesListRadioButton->setChecked(true);

	//maxPointsSpinBox->setValue(maxPoints);
	//paramComboBox->setCurrentIndex(classifParam);
	//maxThreadCountSpinBox->setValue(maxThreadCount);
}

void TemplateAlignmentDialog::saveParamsToPersistentSettings()
{
	QSettings settings("templateAlignment");
	settings.beginGroup("Align");

	//save parameters
	//settings.setValue("MinScale", minScaleDoubleSpinBox->value());
	//settings.setValue("Step", stepScaleDoubleSpinBox->value());
	//settings.setValue("MaxScale", maxScaleDoubleSpinBox->value());
	//settings.setValue("ScalesList", scalesListLineEdit->text());
	//settings.setValue("ScalesRampEnabled", scalesRampRadioButton->isChecked());

	//settings.setValue("MaxPoints", maxPointsSpinBox->value());
	//settings.setValue("ClassifParam", paramComboBox->currentIndex());
	//settings.setValue("MaxThreadCount", maxThreadCountSpinBox->value());
}

QString TemplateAlignmentDialog::getEntityName(ccHObject* obj)
{
	if (!obj)
	{
		assert(false);
		return QString();
	}

	QString name = obj->getName();
	if (name.isEmpty())
		name = tr("unnamed");
	name += QString(" [ID %1]").arg(obj->getUniqueID());

	return name;
}

ccPointCloud* TemplateAlignmentDialog::getCloudFromCombo(QComboBox* comboBox, ccHObject* dbRoot)
{
	assert(comboBox && dbRoot);
	if (!comboBox || !dbRoot)
	{
		assert(false);
		return nullptr;
	}

	//return the cloud currently selected in the combox box
	int index = comboBox->currentIndex();
	if (index < 0)
	{
		assert(false);
		return nullptr;
	}
	unsigned uniqueID = comboBox->itemData(index).toUInt();
	ccHObject* item = dbRoot->find(uniqueID);
	if (!item || !item->isA(CV_TYPES::POINT_CLOUD))
	{
		assert(false);
		return nullptr;
	}
	return static_cast<ccPointCloud*>(item);
}
