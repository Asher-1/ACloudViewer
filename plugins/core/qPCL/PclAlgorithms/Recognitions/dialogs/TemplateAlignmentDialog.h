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

#ifndef Q_PCL_TEMPLATEALIGNMENT_DIALOG_HEADER
#define Q_PCL_TEMPLATEALIGNMENT_DIALOG_HEADER

#include <ui_TemplateAlignmentDialog.h>

//Qt
#include <QDialog>

class ecvMainAppInterface;
class ccPointCloud;
class ccHObject;

//! CANUPO plugin's training dialog
class TemplateAlignmentDialog : public QDialog, public Ui::TemplateAlignmentDialog
{
	Q_OBJECT

public:

	//! Default constructor
	TemplateAlignmentDialog(ecvMainAppInterface* app);

	//! Get template #1 point cloud
	ccPointCloud* getTemplate1Cloud();
	//! Get template #2 point cloud
	ccPointCloud* getTemplate2Cloud();
	//! Get evaluation point cloud
	ccPointCloud* getEvaluationCloud();

	//! Loads parameters from persistent settings
	void loadParamsFromPersistentSettings();
	//! Saves parameters to persistent settings
	void saveParamsToPersistentSettings();

	//! Returns input scales
	bool getScales(std::vector<float>& scales) const;
	//! Returns the max number of threads to use
	int getMaxThreadCount() const;

	//! Returns the Maximum Iterations
	int getMaxIterations() const;

	float getVoxelGridLeafSize() const;

	//! Returns the Normal Radius
	float getNormalRadius() const;
	//! Returns the Feature Radius
	float getFeatureRadius() const;
	//! Returns the Minimum Sample Distance
	float getMinSampleDistance() const;
	//! Returns the Maximum Correspondence Distance
	float getMaxCorrespondenceDistance() const;

	void refreshCloudComboBox();

protected slots:
	void onCloudChanged(int);

protected:

	//! Gives access to the application (data-base, UI, etc.)
	ecvMainAppInterface* m_app;

	//Returns whether the current parameters are valid or not
	bool validParameters() const;

	QString getEntityName(ccHObject* obj);

	ccPointCloud* getCloudFromCombo(QComboBox* comboBox, ccHObject* dbRoot);

};

#endif // Q_PCL_TEMPLATEALIGNMENT_DIALOG_HEADER
