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

#ifndef Q_PCL_CORRESPONDENCEMATCHING_DIALOG_HEADER
#define Q_PCL_CORRESPONDENCEMATCHING_DIALOG_HEADER

#include <ui_CorrespondenceMatchingDialog.h>

//Qt
#include <QDialog>

class ecvMainAppInterface;
class ccPointCloud;
class ccHObject;

//! CANUPO plugin's training dialog
class CorrespondenceMatchingDialog : public QDialog, public Ui::CorrespondenceMatchingDialog
{
	Q_OBJECT

public:

	//! Default constructor
	CorrespondenceMatchingDialog(ecvMainAppInterface* app);

	//! Get model #1 point cloud
	ccPointCloud* getModel1Cloud();
	//! Get model #2 point cloud
	ccPointCloud* getModel2Cloud();

	ccPointCloud* getModelCloudByIndex(int index);

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

	float getVoxelGridLeafSize() const;

	//! Returns the Model Search Radius
	float getModelSearchRadius() const;
	//! Returns the Scene Search Radius
	float getSceneSearchRadius() const;
	//! Returns the Shot Descriptor Radius
	float getShotDescriptorRadius() const;
	//! Returns the normal KSearch
	float getNormalKSearch() const;

	bool getVerificationFlag() const;

	bool isGCActivated() const;

	float getGcConsensusSetResolution() const;
	float getGcMinClusterSize() const;

	float getHoughLRFRadius() const;
	float getHoughBinSize() const;
	float getHoughThreshold() const;

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

#endif // Q_PCL_CORRESPONDENCEMATCHING_DIALOG_HEADER
