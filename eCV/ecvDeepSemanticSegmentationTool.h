//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER PLUGIN: qPCL                         #
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
//#                         COPYRIGHT: DAHAI LU                         #
//#                                                                        #
//##########################################################################
//
#ifndef ECV_DEEP_SEMANTICE_SEGMENTATION_HEADER
#define ECV_DEEP_SEMANTICE_SEGMENTATION_HEADER

//Local
#include <ecvOverlayDialog.h>

// GUI
#include <ui_deepSemanticSegmentationDlg.h>

// ECV_DB_LIB
#include <ecvHObject.h>

// ECV_PYTHON_LIB
#ifdef ECV_PYTHON_USE_AS_DLL
#include <Recognition/PythonInterface.h>

class ecvDeepSemanticSegmentationTool : public ccOverlayDialog, public Ui::DeepSemanticSegmentationDlg
{
	Q_OBJECT

public:
	ecvDeepSemanticSegmentationTool(QWidget* parent = nullptr);
	virtual ~ecvDeepSemanticSegmentationTool();

	//! Adds an entity to the 'selected' entities set
	/** Only the 'selected' entities are moved.
		\return success, if the entity is eligible for graphical transformation
	**/
	bool addEntity(ccHObject* entity);

	//! Returns the number of valid entities (see addEntity)
	unsigned getNumberOfValidEntities() const;

	//inherited from ccOverlayDialog
	virtual bool start() override;
	virtual void stop(bool state) override;
	virtual bool linkWith(QWidget* win) override;

	void clear();

	void getSegmentations(ccHObject::Container &result);

public slots:
	void apply();

	void detect();

	void cancel();

protected:
	bool m_show_progress;

	ccHObject m_entity;
	ccHObject m_selectedEntity;

	std::vector< std::vector<size_t> > m_clusters;

#ifdef ECV_PYTHON_USE_AS_DLL
	std::vector<ClassMap::ClusterMap> m_clusters_map;
#endif

private:
	void doCompute();
	int checkSelected();
	int startDetection();
	int performSegmentation();
	void updateSelectedEntity();
	void refreshSelectedClouds();
	void exportClustersToSF();
	void exportClustersToEntities(ccHObject::Container &result);

	void getSelectedFilterClasses(std::vector<size_t> &filteredClasses);

	void selectAllClasses();
};

#endif

#endif // ECV_DEEP_SEMANTICE_SEGMENTATION_HEADER
