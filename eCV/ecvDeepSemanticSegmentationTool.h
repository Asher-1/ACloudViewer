// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include <ecvOverlayDialog.h>

// GUI
#include <ui_deepSemanticSegmentationDlg.h>

// CV_CORE_LIB
#include <ClassMap.h>

// ECV_DB_LIB
#include <ecvHObject.h>

// ECV_PYTHON_LIB
#ifdef USE_PYTHON_MODULE
#include <recognition/PythonInterface.h>

class ecvDeepSemanticSegmentationTool : public ccOverlayDialog,
                                        public Ui::DeepSemanticSegmentationDlg {
    Q_OBJECT

public:
    ecvDeepSemanticSegmentationTool(QWidget* parent = nullptr);
    virtual ~ecvDeepSemanticSegmentationTool();

    //! Adds an entity to the 'selected' entities set
    /** Only the 'selected' entities are moved.
            \return success, if the entity is eligible for graphical
    transformation
    **/
    bool addEntity(ccHObject* entity);

    //! Returns the number of valid entities (see addEntity)
    unsigned getNumberOfValidEntities() const;

    // inherited from ccOverlayDialog
    virtual bool start() override;
    virtual void stop(bool state) override;
    virtual bool linkWith(QWidget* win) override;

    void clear();

    void getSegmentations(ccHObject::Container& result);

public slots:
    void apply();

    void detect();

    void cancel();

protected:
    bool m_show_progress;

    ccHObject m_entity;
    ccHObject m_selectedEntity;

    std::vector<std::vector<size_t>> m_clusters;

    std::vector<ClassMap::ClusterMap> m_clusters_map;

private:
    void doCompute();
    int checkSelected();
    int startDetection();
    void selectAllClasses();
    int performSegmentation();
    void updateSelectedEntity();
    void refreshSelectedClouds();
    void exportClustersToSF();
    void exportClustersToEntities(ccHObject::Container& result);
    void getSelectedFilterClasses(std::vector<size_t>& filteredClasses);
};

#endif
