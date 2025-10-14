// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_COMPARISON_DIALOG_HEADER
#define CC_COMPARISON_DIALOG_HEADER

// ECV_DB_LIB
#include <ecvOctree.h>

// Qt
#include <ui_comparisonDlg.h>

#include <QDialog>
#include <QString>

class ccHObject;
class ccPointCloud;
class ccGenericMesh;
class ccGenericPointCloud;

//! Dialog for cloud/cloud or cloud/mesh comparison setting
class ccComparisonDlg : public QDialog, public Ui::ComparisonDialog {
    Q_OBJECT

public:
    //! Comparison type
    enum CC_COMPARISON_TYPE {
        CLOUDCLOUD_DIST = 0,
        CLOUDMESH_DIST = 1,
    };

    //! Default constructor
    ccComparisonDlg(ccHObject* compEntity,
                    ccHObject* refEntity,
                    CC_COMPARISON_TYPE cpType,
                    QWidget* parent = nullptr,
                    bool noDisplay = false);

    //! Default destructor
    ~ccComparisonDlg();

    //! Should be called once after the dialog is created
    inline bool initDialog() { return computeApproxDistances(); }

    //! Returns compared entity
    ccHObject* getComparedEntity() const { return m_compEnt; }
    //! Returns compared entity
    ccHObject* getReferenceEntity() { return m_refEnt; }

public slots:
    bool computeDistances();
    void applyAndExit();
    void cancelAndExit();

protected slots:
    void showHisto();
    void locaModelChanged(int);
    void maxDistUpdated();

protected:
    bool isValid();
    bool prepareEntitiesForComparison();
    bool computeApproxDistances();
    int getBestOctreeLevel();
    int determineBestOctreeLevel(double);
    void updateDisplay(bool showSF, bool hideRef);
    void releaseOctrees();

    //! Compared entity
    ccHObject* m_compEnt;
    //! Compared entity equivalent cloud
    ccPointCloud* m_compCloud;
    //! Compared entity's octree
    ccOctree::Shared m_compOctree;
    //! Whether the compared entity octree is partial or not
    bool m_compOctreeIsPartial;
    //! Initial compared entity visibility
    bool m_compSFVisibility;

    //! Reference entity
    ccHObject* m_refEnt;
    //! Reference entity equivalent cloud (if any)
    ccGenericPointCloud* m_refCloud;
    //! Reference entity equivalent mesh (if any)
    ccGenericMesh* m_refMesh;
    //! Reference entity's octree
    ccOctree::Shared m_refOctree;
    //! Whether the reference entity octree is partial or not
    bool m_refOctreeIsPartial;
    //! Initial reference entity visibility
    bool m_refVisibility;

    //! Comparison type
    CC_COMPARISON_TYPE m_compType;

    //! last computed scalar field name
    QString m_sfName;

    //! Initial SF name enabled on the compared entity
    QString m_oldSfName;

    //! Whether a display is active (and should be refreshed) or not
    bool m_noDisplay;

    //! Best octree level (or 0 if none has been guessed already)
    int m_bestOctreeLevel;
};

#endif
