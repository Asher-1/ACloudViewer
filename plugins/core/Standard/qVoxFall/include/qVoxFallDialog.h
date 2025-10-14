// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_VOXFALL_DIALOG_HEADER
#define Q_VOXFALL_DIALOG_HEADER

#include <ui_qVoxFallDialog.h>

// Qt
#include <QSettings>

class ecvMainAppInterface;
class ccMesh;

//! VOXFALL plugin's main dialog
class qVoxFallDialog : public QDialog, public Ui::VoxFallDialog {
    Q_OBJECT

public:
    //! Default constructor
    qVoxFallDialog(ccMesh* mesh1, ccMesh* mesh2, ecvMainAppInterface* app);

    //! Returns mesh #1
    ccMesh* getMesh1() const { return m_mesh1; }
    //! Returns mesh #2
    ccMesh* getMesh2() const { return m_mesh2; }

    //! Returns voxel size
    double getVoxelSize() const;
    //! Returns slope azimuth
    double getAzimuth() const;
    //! Returns whether the blocks will be exported as meshes
    bool getExportMeshesActivation() const;
    //! Labels the blocks as loss or gain clusters
    bool getLossGainActivation() const;

    //! Returns the max number of threads to use
    int getMaxThreadCount() const;

    void loadParamsFromPersistentSettings();
    void loadParamsFrom(const QSettings& settings);
    void saveParamsToPersistentSettings();
    void saveParamsTo(QSettings& settings);

protected:
    void swapMeshes();
    void setMesh1Visibility(bool);
    void setMesh2Visibility(bool);

protected:  // methods
    //! Sets meshes
    void setMeshes(ccMesh* mesh1, ccMesh* mesh2);

protected:  // members
    ecvMainAppInterface* m_app;

    ccMesh* m_mesh1;
    ccMesh* m_mesh2;
};

#endif  // Q_VOXFALL_DIALOG_HEADER
