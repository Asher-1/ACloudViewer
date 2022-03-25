#pragma once

//##########################################################################
//#                                                                        #
//#                   CLOUDCOMPARE PLUGIN: qAnimation                      #
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
//#             COPYRIGHT: Ryan Wicks, 2G Robotics Inc., 2015              #
//#                                                                        #
//##########################################################################

// ECV_DB_LIB
#include <ecvViewportParameters.h>

// Qt
#include <QDialog>

// System
#include <QProgressDialog>
#include <vector>

#include "ui_animationDlg.h"

class ccMesh;
class ccPolyline;
class cc2DViewportObject;
class QListWidgetItem;
#ifdef QFFMPEG_SUPPORT
class QVideoEncoder;
#endif

//! Dialog for qAnimation plugin
class qAnimationDlg : public QDialog, public Ui::AnimationDialog {
    Q_OBJECT

public:
    //! Default constructor
    qAnimationDlg(QWidget* view3d, QWidget* parent = nullptr);

    //! Destrcuctor
    virtual ~qAnimationDlg();

    //! Initialize the dialog with a set of viewports
    bool init(const std::vector<cc2DViewportObject*>& viewports,
              const std::vector<ccMesh*>& meshes);

    ccPolyline* getTrajectory();
    bool exportTrajectoryOnExit();
    bool updateTextures() const;

protected:
    void onFPSChanged(int);
    void onTotalTimeChanged(double);
    void onStepTimeChanged(double);
    void onLoopToggled(bool);
    void onCurrentStepChanged(int);
    void onBrowseButtonClicked();
    void onBrowseTexturesButtonClicked();
    void onAutoStepsDurationToggled(bool);
    void onSmoothTrajectoryToggled(bool);
    void onSmoothRatioChanged(double);

    void preview();
    void renderAnimation() { render(false); }
    void renderFrames() { render(true); }
    void onAccept();
    void onReject();

    void onItemChanged(QListWidgetItem*);

protected:  // methods
    int getCurrentStepIndex();
    size_t countEnabledSteps() const;

    bool smoothModeEnabled() const;

    int countFrames(size_t startIndex = 0);

    void applyViewport(const ecvViewportParameters& viewportParameters);

    double computeTotalTime();

    void updateCurrentStepDuration();
    void updateTotalDuration();
    bool updateCameraTrajectory();
    bool updateSmoothCameraTrajectory();

    bool getNextSegment(size_t& vp1, size_t& vp2) const;

    void render(bool asSeparateFrames);

    bool smoothTrajectory(double ratio, unsigned iterationCount);

    //! Simple step (viewport + time)
    struct Step {
        cc2DViewportObject* viewport = nullptr;
        ecvViewportParameters viewportParams;
        int indexInOriginalTrajectory = -1;
        CCVector3d cameraCenter;
        double duration_sec = 0.0;
        double length = 0.0;
        int indexInSmoothTrajectory = -1;
    };

    typedef std::vector<Step> Trajectory;
    typedef std::vector<ccMesh*> MeshList;

    bool getCompressedTrajectory(Trajectory& compressedTrajectory) const;

    void updateSmoothTrajectoryDurations();

protected:  // members
    //! Animation
    Trajectory m_videoSteps;
    //! Smoothed animation
    Trajectory m_smoothVideoSteps;
    MeshList m_mesh_list;

    //! Associated 3D view
    QWidget* m_view3d;
    void textureAnimationPreview(const QStringList& texture_files,
                                 QProgressDialog& progressDialog);
    bool textureAnimationRender(const QStringList& texture_files,
                                 QProgressDialog& progressDialog,
                                 bool asSeparateFrames
#ifdef QFFMPEG_SUPPORT
                                 ,
                                 QScopedPointer<QVideoEncoder>& encoder
#endif
    );
};
