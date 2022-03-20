//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
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
//#          COPYRIGHT: EDF R&D / TELECOM ParisTech (ENST-TSI)             #
//#                                                                        #
//##########################################################################

#ifndef ECV_ANIMATION_PARAM_DLG_HEADER
#define ECV_ANIMATION_PARAM_DLG_HEADER

// Local
#include "ecvOverlayDialog.h"
#include "ecvPickingListener.h"

// ECV_DB_LIB
#include <ecvGLMatrix.h>
#include <ecvViewportParameters.h>

// system
#include <map>

class QMdiSubWindow;
class ccHObject;
class ccPickingHub;
class AnimationDialogInternal;
class MainWindow;

//! Dialog to interactively edit the camera pose parameters
class ecvAnimationParamDlg : public ccOverlayDialog, public ccPickingListener {
    Q_OBJECT

public:
    //! Default constructor
    explicit ecvAnimationParamDlg(QWidget* parent,
                                  MainWindow* app,
                                  ccPickingHub* pickingHub);

    //! Destructor
    ~ecvAnimationParamDlg() override;

    enum AxisType { AXIS_START, AXIS_END };

    // inherited from ccOverlayDialog
    bool start() override;
    bool linkWith(QWidget* win) override;

    // inherited from ccPickingListener
    void onItemPicked(const PickedItem& pi) override;

public:
    CCVector3d getRotationAxis() const;
    double getRotationAngle() const;
    bool isSavingViewport() const;
    void doCompute();
    inline MainWindow* getMainWindow() { return m_app; }

public slots:

    //! Links this dialog with a given sub-window
    void linkWith(QMdiSubWindow* qWin);

    //! Updates dialog values with axis point
    void updateRotationAxisPoint(AxisType axisType, const CCVector3d& P);

    //! Start animation
    void startAnimation();

    void enableListener(bool state);
    void updateAxisStartToolState(bool state);
    void updateAxisEndToolState(bool state);
    void processPickedItem(ccHObject*, unsigned, int, int, const CCVector3&);

private slots:

    void angleStep();
    void reset();
    void onClose() { reset(); }
    void enablePickRotationAxis(bool state);

protected:
    //! Inits dialog values with specified window
    void initWith(QWidget* win);

    //! Picking hub
    ccPickingHub* m_pickingHub;

    MainWindow* m_app;
    ecvViewportParameters viewportParamsHistory;

private:
    AnimationDialogInternal* Internal;
};

#endif  // ECV_ANIMATION_PARAM_DLG_HEADER
