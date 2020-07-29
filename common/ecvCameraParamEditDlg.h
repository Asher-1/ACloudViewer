//##########################################################################
//#                                                                        #
//#                              CLOUDCOMPARE                              #
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

#ifndef ECV_CAMERA_PARAM_EDIT_DLG_HEADER
#define ECV_CAMERA_PARAM_EDIT_DLG_HEADER

// Local
#include "ecvOverlayDialog.h"
#include "ecvPickingListener.h"

// ECV_DB_LIB
#include <ecvGLMatrix.h>

//system
#include <map>

class MainWindow;
class QMdiSubWindow;
class ccHObject;
class ccPickingHub;
class CameraDialogInternal;
class ecvGenericCameraTool;

//! Dialog to interactively edit the camera pose parameters
class ecvCameraParamEditDlg : public ccOverlayDialog, public ccPickingListener
{
	Q_OBJECT

public:

	//! Default constructor
	explicit ecvCameraParamEditDlg(MainWindow* app, QWidget* parent, ccPickingHub* pickingHub);

	//! Destructor
	~ecvCameraParamEditDlg() override;

	//inherited from ccOverlayDialog
	bool start() override;
	bool linkWith(QWidget* win) override;

	//inherited from ccPickingListener
	void onItemPicked(const PickedItem& pi) override;

	bool setCameraTool(ecvGenericCameraTool* tool);

	void SetCameraGroupsEnabled(bool enabled);

	/**
	 * Open the CustomViewpointDialog to configure customViewpoints
	 */
	static bool ConfigureCustomViewpoints(QWidget* parentWidget);

	/**
	 * Add the current viewpoint to the custom viewpoints
	 */
	static bool AddCurrentViewpointToCustomViewpoints();

	/**
	 * Change camera positing to an indexed custom viewpoints
	 */
	static bool ApplyCustomViewpoint(int CustomViewpointIndex);

	/**
	 * Delete an indexed custom viewpoint
	 */
	static bool DeleteCustomViewpoint(int CustomViewpointIndex);

	/**
	 * Set an indexed custom viewpoint to the current viewpoint
	 */
	static bool SetToCurrentViewpoint(int CustomViewpointIndex);

	/**
	 * Return the list of custom viewpoints tooltups
	 */
	static QStringList CustomViewpointToolTips();

	/**
	 * Return the list of custom viewpoint configurations
	 */
	static QStringList CustomViewpointConfigurations();

public slots:

	//! Links this dialog with a given sub-window
	void linkWith(QMdiSubWindow* qWin);

	//! Updates dialog values with pivot point
	void updatePivotPoint(const CCVector3d& P);
	//! Updates current view mode
	void updateViewMode();
	void pivotChanged();

	void rotationFactorChanged(double);
	void zfactorSliderMoved(int val);

	void pickPointAsPivot(bool);
	void processPickedItem(ccHObject*, unsigned, int, int, const CCVector3&);

private slots:
	// Description:
	// Choose a file and load/save camera properties.
	void saveCameraConfiguration();
	void loadCameraConfiguration();

	// Description:
	// Assign/restore the current camera properties to
	// a custom view button.
	void ConfigureCustomViewpoints();
	void ApplyCustomViewpoint();
	void addCurrentViewpointToCustomViewpoints();
	void updateCustomViewpointButtons();

	void resetViewDirectionPosX();
	void resetViewDirectionNegX();
	void resetViewDirectionPosY();
	void resetViewDirectionNegY();
	void resetViewDirectionPosZ();
	void resetViewDirectionNegZ();

	void resetViewDirection(
		double look_x, double look_y, double look_z, double up_x, double up_y, double up_z);

	void applyCameraRoll();
	void applyCameraElevation();
	void applyCameraAzimuth();
	void applyCameraZoomIn();
	void applyCameraZoomOut();

	void autoPickRotationCenterWithCamera();

	//! Reflects any dialog parameter change
	void reflectParamChange();

protected:

	//! Inits dialog values with specified window
	void initWith(QWidget* win);

	void updateUi();

	//! Type of the pushed matrices map structure
	using PushedMatricesMapType = std::map<QWidget*, ccGLMatrixd>;
	//! Type of an element of the pushed matrices map structure
	using PushedMatricesMapElement = std::pair<QWidget*, ccGLMatrixd>;

	//! Pushed camera matrices (per window)
	PushedMatricesMapType pushedMatrices;

	//! Picking hub
	ccPickingHub* m_pickingHub;

protected slots:
	void updateCamera();
	void cameraChanged();
	
private:
	MainWindow* m_app;
	ecvGenericCameraTool* m_tool;
	CameraDialogInternal* Internal;

	enum CameraAdjustmentType
	{
		Roll = 0,
		Elevation,
		Azimuth,
		Zoom
	};
	void adjustCamera(CameraAdjustmentType enType, double value);
};

#endif // ECV_CAMERA_PARAM_EDIT_DLG_HEADER
