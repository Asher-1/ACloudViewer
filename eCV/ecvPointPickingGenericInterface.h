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
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef ECV_POINT_PICKING_GENERIC_INTERFACE_HEADER
#define ECV_POINT_PICKING_GENERIC_INTERFACE_HEADER

//Local
#include "ecvOverlayDialog.h"
#include "ecvCommon.h"
#include "ecvPickingListener.h"

//cloudViewer
#include <CVGeom.h>

//system
#include <vector>

class ccPointCloud;
class ccHObject;
class ccPickingHub;

/** Generic interface for any dialog/graphical interactor that relies on point picking.
**/
class ccPointPickingGenericInterface : public ccOverlayDialog, public ccPickingListener
{
	Q_OBJECT

public:

	//! Default constructor
	explicit ccPointPickingGenericInterface(ccPickingHub* pickingHub, QWidget* parent = nullptr);
	//! Destructor
	~ccPointPickingGenericInterface() override = default;

	//inherited from ccOverlayDialog
	bool linkWith(QWidget* win) override;
	bool start() override;
	void stop(bool state) override;

	//! Inherited from ccPickingListener
	void onItemPicked(const PickedItem& pi) override;

protected:

	//! Generic method to process picked points
	/** \param cloud picked point cloud
		\param pointIndex point index in cloud
		\param x picked pixel X position
		\param y picked pixel Y position
	**/
	virtual void processPickedPoint(ccPointCloud* cloud, unsigned pointIndex, int x, int y) = 0;

	//! Picking hub
	ccPickingHub* m_pickingHub;
};

#endif // ECV_POINT_PICKING_GENERIC_INTERFACE_HEADER
