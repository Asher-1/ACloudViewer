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

#ifndef ECV_RENDER_SURFACE_TOOL_HEADER
#define ECV_RENDER_SURFACE_TOOL_HEADER

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericPointCloud.h>
#include <ecvWidgetsInterface.h>

// LOCAL
#include <ecvOverlayDialog.h>
#include <ui_graphicalRenderSurfaceWindowDlg.h>

// SYSTEM
#include <vector>

class ccGenericMesh;
class ecvProgressDialog;
class ccHObject;
class ccPolyline;
class ccBBox;

//! Dialog for managing a clipping box
class ecvRenderSurfaceTool : public ccOverlayDialog, public Ui::GraphicalRenderSurfaceWindowDlg
{
	Q_OBJECT
public:

	//! Default constructor
	explicit ecvRenderSurfaceTool(QMainWindow* parent);
	//! Default destructor
	virtual ~ecvRenderSurfaceTool();

	//inherited from ccOverlayDialog
	virtual bool linkWith(QWidget* win) override;
	virtual bool start() override;
	virtual void stop(bool accepted) override;

	bool linkWidgets();

	//! Returns the active 'to be segmented' set
	ccHObject* entities() { return m_filtered; }
	//! Returns the active 'to be segmented' set (const version)
	const ccHObject* entities() const { return m_filtered; }

	//! Adds an entity
	/** \return success, if the entity is eligible for clipping
	**/
	bool addAssociatedEntity(ccHObject* anObject);

	//! Returns the current number of associated entities
	unsigned getNumberOfAssociatedEntity() const;

	ccBBox getOwnBB() const;

	void releaseAssociatedEntities();

	inline ccHObject* getOutput() { return &m_entityContainer; }

protected slots:
	void restoreLastBox();
	void cancel();
	void reset();
	void exportSurface();
	bool updateBBox();
	void exportPointCloud();
	void doSetClippingSelection();

	//! To capture overridden shortcuts (pause button, etc.)
	void onShortcutTriggered(int);

protected:

	//! Associated entities container
	ccHObject m_entityContainer;
	ccHObject* m_filtered;

	//! Segmentation polyline
	ccPolyline* m_segmentationPoly;

	//! Clipping box
	ccBBox m_box;

	//! Show box
	bool m_showBox;

	bool m_deleteHiddenParts;

	bool m_somethingHasChanged;

	DBLib::VTK_WIDGETS_TYPE m_currentMode;

	//! Parent window
	QMainWindow* m_win;
};

#endif // ECV_RENDER_SURFACE_TOOL_HEADER
