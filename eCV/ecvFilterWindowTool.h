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

#ifndef ECV_FILTER_WINDOW_TOOL_HEADER
#define ECV_FILTER_WINDOW_TOOL_HEADER

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericPointCloud.h>
#include <ecvWidgetsInterface.h>

// LOCAL
#include <ecvOverlayDialog.h>
#include <ui_graphicalFilteringWindowDlg.h>

// SYSTEM
#include <vector>

class ccGenericMesh;
class ecvProgressDialog;
class ccHObject;
class ccPolyline;
class ccBBox;

//! Dialog for managing a clipping box
class ecvFilterWindowTool : public ccOverlayDialog, public Ui::GraphicalFilteringWindowDlg
{
	Q_OBJECT
public:

	//! Default constructor
	explicit ecvFilterWindowTool(QMainWindow* parent);
	//! Default destructor
	virtual ~ecvFilterWindowTool();

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
	ccHObject* getSlice(ccHObject* obj, bool silent = true);

	//! Flags the points of a given cloud depending on whether they are inside or outside of this clipping box
	/** \param cloud point cloud
		\param visTable visibility flags
		\param shrink Whether the box is shrinking (faster) or not
	**/
	void flagPointsInside(ccGenericPointCloud* cloud,
		ccGenericPointCloud::VisibilityTableType* visTable,
		bool shrink = false) const;

	//! Extract slices and optionally contours from various clouds and/or clouds
	/** \param clouds input clouds (may be empty if meshes are defined)
		\param meshes input meshes (may be empty if clouds are defined)
		\param clipBox clipping box
		\param singleContourMode if true, a single cut is made (the process is not repeated) and only the contour is extracted (not the slice)
		\param processDimensions If singleContourMode is true: the dimension normal to the slice should be true (and the others false). Otherwise: the dimensions along which to repeat the cuting process should be true.
		\param outputSlices output slices (if successful)
		\param extractContours whether to extract contours or not
		\param maxEdgeLength max contour edge length (the smaller, the tighter the contour will be)
		\param outputContours output contours (if successful)
		\param gap optional gap between each slice
		\param multiPass multi-pass contour extraction
		\param splitContours whether to split the contour when the segment can't be smaller than the specified 'maxEdgeLength'
		\param projectOnBestFitPlane to project the points on the slice best fitting plane (otherwise the plane normal to the 
		\param visualDebugMode displays a 'debugging' window during the contour extraction process
		\param generateRandomColors randomly colors the extracted slices
		\param progressDialog optional progress dialog
	**/
	bool extractSlicesAndContours
		(
		const std::vector<ccGenericPointCloud*>& clouds,
		const std::vector<ccGenericMesh*>& meshes,
		ccBBox& clipBox,
		bool singleContourMode,
		bool processDimensions[3],
		std::vector<ccHObject*>& outputSlices,
		bool extractContours,
		PointCoordinateType maxEdgeLength,
		std::vector<ccPolyline*>& outputContours,
		PointCoordinateType gap = 0,
		bool multiPass = false,
		bool splitContours = false,
		bool projectOnBestFitPlane = false,
		bool visualDebugMode = false,
		bool generateRandomColors = false,
		ecvProgressDialog* progressDialog = 0);

protected slots:
	void restoreLastBox();
	void cancel();
	void reset();
	void doSetClipMode();
	void doSetPolylineMode();
	void extractContour();
	void removeLastContour();
	void exportSlice();
	bool updateBBox();
	void exportMultSlices();

	void doSetContourSelection();
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

	//! Extracts slices and/or contours
	void extractSlicesAndContours(bool extractSlices, bool extractContours, bool singleContourMode);

	bool extractFlatContour(
		ccPointCloud* sliceCloud,
		bool allowMultiPass,
		PointCoordinateType maxEdgeLength,
		std::vector<ccPolyline*>& parts,
		bool allowSplitting = true,
		const PointCoordinateType* preferredNormDim = 0,
		bool enableVisualDebugMode = false);
};

#endif // ECV_FILTER_WINDOW_TOOL_HEADER
