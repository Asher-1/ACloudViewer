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

#ifndef ECV_ANNOTATION_TOOL_HEADER
#define ECV_ANNOTATION_TOOL_HEADER

// LOCAL
#include <ecvOverlayDialog.h>
#include <ui_annotationsDlg.h>

// ECV_DB_LIB
#include <ecvBBox.h>
#include <ecvDisplayTools.h>

// SYSTEM
#include <vector>

class ccGenericPointCloud;
class ecvProgressDialog;
class ccHObject;
class ccBBox;
class ecvGenericAnnotationTool;

//! Dialog for managing a clipping box
class ecvAnnotationsTool : public ccOverlayDialog, public Ui::AnnotationsDlg
{
	Q_OBJECT

public:

	//! Default constructor
	explicit ecvAnnotationsTool(QWidget* parent);
	//! Default destructor
	virtual ~ecvAnnotationsTool();

	//inherited from ccOverlayDialog
	virtual bool linkWith(QWidget* win) override;
	virtual bool start() override;
	virtual void stop(bool state) override;

	bool setAnnotationsTool(ecvGenericAnnotationTool* annotationTool);

	//! Adds an entity
	/** \return success, if the entity is eligible for clipping
	**/
	bool addAssociatedEntity(ccHObject* anObject);

	void releaseAssociatedEntities();

	//! Returns the current number of associated entities
	unsigned getNumberOfAssociatedEntity() const;

public slots:
	void toggleInteractors(bool state);
	void toggleEditMode(bool state);

protected slots:
	//! To capture overridden shortcuts (pause button, etc.)
	void onShortcutTriggered(int);

	void importClassesFromFile();
	void toggleBox(bool);
	void toggleOrigin(bool dummy);

	void onNewMode();
	void onUnionMode();
	void onTrimMode();
	void onIntersectMode();
	void onLabelSelected();
	void onLabelChanged(int index);
	void onItemPicked(bool isPicked);

	void saveAnnotations();
	void reset();
	void closeDialog();
	void exportAnnotationToSF();
	void updateLabelsCombox(const std::vector<std::string>& labels);

	inline void shiftXMinus() { shiftBox(0, true);  }
	inline void shiftXPlus()  { shiftBox(0, false); }
	inline void shiftYMinus() { shiftBox(1, true);  }
	inline void shiftYPlus()  { shiftBox(1, false); }
	inline void shiftZMinus() { shiftBox(2, true);  }
	inline void shiftZPlus()  { shiftBox(2, false); }

	void setFrontView();
	void setBottomView();
	void setTopView();
	void setBackView();
	void setLeftView();
	void setRightView();

protected:

	//! Shift box
	void shiftBox(unsigned char dim, bool minus);

	//! Sets predefined view
	void setView(CC_VIEW_ORIENTATION orientation);

	//! Clipping box
	ecvGenericAnnotationTool* m_annotationTool;

	ccBBox getSelectedEntityBbox();

	//! bounding box
	ccBBox m_box;

	//! Associated entities container
	ccHObject m_entityContainer;

	//! for solving dummy event trigger
	bool m_disabledCombEvent;

	bool m_editMode;

};

#endif // ECV_ANNOTATION_TOOL_HEADER
