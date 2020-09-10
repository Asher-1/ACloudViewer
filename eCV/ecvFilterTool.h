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

#ifndef ECV_FILTER_TOOL_HEADER
#define ECV_FILTER_TOOL_HEADER

// LOCAL
#include <ecvOverlayDialog.h>
#include <ui_filterToolDlg.h>

// ECV_DB_LIB
#include <ecvGenericFiltersTool.h>
#include <ecvDisplayTools.h>

// SYSTEM
#include <vector>

class ccGenericMesh;
class ecvProgressDialog;
class ccHObject;
class ccClipBox;
class ccPolyline;
class ccBBox;

//! Dialog for managing a clipping box
class ecvFilterTool : public ccOverlayDialog, public Ui::FilterToolDlg
{
	Q_OBJECT

public:

	//! Default constructor
	explicit ecvFilterTool(QWidget* parent);
	//! Default destructor
	virtual ~ecvFilterTool();

	//inherited from ccOverlayDialog
	virtual bool linkWith(QWidget* win) override;
	virtual bool start() override;
	virtual void stop(bool state) override;

	void setFilter(ecvGenericFiltersTool* filter);
	ecvGenericFiltersTool* getFilter() const { return m_filter; }

	//! Adds an entity
	/** \return success, if the entity is eligible for clipping
	**/
	bool addAssociatedEntity(ccHObject* anObject);

	//! Returns the current number of associated entities
	unsigned getNumberOfAssociatedEntity() const;

	inline ccHObject::Container getOutputs() const { return m_out_entities; }

protected slots:

	void toggleInteractors(bool);
	void toggleBox(bool);

	void reset();
	void restoreOrigin();
	void closeDialog();
	void exportSlice();

	inline void shiftXMinus() { shiftBox(0, true);  }
	inline void shiftXPlus()  { shiftBox(0, false); }
	inline void shiftYMinus() { shiftBox(1, true);  }
	inline void shiftYPlus()  { shiftBox(1, false); }
	inline void shiftZMinus() { shiftBox(2, true);  }
	inline void shiftZPlus()  { shiftBox(2, false); }

	inline void setTopView()	{ setView(CC_TOP_VIEW);		}
	inline void setBottomView() { setView(CC_BOTTOM_VIEW);	}
	inline void setFrontView()	{ setView(CC_FRONT_VIEW);	}
	inline void setBackView()	{ setView(CC_BACK_VIEW);	}
	inline void setLeftView()	{ setView(CC_LEFT_VIEW);	}
	inline void setRightView()	{ setView(CC_RIGHT_VIEW);	}

	ccBBox getSelectedEntityBbox();

protected:

	//! Releases all associated entities
	/** Warning: resets the current clipping box
	**/
	void releaseAssociatedEntities();

	ccHObject* getSlice(bool silent);

	//! Shift box
	void shiftBox(unsigned char dim, bool minus);

	//! Sets predefined view
	void setView(CC_VIEW_ORIENTATION orientation);

	//! filter tool
	ecvGenericFiltersTool* m_filter;

	ecvGenericFiltersTool::FilterType m_filterType;

	//! Associated entities container
	ccHObject m_entityContainer;

	ccHObject::Container m_out_entities;

};

#endif // ECV_FILTER_TOOL_HEADER
