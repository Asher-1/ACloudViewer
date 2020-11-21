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

#ifndef ECV_DISPLAY_TOOLS_HEADER
#define ECV_DISPLAY_TOOLS_HEADER

// Local
#include "ecvGLMatrix.h"
#include "ecvHObject.h"
#include "ecvGuiParameters.h"
#include "ecvViewportParameters.h"
#include "ecvGenericDisplayTools.h"

// QT
#include <QObject>
#include <QMainWindow>
#include <QElapsedTimer>
#include <QTimer>
#include <QRect>

//System
#include <vector>
#include <list>
#include <unordered_set>

class ccPolyline;
class ccInteractor;
class ecvOrientedBBox;

class ecvGenericVisualizer;
class ecvGenericVisualizer2D;
class ecvGenericVisualizer3D;

class ECV_DB_LIB_API ecvDisplayTools : public QObject, public ecvGenericDisplayTools
{
	Q_OBJECT
public:
	/**
	\param mainWidget MainWindow widget (optional)
	**/
	static void Init(ecvDisplayTools* displayTools, QMainWindow* win, bool stereoMode = false);
	static ecvDisplayTools* TheInstance();

	static void ReleaseInstance();

	//! Destructor
	virtual ~ecvDisplayTools() override;

	//! Schedules a full redraw
	/** Any previously scheduled redraw will be cancelled.
		\warning The redraw will be cancelled if redraw/update is called before.
		\param maxDelay_ms the maximum delay for the call to redraw (in ms)
	**/
	void scheduleFullRedraw(unsigned maxDelay_ms);

	//! Cancels any scheduled redraw
	/** See ccGLWindow::scheduleFullRedraw.
	**/
	void cancelScheduledRedraw();
	
public:
	//! Picking mode
	enum PICKING_MODE {
		NO_PICKING,
		ENTITY_PICKING,
		ENTITY_RECT_PICKING,
		FAST_PICKING,
		POINT_PICKING,
		TRIANGLE_PICKING,
		POINT_OR_TRIANGLE_PICKING,
		LABEL_PICKING,
		DEFAULT_PICKING,
	};

	//! Interaction flags (mostly with the mouse)
	enum INTERACTION_FLAG {

		//no interaction
		INTERACT_NONE = 0,

		//camera interactions
		INTERACT_ROTATE = 1,
		INTERACT_PAN = 2,
		INTERACT_CTRL_PAN = 4,
		INTERACT_ZOOM_CAMERA = 8,
		INTERACT_2D_ITEMS = 16, //labels, etc.
		INTERACT_CLICKABLE_ITEMS = 32, //hot zone

		//options / modifiers
		INTERACT_TRANSFORM_ENTITIES = 64,

		//signals
		INTERACT_SIG_RB_CLICKED = 128,      //right button clicked
		INTERACT_SIG_LB_CLICKED = 256,      //left button clicked
		INTERACT_SIG_MOUSE_MOVED = 512,      //mouse moved (only if a button is clicked)
		INTERACT_SIG_BUTTON_RELEASED = 1024, //mouse button released
		INTERACT_SEND_ALL_SIGNALS = INTERACT_SIG_RB_CLICKED | INTERACT_SIG_LB_CLICKED | INTERACT_SIG_MOUSE_MOVED | INTERACT_SIG_BUTTON_RELEASED,
	};
	Q_DECLARE_FLAGS(INTERACTION_FLAGS, INTERACTION_FLAG)

	//Default interaction modes (with the mouse!)
	static INTERACTION_FLAGS PAN_ONLY();
	static INTERACTION_FLAGS TRANSFORM_CAMERA();
	static INTERACTION_FLAGS TRANSFORM_ENTITIES();

	//! Default message positions on screen
	enum MessagePosition {
		LOWER_LEFT_MESSAGE,
		UPPER_CENTER_MESSAGE,
		SCREEN_CENTER_MESSAGE,
	};

	//! Message type
	enum MessageType {
		CUSTOM_MESSAGE,
		SCREEN_SIZE_MESSAGE,
		PERSPECTIVE_STATE_MESSAGE,
		SUN_LIGHT_STATE_MESSAGE,
		CUSTOM_LIGHT_STATE_MESSAGE,
		MANUAL_TRANSFORMATION_MESSAGE,
		MANUAL_SEGMENTATION_MESSAGE,
		ROTAION_LOCK_MESSAGE,
		FULL_SCREEN_MESSAGE,
	};

	//! Pivot symbol visibility
	enum PivotVisibility {
		PIVOT_HIDE,
		PIVOT_SHOW_ON_MOVE,
		PIVOT_ALWAYS_SHOW,
	};

	//! Temporary Message to display in the lower-left corner
	struct ECV_DB_LIB_API MessageToDisplay
	{
		MessageToDisplay()
			: messageValidity_sec(0)
			, position(LOWER_LEFT_MESSAGE)
			, type(CUSTOM_MESSAGE)
		{}

		//! Message
		QString message;
		//! Message end time (sec)
		qint64 messageValidity_sec;
		//! Message position on screen
		MessagePosition position;
		//! Message type
		MessageType type;
	};

	//! List of messages to display
	std::list<MessageToDisplay> m_messagesToDisplay;

	//! Optional output metrics (from computeProjectionMatrix)
	struct ECV_DB_LIB_API ProjectionMetrics
	{
		ProjectionMetrics()
			: zNear(0.0)
			, zFar(0.0)
			, cameraToBBCenterDist(0.0)
			, bbHalfDiag(0.0)
		{}

		double zNear;
		double zFar;
		double cameraToBBCenterDist;
		double bbHalfDiag;
	};

	//! Precomputed stuff for the 'hot zone'
	struct ECV_DB_LIB_API HotZone
	{
		//display font
		QFont font;
		//text height
		int textHeight;
		//text shift
		int yTextBottomLineShift;
		//default color
		unsigned char color[3];

		//bubble-view label rect.
		QString bbv_label;
		//bubble-view label rect.
		QRect bbv_labelRect;
		//bubble-view row width
		int bbv_totalWidth;

		//fullscreen label rect.
		QString fs_label;
		//fullscreen label rect.
		QRect fs_labelRect;
		//fullscreen row width
		int fs_totalWidth;

		//point size label
		QString psi_label;
		//point size label rect.
		QRect psi_labelRect;
		//point size row width
		int psi_totalWidth;

		//line size label
		QString lsi_label;
		//line size label rect.
		QRect lsi_labelRect;
		//line size row width
		int lsi_totalWidth;

		int margin;
		int iconSize;
		QPoint topCorner;

		explicit HotZone(QWidget* win)
			: textHeight(0)
			, yTextBottomLineShift(0)
			, bbv_label("bubble-view mode")
			, fs_label("fullscreen mode")
			, psi_label("default point size")
			, lsi_label("default line width")
			, margin(10)
			, iconSize(16)
			, topCorner(0, 0)
		{
			//default color ("greenish")
			color[0] = 133;
			color[1] = 193;
			color[2] = 39;

			if (win)
			{
				font = win->font();
				int retinaScale = win->devicePixelRatio();
				font.setPointSize(12 * retinaScale);
				margin *= retinaScale;
				iconSize *= retinaScale;
				font.setBold(true);
			}

			QFontMetrics metrics(font);
			bbv_labelRect = metrics.boundingRect(bbv_label);
			fs_labelRect = metrics.boundingRect(fs_label);
			psi_labelRect = metrics.boundingRect(psi_label);
			lsi_labelRect = metrics.boundingRect(lsi_label);

			psi_totalWidth = /*margin() + */psi_labelRect.width() + margin + iconSize + margin + iconSize/* + margin*/;
			lsi_totalWidth = /*margin() + */lsi_labelRect.width() + margin + iconSize + margin + iconSize/* + margin*/;
			bbv_totalWidth = /*margin() + */bbv_labelRect.width() + margin + iconSize/* + margin*/;
			fs_totalWidth = /*margin() + */fs_labelRect.width() + margin + iconSize/* + margin*/;

			textHeight = std::max(psi_labelRect.height(), bbv_labelRect.height());
			textHeight = std::max(lsi_labelRect.height(), textHeight);
			textHeight = std::max(fs_labelRect.height(), textHeight);
			textHeight = (3 * textHeight) / 4; // --> factor: to recenter the baseline a little
			yTextBottomLineShift = (iconSize / 2) + (textHeight / 2);
		}

		QRect rect(bool clickableItemsVisible, bool bubbleViewModeEnabled, bool fullScreenEnabled) const
		{
			//total hot zone area size (without margin)
			int totalWidth = 0;
			if (clickableItemsVisible)
				totalWidth = std::max(psi_totalWidth, lsi_totalWidth);
			if (bubbleViewModeEnabled)
				totalWidth = std::max(totalWidth, bbv_totalWidth);
			if (fullScreenEnabled)
				totalWidth = std::max(totalWidth, fs_totalWidth);

			QPoint minAreaCorner(0, std::min(0, yTextBottomLineShift - textHeight));
			QPoint maxAreaCorner(totalWidth, std::max(iconSize, yTextBottomLineShift));
			int rowCount = clickableItemsVisible ? 2 : 0;
			rowCount += bubbleViewModeEnabled ? 1 : 0;
			rowCount += fullScreenEnabled ? 1 : 0;
			maxAreaCorner.setY(maxAreaCorner.y() + (iconSize + margin) * (rowCount - 1));

			QRect areaRect(minAreaCorner - QPoint(margin, margin) / 2,
						   maxAreaCorner + QPoint(margin, margin) / 2);

			return areaRect;
		}
	};

	//! Text alignment
	enum TextAlign { ALIGN_HLEFT	= 1,
					 ALIGN_HMIDDLE	= 2,
					 ALIGN_HRIGHT	= 4,
					 ALIGN_VTOP		= 8,
					 ALIGN_VMIDDLE	= 16,
					 ALIGN_VBOTTOM	= 32,
					 ALIGN_DEFAULT	= 1 | 8};

	//! Displays a string at a given 2D position
	/** This method should be called solely during 2D pass rendering.
		The coordinates are expressed relatively to the current viewport (y = 0 at the top!).
		\param text string
		\param x horizontal position of string origin
		\param y vertical position of string origin
		\param align alignment position flags
		\param bkgAlpha background transparency (0 by default)
		\param rgbColor text color (optional)
		\param font optional font (otherwise default one will be used)
	**/
	static void DisplayText(QString text,
		int x,
		int y,
		unsigned char align = ALIGN_DEFAULT,
		float bkgAlpha = 0.0f,
		const unsigned char* rgbColor = nullptr,
		const QFont* font = nullptr,
		QString id = "");

	static void DisplayText(const CC_DRAW_CONTEXT&  CONTEXT) { TheInstance()->displayText(CONTEXT); }
	inline virtual void displayText(const CC_DRAW_CONTEXT&  CONTEXT) { /* do nothing */ }

	//! Displays a string at a given 3D position
	/** This method should be called solely during 3D pass rendering (see paintGL).
		\param str string
		\param pos3D 3D position of string origin
		\param rgbColor color (optional: if let to 0, default text rendering color is used)
		\param font font (optional)
	**/
	static void Display3DLabel(const QString& str,
		const CCVector3& pos3D,
		const unsigned char* rgbColor = nullptr,
		const QFont& font = QFont());

public: //! Draws the main 3D layer
	static void SetFocusToScreen();
	static void ToBeRefreshed();
	static void RefreshDisplay(bool only2D = false, bool forceRedraw = true);
	static void RedrawDisplay(bool only2D = false, bool forceRedraw = true);
	static void CheckIfRemove();
	inline static void Draw(CC_DRAW_CONTEXT& context, const ccHObject * obj) { TheInstance()->draw(context, obj); }
	inline virtual void draw(CC_DRAW_CONTEXT& context, const ccHObject * obj) { /* do nothing */ }
	inline static void DrawBBox(CC_DRAW_CONTEXT& context, const ccBBox * bbox) { TheInstance()->drawBBox(context, bbox); }
	inline virtual void drawBBox(CC_DRAW_CONTEXT& context, const ccBBox * bbox) { /* do nothing */ }
	inline static void DrawOrientedBBox(CC_DRAW_CONTEXT& context, const ecvOrientedBBox * obb) { TheInstance()->drawOrientedBBox(context, obb); }
	inline virtual void drawOrientedBBox(CC_DRAW_CONTEXT& context, const ecvOrientedBBox * obb) { /* do nothing */ }
	static void RemoveBB(CC_DRAW_CONTEXT context);
	static void RemoveBB(const QString& viewId);
	static void ChangeEntityProperties(PROPERTY_PARAM& propertyParam, bool autoUpdate = true);
    inline virtual void changeEntityProperties(PROPERTY_PARAM& propertyParam) { /* do nothing */ }
	static void DrawWidgets(const WIDGETS_PARAMETER& param, bool update = false);
    inline virtual void drawWidgets(const WIDGETS_PARAMETER& param) { /* do nothing */ }
	static void RemoveWidgets(const WIDGETS_PARAMETER& param, bool update = false);

	inline static void DrawCoordinates(double scale = 1.0,
		const std::string &id = "reference", int viewport = 0) {
		TheInstance()->drawCoordinates(scale, id, viewport);
	}

	inline virtual void drawCoordinates(double scale = 1.0,
		const std::string &id = "reference", int viewport = 0) { /* do nothing */ }	

	inline static void ToggleOrientationMarker(bool state = true) {
		TheInstance()->toggleOrientationMarker(state);
		UpdateScreen();
	}
	inline virtual void toggleOrientationMarker(bool state = true) { /* do nothing */ }

private:
	static void Draw3D(CC_DRAW_CONTEXT& CONTEXT);

public: // main interface
	inline static ecvGenericVisualizer3D* GetVisualizer3D() { return TheInstance()->getVisualizer3D(); }
	inline virtual ecvGenericVisualizer3D* getVisualizer3D() { return nullptr; /* do nothing */ }
	inline static ecvGenericVisualizer2D* GetVisualizer2D() { return TheInstance()->getVisualizer2D(); }
	inline virtual ecvGenericVisualizer2D* getVisualizer2D() { return nullptr; /* do nothing */ }

	inline static QWidget * GetCurrentScreen() 
	{ 
		if (!TheInstance()) return nullptr;
		return TheInstance()->m_currentScreen;
	}
	static void SetCurrentScreen(QWidget* widget);
	inline static QWidget* GetMainScreen() 
	{ 
		if (!TheInstance()) return nullptr;
		return TheInstance()->m_mainScreen;
	}
	inline static void SetMainScreen(QWidget* widget) { TheInstance()->m_mainScreen = widget; }

	inline static QMainWindow* GetMainWindow() { return TheInstance()->m_win; }
	inline static void SetMainWindow(QMainWindow* win) { TheInstance()->m_win = win; }

	static QPointF ToCenteredGLCoordinates(int x, int y);
	static CCVector3d ToVtkCoordinates(int x, int y, int z = 0);
	static void ToVtkCoordinates(CCVector3d & sP);

	//! Returns window own DB
	inline static ccHObject* GetOwnDB() { return TheInstance()->m_winDBRoot; }
	//! Adds an entity to window own DB
	/** By default no dependency link is established between the entity and the window (DB).
	**/
	static void AddToOwnDB(ccHObject* obj, bool noDependency = true);

	//! Removes an entity from window own DB
	static void RemoveFromOwnDB(ccHObject* obj);

	static void SetSceneDB(ccHObject* root);
	inline static ccHObject* GetSceneDB() { return TheInstance()->m_globalDBRoot; }

	static void UpdateNamePoseRecursive();

	static void SetRedrawRecursive(bool redraw = false);
	static void SetRedrawRecursive(ccHObject* obj, bool redraw = false);

	//! Returns the visible objects bounding-box
	static void GetVisibleObjectsBB(ccBBox& box);

	//! Rotates the base view matrix
	/** Warning: 'base view' marix is either:
		- the rotation around the object in object-centered mode
		- the rotation around the camera center in viewer-centered mode
		(see setPerspectiveState).
	**/
	static void RotateBaseViewMat(const ccGLMatrixd& rotMat);

	inline static ccGLMatrixd& GetBaseViewMat() { return TheInstance()->m_viewportParams.viewMat; }
	static void SetBaseViewMat(ccGLMatrixd& mat);

	static void SetRemoveViewIDs(std::vector<removeInfo> & removeinfos);
	inline static void SetRemoveAllFlag(bool state) { TheInstance()->m_removeAllFlag = state; }

	inline static void TransformCameraView(const ccGLMatrixd & viewMat) { TheInstance()->transformCameraView(viewMat); }
    inline virtual void transformCameraView(const ccGLMatrixd & viewMat) { /* do nothing */ }
	inline static void TransformCameraProjection(const ccGLMatrixd & projMat) { TheInstance()->transformCameraProjection(projMat); }
    inline virtual void transformCameraProjection(const ccGLMatrixd & projMat) { /* do nothing */ }

	static inline int GetDevicePixelRatio() { 
		//return TheInstance()->getDevicePixelRatio(); 
		return GetMainWindow()->devicePixelRatio();
	}

	inline static QRect GetScreenRect() { 
		QRect screenRect = GetCurrentScreen()->geometry();
		QPoint globalPosition = GetCurrentScreen()->mapToGlobal(screenRect.topLeft());
		screenRect.setTopLeft(globalPosition);
		return screenRect;
	}
	inline static void SetScreenSize(int xw, int yw) { GetCurrentScreen()->resize(QSize(xw, yw)); }
	inline static QSize GetScreenSize() { return GetCurrentScreen()->size(); }

	inline static void SetRenderWindowSize(int xw, int yw) { TheInstance()->setRenderWindowSize(xw, yw); }
	inline virtual void setRenderWindowSize(int xw, int yw) {/* do nothing */}

	inline static void FullScreen(bool state) { TheInstance()->fullScreen(state); }
	inline virtual void fullScreen(bool state) {/* do nothing */}

	inline static void GetCameraPos(double *pos, int viewPort = 0) { TheInstance()->getCameraPos(pos, viewPort); }
    inline virtual void getCameraPos(double *pos, int viewPort = 0) { /* do nothing */ }
	inline static void GetCameraFocal(double *focal, int viewPort = 0) { TheInstance()->getCameraFocal(focal, viewPort); }
    inline virtual void getCameraFocal(double *focal, int viewPort = 0) { /* do nothing */ }
	inline static void GetCameraUp(double *up, int viewPort = 0) { TheInstance()->getCameraUp(up, viewPort); }
	virtual void getCameraUp(double *up, int viewPort = 0) { /* do nothing */ }

	inline static void SetCameraPosition(const CCVector3d& pos, int viewPort = 0) { 
        TheInstance()->setCameraPosition(pos, viewPort); }
    inline virtual void setCameraPosition(const CCVector3d& pos, int viewPort = 0) { /* do nothing */ }
	inline static void SetCameraPosition(const double *pos, const double *focal, const double *up, int viewPort = 0) { 
		TheInstance()->setCameraPosition(pos, focal, up, viewPort); }
    inline virtual void setCameraPosition(const double *pos, const double *focal, const double *up, int viewPort = 0) { /* do nothing */ }
	inline static void SetCameraPosition(const double *pos, const double *up, int viewPort = 0) { 
		TheInstance()->setCameraPosition(pos, up, viewPort); }
    inline virtual void setCameraPosition(const double *pos, const double *up, int viewPort = 0) { /* do nothing */ }
	inline static void SetCameraPosition(double pos_x, double pos_y, double pos_z,
		double view_x, double view_y, double view_z, double up_x, double up_y, double up_z, int viewPort = 0) {
		TheInstance()->setCameraPosition(pos_x, pos_y, pos_z,
			view_x, view_y, view_z, up_x, up_y, up_z, viewPort);
	}
	inline virtual void setCameraPosition(double pos_x, double pos_y, double pos_z,
		double view_x, double view_y, double view_z,
		double up_x, double up_y, double up_z, int viewPort = 0) { /* do nothing */ }

	// set and get clip distances (near and far)
	inline static void GetCameraClip(double *clipPlanes, int viewPort = 0) { TheInstance()->getCameraClip(clipPlanes, viewPort); }
	virtual void getCameraClip(double *clipPlanes, int viewPort = 0) { /* do nothing */ }
	inline static void SetCameraClip(double znear, double zfar, int viewport = 0) {
		TheInstance()->m_viewportParams.zNear = znear;
		TheInstance()->m_viewportParams.zFar = zfar;
		TheInstance()->setCameraClip(znear, zfar, viewport); 
	}
	virtual void setCameraClip(double znear, double zfar, int viewport = 0) { /* do nothing */ }

	inline static void ResetCameraClippingRange() { TheInstance()->resetCameraClippingRange(); }
	inline virtual void resetCameraClippingRange() { /* do nothing */ }

	// set and get view angle in y direction
	inline static double GetCameraFovy(int viewPort = 0) { return TheInstance()->getCameraFovy(viewPort); }
	inline virtual double getCameraFovy(int viewPort = 0) { return 0; /* do nothing */ }
	inline static void SetCameraFovy(double fovy, int viewport = 0) { 
		TheInstance()->m_viewportParams.fov = fovy;
		TheInstance()->setCameraFovy(fovy, viewport); 
	}
	inline virtual void setCameraFovy(double fovy, int viewport = 0) { /* do nothing */ }

	inline static void GetViewerPos(int * viewPos, int viewPort = 0) {
		viewPos[0] = 0;
		viewPos[1] = 0;
		viewPos[2] = Width();
		viewPos[3] = Height();
	}

	/** \brief Save the current rendered image to disk, as a PNG screenshot.
	   * \param[in] file the name of the PNG file
	   */
	inline static void SaveScreenshot(const std::string &file) { TheInstance()->saveScreenshot(file); }
	inline virtual void saveScreenshot(const std::string &file) { /* do nothing */ }
	
	/** \brief Save or Load the current rendered camera parameters to disk or current camera.
	* \param[in] file the name of the param file
	*/
	inline static void SaveCameraParameters(const std::string &file) { TheInstance()->saveCameraParameters(file); }
	inline virtual void saveCameraParameters(const std::string &file) { /* do nothing */ }

	inline static void LoadCameraParameters(const std::string &file) { TheInstance()->loadCameraParameters(file); }
	inline virtual void loadCameraParameters(const std::string &file) { /* do nothing */ }

	inline static void ShowOrientationMarker() 
	{ 
		TheInstance()->showOrientationMarker();
		UpdateScreen();
	}
	inline virtual void showOrientationMarker() { /* do nothing */ }

	inline static void SetOrthoProjection(int viewport = 0) 
	{ 
		TheInstance()->setOrthoProjection(viewport);
		UpdateScreen(); 
	}
	inline virtual void setOrthoProjection(int viewport = 0) { /* do nothing */ }
	inline static void SetPerspectiveProjection(int viewport = 0) 
	{ 
		TheInstance()->setPerspectiveProjection(viewport);
		UpdateScreen(); 
	}
	inline virtual void setPerspectiveProjection(int viewport = 0) { /* do nothing */ }

	/** \brief Use Vertex Buffer Objects renderers.
	  * This is an optimization for the obsolete OpenGL backend. Modern OpenGL2 backend (VTK \A1\DD 6.3) uses vertex
	  * buffer objects by default, transparently for the user.
	  * \param[in] use_vbos set to true to use VBOs
	  */
	inline static void SetUseVbos(bool useVbos) { TheInstance()->setUseVbos(useVbos); }
	inline virtual void setUseVbos(bool useVbos) { /* do nothing */ }

	/** \brief Set the ID of a cloud or shape to be used for LUT display
	  * \param[in] id The id of the cloud/shape look up table to be displayed
	  * The look up table is displayed by pressing 'u' in the PCLVisualizer */
	inline static void SetLookUpTableID(const std::string & viewID) { TheInstance()->setLookUpTableID(viewID); }
	inline virtual void setLookUpTableID(const std::string & viewID) { /* do nothing */ }

	inline static void GetProjectionMatrix(double * projArray, int viewPort = 0) {
		TheInstance()->getProjectionMatrix(projArray, viewPort);
	}
	inline virtual void getProjectionMatrix(double * projArray, int viewPort = 0) { /* do nothing */ }
	inline static void GetViewMatrix(double * viewArray, int viewPort = 0) { 
		TheInstance()->getViewMatrix(viewArray, viewPort);
	}
	inline virtual void getViewMatrix(double * viewArray, int viewPort = 0) { /* do nothing */ }

	inline static bool HideShowEntities(CC_DRAW_CONTEXT& CONTEXT) { return TheInstance()->hideShowEntities(CONTEXT); }
	static void HideShowEntities(const QStringList & viewIDs, ENTITY_TYPE hideShowEntityType, bool visibility = false);
	inline virtual bool hideShowEntities(CC_DRAW_CONTEXT& CONTEXT) { return true; /* do nothing */ }

	inline static void RemoveEntities(CC_DRAW_CONTEXT& CONTEXT) { TheInstance()->removeEntities(CONTEXT); }
	static void RemoveEntities(const QStringList & viewIDs, ENTITY_TYPE removeEntityType);
	inline virtual void removeEntities(CC_DRAW_CONTEXT& CONTEXT) { /* do nothing */ }

	static void DrawBackground(CC_DRAW_CONTEXT& CONTEXT);
	static void DrawForeground(CC_DRAW_CONTEXT& CONTEXT);
	static void Update2DLabel(bool immediateUpdate = false);
	static void Pick2DLabel(int x, int y);
	virtual QString pick2DLabel(int x, int y) { return QString(); /* do nothing */ }
	static void Redraw2DLabel();

	static QString Pick3DItem(int x = -1, int y = -1) { return TheInstance()->pick3DItem(x, y); }
	virtual QString pick3DItem(int x = -1, int y = -1) { return QString(); /* do nothing */ }

	static void FilterByEntityType(ccHObject::Container& labels, CV_CLASS_ENUM type);

	inline virtual void setBackgroundColor(CC_DRAW_CONTEXT& CONTEXT) { /* do nothing */ }

	/** \brief Create a new viewport from [xmin,ymin] -> [xmax,ymax].
	  * \param[in] xmin the minimum X coordinate for the viewport (0.0 <= 1.0)
	  * \param[in] ymin the minimum Y coordinate for the viewport (0.0 <= 1.0)
	  * \param[in] xmax the maximum X coordinate for the viewport (0.0 <= 1.0)
	  * \param[in] ymax the maximum Y coordinate for the viewport (0.0 <= 1.0)
	  * \param[in] viewport the id of the new viewport
	  *
	  * \note If no renderer for the current window exists, one will be created, and
	  * the viewport will be set to 0 ('all'). In case one or multiple renderers do
	  * exist, the viewport ID will be set to the total number of renderers - 1.
	  */
	inline static void CreateViewPort(double xmin, double ymin, double xmax, double ymax, int &viewport) {
		TheInstance()->createViewPort(xmin, ymin, xmax, ymax, viewport);
	}
	inline virtual void createViewPort(double xmin, double ymin, double xmax, double ymax, int &viewport) { /* do nothing */ }

	inline static void ResetCameraViewpoint(const std::string & viewID) { TheInstance()->resetCameraViewpoint(viewID); }
	inline virtual void resetCameraViewpoint(const std::string & viewID) { /* do nothing */ }

	static void SetPointSize(float size, bool silent = false, int viewport = 0);
	static void SetPointSizeRecursive(int size);

	//! Sets line width
	/** \param width lines width (between MIN_LINE_WIDTH_F and MAX_LINE_WIDTH_F)
	**/
	static void SetLineWidth(float width, bool silent = false, int viewport = 0);
	static void SetLineWithRecursive(PointCoordinateType with);

	inline static void Toggle2Dviewer(bool state) { TheInstance()->toggle2Dviewer(state); }
	inline virtual void toggle2Dviewer(bool state) { /* do nothing */ }

public: // visualization matrix transformation
	//! Displays a status message in the bottom-left corner
	/** WARNING: currently, 'append' is not supported for SCREEN_CENTER_MESSAGE
		\param message message (if message is empty and append is 'false', all messages will be cleared)
		\param pos message position on screen
		\param append whether to append the message or to replace existing one(s) (only messages of the same type are impacted)
		\param displayMaxDelay_sec minimum display duration
		\param type message type (if not custom, only one message of this type at a time is accepted)
	**/
	static void DisplayNewMessage(const QString& message,
		MessagePosition pos,
		bool append = false,
		int displayMaxDelay_sec = 2,
		MessageType type = CUSTOM_MESSAGE);

	//! Returns current parameters for this display (const version)
	/** Warning: may return overridden parameters!
	**/
	static const ecvGui::ParamStruct& GetDisplayParameters();

	//! Sets current parameters for this display
	static void SetDisplayParameters(const ecvGui::ParamStruct& params);

	static void UpdateDisplayParameters();

	static void SetupProjectiveViewport(const ccGLMatrixd& cameraMatrix, float fov_deg = 0.0f,
		float ar = 1.0f, bool viewerBasedPerspective = true,
		bool bubbleViewMode = false);

	//! Sets current camera aspect ratio (width/height)
	/** AR is only used in perspective mode.
	**/
	static void SetAspectRatio(float ar);

	//! Sets current zoom
	/** Warning: has no effect in viewer-centered perspective mode
	**/
	static void ResizeGL(int w, int h);
	static void UpdateScreenSize();
    inline static void Update() { GetCurrentScreen()->update(); UpdateCamera(); }
    inline static void UpdateScreen() { GetCurrentScreen()->update(); UpdateScene(); }
	inline static void ResetCamera(const ccBBox * bbox)  { TheInstance()->resetCamera(bbox);  UpdateScreen(); }
	inline virtual void resetCamera(const ccBBox * bbox) { /* do nothing */ }
	inline static void ResetCamera() { TheInstance()->resetCamera(); UpdateScreen(); }
	inline virtual void resetCamera() { /* do nothing */ }
	inline static void UpdateCamera() { TheInstance()->updateCamera(); UpdateScreen(); }
	inline virtual void updateCamera() { /* do nothing */ }

	inline static void UpdateScene() { TheInstance()->updateScene(); }
	inline virtual void updateScene() { /* do nothing */ }

	inline static void SetAutoUpateCameraPos(bool state) { TheInstance()->setAutoUpateCameraPos(state); }
	inline virtual void setAutoUpateCameraPos(bool state) { /* do nothing */ }

	/**
	 * Get the current center of rotation
	 */
	inline static void GetCenterOfRotation(double center[3]) { TheInstance()->getCenterOfRotation(center); }
	inline static void GetCenterOfRotation(CCVector3d& center) { TheInstance()->getCenterOfRotation(center.u); }
	inline virtual void getCenterOfRotation(double center[3]) { /* do nothing */ }

	/**
	 * Resets the center of rotation to the focal point.
	 */
	inline static void ResetCenterOfRotation() {
		TheInstance()->resetCenterOfRotation(); 
		UpdateScreen();
	}
	inline virtual void resetCenterOfRotation() { /* do nothing */ }

	/**
	 * Set the center of rotation. For this to work,
	 * one should have appropriate interaction style
	 * and camera manipulators that use the center of rotation
	 * They are setup correctly by default
	 */
	inline static void SetCenterOfRotation(double x, double y, double z) 
	{
		TheInstance()->setCenterOfRotation(x, y, z);
	}
	inline virtual void setCenterOfRotation(double x, double y, double z) { /* do nothing */ }

	inline static void SetCenterOfRotation(const double xyz[3])
	{
		SetCenterOfRotation(xyz[0], xyz[1], xyz[2]);
	}

	inline static void SetCenterOfRotation(const CCVector3d& center) 
	{ 
		SetCenterOfRotation(center.u);
	}

	inline static double GetGLDepth(int x, int y) { return TheInstance()->getGLDepth(x, y); }
	inline virtual double getGLDepth(int x, int y) { return 1.0; /* do nothing */ }

	//! Converts a given (mouse) position in pixels to an orientation
	/** The orientation vector origin is the current pivot point!
	**/
	static CCVector3d ConvertMousePositionToOrientation(int x, int y);

	//! Updates currently active items list (m_activeItems)
	/** The items must be currently displayed in this context
		AND at least one of them must be under the mouse cursor.
	**/
	static void UpdateActiveItemsList(int x, int y, bool extendToSelectedLabels = false);

	//! Sets the OpenGL viewport (shortut)
	inline static void SetGLViewport(int x, int y, int w, int h) { SetGLViewport(QRect(x, y, w, h)); }
	//! Sets the OpenGL viewport
	static void SetGLViewport(const QRect& rect);

	//Graphical features controls
	static void drawCross();
	static void drawTrihedron();

	//! Renders screen to a file
	static bool RenderToFile(QString filename,
		float zoomFactor = 1.0f,
		bool dontScaleFeatures = false,
		bool renderOverlayItems = false);

	static void DrawScale(const ecvColor::Rgbub& color);

	static void DisplayTexture2DPosition(QImage image, const QString& id, int x, int y, int w, int h, unsigned char alpha=255);
	//! Draws the 'hot zone' (+/- icons for point size), 'leave bubble-view' button, etc.
	static void DrawClickableItems(int xStart, int& yStart);
	static void RenderText(
		int x, int y, const QString & str, 
		const QFont & font = QFont(), 
		ecvColor::Rgbub color = ecvColor::defaultLabelBkgColor,
		QString id = "");
	static void RenderText(
		double x, double y, double z, 
		const QString & str,
		const QFont & font = QFont(),
		ecvColor::Rgbub color = ecvColor::defaultLabelBkgColor,
		QString id = "");

	//! Toggles (exclusive) full-screen mode
    inline static void ToggleExclusiveFullScreen(bool state) { TheInstance()->toggleExclusiveFullScreen(state); }
    inline virtual void toggleExclusiveFullScreen(bool state) {/* in this do nothing */}


	//! Returns whether the window is in exclusive full screen mode or not
	inline static bool ExclusiveFullScreen() { return TheInstance()->m_exclusiveFullscreen; }
	inline static void SetExclusiveFullScreenFlage(bool state) { TheInstance()->m_exclusiveFullscreen = state; }

	//! Sets pixel size (i.e. zoom base)
	/** Emits the 'pixelSizeChanged' signal.
	**/
	static void SetPixelSize(float pixelSize);

	//! Center and zoom on a given bounding box
	/** If no bounding box is defined, the current displayed 'scene graph'
		bounding box is taken.
	**/
	static void UpdateConstellationCenterAndZoom(const ccBBox* aBox = nullptr, bool redraw = true);

	//! Returns context information
	static void GetContext(CC_DRAW_CONTEXT& CONTEXT);

	//! Returns the current OpenGL camera parameters
	static void GetGLCameraParameters(ccGLCameraParameters& params);

	static void SetInteractionMode(INTERACTION_FLAGS flags);

	static void SetView(CC_VIEW_ORIENTATION orientation, ccBBox* bbox);
	static void SetView(CC_VIEW_ORIENTATION orientation, bool forceRedraw = true);

	//! Returns a 4x4 'OpenGL' matrix corresponding to a default 'view' orientation
	/** \param orientation view orientation
		\return corresponding GL matrix
	**/
	static ccGLMatrixd GenerateViewMat(CC_VIEW_ORIENTATION orientation);

	//! Set perspective state/mode
	/** Persepctive mode can be:
		- object-centered (moving the mouse make the object rotate)
		- viewer-centered (moving the mouse make the camera move)
		\warning Disables bubble-view mode automatically

		\param state whether perspective mode is enabled or not
		\param objectCenteredView whether view is object- or viewer-centered (forced to true in ortho. mode)
	**/
	static void SetPerspectiveState(bool state, bool objectCenteredView);

	inline static bool GetPerspectiveState() { return TheInstance()->getPerspectiveState(); }
	inline virtual bool getPerspectiveState() const override { return TheInstance()->m_viewportParams.perspectiveView; }

	//! Returns the zoom value equivalent to the current camera position (perspective only)
	static float ComputePerspectiveZoom();

	//! Sets current camera f.o.v. (field of view) in degrees
	/** FOV is only used in perspective mode.
	**/
	static void SetFov(float fov);
	//! Returns the current f.o.v. (field of view) in degrees
	static float GetFov();

	inline static void ZoomGlobal() { UpdateConstellationCenterAndZoom(); }

	//! Returns current viewing direction
	/** This is the direction normal to the screen
		(pointing 'inside') in world base.
	**/
	static CCVector3d GetCurrentViewDir();

	//! Returns current up direction
	/** This is the vertical direction of the screen
		(pointing 'upward') in world base.
	**/
	static CCVector3d GetCurrentUpDir();

	//! Sets camera position
	/** Emits the 'cameraPosChanged' signal.
	**/
	static void SetCameraPos(const CCVector3d& P);

	//! Displaces camera
	/** Values are given in objects world along the current camera
		viewing directions (we use the right hand rule):
		* X: horizontal axis (right)
		* Y: vertical axis (up)
		* Z: depth axis (pointing out of the screen)
	**/
	static void MoveCamera(float dx, float dy, float dz);

	static void SetPickingMode(PICKING_MODE mode = DEFAULT_PICKING);

	//! Sets current zoom
	/** Warning: has no effect in viewer-centered perspective mode
	**/
	static void SetZoom(float value);

	//! Updates current zoom
	/** Warning: has no effect in viewer-centered perspective mode
	**/
	static void UpdateZoom(float zoomFactor);

	//! Sets pivot point
	/** Emits the 'pivotPointChanged' signal.
	**/
	static void SetPivotPoint(const CCVector3d& P, bool autoUpdateCameraPos = false, bool verbose = false);

	//! Sets pivot visibility
	static void SetPivotVisibility(PivotVisibility vis);
	static void SetPivotVisibility(bool state) { TheInstance()->setPivotVisibility(state); }
	virtual void setPivotVisibility(bool state) { /*do nothing here*/ }

	//! Returns pivot visibility
	inline static PivotVisibility GetPivotVisibility() { return TheInstance()->m_pivotVisibility; }

	//! Shows or hide the pivot symbol
	/** Warnings:
		- not to be mistaken with setPivotVisibility
		- only taken into account if pivot visibility is set to PIVOT_SHOW_ON_MOVE
	**/
	static void ShowPivotSymbol(bool state);

	static void SetViewportParameters(const ecvViewportParameters& params);
	static const ecvViewportParameters& GetViewportParameters();

	inline static double GetParallelScale(int viewPort = 0) { return TheInstance()->getParallelScale(); }
	inline virtual double getParallelScale(int viewPort = 0) { return -1.0; }

	static ccGLMatrixd& GetModelViewMatrix();
	static ccGLMatrixd& GetProjectionMatrix();

	static void UpdateModelViewMatrix();
	static void UpdateProjectionMatrix();
	static ccGLMatrixd ComputeModelViewMatrix(const CCVector3d& cameraCenter);
	static ccGLMatrixd ComputeProjectionMatrix(const CCVector3d& cameraCenter, bool withGLfeatures, 
											   ProjectionMetrics* metrics/*=0*/, double* eyeOffset/*=0*/);

	inline static void Deprecate3DLayer() { TheInstance()->m_updateFBO = true; }
	inline static void InvalidateViewport() { TheInstance()->m_validProjectionMatrix = false; }
	inline static void InvalidateVisualization() { TheInstance()->m_validModelviewMatrix = false; }

	static CCVector3d GetRealCameraCenter();

	//! Returns the actual pixel size on screen (taking zoom or perspective parameters into account)
	/** In perspective mode, this value is approximate.
	**/
	static double ComputeActualPixelSize();

	//! Sets bubble-view mode state
	/** Bubble-view is a kind of viewer-based perspective mode where
		the user can't displace the camera (apart from up-down or
		left-right rotations). The f.o.v. is also maximized.

		\warning Any call to a method that changes the perpsective will
		automatically disable this mode.
	**/
	static void SetBubbleViewMode(bool state);
	//! Returns whether bubble-view mode is enabled or no
	inline static bool BubbleViewModeEnabled() { return TheInstance()->m_bubbleViewModeEnabled; }
	//! Set bubble-view f.o.v. (in degrees)
	static void SetBubbleViewFov(float fov_deg);

	//! Sets whether to display the coordinates of the point below the cursor position
	inline static void ShowCursorCoordinates(bool state) { TheInstance()->m_showCursorCoordinates = state; }
	//! Whether the coordinates of the point below the cursor position are displayed or not
	inline static bool CursorCoordinatesShown() { return TheInstance()->m_showCursorCoordinates; }

	//! Toggles the automatic setting of the pivot point at the center of the screen
	static void SetAutoPickPivotAtCenter(bool state);
	static void SendAutoPickPivotAtCenter(bool state) { emit TheInstance()->autoPickPivot(state); }
	//! Whether the pivot point is automatically set at the center of the screen
	inline static bool AutoPickPivotAtCenter() { return TheInstance()->m_autoPickPivotAtCenter; }

	 //! Returns the approximate 3D position of the clicked pixel
	static bool GetClick3DPos(int x, int y, CCVector3d& P3D);

	static void DrawPivot();

	//debug traces on screen
	//! Shows debug info on screen
	inline static void EnableDebugTrace(bool state) { TheInstance()->m_showDebugTraces = state; }

	//! Toggles debug info on screen
	inline static void ToggleDebugTrace() { TheInstance()->m_showDebugTraces = !TheInstance()->m_showDebugTraces; }


	//! Picking parameters
	struct PickingParameters
	{
		//! Default constructor
		PickingParameters(	PICKING_MODE _mode = NO_PICKING,
							int _centerX = 0,
							int _centerY = 0,
							int _pickWidth = 5,
							int _pickHeight = 5,
							bool _pickInSceneDB = true,
							bool _pickInLocalDB = true)
			: mode(_mode)
			, centerX(_centerX)
			, centerY(_centerY)
			, pickWidth(_pickWidth)
			, pickHeight(_pickHeight)
			, pickInSceneDB(_pickInSceneDB)
			, pickInLocalDB(_pickInLocalDB)
		{}

		PICKING_MODE mode;
		int centerX;
		int centerY;
		int pickWidth;
		int pickHeight;
		bool pickInSceneDB;
		bool pickInLocalDB;
	};

	//! Processes the clickable items
	/** \return true if an item has been clicked
	**/
	static bool ProcessClickableItems(int x, int y);

	//! Sets current camera 'zNear' coefficient
	/** zNear coef. is only used in perspective mode.
	**/
	static void SetZNearCoef(double coef);

	//! Starts picking process
	/** \param params picking parameters
	**/
	static void StartPicking(PickingParameters& params);

	static ccHObject* GetPickedEntity(const PickingParameters& params);

	//! Performs the picking with OpenGL
	static void StartOpenGLPicking(const PickingParameters& params);

	//! Starts OpenGL picking process
	static void StartCPUBasedPointPicking(const PickingParameters& params);

	//! Processes the picking process result and sends the corresponding signal
	static void ProcessPickingResult(const PickingParameters& params,
								ccHObject* pickedEntity,
								int pickedItemIndex,
								const CCVector3* nearestPoint = nullptr,
								const std::unordered_set<int>* selectedIDs = nullptr);

	//! Sets current font size
	/** Warning: only used internally.
		Change 'defaultFontSize' with setDisplayParameters instead!
	**/
	inline static void SetFontPointSize(int pixelSize) { TheInstance()->m_font.setPointSize(pixelSize); }
	//! Returns current font size
	static int GetFontPointSize();
	//! Returns current font size for labels
	static int GetLabelFontPointSize();

	//takes rendering zoom into account!
	static QFont GetLabelDisplayFont();
	//takes rendering zoom into account!
	inline static QFont GetTextDisplayFont() { return TheInstance()->m_font; }

	static ENTITY_TYPE ConVertToEntityType(const CV_CLASS_ENUM & type);

	//! Default picking radius value
	static const int DefaultPickRadius = 5;

	//! Sets picking radius
	inline static void SetPickingRadius(int radius) { TheInstance()->m_pickRadius = radius; }
	//! Returns the current picking radius
	inline static int GetPickingRadius() { return TheInstance()->m_pickRadius; }

	//! Sets whether overlay entities (scale, tetrahedron, etc.) should be displayed or not
	inline static void DisplayOverlayEntities(bool state) { TheInstance()->m_displayOverlayEntities = state; }

	//! Returns whether overlay entities (scale, tetrahedron, etc.) are displayed or not
	inline static bool OverlayEntitiesAreDisplayed() { return TheInstance()->m_displayOverlayEntities; }

	//! Currently active items
	/** Active items can be moved with mouse, etc.
	**/
	std::list<ccInteractor*> m_activeItems;

protected:
	ecvDisplayTools() = default;
	//! register visualizer callback function
	virtual void registerVisualizer(QMainWindow * win, bool stereoMode = false) = 0;

	QWidget* m_currentScreen;
	QWidget* m_mainScreen;
	QMainWindow* m_win;
public:
	//! Viewport parameters (zoom, etc.)
	ecvViewportParameters m_viewportParams;

	//! Clickable item
	struct ClickableItem
	{
		enum Role {
			NO_ROLE,
			INCREASE_POINT_SIZE,
			DECREASE_POINT_SIZE,
			INCREASE_LINE_WIDTH,
			DECREASE_LINE_WIDTH,
			LEAVE_BUBBLE_VIEW_MODE,
			LEAVE_FULLSCREEN_MODE,
		};

		ClickableItem() : role(NO_ROLE) {}
		ClickableItem(Role _role, QRect _area) : role(_role), area(_area) {}

		Role role;
		QRect area;
	};

	//! Currently displayed clickable items
	std::vector<ClickableItem> m_clickableItems;

	//! Whether clickable items are visible (= mouse over) or not
	bool m_clickableItemsVisible;

	//! Current intercation flags
	INTERACTION_FLAGS m_interactionFlags;

	PICKING_MODE m_pickingMode;
	//! Whether picking mode is locked or not
	bool m_pickingModeLocked;

	//! Internal timer
	QElapsedTimer m_timer;

	//! Touch event in progress
	bool m_touchInProgress;
	//! Touch gesture initial distance
	qreal m_touchBaseDist;

	//! Scheduler timer
	QTimer m_scheduleTimer;
	//! Scheduled full redraw (no LOD)
	qint64 m_scheduledFullRedrawTime;

	//! Overridden display parameter 
	ecvGui::ParamStruct m_overridenDisplayParameters;

	//! Whether to display overlay entities or not (scale, tetrahedron, etc.)
	bool m_displayOverlayEntities;

	//! Whether display parameters are overidden for this window
	bool m_overridenDisplayParametersEnabled;

	//! Wether exclusive full screen is enabled or not
	bool m_exclusiveFullscreen;

	//! Whether to display the coordinates of the point below the cursor position
	bool m_showCursorCoordinates;

	//! Whether the pivot point is automatically picked at the center of the screen (when possible)
	bool m_autoPickPivotAtCenter;

	//! Whether the display should be refreshed on next call to 'refresh'
	bool m_shouldBeRefreshed;

	//! Candidate pivot point (will be used when the mouse is released)
	CCVector3d m_autoPivotCandidate;

	//! viewport
	QRect m_glViewport;

	//! Debug traces visibility
	bool m_showDebugTraces;

	//! Picking radius (pixels)
	int m_pickRadius;

	//! Auto-refresh mode
	bool m_autoRefresh;

	//! Wheter the rotation axis is locked or not
	bool m_rotationAxisLocked;
	//! Locked rotation axis
	CCVector3d m_lockedRotationAxis;

	//! Complete visualization matrix (GL style - double version)
	ccGLMatrixd m_viewMatd;
	//! Whether the model veiw matrix is valid (or need to be recomputed)
	bool m_validModelviewMatrix;
	//! Projection matrix (GL style - double version)
	ccGLMatrixd m_projMatd;
	//! Whether the projection matrix is valid (or need to be recomputed)
	bool m_validProjectionMatrix;
	//! Distance between the camera and the displayed objects bounding-box
	double m_cameraToBBCenterDist;
	//! Half size of the displayed objects bounding-box
	double m_bbHalfDiag;

	//! Pivot symbol visibility
	PivotVisibility m_pivotVisibility;

	//! Whether pivot symbol should be shown or not
	bool m_pivotSymbolShown;

	//! Whether rectangular picking is allowed or not
	bool m_allowRectangularEntityPicking;

	//! Rectangular picking polyline
	ccPolyline* m_rectPickingPoly;

	//! Window own DB
	ccHObject* m_winDBRoot;

	//! CV main DB
	ccHObject* m_globalDBRoot;

	bool m_removeFlag;
	bool m_removeAllFlag;
	std::vector<removeInfo> m_removeInfos;

	//! Whether to always use FBO or only for GL filters
	bool m_alwaysUseFBO;

	//! Whether FBO should be updated (or simply displayed as a texture = faster!)
	bool m_updateFBO;

	//! Sun light position
	/** Relative to screen.
	**/
	float m_sunLightPos[4];

	//! Whether sun light is enabled or not
	bool m_sunLightEnabled;

	//! Custom light position
	/** Relative to object.
	**/
	float m_customLightPos[4];

	//! Whether custom light is enabled or not
	bool m_customLightEnabled;

	//! Bubble-view mode state
	bool m_bubbleViewModeEnabled;

	//! Bubble-view mode f.o.v. (degrees)
	float m_bubbleViewFov_deg;

	//! Pre-bubble-view camera parameters (backup)
	ecvViewportParameters m_preBubbleViewParameters;

	//! Unique ID
	int m_uniqueID;

	//! Default font
	QFont m_font;

	//! Display capturing mode options
	struct CaptureModeOptions
	{
		//! Default constructor
		CaptureModeOptions()
			: enabled(false)
			, zoomFactor(1.0f)
			, renderOverlayItems(false)
		{}

		bool enabled;
		float zoomFactor;
		bool renderOverlayItems;
	};

	//! Display capturing mode options
	CaptureModeOptions m_captureMode;

	//! Deferred picking
	QTimer m_deferredPickingTimer;

public: // event representation
	static bool USE_2D;
	static bool USE_VTK_PICK;

	CCVector3 m_last_picked_point;
	int m_last_point_index = -1;
	QString m_last_picked_id = QString();

	//! Last click time (msec)
	qint64 m_lastClickTime_ticks;

	//! Hot zone
	HotZone* m_hotZone;

	//! Last mouse position
	QPoint m_lastMousePos;

	QPoint m_lastMouseMovePos;

	QStringList m_diagStrings;

	//! Whether the mouse (cursor) has moved after being pressed or not
	bool m_mouseMoved;
	//! Whether the mouse is currently pressed or not
	bool m_mouseButtonPressed;

	//! Ignore next mouse release event
	bool m_ignoreMouseReleaseEvent;

	static int Width() { return size().width(); }
	static int Height() { return size().height(); }
	static QSize size() { return GetScreenSize(); }

	//! Returns the OpenGL context width
	static int GlWidth() { return TheInstance()->m_glViewport.width(); }
	//! Returns the OpenGL context height
	static int GlHeight() { return TheInstance()->m_glViewport.height(); }
	//! Returns the OpenGL context size
	static QSize GlSize() { return TheInstance()->m_glViewport.size(); }

public slots:

	//! Reacts to the itemPickedFast signal
	void onItemPickedFast(ccHObject* pickedEntity, int pickedItemIndex, int x, int y);

	void onPointPicking(const CCVector3& p, int index, const std::string& id);

	//! Checks for scheduled redraw
	void checkScheduledRedraw();

	//! Performs standard picking at the last clicked mouse position (see m_lastMousePos)
	void doPicking();

	//called when receiving mouse wheel is rotated
	void onWheelEvent(float wheelDelta_deg);

signals:

	//! Signal emitted when an entity is selected in the 3D view
	void entitySelectionChanged(ccHObject* entity);
	//! Signal emitted when multiple entities are selected in the 3D view
	void entitiesSelectionChanged(std::unordered_set<int> entIDs);

	//! Signal emitted when a point (or a triangle) is picked
	/** \param entity 'picked' entity
		\param subEntityID point or triangle index in entity
		\param x mouse cursor x position
		\param y mouse cursor y position
		\param P the picked point
	**/
	void itemPicked(ccHObject* entity, unsigned subEntityID, int x, int y, const CCVector3& P);

	//! Signal emitted when an item is picked (FAST_PICKING mode only)
	/** \param entity entity
		\param subEntityID point or triangle index in entity
		\param x mouse cursor x position
		\param y mouse cursor y position
	**/
	void itemPickedFast(ccHObject* entity, int subEntityID, int x, int y);

	//! Signal emitted when fast picking is finished (FAST_PICKING mode only)
	void fastPickingFinished();

	/*** Camera link mode (interactive modifications of the view/camera are echoed to other windows) ***/

	//! Signal emitted when the window 'model view' matrix is interactively changed
	void viewMatRotated(const ccGLMatrixd& rotMat);
	//! Signal emitted when the camera is interactively displaced
	void cameraDisplaced(float ddx, float ddy);
	//! Signal emitted when the mouse wheel is rotated
	void mouseWheelRotated(float wheelDelta_deg);

	//! Signal emitted when the perspective state changes (see setPerspectiveState)
	void perspectiveStateChanged();

	//! Signal emitted when the window 'base view' matrix is changed
	void baseViewMatChanged(const ccGLMatrixd& newViewMat);

	//! Signal emitted when the pixel size is changed
	void pixelSizeChanged(float pixelSize);

	//! Signal emitted when the f.o.v. changes
	void fovChanged(float fov);

	//! Signal emitted when the zNear coef changes
	void zNearCoefChanged(float coef);

	//! Signal emitted when the pivot point is changed
	void pivotPointChanged(const CCVector3d&);

	//! Signal emitted when the camera position is changed
	void cameraPosChanged(const CCVector3d&);

	//! Signal emitted when the selected object is translated by the user
	void translation(const CCVector3d& t);

	//! Signal emitted when the selected object is rotated by the user
	/** \param rotMat rotation applied to current viewport (4x4 OpenGL matrix)
	**/
	void rotation(const ccGLMatrixd& rotMat);

	//! Signal emitted when the left mouse button is cliked on the window
	/** See INTERACT_SIG_LB_CLICKED.
		Arguments correspond to the clicked point coordinates (x,y) in
		pixels relative to the window corner!
	**/
	void leftButtonClicked(int x, int y);

	//! Signal emitted when the right mouse button is cliked on the window
	/** See INTERACT_SIG_RB_CLICKED.
		Arguments correspond to the clicked point coordinates (x,y) in
		pixels relative to the window corner!
	**/
	void rightButtonClicked(int x, int y);

	//! Signal emitted when the double mouse button is cliked on the window
	/** See INTERACT_SIG_LB_CLICKED.
		Arguments correspond to the clicked point coordinates (x,y) in
		pixels relative to the window corner!
	**/
	void doubleButtonClicked(int x, int y);

	//! Signal emitted when the mouse is moved
	/** See INTERACT_SIG_MOUSE_MOVED.
		The two first arguments correspond to the current cursor coordinates (x,y)
		relative to the window corner!
	**/
	void mouseMoved(int x, int y, Qt::MouseButtons buttons);

	//! Signal emitted when a mouse button is released (cursor on the window)
	/** See INTERACT_SIG_BUTTON_RELEASED.
	**/
	void buttonReleased();

	//! Signal emitted during 3D pass of OpenGL display process
	/** Any object connected to this slot can draw additional stuff in 3D.
		Depth buffering, lights and shaders are enabled by default.
	**/
	void drawing3D();

	//! Signal emitted when files are dropped on the window
	void filesDropped(const QStringList& filenames);

	//! Signal emitted when a new label is created
	void newLabel(ccHObject* obj);

	//! Signal emitted when the exclusive fullscreen is toggled
	void exclusiveFullScreenToggled(bool exclusive);
	void autoPickPivot(bool state);

	void labelmove2D(int x, int y, int dx, int dy);

	void mousePosChanged(const QPoint& pos);

	void pointPicked(double x, double y, double z);

	void cameraParamChanged();

};

Q_DECLARE_OPERATORS_FOR_FLAGS(ecvDisplayTools::INTERACTION_FLAGS);

#endif // ECV_DISPLAY_TOOLS_HEADER
