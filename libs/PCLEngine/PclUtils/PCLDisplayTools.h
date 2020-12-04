//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER BACKEND : qPCL                       #
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
//#                         COPYRIGHT: DAHAI LU                         #
//#                                                                        #
//##########################################################################
//
#ifndef QPCL_DISPLAY_TOOLS_HEADER
#define QPCL_DISPLAY_TOOLS_HEADER

//Local
#include "qPCL.h"
#include "PCLCloud.h"
#include "PCLVis.h"
#include "ImageVis.h"
#include "Tools/ecvTools.h"
#include "VTKExtensions/Widgets/QVTKWidgetCustom.h"

#include "CVMath.h"

// ECV_DB_LIB
#include <ecvDisplayTools.h>

//system
#include <list>
#include <string>

class ccHObject;
class ImageVis;
class ccGenericMesh;
class ccImage;
class ecvOrientedBBox;
class ccPointCloud;
class QMainWindow;

//! CC to PCL cloud converter
class QPCL_ENGINE_LIB_API PCLDisplayTools : public ecvDisplayTools
{
public:
	//! Constructor
	PCLDisplayTools() = default;

	//! Destructor
	virtual ~PCLDisplayTools() override;

public: // inherit from ecvDisplayTools

	inline virtual ecvGenericVisualizer3D* getVisualizer3D() override { return get3DViewer(); }
	inline virtual ecvGenericVisualizer2D* getVisualizer2D() override { return get2DViewer(); }

	inline QVTKWidgetCustom * getQVtkWidget() { return this->m_vtkWidget; }

	inline virtual void toWorldPoint(const CCVector3d& input2D, CCVector3d& output3D) override { getQVtkWidget()->toWorldPoint(input2D, output3D); }
	inline virtual void toWorldPoint(const CCVector3& input2D, CCVector3d& output3D) override { getQVtkWidget()->toWorldPoint(input2D, output3D); }
	
	virtual void toDisplayPoint(const CCVector3d & worldPos, CCVector3d & displayPos) override 
	{
		getQVtkWidget()->toDisplayPoint(worldPos, displayPos);
	}
	virtual void toDisplayPoint(const CCVector3 & worldPos, CCVector3d & displayPos) override
	{
		getQVtkWidget()->toDisplayPoint(worldPos, displayPos);
	}

	virtual void displayText(const CC_DRAW_CONTEXT& CONTEXT) override;

	virtual void toggle2Dviewer(bool state) override;

	virtual void drawWidgets(const WIDGETS_PARAMETER& param) override;
	virtual void changeEntityProperties(PROPERTY_PARAM& param) override;

    virtual void transformCameraView(const ccGLMatrixd & viewMat) override;
    virtual void transformCameraProjection(const ccGLMatrixd & projMat) override;

	virtual void draw(CC_DRAW_CONTEXT& CONTEXT, const ccHObject * obj) override;

	bool checkEntityNeedUpdate(std::string& viewID, const ccHObject* obj);

	virtual void drawBBox(CC_DRAW_CONTEXT& context, const ccBBox * bbox) override;

	virtual void drawOrientedBBox(CC_DRAW_CONTEXT& context, const ecvOrientedBBox * obb) override;

	virtual void toggleOrientationMarker(bool state) override;

	virtual void removeEntities(CC_DRAW_CONTEXT& CONTEXT) override;
	virtual bool hideShowEntities(CC_DRAW_CONTEXT& CONTEXT) override;

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
	inline virtual void createViewPort(double xmin, double ymin, double xmax, double ymax, int &viewport) override {
		m_visualizer3D->createViewPort(xmin, ymin, xmax, ymax, viewport);
	}

	inline virtual void resetCameraViewpoint(const std::string & viewID) override {
		m_visualizer3D->resetCameraViewpoint(viewID);
	}

	inline virtual void setBackgroundColor(CC_DRAW_CONTEXT& CONTEXT) override {
		getQVtkWidget()->setBackgroundColor(ecvTools::TransFormRGB(CONTEXT.backgroundCol),
			ecvTools::TransFormRGB(CONTEXT.backgroundCol2), CONTEXT.drawBackgroundGradient);
	}

	inline virtual void showOrientationMarker() override {
		m_visualizer3D->showPclMarkerAxes(m_visualizer3D->getRenderWindowInteractor());
	}

	inline virtual void drawCoordinates(double scale = 1.0,
		const std::string &id = "reference", int viewport = 0) override 
	{
		m_visualizer3D->addCoordinateSystem(scale, id, viewport);
	}

public:
	// set and get camera parameters
	inline virtual void resetCamera() override { m_visualizer3D->resetCamera(); }
	inline virtual void resetCamera(const ccBBox * bbox) override 
	{ 
		m_visualizer3D->resetCamera(bbox);
	}
	inline virtual void updateCamera() override { m_visualizer3D->updateCamera(); }

	inline virtual void updateScene() override { getQVtkWidget()->updateScene(); }

	inline virtual void setAutoUpateCameraPos(bool state) override
	{ 
		if (this->m_visualizer3D)
		{
			this->m_visualizer3D->setAutoUpateCameraPos(state);
		}
	}

	/**
	 * Get the current center of rotation
	 */
	inline virtual void getCenterOfRotation(double center[3]) override
	{
		if (this->m_visualizer3D)
		{
			this->m_visualizer3D->getCenterOfRotation(center);
		}
	}

	/**
	 * Resets the center of rotation to the focal point.
	 */
	inline virtual void resetCenterOfRotation() override 
	{
		if (this->m_visualizer3D)
		{
			this->m_visualizer3D->resetCenterOfRotation();
		}
	}

	/**
	 * Set the center of rotation. For this to work,
	 * one should have appropriate interaction style
	 * and camera manipulators that use the center of rotation
	 * They are setup correctly by default
	 */
	inline virtual void setCenterOfRotation(double x, double y, double z) override
	{
		if (this->m_visualizer3D)
		{
			this->m_visualizer3D->setCenterOfRotation(x, y, z);
		}
	}
	inline void setCenterOfRotation(const double xyz[3])
	{
		this->setCenterOfRotation(xyz[0], xyz[1], xyz[2]);
	}

	virtual void setPivotVisibility(bool state) override
	{
		if (this->m_visualizer3D)
		{
			this->m_visualizer3D->setCenterAxesVisibility(state);
		}
	}

	inline virtual void resetCameraClippingRange() override
	{ 
		if (m_visualizer3D)
		{
			m_visualizer3D->resetCameraClippingRange();
		}
	}

	inline virtual double getGLDepth(int x, int y) override
	{
		if (m_visualizer3D)
		{
			return m_visualizer3D->getGLDepth(x, y);
		}
		else
		{
			return 1.0;
		}
	}

	inline virtual void getCameraPos(double *pos, int viewPort = 0) override {
		const pcl::visualization::Camera& cam = m_visualizer3D->getCamera(viewPort);
		pos[0] = cam.pos[0];
		pos[1] = cam.pos[1];
		pos[2] = cam.pos[2];
	}
	inline virtual void getCameraFocal(double *focal, int viewPort = 0) override { 
		const pcl::visualization::Camera& cam = m_visualizer3D->getCamera(viewPort);
		focal[0] = cam.focal[0];
		focal[1] = cam.focal[1];
		focal[2] = cam.focal[2];
	}
	inline virtual void getCameraUp(double *up, int viewPort = 0) override {
		const pcl::visualization::Camera& cam = m_visualizer3D->getCamera(viewPort);
		up[0] = cam.view[0];
		up[1] = cam.view[1];
		up[2] = cam.view[2];
	}

    inline virtual void setCameraPosition( const CCVector3d& pos, int viewPort = 0 ) override {
		getQVtkWidget()->setCameraPosition(pos);
	}

	inline virtual void setCameraPosition(const double *pos, const double *focal, const double *up, int viewPort = 0) override {
		m_visualizer3D->setCameraPosition(
			pos[0], pos[1], pos[2],
			focal[0], focal[1], focal[2],
			up[0], up[1], up[2], viewPort
		);
	}

	inline virtual void setCameraPosition(const double *pos, const double *up, int viewPort = 0) override {
		m_visualizer3D->setCameraPosition(
			pos[0], pos[1], pos[2],
			up[0], up[1], up[2], viewPort
		);
	}

	inline virtual void setCameraPosition(double pos_x, double pos_y, double pos_z,
		double view_x, double view_y, double view_z,
		double up_x, double up_y, double up_z, int viewPort = 0)  override {
		m_visualizer3D->setCameraPosition(pos_x, pos_y, pos_z,
			view_x, view_y, view_z, up_x, up_y, up_z, viewPort);
	}

    inline virtual void setRenderWindowSize(int xw, int yw) override
	{
		getQVtkWidget()->GetRenderWindow()->SetPosition(0, 0);
		getQVtkWidget()->GetRenderWindow()->SetSize(xw, yw);
	}

	inline virtual void fullScreen(bool state) override 
	{ 
		m_visualizer3D->setFullScreen(state);
	}

	inline virtual void setOrthoProjection(int viewport = 0) override
	{ 
		if (m_visualizer3D)
		{
			m_visualizer3D->setOrthoProjection(viewport);
			m_visualizer3D->synchronizeGeometryBounds();
		}
	}
	inline virtual void setPerspectiveProjection(int viewport = 0) override 
	{ 
		if (m_visualizer3D)
		{
			m_visualizer3D->setPerspectiveProjection(viewport);
			m_visualizer3D->synchronizeGeometryBounds();
		}
	}

	// set and get clip distances (near and far)
	inline virtual void getCameraClip(double *clipPlanes, int viewPort = 0) override 
	{ 
		const pcl::visualization::Camera& cam = m_visualizer3D->getCamera(viewPort);
		clipPlanes[0] = cam.clip[0];
		clipPlanes[1] = cam.clip[1];
	}
	inline virtual void setCameraClip(double znear, double zfar, int viewport = 0) override 
	{ 
		if (m_visualizer3D)
		{
			m_visualizer3D->setCameraClipDistances(znear, zfar, viewport);
		}
	}

	// set and get view angle in y direction or zoom factor in perspective mode
    inline virtual double getCameraFovy(int viewPort = 0) override { return CVLib::RadiansToDegrees(m_visualizer3D->getCamera(viewPort).fovy); }
    inline virtual void setCameraFovy(double fovy, int viewport = 0) override { m_visualizer3D->setCameraFieldOfView(CVLib::RadiansToDegrees(fovy), viewport); }

	// get zoom factor in parallel mode
	virtual double getParallelScale(int viewPort = 0) override;

	/** \brief Save the current rendered image to disk, as a PNG screen shot.
	   * \param[in] file the name of the PNG file
	   */
	inline virtual void saveScreenshot(const std::string &file) override { m_visualizer3D->saveScreenshot(file); }
	
	/** \brief Save or Load the current rendered camera parameters to disk or current camera.
	* \param[in] file the name of the param file
	*/
	inline virtual void saveCameraParameters(const std::string &file) override { m_visualizer3D->saveCameraParameters(file); }
	inline virtual void loadCameraParameters(const std::string &file) override { m_visualizer3D->loadCameraParameters(file); }
	
	/** \brief Use Vertex Buffer Objects renderers.
	  * This is an optimization for the obsolete OpenGL backend. Modern OpenGL2 backend (VTK6.3) uses vertex
	  * buffer objects by default, transparently for the user.
	  * \param[in] use_vbos set to true to use VBOs
	  */
	inline virtual void setUseVbos(bool useVbos) override { m_visualizer3D->setUseVbos(useVbos); }

	/** \brief Set the ID of a cloud or shape to be used for LUT display
	  * \param[in] id The id of the cloud/shape look up table to be displayed
	  * The look up table is displayed by pressing 'u' in the PCLVisualizer */
	inline virtual void setLookUpTableID(const std::string & viewID) override { m_visualizer3D->setLookUpTableID(viewID); }

	inline virtual void getProjectionMatrix(double * projArray, int viewPort = 0) override;
	inline virtual void getViewMatrix(double * ViewArray, int viewPort = 0) override;

	virtual void setViewMatrix(double* viewArray, int viewPort = 0);

public:
	inline PclUtils::PCLVis* get3DViewer() { return m_visualizer3D.get(); }
	inline PclUtils::ImageVis* get2DViewer() { return m_visualizer2D.get(); }

	virtual QString pick2DLabel(int x, int y) override;

	virtual QString pick3DItem(int x = -1, int y = -1) override;
private:
	void drawPointCloud(CC_DRAW_CONTEXT& CONTEXT, ccPointCloud * ecvCloud);
    void drawMesh(CC_DRAW_CONTEXT& CONTEXT, ccGenericMesh* mesh);
	void drawPolygon(CC_DRAW_CONTEXT& CONTEXT, ccPolyline* polyline);
	void drawImage(CC_DRAW_CONTEXT& CONTEXT, ccImage* image);

	bool updateEntityColor(CC_DRAW_CONTEXT& CONTEXT, ccHObject* ent);

protected:
	// QVTKOpenGLNativeWidget
	QVTKWidgetCustom* m_vtkWidget = nullptr;

	PclUtils::ImageVisPtr m_visualizer2D = nullptr;

	PclUtils::PCLVisPtr m_visualizer3D = nullptr;

	virtual void registerVisualizer(QMainWindow * widget, bool stereoMode = false) override;
};

#endif // QPCL_DISPLAY_TOOLS_HEADER
