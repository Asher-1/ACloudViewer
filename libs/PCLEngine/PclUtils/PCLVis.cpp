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

//Local
#include "PCLVis.h"
#include "PCLConv.h"
#include "Tools/ecvTools.h"
#include "Tools/PclTools.h"

#include "VtkUtils/vtkutils.h"

// CV_CORE_LIB
#include <CVTools.h>
#include <ecvGLMatrix.h>

// ECV_DB_LIB
#include <ecvBBox.h>
#include <ecvColorScale.h>
#include <ecvScalarField.h>
#include <ecvDisplayTools.h>

// VTK Extension
#include <VTKExtensions/Views/vtkPVCenterAxesActor.h>
#include <VTKExtensions/Core/vtkMemberFunctionCommand.h>
#include <VTKExtensions/Widgets/CustomVtkCaptionWidget.h>
#include <VTKExtensions/Widgets/vtkScalarBarWidgetCustom.h>
#include "VTKExtensions/InteractionStyle/vtkPVTrackballMultiRotate.h"
#include "VTKExtensions/InteractionStyle/vtkPVTrackballRoll.h"
#include "VTKExtensions/InteractionStyle/vtkPVTrackballRotate.h"
#include "VTKExtensions/InteractionStyle/vtkTrackballPan.h"
#include "VTKExtensions/InteractionStyle/vtkPVTrackballZoom.h"
#include "VTKExtensions/InteractionStyle/vtkPVTrackballZoomToMouse.h"
#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"

// VTK
#include <vtkPointPicker.h>
#include <vtkAreaPicker.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>
#include <vtkRendererCollection.h>
#include <vtkCamera.h>
#include <vtkAxes.h>
#include <vtkTubeFilter.h>

#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkCaptionActor2D.h>
#include <vtkCaptionRepresentation.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkWidgetEvent.h>
#include <vtkWidgetEventTranslator.h>

#include <vtkTexture.h>
#include <vtkAxesActor.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkPropAssembly.h>
#include <vtkUnsignedCharArray.h>
#include <vtkAnnotatedCubeActor.h>
#include <vtkPropPicker.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkJPEGReader.h>
#include <vtkBMPReader.h>
#include <vtkPNMReader.h>
#include <vtkPNGReader.h>
#include <vtkTIFFReader.h>
#include <vtkLookupTable.h>
#include <vtkTextureUnitManager.h>

// PCL
#include <pcl/common/transforms.h>
#include <pcl/visualization/common/actor_map.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>

// Support for VTK 7.1 upwards
#ifdef vtkGenericDataArray_h
#define SetTupleValue SetTypedTuple
#define InsertNextTupleValue InsertNextTypedTuple
#define GetTupleValue GetTypedTuple
#endif

#define ORIENT_MODE 0
#define SELECT_MODE 1

namespace PclUtils
{
PCLVis::PCLVis(vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle> interactor_style,
               const std::string& viewerName/* = ""*/,
               bool initIterator/* = false*/,
               int argc/* = 0*/,				// unused
               char** argv/* = nullptr*/ )		// unused
		: pcl::visualization::PCLVisualizer(argc, argv, viewerName, interactor_style, initIterator)
		, m_widget_map(new WidgetActorMap)
		, m_prop_map(new PropActorMap)
		, m_x_pressNum(0)
		, m_currentMode(ORIENT_MODE)
		, m_autoUpdateCameraPos(false)
		, m_pointPickingEnabled(true)
		, m_areaPickingEnabled(false)
		, m_actorPickingEnabled(false)
	{
		// disable warnings!
		getRenderWindow()->GlobalWarningDisplayOff();

		// config center axes!
		this->configCenterAxes();

        // config interactor style!
		this->configInteractorStyle(interactor_style);
	}

	PCLVis::PCLVis(vtkSmartPointer<vtkRenderer> ren,
				   vtkSmartPointer<vtkRenderWindow> wind,
				   vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle> interactor_style,
				   const std::string& viewerName,
                   bool initIterator/* = false*/,
                   int argc /* = 0*/,           // unused
                   char** argv /* = nullptr*/)  // unused
        : pcl::visualization::PCLVisualizer(argc, argv, ren, wind, viewerName, interactor_style, initIterator)
		, m_widget_map(new WidgetActorMap)
		, m_prop_map(new PropActorMap)
		, m_x_pressNum(0)
		, m_currentMode(ORIENT_MODE)
		, m_autoUpdateCameraPos(false)
		, m_pointPickingEnabled(true)
		, m_areaPickingEnabled(false)
		, m_actorPickingEnabled(false)
	{
        // disable warnings!
		getRenderWindow()->GlobalWarningDisplayOff();

		// config center axes!
        this->configCenterAxes();

        // config interactor style!
        this->configInteractorStyle(interactor_style);
	}

	PCLVis::~PCLVis()
	{
		if (isPointPickingEnabled())
		{
			setPointPickingEnabled(false);
		}

		if (isAreaPickingEnabled())
		{
			setAreaPickingEnabled(false);
		}

		if (isActorPickingEnabled())
		{
			setActorPickingEnabled(false);
		}

		if (m_centerAxes)
		{
			this->m_centerAxes->Delete();
		}
	}

	void PCLVis::configCenterAxes()
	{
        this->m_centerAxes = VTKExtensions::vtkPVCenterAxesActor::New();
        this->m_centerAxes->SetComputeNormals(0);
        this->m_centerAxes->SetPickable(0);
        this->m_centerAxes->SetScale(0.25, 0.25, 0.25);
        addActorToRenderer(this->m_centerAxes);
    }

    void PCLVis::configInteractorStyle(vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle> interactor_style) 
	{
        this->TwoDInteractorStyle = vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>::New();
        this->m_interactorStyle = this->ThreeDInteractorStyle = interactor_style;

        // add some default manipulators. Applications can override them without much ado.
        registerInteractorStyle(false);
        // getInteractorStyle()->setKeyboardModifier(pcl::visualization::INTERACTOR_KB_MOD_SHIFT);
	}

	void PCLVis::initialize()
	{
		registerMouse();
		registerKeyboard();
		registerPointPicking();
		registerAreaPicking();

		vtkMemberFunctionCommand<PCLVis>* observer =
			vtkMemberFunctionCommand<PCLVis>::New();
		observer->SetCallback(*this, &PCLVis::resetCameraClippingRange);
		this->getCurrentRenderer()->AddObserver(vtkCommand::ResetCameraClippingRangeEvent, observer);
		observer->FastDelete();
    }

	void PCLVis::getCenterOfRotation(double center[3])
	{
		if (this->ThreeDInteractorStyle)
		{
			this->ThreeDInteractorStyle->GetCenterOfRotation(center);
		}
	}

	void PCLVis::resetCenterOfRotation()
	{
		double fP[4];
		double center[3];
		getVtkCamera()->GetFocalPoint(fP);
		for (int i = 0; i < 3; ++i)
		{
			center[i] = fP[i] / fP[3];
		}
		setCenterOfRotation(center[0], center[1], center[2]);
	}

	void PCLVis::setCenterOfRotation(double x, double y, double z)
	{
		this->m_centerAxes->SetPosition(x, y, z);
		if (this->TwoDInteractorStyle)
		{
			this->TwoDInteractorStyle->SetCenterOfRotation(x, y, z);
		}
		if (this->ThreeDInteractorStyle)
		{
			this->ThreeDInteractorStyle->SetCenterOfRotation(x, y, z);
		}
	}

	//----------------------------------------------------------------------------
	void PCLVis::setRotationFactor(double factor)
	{
		if (this->TwoDInteractorStyle)
		{
			this->TwoDInteractorStyle->SetRotationFactor(factor);
		}
		if (this->ThreeDInteractorStyle)
		{
			this->ThreeDInteractorStyle->SetRotationFactor(factor);
		}
	}

	double PCLVis::getRotationFactor()
	{
		if (this->ThreeDInteractorStyle)
		{
			return this->ThreeDInteractorStyle->GetRotationFactor();
		}
		else
		{
			return 1.0;
		}
	}

	//----------------------------------------------------------------------------
	void PCLVis::setCamera2DManipulators(const int manipulators[9])
	{
		this->setCameraManipulators(this->TwoDInteractorStyle, manipulators);
	}

	//----------------------------------------------------------------------------
	void PCLVis::setCamera3DManipulators(const int manipulators[9])
	{
		this->setCameraManipulators(this->ThreeDInteractorStyle, manipulators);
	}

	//----------------------------------------------------------------------------
	void PCLVis::setCameraManipulators(
		VTKExtensions::vtkCustomInteractorStyle* style, 
		const int manipulators[9])
	{
		if (!style)
		{
			return;
		}

		style->RemoveAllManipulators();
		enum
		{
			NONE = 0,
			SHIFT = 1,
			CTRL = 2
		};

		enum
		{
			PAN = 1,
			ZOOM = 2,
			ROLL = 3,
			ROTATE = 4,
			MULTI_ROTATE = 5,
			ZOOM_TO_MOUSE = 6
		};

		for (int manip = NONE; manip <= CTRL; manip++)
		{
			for (int button = 0; button < 3; button++)
			{
				int manipType = manipulators[3 * manip + button];
				vtkSmartPointer<vtkCameraManipulator> cameraManipulator;
				switch (manipType)
				{
				case PAN:
					cameraManipulator = vtkSmartPointer<vtkTrackballPan>::New();
					break;
				case ZOOM:
					cameraManipulator = vtkSmartPointer<vtkPVTrackballZoom>::New();
					break;
				case ROLL:
					cameraManipulator = vtkSmartPointer<vtkPVTrackballRoll>::New();
					break;
				case ROTATE:
					cameraManipulator = vtkSmartPointer<vtkPVTrackballRotate>::New();
					break;
				case MULTI_ROTATE:
					cameraManipulator = vtkSmartPointer<vtkPVTrackballMultiRotate>::New();
					break;
				case ZOOM_TO_MOUSE:
					cameraManipulator = vtkSmartPointer<vtkPVTrackballZoomToMouse>::New();
					break;
				}
				if (cameraManipulator)
				{
					cameraManipulator->SetButton(button + 1); // since button starts with 1.
					cameraManipulator->SetControl(manip == CTRL ? 1 : 0);
					cameraManipulator->SetShift(manip == SHIFT ? 1 : 0);
					style->AddManipulator(cameraManipulator);
				}
			}
		}
	}

	//----------------------------------------------------------------------------
	void PCLVis::setCamera2DMouseWheelMotionFactor(double factor)
	{
		if (this->TwoDInteractorStyle)
		{
			this->TwoDInteractorStyle->SetMouseWheelMotionFactor(factor);
		}
	}

	//----------------------------------------------------------------------------
	void PCLVis::setCamera3DMouseWheelMotionFactor(double factor)
	{
		if (this->ThreeDInteractorStyle)
		{
			this->ThreeDInteractorStyle->SetMouseWheelMotionFactor(factor);
		}
	}

	//----------------------------------------------------------------------------
	void PCLVis::updateCenterAxes()
	{
		vtkBoundingBox bbox(this->GeometryBounds);

		// include the center of rotation in the axes size determination.
		bbox.AddPoint(this->m_centerAxes->GetPosition());

		double widths[3];
		bbox.GetLengths(widths);

		// lets make some thickness in all directions
		double diameterOverTen = bbox.GetMaxLength() > 0 ? bbox.GetMaxLength() / 10.0 : 1.0;
		widths[0] = widths[0] < diameterOverTen ? diameterOverTen : widths[0];
		widths[1] = widths[1] < diameterOverTen ? diameterOverTen : widths[1];
		widths[2] = widths[2] < diameterOverTen ? diameterOverTen : widths[2];

		widths[0] *= 0.25;
		widths[1] *= 0.25;
		widths[2] *= 0.25;
		this->m_centerAxes->SetScale(widths);
	}

	//----------------------------------------------------------------------------
	void PCLVis::synchronizeGeometryBounds()
	{
		// get local bounds to consider 3D widgets correctly.
		// if ComputeVisiblePropBounds is called when there's no real window on the
		// local process, all vtkWidgetRepresentations return wacky Z bounds which
		// screws up the renderer and we don't see any images. Hence we skip this on
		// non-rendering nodes.
		this->m_centerAxes->SetUseBounds(0);
		double prop_bounds[6];
		this->getCurrentRenderer()->ComputeVisiblePropBounds(prop_bounds);
		this->m_centerAxes->SetUseBounds(1);

		// sync up bounds across all processes when doing distributed rendering.
		this->GeometryBounds.Reset();
		this->GeometryBounds.AddBounds(prop_bounds);
		if (!this->GeometryBounds.IsValid())
		{
			this->GeometryBounds.SetBounds(-1, 1, -1, 1, -1, 1);
		}

		this->updateCenterAxes();
		this->resetCameraClippingRange();
	}

	//*****************************************************************
	// Forwarded to center axes.
	//----------------------------------------------------------------------------
	void PCLVis::setCenterAxesVisibility(bool v)
	{
		if (this->m_centerAxes)
		{
			this->m_centerAxes->SetVisibility(v);
		}
	}

	double PCLVis::getGLDepth(int x, int y)
	{
		// Get camera focal point and position. Convert to display (screen)
		// coordinates. We need a depth value for z-buffer.
		//
		double cameraFP[4];
        getVtkCamera()->GetFocalPoint(cameraFP);
        cameraFP[3] = 1.0;
        getCurrentRenderer()->SetWorldPoint(cameraFP);
        getCurrentRenderer()->WorldToDisplay();
        double* displayCoord = getCurrentRenderer()->GetDisplayPoint();
        double z_buffer = displayCoord[2];
		
		return z_buffer;
	}

	void PCLVis::getProjectionTransformMatrix(Eigen::Matrix4d& proj)
	{
		vtkMatrix4x4 * pMat = getVtkCamera()->GetProjectionTransformMatrix(getCurrentRenderer());
		proj = Eigen::Matrix4d(pMat->GetData());
		proj.transposeInPlace();
	}

	void PCLVis::getModelViewTransformMatrix(Eigen::Matrix4d& view)
	{
		vtkMatrix4x4 * vMat = getVtkCamera()->GetModelViewTransformMatrix();
		view = Eigen::Matrix4d(vMat->GetData());
		view.transposeInPlace();
	}

	void PCLVis::resetCamera(const ccBBox * bbox)
	{
		if (!bbox) return;
		double bounds[6];
		bbox->getBounds(bounds);
		resetCamera(bounds);
	}

	void PCLVis::resetCameraClippingRange()
	{
		if (this->GeometryBounds.IsValid())
		{
			double originBounds[6];
			this->GeometryBounds.GetBounds(originBounds);
			this->GeometryBounds.Scale(1.1, 1.1, 1.1);
			double bounds[6];
			this->GeometryBounds.GetBounds(bounds);
			this->getCurrentRenderer()->ResetCameraClippingRange(bounds);
			this->GeometryBounds.SetBounds(originBounds);
		}
	}

	void PCLVis::resetCamera(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax)
	{
		// Update the camera parameters
		getRendererCollection()->InitTraversal();
		vtkRenderer* renderer = NULL;
		while ((renderer = getRendererCollection()->GetNextItem()) != NULL)
		{
			renderer->ResetCamera(xMin, xMax, yMin, yMax, zMin, zMax);
		}
	}

	// Reset the camera clipping range to include this entire bounding box
	void PCLVis::getReasonableClippingRange(double range[2])
	{
		double vn[3], position[3], a, b, c, d;
		double dist;
		int i, j, k;

		this->synchronizeGeometryBounds();

		double bounds[6];
		this->GeometryBounds.GetBounds(bounds);

		// Don't reset the clipping range when we don't have any 3D visible props
		if (!vtkMath::AreBoundsInitialized(bounds))
		{
			return;
		}

		if (getVtkCamera() == nullptr)
		{
			CVLog::Warning("Trying to reset clipping range of non-existent camera");
			return;
		}

		double expandedBounds[6] = { bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5] };
		if (!getVtkCamera()->GetUseOffAxisProjection())
		{
			getVtkCamera()->GetViewPlaneNormal(vn);
			getVtkCamera()->GetPosition(position);
			this->expandBounds(expandedBounds, getVtkCamera()->GetModelTransformMatrix());
		}
		else
		{
			getVtkCamera()->GetEyePosition(position);
			getVtkCamera()->GetEyePlaneNormal(vn);
			this->expandBounds(expandedBounds, getVtkCamera()->GetModelViewTransformMatrix());
		}

		a = -vn[0];
		b = -vn[1];
		c = -vn[2];
		d = -(a * position[0] + b * position[1] + c * position[2]);

		// Set the max near clipping plane and the min far clipping plane
		range[0] = a * expandedBounds[0] + b * expandedBounds[2] + c * expandedBounds[4] + d;
		range[1] = 1e-18;

		// Find the closest / farthest bounding box vertex
		for (k = 0; k < 2; k++)
		{
			for (j = 0; j < 2; j++)
			{
				for (i = 0; i < 2; i++)
				{
					dist = a * expandedBounds[i] + b * expandedBounds[2 + j] + c * expandedBounds[4 + k] + d;
					range[0] = (dist < range[0]) ? (dist) : (range[0]);
					range[1] = (dist > range[1]) ? (dist) : (range[1]);
				}
			}
		}

		// do not let far - near be less than 0.1 of the window height
		// this is for cases such as 2D images which may have zero range
		double minGap = 0.0;
		if (getVtkCamera()->GetParallelProjection())
		{
			minGap = 0.1 * getVtkCamera()->GetParallelScale();
		}
		else
		{
			double angle = vtkMath::RadiansFromDegrees(getVtkCamera()->GetViewAngle());
			minGap = 0.2 * tan(angle / 2.0) * range[1];
		}
		if (range[1] - range[0] < minGap)
		{
			minGap = minGap - range[1] + range[0];
			range[1] += minGap / 2.0;
			range[0] -= minGap / 2.0;
		}

		// Do not let the range behind the camera throw off the calculation.
		if (range[0] < 0.0)
		{
			range[0] = 0.0;
		}

		// Give ourselves a little breathing room
		range[0] = 0.99 * range[0] - (range[1] - range[0]) * getCurrentRenderer()->GetClippingRangeExpansion();
		range[1] = 1.01 * range[1] + (range[1] - range[0]) * getCurrentRenderer()->GetClippingRangeExpansion();

		// Make sure near is not bigger than far
		range[0] = (range[0] >= range[1]) ? (0.01 * range[1]) : (range[0]);

		// Make sure near is at least some fraction of far - this prevents near
		// from being behind the camera or too close in front. How close is too
		// close depends on the resolution of the depth buffer
		if (!getCurrentRenderer()->GetNearClippingPlaneTolerance())
		{
			getCurrentRenderer()->SetNearClippingPlaneTolerance(0.01);
			if (this->getRenderWindow())
			{
				int ZBufferDepth = this->getRenderWindow()->GetDepthBufferSize();
				if (ZBufferDepth > 16)
				{
					getCurrentRenderer()->SetNearClippingPlaneTolerance(0.001);
				}
			}
		}

		// make sure the front clipping range is not too far from the far clippnig
		// range, this is to make sure that the zbuffer resolution is effectively
		// used
		if (range[0] < getCurrentRenderer()->GetNearClippingPlaneTolerance() * range[1])
		{
			range[0] = getCurrentRenderer()->GetNearClippingPlaneTolerance() * range[1];
		}
	}

	//----------------------------------------------------------------------------
	void PCLVis::expandBounds(double bounds[6], vtkMatrix4x4* matrix)
	{
		if (!bounds)
		{
			CVLog::Error("ERROR: Invalid bounds");
			return;
		}

		if (!matrix)
		{
			CVLog::Error("<<ERROR: Invalid matrix");
			return;
		}

		// Expand the bounding box by model view transform matrix.
		double pt[8][4] = { { bounds[0], bounds[2], bounds[5], 1.0 },
		  { bounds[1], bounds[2], bounds[5], 1.0 }, { bounds[1], bounds[2], bounds[4], 1.0 },
		  { bounds[0], bounds[2], bounds[4], 1.0 }, { bounds[0], bounds[3], bounds[5], 1.0 },
		  { bounds[1], bounds[3], bounds[5], 1.0 }, { bounds[1], bounds[3], bounds[4], 1.0 },
		  { bounds[0], bounds[3], bounds[4], 1.0 } };

		// \note: Assuming that matrix doesn not have projective component. Hence not
		// dividing by the homogeneous coordinate after multiplication
		for (int i = 0; i < 8; ++i)
		{
			matrix->MultiplyPoint(pt[i], pt[i]);
		}

		// min = mpx = pt[0]
		double min[4], max[4];
		for (int i = 0; i < 4; ++i)
		{
			min[i] = pt[0][i];
			max[i] = pt[0][i];
		}

		for (int i = 1; i < 8; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				if (min[j] > pt[i][j])
					min[j] = pt[i][j];
				if (max[j] < pt[i][j])
					max[j] = pt[i][j];
			}
		}

		// Copy values back to bounds.
		bounds[0] = min[0];
		bounds[2] = min[1];
		bounds[4] = min[2];

		bounds[1] = max[0];
		bounds[3] = max[1];
		bounds[5] = max[2];
	}

	/********************************Draw Entities*********************************/
	void PCLVis::draw(CC_DRAW_CONTEXT& CONTEXT, PCLCloud::Ptr smCloud)
	{
		const std::string viewID = CVTools::fromQString(CONTEXT.viewID);
		int viewPort = CONTEXT.defaultViewPort;

		if (CONTEXT.drawParam.showColors || CONTEXT.drawParam.showSF)
		{
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
			FROM_PCL_CLOUD(*smCloud, *cloud_rgb);

			if (cloud_rgb)
			{
				if (!updatePointCloud(cloud_rgb, viewID))
				{
					addPointCloud(cloud_rgb, viewID, viewPort);
				}
			}
		}
		else
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
			FROM_PCL_CLOUD(*smCloud, *cloud_xyz);
			if (cloud_xyz)
			{
				ecvColor::Rgbub col = CONTEXT.pointsDefaultCol;
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_xyz, col.r, col.g, col.b);
				if (!updatePointCloud<pcl::PointXYZ>(cloud_xyz, single_color, viewID))
				{
					addPointCloud<pcl::PointXYZ>(cloud_xyz, single_color, viewID, viewPort);
				}
			}
		}
	}

	void PCLVis::updateNormals(CC_DRAW_CONTEXT& CONTEXT, PCLCloud::Ptr smCloud)
	{	
		const std::string viewID = CVTools::fromQString(CONTEXT.viewID);
		int viewPort = CONTEXT.defaultViewPort;

		if (CONTEXT.drawParam.showNorms)
		{
			const std::string normalID = viewID + "-normal";
			pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
			FROM_PCL_CLOUD(*smCloud, *cloud_normals);
			int normalDensity = CONTEXT.normalDensity > 20 ? CONTEXT.normalDensity : 20;
			float normalScale = CONTEXT.normalScale > 0.02f ? CONTEXT.normalScale : 0.02f;
			if (contains(normalID))
			{
				removePointClouds(normalID, viewPort);
			}
			addPointCloudNormals<pcl::PointNormal>(cloud_normals, normalDensity, normalScale, normalID, viewPort);
			setPointCloudUniqueColor(0.0, 0.0, 1.0, normalID, viewPort);
		}
		else
		{
			const std::string normalID = viewID + "-normal";
			if (contains(normalID))
			{
				removePointClouds(normalID, viewPort);
			}
		}
	}

	void PCLVis::draw(CC_DRAW_CONTEXT& CONTEXT, PCLMesh::Ptr pclMesh)
	{
		std::string viewID = CVTools::fromQString(CONTEXT.viewID);
		int viewPort = CONTEXT.defaultViewPort;
		if (CONTEXT.visFiltering)
		{
			if (contains(viewID))
			{
				removeMesh(viewID, viewPort);
			}
			addPolygonMesh(*pclMesh, viewID, viewPort);
		}
		else
		{
			if (!updatePolygonMesh(*pclMesh, viewID))
			{
				addPolygonMesh(*pclMesh, viewID, viewPort);
			}
		}
	}

	void PCLVis::draw(CC_DRAW_CONTEXT& CONTEXT, PCLTextureMesh::Ptr textureMesh)
	{
		std::string viewID = CVTools::fromQString(CONTEXT.viewID);
		int viewPort = CONTEXT.defaultViewPort;
		if (contains(viewID))
		{
			removeMesh(viewID, viewPort);
		}
		addTextureMesh(*textureMesh, viewID, viewPort);
	}

	void PCLVis::transformEntities(CC_DRAW_CONTEXT & CONTEXT)
	{
		if (CONTEXT.transformInfo.isApplyTransform())
		{
			std::string viewID = CVTools::fromQString(CONTEXT.viewID);
			vtkActor* actor = getActorById(viewID);
			if (actor)
			{
				// remove invalid bbox
				std::string bboxId = CVTools::fromQString(QString("BBox-") + CONTEXT.viewID);
				removeShapes(bboxId, CONTEXT.defaultViewPort);

				TransformInfo transInfo = CONTEXT.transformInfo;
				CCVector3 transVecStart = transInfo.transVecStart;
				CCVector3 transVecEnd = transInfo.transVecEnd;
				CCVector3 scaleXYZ = transInfo.scaleXYZ;
				CCVector3d position = CCVector3d::fromArray(transInfo.position.u);
				ccGLMatrixd rotMat = transInfo.rotMat;

				vtkSmartPointer<vtkTransform> trans = vtkSmartPointer<vtkTransform>::New();
				if (transInfo.isPositionChanged)
				{
					actor->AddPosition(position.u);
				}

				double * origin = actor->GetOrigin();

				trans->PostMultiply();
				trans->Translate(-origin[0], -origin[1], -origin[2]);

				if (transInfo.isScale)
					trans->Scale(scaleXYZ.u);

				if (transInfo.isRotate && transInfo.RotMatrixMode)
				{
					trans->SetMatrix(rotMat.data());
				}
				else if (transInfo.isRotate && !transInfo.RotMatrixMode)
				{
					trans->RotateWXYZ(
						transInfo.rotateParam.angle,
						transInfo.rotateParam.rotAxis.u);
				}

				trans->Translate(origin);

				if (transInfo.isTranslate)
				{
					trans->Translate(transInfo.transVecStart.u);
					trans->Translate(transInfo.transVecEnd.u);
				}

				actor->SetUserTransform(trans);
				actor->Modified();
			}
		}
	}

	void PCLVis::draw(const CC_DRAW_CONTEXT& CONTEXT, PCLPolygon::Ptr pclPolygon, bool closed)
	{
		std::string viewID = CVTools::fromQString(CONTEXT.viewID);
		int viewPort = CONTEXT.defaultViewPort;
		ecvColor::Rgbf polygonColor = ecvTools::TransFormRGB(CONTEXT.defaultPolylineColor);

		removeShape(viewID);

		if (closed)
		{
			addPolygon<PointT>(*pclPolygon, polygonColor.r, polygonColor.g, polygonColor.b, viewID, viewPort);
		}
		else
		{
			addPolyline(pclPolygon, polygonColor.r, polygonColor.g, polygonColor.b, CONTEXT.defaultLineWidth, viewID, viewPort);
		}
	}

	bool PCLVis::updateScalarBar(const CC_DRAW_CONTEXT& CONTEXT)
	{
		std::string viewID = CVTools::fromQString(CONTEXT.viewID);
		int viewPort = CONTEXT.defaultViewPort;
		vtkAbstractWidget* widget = getWidgetById(viewID);
		if (!widget || !PclTools::UpdateScalarBar(widget, CONTEXT))
		{
			this->removeWidgets(viewID, viewPort);
			return false;
		}
		
		return true;
	}
	
	bool PCLVis::addScalarBar(const CC_DRAW_CONTEXT& CONTEXT)
	{
		std::string viewID = CVTools::fromQString(CONTEXT.viewID);
		if (containWidget(viewID))
		{
			CVLog::Warning("[PCLVis::addScalarBar] The id <%s> already exists! Please choose a different id and retry.", viewID.c_str());
			return false;
		}

		vtkSmartPointer<vtkScalarBarWidgetCustom> scalarBarWidget =
			vtkSmartPointer<vtkScalarBarWidgetCustom>::New();
		scalarBarWidget->SetInteractor(getRenderWindowInteractor());
		scalarBarWidget->CreateDefaultRepresentation();
		scalarBarWidget->CreateDefaultScalarBarActor();
		scalarBarWidget->On();

		// Save the pointer/ID pair to the global actor map
		(*m_widget_map)[viewID].widget = scalarBarWidget;

		return updateScalarBar(CONTEXT);
	}

	bool PCLVis::updateCaption(
		const std::string & text,
		const CCVector2& pos2D,
		const CCVector3& anchorPos, 
		double r, double g, double b, double a, 
		int fontSize, const std::string & viewID, int viewport)
	{
		vtkAbstractWidget* widget = getWidgetById(viewID);
		if (!widget) return false;

		CustomVtkCaptionWidget* captionWidget = CustomVtkCaptionWidget::SafeDownCast(widget);
		if (!captionWidget) return false;

		vtkCaptionRepresentation* rep = vtkCaptionRepresentation::SafeDownCast(captionWidget->GetRepresentation());
		if (rep)
		{
			//rep->SetPosition(1.0 * pos2D.x / getRenderWindow()->GetSize()[0], 1.0 * pos2D.y / getRenderWindow()->GetSize()[1]);
			rep->SetAnchorPosition(CCVector3d::fromArray(anchorPos.u).u);
			vtkCaptionActor2D* actor2D = rep->GetCaptionActor2D();
			actor2D->SetCaption(text.c_str());

			vtkTextProperty* textProperty = actor2D->GetTextActor()->GetTextProperty();
			textProperty->SetColor(r, g, b);
			textProperty->SetFontSize(fontSize);
		}

		return true;
	}

	bool PCLVis::addCaption(const std::string & text,
		const CCVector2& pos2D, 
		const CCVector3& anchorPos,
		double r, double g, double b, double a, 
		int fontSize, const std::string & viewID,
		bool anchorDragable, int viewport)
	{
		if (containWidget(viewID))
		{
			CVLog::Warning("[PCLVis::addCaption] The id <%s> already exists! Please choose a different id and retry.", viewID.c_str());
			return false;
		}

		vtkSmartPointer<vtkCaptionRepresentation> captionRepresentation =
			vtkSmartPointer<vtkCaptionRepresentation>::New();
		{
			captionRepresentation->SetAnchorPosition(CCVector3d::fromArray(anchorPos.u).u);
			captionRepresentation->SetPosition(
				1.0 * pos2D.x / getRenderWindow()->GetSize()[0],
				1.0 * pos2D.y / getRenderWindow()->GetSize()[1]);
		}

		vtkCaptionActor2D* actor2D = captionRepresentation->GetCaptionActor2D();
		actor2D->SetCaption(text.c_str());
		actor2D->ThreeDimensionalLeaderOff();
		actor2D->BorderOff();
		actor2D->LeaderOn();
		actor2D->SetLeaderGlyphSize(10.0);
		actor2D->SetMaximumLeaderGlyphSize(10.0);

		const ecvColor::Rgbf& actorColor = ecvColor::FromRgb(ecvColor::red);
		actor2D->GetProperty()->SetColor(actorColor.r, actorColor.g, actorColor.b);
		
		vtkTextProperty* textProperty = actor2D->GetTextActor()->GetTextProperty();
		textProperty->SetColor(r, g, b);
		textProperty->SetFontSize(fontSize);
		const ecvColor::Rgbf& col = ecvColor::FromRgb(ecvColor::white);
		textProperty->SetBackgroundColor(col.r, col.g, col.b);
		textProperty->SetBackgroundOpacity(a);
		textProperty->FrameOn();
		textProperty->SetFrameColor(actorColor.r, actorColor.g, actorColor.b);
		textProperty->SetFrameWidth(2);
		textProperty->BoldOff();
		textProperty->ItalicOff();
		textProperty->SetFontFamilyToArial();
		textProperty->SetJustificationToLeft();
		textProperty->SetVerticalJustificationToCentered();

		vtkSmartPointer<CustomVtkCaptionWidget> captionWidget =
			vtkSmartPointer<CustomVtkCaptionWidget>::New();
		captionWidget->SetHandleEnabled(anchorDragable);
		captionWidget->SetInteractor(getRenderWindowInteractor());
		captionWidget->SetRepresentation(captionRepresentation);
		captionWidget->On(); 

		// Save the pointer/ID pair to the global actor map
		(*m_widget_map)[viewID].widget = captionWidget;
		return true;
	}

	bool PCLVis::addPolyline(
		const PCLPolygon::ConstPtr pclPolygon,
		double r, double g, double b, float width,
		const std::string &id, int viewport)
	{
		vtkSmartPointer<vtkPoints> poly_points = vtkSmartPointer<vtkPoints>::New();
		poly_points->SetNumberOfPoints(pclPolygon->getContour().size());
		size_t i;
		for (i = 0; i < pclPolygon->getContour().size(); ++i)
		{
			poly_points->SetPoint(i,
				pclPolygon->getContour()[i].x,
				pclPolygon->getContour()[i].y,
				pclPolygon->getContour()[i].z);
		}

		// Check to see if this ID entry already exists (has it been already added to the visualizer?)
		pcl::visualization::ShapeActorMap::iterator am_it = getShapeActorMap()->find(id);
		if (am_it != getShapeActorMap()->end())
		{
			CVLog::Warning("[addPolyline] A shape with id <%s> already exists! Please choose a different id and retry.", id.c_str());
			return (false);
		}

		if (poly_points->GetNumberOfPoints() < 2)
		{
			CVLog::Warning("[addPolyline] point size less 2.");
			return (false);
		}

		vtkSmartPointer<vtkDataSet> data = PclTools::CreateLine(poly_points);

		// Create an Actor
		vtkSmartPointer<vtkLODActor> actor;
		PclTools::CreateActorFromVTKDataSet(data, actor);
		actor->GetProperty()->SetRepresentationToSurface();
		actor->GetProperty()->SetLineWidth(width);
		actor->GetProperty()->SetColor(r, g, b);
		addActorToRenderer(actor, viewport);

		// Save the pointer/ID pair to the global actor map
		(*getShapeActorMap())[id] = actor;
		return (true);
	}

	void PCLVis::displayText(const CC_DRAW_CONTEXT& CONTEXT)
	{
		ecvTextParam textParam = CONTEXT.textParam;
		if (textParam.text.isEmpty())
		{
			CVLog::Warning("empty text detected!");
			return;
		}
		std::string text = CVTools::fromQString(textParam.text);
		//std::string viewID = CVTools::fromQString(CONTEXT.viewID);
		std::string viewID = text;

		int viewPort = CONTEXT.defaultViewPort;
		int xPos = static_cast<int>(textParam.textPos.x);
		int yPos = static_cast<int>(textParam.textPos.y);
		ecvColor::Rgbf textColor = ecvTools::TransFormRGB(CONTEXT.textDefaultCol);
		if (textParam.display3D)
		{
			pcl::PointXYZ position;
			position.x = textParam.textPos.x;
			position.y = textParam.textPos.y;
			position.z = textParam.textPos.z;
			if (contains(viewID))
			{
				removeShapes(viewID, viewPort);
			}

			addText3D(text, position, textParam.textScale,
				textColor.r, textColor.g, textColor.b, viewID, viewPort);
		}
		else
		{
			if (!updateText(text, xPos, yPos, viewID))
			{
				addText(text, xPos, yPos, textParam.font.pointSize(),
					textColor.r, textColor.g, textColor.b, viewID, viewPort);
			}
		}
	}

	int PCLVis::textureFromTexMaterial(const pcl::TexMaterial& tex_mat, vtkTexture* vtk_tex) const
	{
		if (tex_mat.tex_file == "")
		{
			return (-1);
		}

		boost::filesystem::path full_path(tex_mat.tex_file.c_str());
		if (!boost::filesystem::exists(full_path))
		{
			boost::filesystem::path parent_dir = full_path.parent_path();
			std::string upper_filename = tex_mat.tex_file;
			boost::to_upper(upper_filename);
			std::string real_name = "";

			try
			{
				if (!boost::filesystem::exists(parent_dir))
				{
					CVLog::Warning("[PCLVis::textureFromTexMaterial] Parent directory '%s' doesn't exist!",
						parent_dir.string().c_str());
					return (-1);
				}

				if (!boost::filesystem::is_directory(parent_dir))
				{
					CVLog::Warning("[PCLVis::textureFromTexMaterial] Parent '%s' is not a directory !",
						parent_dir.string().c_str());
					return (-1);
				}

				typedef std::vector<boost::filesystem::path> paths_vector;
				paths_vector paths;
				std::copy(boost::filesystem::directory_iterator(parent_dir),
					boost::filesystem::directory_iterator(),
					back_inserter(paths));

				for (paths_vector::const_iterator it = paths.begin(); it != paths.end(); ++it)
				{
					if (boost::filesystem::is_regular_file(*it))
					{
						std::string name = it->string();
						boost::to_upper(name);
						if (name == upper_filename)
						{
							real_name = it->string();
							break;
						}
					}
				}
				// Check texture file existence
				if (real_name == "")
				{
					CVLog::Warning("[PCLVis::textureFromTexMaterial] Can not find texture file %s!",
						tex_mat.tex_file.c_str());
					return (-1);
				}
			}
			catch (const boost::filesystem::filesystem_error& ex)
			{

				CVLog::Warning("[PCLVis::textureFromTexMaterial] Error %s when looking for file %s!",
					ex.what(), tex_mat.tex_file.c_str());
				return (-1);
			}

			//Save the real path
			full_path = real_name.c_str();
		}

		std::string extension = full_path.extension().string();
		//!!! nizar 20131206 : The list is far from being exhaustive I am afraid.
		if ((extension == ".jpg") || (extension == ".JPG"))
		{
			vtkSmartPointer<vtkJPEGReader> jpeg_reader = vtkSmartPointer<vtkJPEGReader>::New();
			jpeg_reader->SetFileName(full_path.string().c_str());
			jpeg_reader->Update();
			vtk_tex->SetInputConnection(jpeg_reader->GetOutputPort());
		}
		else if ((extension == ".bmp") || (extension == ".BMP"))
		{
			vtkSmartPointer<vtkBMPReader> bmp_reader = vtkSmartPointer<vtkBMPReader>::New();
			bmp_reader->SetFileName(full_path.string().c_str());
			bmp_reader->Update();
			vtk_tex->SetInputConnection(bmp_reader->GetOutputPort());
		}
		else if ((extension == ".pnm") || (extension == ".PNM"))
		{
			vtkSmartPointer<vtkPNMReader> pnm_reader = vtkSmartPointer<vtkPNMReader>::New();
			pnm_reader->SetFileName(full_path.string().c_str());
			pnm_reader->Update();
			vtk_tex->SetInputConnection(pnm_reader->GetOutputPort());
		}
		else if ((extension == ".png") || (extension == ".PNG"))
		{
			vtkSmartPointer<vtkPNGReader> png_reader = vtkSmartPointer<vtkPNGReader>::New();
			png_reader->SetFileName(full_path.string().c_str());
			png_reader->Update();
			vtk_tex->SetInputConnection(png_reader->GetOutputPort());
		}
		else if ((extension == ".tiff") || (extension == ".TIFF"))
		{
			vtkSmartPointer<vtkTIFFReader> tiff_reader = vtkSmartPointer<vtkTIFFReader>::New();
			tiff_reader->SetFileName(full_path.string().c_str());
			tiff_reader->Update();
			vtk_tex->SetInputConnection(tiff_reader->GetOutputPort());
		}
		else
		{
			CVLog::Warning("[PCLVis::textureFromTexMaterial] Unhandled image %s for material %s!",
				full_path.c_str(), tex_mat.tex_name.c_str());
			return (-1);
		}

		return (1);
	}

	bool PCLVis::addTextureMesh(const PCLTextureMesh &mesh, const std::string &id, int viewport)
	{
		pcl::visualization::CloudActorMap::iterator am_it = getCloudActorMap()->find(id);
		if (am_it != getCloudActorMap()->end())
		{
			CVLog::Error("[PCLVis::addTextureMesh] A shape with id <%s> already exists!"
				" Please choose a different id and retry.",
				id.c_str());
			return (false);
		}
		// no texture materials --> exit
		if (mesh.tex_materials.size() == 0)
		{
			CVLog::Error("[PCLVis::addTextureMesh] No textures found!");
			return (false);
		}
		// polygons are mapped to texture materials
		if (mesh.tex_materials.size() != mesh.tex_polygons.size())
		{
			CVLog::Error("[PCLVis::addTextureMesh] Materials number %lu differs from polygons number %lu!",
				mesh.tex_materials.size(), mesh.tex_polygons.size());
			return (false);
		}
		// each texture material should have its coordinates set
		if (mesh.tex_materials.size() != mesh.tex_coordinates.size())
		{
			CVLog::Error("[PCLVis::addTextureMesh] Coordinates number %lu differs from materials number %lu!",
				mesh.tex_coordinates.size(), mesh.tex_materials.size());
			return (false);
		}
		// total number of vertices
		std::size_t nb_vertices = 0;
		for (std::size_t i = 0; i < mesh.tex_polygons.size(); ++i)
			nb_vertices += mesh.tex_polygons[i].size();
		// no vertices --> exit
		if (nb_vertices == 0)
		{
			CVLog::Error("[PCLVis::addTextureMesh] No vertices found!");
			return (false);
		}
		// total number of coordinates
		std::size_t nb_coordinates = 0;
		for (std::size_t i = 0; i < mesh.tex_coordinates.size(); ++i)
			nb_coordinates += mesh.tex_coordinates[i].size();
		
		// no texture coordinates --> exit
		if (nb_coordinates == 0)
		{
			CVLog::Error("[PCLVis::addTextureMesh] No textures coordinates found!");
			return (false);
		}

		// Create points from mesh.cloud
		vtkSmartPointer<vtkPoints> poly_points = vtkSmartPointer<vtkPoints>::New();
		vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
		bool has_color = false;
		vtkSmartPointer<vtkMatrix4x4> transformation = vtkSmartPointer<vtkMatrix4x4>::New();
		if ((pcl::getFieldIndex(mesh.cloud, "rgba") != -1) ||
			(pcl::getFieldIndex(mesh.cloud, "rgb") != -1))
		{
			pcl::PointCloud<pcl::PointXYZRGB> cloud;
			pcl::fromPCLPointCloud2(mesh.cloud, cloud);
			if (cloud.points.size() == 0)
			{
				CVLog::Error("[PCLVis::addTextureMesh] Cloud is empty!");
				return (false);
			}
			convertToVtkMatrix(cloud.sensor_origin_, cloud.sensor_orientation_, transformation);
			has_color = true;
			colors->SetNumberOfComponents(3);
			colors->SetName("Colors");
			poly_points->SetNumberOfPoints(cloud.size());
			for (std::size_t i = 0; i < cloud.points.size(); ++i)
			{
				const pcl::PointXYZRGB &p = cloud.points[i];
				poly_points->InsertPoint(i, p.x, p.y, p.z);
				const unsigned char color[3] = { p.r, p.g, p.b };
				colors->InsertNextTupleValue(color);
			}
		}
		else
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
			// no points --> exit
			if (cloud->points.size() == 0)
			{
				CVLog::Error("[PCLVis::addTextureMesh] Cloud is empty!");
				return (false);
			}
			convertToVtkMatrix(cloud->sensor_origin_, cloud->sensor_orientation_, transformation);
			poly_points->SetNumberOfPoints(cloud->points.size());
			for (std::size_t i = 0; i < cloud->points.size(); ++i)
			{
				const pcl::PointXYZ &p = cloud->points[i];
				poly_points->InsertPoint(i, p.x, p.y, p.z);
			}
		}

		//create polys from polyMesh.tex_polygons
		vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
		for (const auto& tex_polygon : mesh.tex_polygons)
		{
			for (const auto& vertex : tex_polygon)
			{
				polys->InsertNextCell(static_cast<int> (vertex.vertices.size()));
				for (const auto& point : vertex.vertices)
					polys->InsertCellPoint(point);
			}
		}

		vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
		polydata->SetPolys(polys);
		polydata->SetPoints(poly_points);
		if (has_color)
			polydata->GetPointData()->SetScalars(colors);

		vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		mapper->SetInputData(polydata);

		vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
		vtkTextureUnitManager* tex_manager = vtkOpenGLRenderWindow::SafeDownCast(getRenderWindow())->GetTextureUnitManager();
		if (!tex_manager)
			return (false);
		// hardware always supports multitexturing of some degree
		int texture_units = tex_manager->GetNumberOfTextureUnits();
		if ((size_t)texture_units < mesh.tex_materials.size())
			CVLog::Warning("[PCLVis::addTextureMesh] GPU texture units %d < mesh textures %d!",
				texture_units, mesh.tex_materials.size());
		// Load textures
		std::size_t last_tex_id = std::min(static_cast<int> (mesh.tex_materials.size()), texture_units);
		std::size_t tex_id = 0;
		while (tex_id < last_tex_id)
		{
#if (VTK_MAJOR_VERSION == 8 && VTK_MINOR_VERSION >= 2) || VTK_MAJOR_VERSION > 8
			const char *tu = mesh.tex_materials[tex_id].tex_name.c_str();
#else
			int tu = vtkProperty::VTK_TEXTURE_UNIT_0 + tex_id;
#endif
			vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
			if (!textureFromTexMaterial(mesh.tex_materials[tex_id], texture))
			{
				CVLog::Warning("[PCLVisualizer::addTextureMesh] Failed to load texture %s, skipping!\n",
					mesh.tex_materials[tex_id].tex_name.c_str());
				continue;
			}
			// the first texture is in REPLACE mode others are in ADD mode
			if (tex_id == 0)
				texture->SetBlendingMode(vtkTexture::VTK_TEXTURE_BLENDING_MODE_REPLACE);
			else
				texture->SetBlendingMode(vtkTexture::VTK_TEXTURE_BLENDING_MODE_ADD);
			// add a texture coordinates array per texture
			vtkSmartPointer<vtkFloatArray> coordinates = vtkSmartPointer<vtkFloatArray>::New();
			coordinates->SetNumberOfComponents(2);
			std::stringstream ss; ss << "TCoords" << tex_id;
			std::string this_coordinates_name = ss.str();
			coordinates->SetName(this_coordinates_name.c_str());

			for (std::size_t t = 0; t < mesh.tex_coordinates.size(); ++t)
				if (t == tex_id)
					for (std::size_t tc = 0; tc < mesh.tex_coordinates[t].size(); ++tc)
						coordinates->InsertNextTuple2(mesh.tex_coordinates[t][tc][0],
							mesh.tex_coordinates[t][tc][1]);
				else
					for (std::size_t tc = 0; tc < mesh.tex_coordinates[t].size(); ++tc)
						coordinates->InsertNextTuple2(-1.0, -1.0);
			mapper->MapDataArrayToMultiTextureAttribute(tu,
				this_coordinates_name.c_str(),
				vtkDataObject::FIELD_ASSOCIATION_POINTS);

			polydata->GetPointData()->AddArray(coordinates);
			actor->GetProperty()->SetTexture(tu, texture);
			const PCLMaterial::RGB & ambientColor = mesh.tex_materials[tex_id].tex_Ka;
			const PCLMaterial::RGB & diffuseColor = mesh.tex_materials[tex_id].tex_Kd;
			const PCLMaterial::RGB & specularColor = mesh.tex_materials[tex_id].tex_Ks;
			actor->GetProperty()->SetAmbientColor(ambientColor.r, ambientColor.g, ambientColor.b);
			actor->GetProperty()->SetDiffuseColor(diffuseColor.r, diffuseColor.g, diffuseColor.b);
			actor->GetProperty()->SetSpecularColor(specularColor.r, specularColor.g, specularColor.b);
            actor->GetProperty()->SetMaterialName(mesh.tex_materials[tex_id].tex_name.c_str());
			++tex_id;
		}

		// set mapper
		actor->SetMapper(mapper);
		addActorToRenderer(actor, viewport);

		// Save the pointer/ID pair to the global actor map
		(*getCloudActorMap())[id].actor = actor;

		// Save the viewpoint transformation matrix to the global actor map
		(*getCloudActorMap())[id].viewpoint_transformation_ = transformation;

		return (true);
	}

	bool PCLVis::addOrientedCube(
			const Eigen::Vector3f &translation, const Eigen::Quaternionf &rotation,
			double width, double height, double depth, double r, double g, double b,
			const std::string &id, int viewport)
	{
		// Check to see if this ID entry already exists (has it been already added to the visualizer?)
		pcl::visualization::ShapeActorMap::iterator am_it = getShapeActorMap()->find(id);
		if (am_it != getShapeActorMap()->end())
		{
			CVLog::Error("[PCLVis::addCube] A shape with id <%s> already exists!"
				" Please choose a different id and retry.",
				id.c_str());
			return (false);
		}
		
		vtkSmartPointer<vtkDataSet> data = pcl::visualization::createCube(translation, rotation, width, height, depth);

		// Create an Actor
		vtkSmartPointer<vtkLODActor> actor;
		PclTools::CreateActorFromVTKDataSet(data, actor);
		actor->GetProperty()->SetRepresentationToSurface();
		actor->GetProperty()->SetColor(r, g, b);
		addActorToRenderer(actor, viewport);

		// Save the pointer/ID pair to the global actor map
		(*getShapeActorMap())[id] = actor;

		return (true);
	}
	/********************************Draw Entities*********************************/

	/******************************** Entities Removement *********************************/
	void PCLVis::hideShowActors(bool visibility, const std::string & viewID, int viewport)
	{
		double opacity = visibility ? 1.0 : 0.0;
		vtkActor* actor = getActorById(viewID);
		if (actor)
		{
			actor->SetVisibility(opacity);
			actor->Modified();
		}
	}

	void PCLVis::hideShowWidgets(bool visibility, const std::string & viewID, int viewport)
	{
		// Extra step: check if there is a widget with the same ID
		WidgetActorMap::iterator wa_it = getWidgetActorMap()->find(viewID);
		if (wa_it == getWidgetActorMap()->end()) return;

		assert(wa_it->second.widget);

		visibility ? wa_it->second.widget->On() : wa_it->second.widget->Off();
	}

	bool PCLVis::removeEntities(const CC_DRAW_CONTEXT& CONTEXT)
	{
		std::string removeViewID = CVTools::fromQString(CONTEXT.removeViewID);

		bool removeFlag = false;

		int viewPort = CONTEXT.defaultViewPort;
		switch (CONTEXT.removeEntityType)
		{
		case ENTITY_TYPE::ECV_POINT_CLOUD:
		{
			removePointClouds(removeViewID, viewPort);
			removeFlag = true;
			break;
		}
		case ENTITY_TYPE::ECV_MESH:
		{
			removeMesh(removeViewID, viewPort);
			removePointClouds(removeViewID, viewPort);
			removeFlag = true;
		}
		break;
		case ENTITY_TYPE::ECV_LINES_3D:
		case ENTITY_TYPE::ECV_POLYLINE_2D:
		case ENTITY_TYPE::ECV_SHAPE:
		{
			removeShapes(removeViewID, viewPort);
			removePointClouds(removeViewID, viewPort);
			removeFlag = true;
		}
		break;
		case ENTITY_TYPE::ECV_TEXT3D:
		{
			removeText3D(removeViewID, viewPort);
			removeFlag = true;
			break;
		}
		case ENTITY_TYPE::ECV_TEXT2D:
		{
			removeText2D(removeViewID, viewPort);
			break;
		}
		case ENTITY_TYPE::ECV_CAPTION:
		case ENTITY_TYPE::ECV_SCALAR_BAR:
		{
			removeWidgets(removeViewID, viewPort);
		}
		break;
		case ENTITY_TYPE::ECV_ALL:
		{
			removeALL(viewPort);
			removeFlag = true;
			break;
		}
		default:
			break;
		}

		return removeFlag;
	}

	bool PCLVis::removeWidgets(const std::string & viewId, int viewport)
	{
		// Check to see if the given ID entry exists
		PclUtils::WidgetActorMap::iterator wa_it = m_widget_map->find(viewId);

		if (wa_it == m_widget_map->end())
			return (false);

		// Remove it from all renderers
		if (wa_it->second.widget)
		{
			// Remove the pointer/ID pair to the global actor map
			wa_it->second.widget->Off();
			m_widget_map->erase(wa_it);
			return (true);
		}
		return (false);
	}

	void PCLVis::removePointClouds(const std::string & viewId, int viewport)
	{
		if (contains(viewId))
		{
			removePointCloud(viewId, viewport);
		}

		// for normals case
		std::string normalViewId = viewId + "-normal";
		if (contains(normalViewId))
		{
			removePointCloud(normalViewId, viewport);
		}
	}

	void PCLVis::removeShapes(const std::string & viewId, int viewport)
	{
		if (contains(viewId))
		{
			removeShape(viewId, viewport);
		}
	}

	void PCLVis::removeMesh(const std::string & viewId, int viewport)
	{
		if (contains(viewId))
		{
			removePolygonMesh(viewId, viewport);
		}
	}

	void PCLVis::removeText3D(const std::string & viewId, int viewport)
	{
		if (contains(viewId))
		{
			removeText3D(viewId, viewport);
		}
	}

	void PCLVis::removeText2D(const std::string & viewId, int viewport)
	{
		if (contains(viewId))
		{
			removeShapes(viewId, viewport);
		}
	}

	void PCLVis::removeALL(int viewport)
	{
		removeAllPointClouds(viewport);
		removeAllShapes(viewport);
	}
	/******************************** Entities Removement *********************************/

	/******************************** Entities Setting *********************************/
	void PCLVis::setPointCloudUniqueColor(float r, float g, float b, const std::string &viewID, int viewPort)
	{
		if (contains(viewID))
		{
			setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, viewID, viewPort);
		}
	}

	void PCLVis::resetScalarColor(const std::string &viewID, bool flag/* = true*/, int viewPort/* = 0*/)
	{
		vtkLODActor* actor = vtkLODActor::SafeDownCast(getActorById(viewID));

		if (actor)
		{
			if (flag)
			{
				actor->GetMapper()->ScalarVisibilityOn();
			}
			else
			{
				actor->GetMapper()->ScalarVisibilityOff();
			}
			actor->Modified();
		}
	}

	void PCLVis::setShapeUniqueColor(float r, float g, float b, const std::string &viewID, int viewPort)
	{
		if (contains(viewID))
		{
			setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, viewID, viewPort);
		}
	}

	void PCLVis::setPointSize(const unsigned char pointSize, const std::string & viewID, int viewport)
	{
		unsigned char size = pointSize > 16 ? 16 : pointSize;
		size = pointSize < 1 ? 1 : pointSize;
		if (contains(viewID))
		{
			setPointCloudRenderingProperties(
				pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, viewID, viewport);
		}
	}

	void PCLVis::setPointCloudOpacity(double opacity, const std::string & viewID, int viewport)
	{
		if (contains(viewID))
		{
			double lastOpacity;
			getPointCloudRenderingProperties(
				pcl::visualization::RenderingProperties::PCL_VISUALIZER_OPACITY, lastOpacity, viewID);
			if (opacity != lastOpacity)
			{
				setPointCloudRenderingProperties(
					pcl::visualization::RenderingProperties::PCL_VISUALIZER_OPACITY, opacity, viewID, viewport);
			}
		}
	}

	void PCLVis::setShapeOpacity(double opacity, const std::string & viewID, int viewport)
	{
		if (contains(viewID))
		{
			setShapeRenderingProperties(
				pcl::visualization::RenderingProperties::PCL_VISUALIZER_OPACITY, opacity, viewID, viewport);
		}
	}

	void PCLVis::setMeshRenderingMode(MESH_RENDERING_MODE mode, const std::string & viewID, int viewport)
	{
		vtkActor* actor = getActorById(viewID);
		if (!actor)
		{
			CVLog::Warning("[PCLVis::SetMeshRenderingMode] Requested viewID not found, please check again...");
			return;
		}
		switch (mode)
		{
		case MESH_RENDERING_MODE::ECV_POINTS_MODE:
			actor->GetProperty()->SetRepresentationToPoints();
			actor->Modified();
			break;
		case MESH_RENDERING_MODE::ECV_WIREFRAME_MODE:
			actor->GetProperty()->SetRepresentationToWireframe();
			actor->Modified();
			break;
		case MESH_RENDERING_MODE::ECV_SURFACE_MODE:
			actor->GetProperty()->SetRepresentationToSurface();
			actor->Modified();
			break;
		default:
			break;
		}
	}

	void PCLVis::setLightMode(const std::string & viewID, int viewport)
	{
		vtkActor* actor = getActorById(viewID);
		if (!actor)
		{
			CVLog::Warning(tr("[PCLVis::SetLightMode] Requested viewID [%1] not found ...").arg(viewID.c_str()));
			return;
		}
		//actor->GetProperty()->SetAmbient(1.0);
		actor->GetProperty()->SetLighting(false);
	}

	void PCLVis::setLineWidth(const unsigned char lineWidth, const std::string & viewID, int viewport)
	{
		if (contains(viewID))
		{
			setShapeRenderingProperties(
				pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, lineWidth, viewID, viewport);
		}
	}
	/******************************** Entities Setting *********************************/

	/******************************** Camera Tools *********************************/
	pcl::visualization::Camera PCLVis::getCamera(int viewport)
	{
		// get camera param in all viewport
		pcl::visualization::Camera camera;
		this->getCameraParameters(camera);
		return camera;
	}

	vtkSmartPointer<vtkCamera> PCLVis::getVtkCamera(int viewport)
	{
		return getCurrentRenderer()->GetActiveCamera();
	}

	double PCLVis::getParallelScale()
	{
		vtkSmartPointer<vtkCamera> cam = this->getRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera();
		return cam->GetParallelScale();
	}

	void PCLVis::setOrthoProjection(int viewport)
	{
		vtkSmartPointer<vtkCamera> cam = getVtkCamera();
		int flag = cam->GetParallelProjection();
		if (!flag)
		{
			cam->SetParallelProjection(true);
			getCurrentRenderer()->SetActiveCamera(cam);
			getRenderWindow()->Render();
		}
	}

	void PCLVis::setPerspectiveProjection(int viewport)
	{
		vtkSmartPointer<vtkCamera> cam = getVtkCamera();
		int flag = cam->GetParallelProjection();
		if (flag)
		{
			cam->SetParallelProjection(false);
			getCurrentRenderer()->SetActiveCamera(cam);
			getRenderWindow()->Render();
		}
	}

	/******************************** Camera Tools *********************************/

	/******************************** Util Tools *********************************/
	void PCLVis::setAreaPickingMode(bool state)
	{
		if (state)
		{
			if (m_currentMode == ORIENT_MODE)
			{
				m_currentMode = SELECT_MODE;
				// Save the point picker
				m_point_picker = static_cast<vtkPointPicker*> (getInteractorStyle()->GetInteractor()->GetPicker());
				// Switch for an area picker
				m_area_picker = vtkSmartPointer<vtkAreaPicker>::New();
				getInteractorStyle()->GetInteractor()->SetPicker(m_area_picker);
			}
		}
		else
		{
			if (m_currentMode == SELECT_MODE)
			{
				m_currentMode = ORIENT_MODE;
				// Restore point picker
				getInteractorStyle()->GetInteractor()->SetPicker(m_point_picker);
			}
		}

		getRendererCollection()->Render();
		getInteractorStyle()->GetInteractor()->Render();
	}

    void PCLVis::toggleAreaPicking()
	{
		setAreaPickingEnabled(!isAreaPickingEnabled());
		if (this->ThreeDInteractorStyle)
		{
			this->ThreeDInteractorStyle->toggleAreaPicking();
		}
	}

	void PCLVis::exitCallbackProcess()
	{
		getPCLInteractorStyle()->GetInteractor()->ExitCallback();
	}
	/******************************** Util Tools *********************************/

	/********************************MarkerAxes*********************************/
	void PCLVis::showPclMarkerAxes(vtkRenderWindowInteractor* interactor)
	{
		if (!interactor) return;
		showOrientationMarkerWidgetAxes(interactor);
		CVLog::Print("Show Orientation Marker Widget Axes!");
	}

	void PCLVis::hidePclMarkerAxes()
	{
		//removeOrientationMarkerWidgetAxes();
		hideOrientationMarkerWidgetAxes();
		CVLog::Print("Hide Orientation Marker Widget Axes!");
	}

	void PCLVis::hideOrientationMarkerWidgetAxes()
	{
		if (m_axes_widget)
		{
			if (m_axes_widget->GetEnabled())
				m_axes_widget->SetEnabled(false);
			else
				CVLog::Warning("Orientation Widget Axes was already disabled, doing nothing.");
		}
		else
		{
			CVLog::Warning("Attempted to delete Orientation Widget Axes which does not exist!");
		}
	}

	void PCLVis::showOrientationMarkerWidgetAxes(vtkRenderWindowInteractor* interactor)
	{
		if (!m_axes_widget)
		{
			vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
			axes->SetShaftTypeToCylinder();
			axes->SetXAxisLabelText("x");
			axes->SetYAxisLabelText("y");
			axes->SetZAxisLabelText("z");
			axes->SetTotalLength(1.5, 1.5, 1.5);

			vtkSmartPointer<vtkAnnotatedCubeActor> cube = vtkSmartPointer<vtkAnnotatedCubeActor>::New();
			cube->SetXPlusFaceText("R");
			cube->SetXMinusFaceText("L");
			cube->SetYPlusFaceText("A");
			cube->SetYMinusFaceText("P");
			cube->SetZPlusFaceText("I");
			cube->SetZMinusFaceText("S");
			cube->SetXFaceTextRotation(180);
			cube->SetYFaceTextRotation(180);
			cube->SetZFaceTextRotation(-90);
			cube->SetFaceTextScale(0.65);
			cube->GetCubeProperty()->SetColor(0.5, 1, 1);
			cube->GetTextEdgesProperty()->SetLineWidth(1);
			cube->GetTextEdgesProperty()->SetDiffuse(0);
			cube->GetTextEdgesProperty()->SetAmbient(1);
			cube->GetTextEdgesProperty()->SetColor(0.1800, 0.2800, 0.2300);
			// this static function improves the appearance of the text edges
			// since they are overlaid on a surface rendering of the cube's faces
			vtkMapper::SetResolveCoincidentTopologyToPolygonOffset();

			cube->GetXPlusFaceProperty()->SetColor(1, 0, 0);
			cube->GetXPlusFaceProperty()->SetInterpolationToFlat();
			cube->GetXMinusFaceProperty()->SetColor(1, 0, 0);
			cube->GetXMinusFaceProperty()->SetInterpolationToFlat();
			cube->GetYPlusFaceProperty()->SetColor(0, 1, 0);
			cube->GetYPlusFaceProperty()->SetInterpolationToFlat();
			cube->GetYMinusFaceProperty()->SetColor(0, 1, 0);
			cube->GetYMinusFaceProperty()->SetInterpolationToFlat();
			cube->GetZPlusFaceProperty()->SetColor(0, 0, 1);
			cube->GetZPlusFaceProperty()->SetInterpolationToFlat();
			cube->GetZMinusFaceProperty()->SetColor(0, 0, 1);
			cube->GetZMinusFaceProperty()->SetInterpolationToFlat();

			vtkSmartPointer<vtkTextProperty> tprop4 = vtkSmartPointer<vtkTextProperty>::New();
			// tprop.ItalicOn();
			tprop4->ShadowOn();
			tprop4->SetFontFamilyToArial();

			// tprop.SetFontFamilyToTimes();
			axes->GetXAxisCaptionActor2D()->SetCaptionTextProperty(tprop4);
			//
			vtkSmartPointer<vtkTextProperty> tprop2 = vtkSmartPointer<vtkTextProperty>::New();
			tprop2->ShallowCopy(tprop4);
			axes->GetYAxisCaptionActor2D()->SetCaptionTextProperty(tprop2);
			//
			vtkSmartPointer<vtkTextProperty> tprop3 = vtkSmartPointer<vtkTextProperty>::New();
			tprop3->ShallowCopy(tprop4);
			axes->GetZAxisCaptionActor2D()->SetCaptionTextProperty(tprop3);

			vtkSmartPointer<vtkPropAssembly> assembly = vtkSmartPointer<vtkPropAssembly>::New();
			assembly->AddPart(cube);
			assembly->AddPart(axes);

			m_axes_widget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
			m_axes_widget->SetOutlineColor(0.9300, 0.5700, 0.1300);
			m_axes_widget->SetOrientationMarker(assembly);
			m_axes_widget->SetInteractor(interactor);
			m_axes_widget->SetViewport(0.8, 0.0, 1.0, 0.2);
			m_axes_widget->SetEnabled(true);
			m_axes_widget->InteractiveOff();
		}
		else
		{
			m_axes_widget->SetEnabled(true);
			CVLog::Print("Show Orientation Marker Widget Axes!");
		}
	}

	void PCLVis::toggleOrientationMarkerWidgetAxes()
	{
		if (m_axes_widget)
		{
			if (m_axes_widget->GetEnabled())
			{
				m_axes_widget->SetEnabled(false);
				CVLog::Print("Hide Orientation Marker Widget Axes!");
			}
			else
			{
				m_axes_widget->SetEnabled(true);
				CVLog::Print("Show Orientation Marker Widget Axes!");
			}
		}
		else
		{
			CVLog::Warning("Attempted to delete Orientation Widget Axes which does not exist!");
		}
	}
	/********************************MarkerAxes*********************************/

	/********************************Actor Function*********************************/
	vtkRenderer * PCLVis::getCurrentRenderer()
	{
		return getRendererCollection()->GetFirstRenderer();
	}

	vtkSmartPointer<pcl::visualization::PCLVisualizerInteractorStyle>
		PCLVis::getPCLInteractorStyle()
	{
		return getInteractorStyle();
	}

	vtkActor* PCLVis::getActorById(const std::string & viewId)
	{
		// Check to see if this ID entry already exists (has it been already added to the visualizer?)
		 // Check to see if the given ID entry exists
		pcl::visualization::CloudActorMap::iterator ca_it = getCloudActorMap()->find(viewId);
		// Extra step: check if there is a cloud with the same ID
		pcl::visualization::ShapeActorMap::iterator am_it = getShapeActorMap()->find(viewId);

		bool shape = true;
		// Try to find a shape first
		if (am_it == getShapeActorMap()->end())
		{
			// There is no cloud or shape with this ID, so just exit
			if (ca_it == getCloudActorMap()->end())
			{
#ifdef QT_DEBUG
				CVLog::Error("[getActorById] Could not find any PointCloud or Shape datasets with id <%s>!", viewId.c_str());
#endif
				return nullptr;
			}

			// Cloud found, set shape to false
			shape = false;
		}

		vtkActor* actor;

		// Remove the pointer/ID pair to the global actor map
		if (shape)
		{
			actor = vtkLODActor::SafeDownCast(am_it->second);
			if (!actor)
			{
				actor = vtkActor::SafeDownCast(am_it->second);
			}
		}
		else
		{
			actor = vtkLODActor::SafeDownCast(ca_it->second.actor);
			if (!actor)
			{
				actor = vtkActor::SafeDownCast(ca_it->second.actor);
			}
		}

		// Get the actor pointer

		if (!actor)
			return nullptr;

		return actor;
	}

	vtkAbstractWidget* PCLVis::getWidgetById(const std::string & viewId)
	{
		// Extra step: check if there is a widget with the same ID
		WidgetActorMap::iterator wa_it = getWidgetActorMap()->find(viewId);
		if (wa_it == getWidgetActorMap()->end()) return nullptr;
		return wa_it->second.widget;
	}

	std::string PCLVis::getIdByActor(vtkActor* actor)
	{
		// Check to see if this ID entry already exists (has it been already added to the visualizer?)
		 // Check to see if the given ID entry exists
		pcl::visualization::CloudActorMap::iterator cloudIt = getCloudActorMap()->begin();
		// Extra step: check if there is a cloud with the same ID
		pcl::visualization::ShapeActorMap::iterator shapeIt = getShapeActorMap()->begin();

		for (; cloudIt != getCloudActorMap()->end(); cloudIt++)
		{
			vtkActor* tempActor = vtkLODActor::SafeDownCast(cloudIt->second.actor);
			if (tempActor && tempActor == actor)
			{
				return cloudIt->first;
			}
		}

		for (; shapeIt != getShapeActorMap()->end(); shapeIt++)
		{
			vtkActor* tempActor = vtkLODActor::SafeDownCast(shapeIt->second);
			if (tempActor && tempActor == actor)
			{
				return shapeIt->first;
			}
		}

		return std::string("");
	}

	bool PCLVis::removeActorFromRenderer(const vtkSmartPointer<vtkProp> &actor, int viewport)
	{
		vtkProp* actor_to_remove = vtkProp::SafeDownCast(actor);

		// Initialize traversal
		getRendererCollection()->InitTraversal();
		vtkRenderer* renderer = NULL;
		int i = 0;
		while ((renderer = getRendererCollection()->GetNextItem()) != NULL)
		{
			// Should we remove the actor from all renderers?
			if (viewport == 0)
			{
				renderer->RemoveActor(actor);
			}
			else if (viewport == i)               // add the actor only to the specified viewport
			{
				// Iterate over all actors in this renderer
				vtkPropCollection* actors = renderer->GetViewProps();
				actors->InitTraversal();
				vtkProp* current_actor = NULL;
				while ((current_actor = actors->GetNextProp()) != NULL)
				{
					if (current_actor != actor_to_remove)
						continue;
					renderer->RemoveActor(actor);
					// Found the correct viewport and removed the actor
					return (true);
				}
			}
			++i;
		}
		if (viewport == 0) return (true);
		return (false);
	}

	void PCLVis::addActorToRenderer(const vtkSmartPointer<vtkProp> &actor, int viewport) {
		// Add it to all renderers
		getRendererCollection()->InitTraversal();
		vtkRenderer* renderer = NULL;
		int i = 0;
		while ((renderer = getRendererCollection()->GetNextItem()) != NULL)
		{
			// Should we add the actor to all renderers?
			if (viewport == 0)
			{
				renderer->AddActor(actor);
			}
			else if (viewport == i)  // add the actor only to the specified viewport
			{
				renderer->AddActor(actor);
			}
			++i;
		}
	}

    void PCLVis::setupInteractor(vtkRenderWindowInteractor* iren, vtkRenderWindow* win)
	{
		this->interactor_ = iren;
		pcl::visualization::PCLVisualizer::setupInteractor(iren, win);
	}
	/********************************Actor Function*********************************/

	/********************************Interactor Function*********************************/

	void PCLVis::registerKeyboard()
	{
		m_cloud_mutex.lock();    // for not overwriting the point m_baseCloud
		registerKeyboardCallback(&PCLVis::keyboardEventProcess, *this);
		CVLog::Print("[keyboard Event] press Delete to remove annotations");
		m_cloud_mutex.unlock();
	}

	void PCLVis::registerMouse()
	{
		m_cloud_mutex.lock();    // for not overwriting the point m_baseCloud
		registerMouseCallback(&PCLVis::mouseEventProcess, *this);
		CVLog::Print("[mouse Event] click left button to pick annotation");
		m_cloud_mutex.unlock();
	}

	void PCLVis::registerPointPicking()
	{
		m_cloud_mutex.lock();    // for not overwriting the point m_baseCloud 
		registerPointPickingCallback(&PCLVis::pointPickingProcess, *this);
		CVLog::Print("[pointPicking] Shift+click on three floor points!");
		m_cloud_mutex.unlock();
	}

	void PCLVis::registerInteractorStyle(bool useDefault)
	{
		if (useDefault)
		{
			if (this->ThreeDInteractorStyle)
			{
				vtkPVTrackballRotate* manip = vtkPVTrackballRotate::New();
				manip->SetButton(1);
				this->ThreeDInteractorStyle->AddManipulator(manip);
				manip->Delete();

				vtkPVTrackballZoom* manip2 = vtkPVTrackballZoom::New();
				manip2->SetButton(3);
				this->ThreeDInteractorStyle->AddManipulator(manip2);
				manip2->Delete();

				vtkTrackballPan* manip3 = vtkTrackballPan::New();
				manip3->SetButton(2);
				this->ThreeDInteractorStyle->AddManipulator(manip3);
				manip3->Delete();
			}
			else
			{
				CVLog::Warning("register default 3D interactor styles failed!");
			}

			if (this->TwoDInteractorStyle)
			{
				vtkTrackballPan* manip4 = vtkTrackballPan::New();
				manip4->SetButton(1);
				this->TwoDInteractorStyle->AddManipulator(manip4);
				manip4->Delete();

				vtkPVTrackballZoom* manip5 = vtkPVTrackballZoom::New();
				manip5->SetButton(3);
				this->TwoDInteractorStyle->AddManipulator(manip5);
				manip5->Delete();
			}
			else
			{
				CVLog::Warning("register default 2D interactor styles failed!");
			}
		}
		else
		{
			int manipulators[9];
			// left button
			manipulators[0] = 4;	// no special key -> Rotate
			manipulators[3] = 1;	// shift key -> Pan
			manipulators[6] = 3;	// ctrl key -> Roll(Spin)
			// middle button
			manipulators[1] = 1;	// no special key -> Pan
			manipulators[4] = 4;	// shift key -> Rotate
			manipulators[7] = 3;	// ctrl key -> Roll(Spin)
			// right button
			manipulators[2] = 2;	// no special key -> Zoom
			manipulators[5] = 1;	// shift key -> Pan
			manipulators[8] = 6;	// ctrl key -> Zoom to Mouse
			setCamera3DManipulators(manipulators);

			// left button
			manipulators[0] = 1;	// no special key -> Pan
			manipulators[3] = 2;	// shift key -> Zoom
			manipulators[6] = 3;	// ctrl key -> Roll(Spin)
			// middle button
			manipulators[1] = 3;	// no special key -> Roll(Spin)
			manipulators[4] = 2;	// shift key -> Zoom
			manipulators[7] = 1;	// ctrl key -> Pan
			// right button
			manipulators[2] = 2;	// no special key -> Zoom
			manipulators[5] = 6;	// shift key -> Zoom to Mouse
			manipulators[8] = 4;	// ctrl key -> Rotate
			setCamera2DManipulators(manipulators);
		}
	}

	void PCLVis::registerAreaPicking()
	{
		m_cloud_mutex.lock();    // for not overwriting the point m_baseCloud 
		registerAreaPickingCallback(&PCLVis::areaPickingEventProcess, *this);
		CVLog::Print("[areaPicking] press A to start or ending picking!");
		m_cloud_mutex.unlock();
	}

	void PCLVis::pointPickingProcess(const pcl::visualization::PointPickingEvent & event, void * args)
	{
		if (!m_pointPickingEnabled) return;
		
		int idx = event.getPointIndex();
		if (idx == -1) return;

		// Because VTK/OpenGL stores data without NaN, we lose the 1-1 correspondence, so we must search for the real point
		CCVector3 picked_pt;
		event.getPoint(picked_pt.x, picked_pt.y, picked_pt.z);
		std::string id = pickItem();
		emit interactorPointPickedEvent(picked_pt, idx, id);
	}

	void PCLVis::areaPickingEventProcess(const pcl::visualization::AreaPickingEvent& event, void * args)
	{
		if (!m_areaPickingEnabled) return;

		m_selected_slice.clear();
		m_selected_slice.resize(0);
		event.getPointsIndices(m_selected_slice);

		if (m_selected_slice.empty()) return;

		emit interactorAreaPickedEvent(m_selected_slice);
	}

	void PCLVis::keyboardEventProcess(const pcl::visualization::KeyboardEvent& event, void * args)
	{
		if (event.getKeySym() == "x" || event.getKeySym() == "X")
		{
			m_x_pressNum++;
		}

		// delete annotation
		if (event.keyDown()) // avoid double emitting
		{
			emit interactorKeyboardEvent(event.getKeySym());
		}
	}

	void PCLVis::mouseEventProcess(const pcl::visualization::MouseEvent& event, void * args)
	{
		if (event.getButton() == pcl::visualization::MouseEvent::LeftButton
			&& event.getType() == pcl::visualization::MouseEvent::MouseButtonPress) 
		{
			if (m_actorPickingEnabled)
			{
				pickActor(event.getX(), event.getY());
			}
		}
	}

	void PCLVis::pickActor(double x, double y) {
		if (!m_propPicker)
		{
			m_propPicker = vtkSmartPointer<vtkPropPicker>::New();
		}

		m_propPicker->Pick(x, y, 0, getRendererCollection()->GetFirstRenderer());
		vtkActor* pickedActor = m_propPicker->GetActor();
		if (pickedActor)
		{
			emit interactorPickedEvent(pickedActor);
		}
	}

	std::string PCLVis::pickItem(double x, double y)
	{
		if (!m_area_picker)
		{
			m_area_picker = vtkSmartPointer<vtkAreaPicker>::New();
		}
		int * pos = getRenderWindowInteractor()->GetEventPosition();

		m_area_picker->AreaPick(pos[0], pos[1], pos[0] + 5.0, pos[1] + 5.0, getRendererCollection()->GetFirstRenderer());
		vtkActor* pickedActor = m_area_picker->GetActor();
		if (pickedActor)
		{
			return getIdByActor(pickedActor);
		}
		else
		{
			return std::string("");
		}
	}

	/********************************Interactor Function*********************************/

}
