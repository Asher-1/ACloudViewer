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

#include "PCLDisplayTools.h"

//PCLModules
#include "PCLConv.h"
#include "sm2cc.h"
#include "cc2sm.h"

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

// CV_CORE_LIB
#include <CVGeom.h>
#include <CVTools.h>
#include <ecvGLMatrix.h>

// ECV_DB_LIB
#include <ecvBBox.h>
#include <ecvImage.h>
#include <ecvMesh.h>
#include <ecvPolyline.h>
#include <ecvSensor.h>
#include <ecvSingleton.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvHObjectCaster.h>

#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"

//system
#include <assert.h>
#include <vtkGenericOpenGLRenderWindow.h>

void PCLDisplayTools::registerVisualizer(QMainWindow * win, bool stereoMode)
{
	this->m_vtkWidget = new QVTKWidgetCustom(win, this, stereoMode);
	SetMainScreen(this->m_vtkWidget);
	SetCurrentScreen(this->m_vtkWidget);

	if (!m_visualizer3D)
	{
		auto renderer = vtkSmartPointer<vtkRenderer>::New();
		auto window = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
        auto interactorStyle = vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>::New();
        m_visualizer3D.reset(new PclUtils::PCLVis(renderer, window, interactorStyle, "3Dviewer", false));
        //m_visualizer3D.reset(new PclUtils::PCLVis(interactorStyle, "3Dviewer", false)); // deprecated!
	}

	getQVtkWidget()->SetRenderWindow(m_visualizer3D->getRenderWindow());
    m_visualizer3D->setupInteractor(getQVtkWidget()->GetInteractor(), getQVtkWidget()->GetRenderWindow());
	getQVtkWidget()->initVtk(m_visualizer3D->getRenderWindowInteractor(), false);
	m_visualizer3D->initialize();

	if (ecvDisplayTools::USE_2D)
	{
		if (!m_visualizer2D)
		{
			m_visualizer2D.reset(new PclUtils::ImageVis("2Dviewer", false));
		}

		m_visualizer2D->setRender(getQVtkWidget()->getVtkRender());
		m_visualizer2D->setupInteractor(getQVtkWidget()->GetInteractor(), getQVtkWidget()->GetRenderWindow());
		{
			//// test
			//QImage image("G:/develop/pcl_projects/cloud/obj/test/maps/test_albedo.jpg");
			//m_visualizer2D->showRGBImage(image.bits(), image.width(), image.height(), "image", 1.0);
			//m_visualizer2D->spin();
		}
	}
	else
	{
		m_visualizer2D = nullptr;
	}
}

PCLDisplayTools::~PCLDisplayTools()
{
	if (this->m_vtkWidget)
	{
		delete this->m_vtkWidget;
		this->m_vtkWidget = nullptr;
	}
}

void PCLDisplayTools::drawPointCloud(CC_DRAW_CONTEXT & CONTEXT, ccPointCloud * ecvCloud)
{
	std::string viewID = CVTools::fromQString(CONTEXT.viewID);
	int viewPort = CONTEXT.defaultViewPort;
	bool firstShow = !m_visualizer3D->contains(viewID);

	if (ecvCloud->isRedraw() ||  firstShow)
	{
		if (firstShow || checkEntityNeedUpdate(viewID, ecvCloud))
		{
			PCLCloud::Ptr pclCloud = cc2smReader(ecvCloud, true).getAsSM(!CONTEXT.drawParam.showSF);
			if (!pclCloud) { return; }
			m_visualizer3D->draw(CONTEXT, pclCloud);
			m_visualizer3D->updateNormals(CONTEXT, pclCloud);
		}
		else
		{
			m_visualizer3D->resetScalarColor(viewID, true, viewPort);
			if (!updateEntityColor(CONTEXT, ecvCloud))
			{
				PCLCloud::Ptr pclCloud = cc2smReader(ecvCloud, true).getAsSM(!CONTEXT.drawParam.showSF);
				if (!pclCloud) { return; }
				m_visualizer3D->draw(CONTEXT, pclCloud);
				m_visualizer3D->updateNormals(CONTEXT, pclCloud);
			}
			else
			{
				if (CONTEXT.drawParam.showNorms)
				{
					PCLCloud::Ptr pointNormals = cc2smReader(ecvCloud).getPointNormals();
					m_visualizer3D->updateNormals(CONTEXT, pointNormals);
				}
				else
				{
					m_visualizer3D->updateNormals(CONTEXT, nullptr);
				}

			}
		}
	}

	if (m_visualizer3D->contains(viewID))
	{
		m_visualizer3D->setPointSize(CONTEXT.defaultPointSize, viewID, viewPort);

		if ((!CONTEXT.drawParam.showColors && !CONTEXT.drawParam.showSF)
			|| ecvCloud->isColorOverriden())
		{
			ecvColor::Rgbf pointUniqueColor =
				ecvTools::TransFormRGB(CONTEXT.pointsCurrentCol);
			m_visualizer3D->setPointCloudUniqueColor(
				pointUniqueColor.r, pointUniqueColor.g,
				pointUniqueColor.b, viewID, viewPort);
		}
	}

}

void PCLDisplayTools::drawMesh(CC_DRAW_CONTEXT& CONTEXT, ccMesh* mesh)
{
	std::string viewID = CVTools::fromQString(CONTEXT.viewID);
	int viewPort = CONTEXT.defaultViewPort;
	const ccGenericPointCloud::VisibilityTableType& verticesVisibility = mesh->getAssociatedCloud()->getTheVisibilityArray();
	CONTEXT.visFiltering = (verticesVisibility.size() >= mesh->getAssociatedCloud()->size());
	bool firstShow = !m_visualizer3D->contains(viewID);
	if (mesh->isRedraw() || firstShow)
	{
		ccPointCloud* ecvCloud = ccHObjectCaster::ToPointCloud(mesh);
		if (!ecvCloud) return;

		//materials & textures
		bool lodEnabled = false;
		bool showTextures = (mesh->hasTextures() && mesh->materialsShown() && !lodEnabled);

		if (showTextures)
		{
			PCLTextureMesh::Ptr textureMesh = cc2smReader(ecvCloud, true).getPclTextureMesh(mesh);
			if (!textureMesh)
			{
				// try convert to polygonmesh
				CVLog::Warning("[PCLDisplayTools::drawMesh] try convert to polygonmesh");
				PCLMesh::Ptr pclMesh = cc2smReader(ecvCloud, true).getPclMesh(mesh);
				if (!pclMesh) return;
				m_visualizer3D->draw(CONTEXT, pclMesh);
			}
			else
			{
				m_visualizer3D->draw(CONTEXT, textureMesh);
			}
		}
		else
		{
			if (firstShow || checkEntityNeedUpdate(viewID, ecvCloud))
			{
				PCLMesh::Ptr pclMesh = cc2smReader(ecvCloud, true).getPclMesh(mesh);
				if (!pclMesh) return;

				m_visualizer3D->draw(CONTEXT, pclMesh);
			}
			else
			{
				m_visualizer3D->resetScalarColor(viewID, true, viewPort);
				if (!updateEntityColor(CONTEXT, ecvCloud))
				{
					PCLMesh::Ptr pclMesh = cc2smReader(ecvCloud, true).getPclMesh(mesh);
					if (!pclMesh) return;

					m_visualizer3D->draw(CONTEXT, pclMesh);
				}
			}

		}
		m_visualizer3D->transformEntities(CONTEXT);
	}

	if (m_visualizer3D->contains(viewID))
	{
		m_visualizer3D->setMeshRenderingMode(CONTEXT.meshRenderingMode, viewID, viewPort);

		if ((!CONTEXT.drawParam.showColors && !CONTEXT.drawParam.showSF)
			|| mesh->isColorOverriden())
		{
			ecvColor::Rgbf meshColor = ecvTools::TransFormRGB(CONTEXT.defaultMeshColor);
			m_visualizer3D->setPointCloudUniqueColor(meshColor.r, meshColor.g, meshColor.b, viewID, viewPort);
		}
		m_visualizer3D->setPointCloudOpacity(CONTEXT.opacity, viewID, viewPort);
	}
}

void PCLDisplayTools::drawPolygon(CC_DRAW_CONTEXT& CONTEXT, ccPolyline* polyline)
{
	std::string viewID = CVTools::fromQString(CONTEXT.viewID);
	bool firstShow = !m_visualizer3D->contains(viewID);
	int viewPort = CONTEXT.defaultViewPort;

	if (polyline->isRedraw() || firstShow)
	{
		PCLPolygon::Ptr pclPolygon = cc2smReader(true).getPclPolygon(polyline);
		if (!pclPolygon) return;
		m_visualizer3D->draw(CONTEXT, pclPolygon, polyline->isClosed());
	}

	if (m_visualizer3D->contains(viewID))
	{
		ecvColor::Rgbf polygonColor = ecvTools::TransFormRGB(CONTEXT.defaultPolylineColor);
		m_visualizer3D->setShapeUniqueColor(polygonColor.r, polygonColor.g, polygonColor.b, viewID, viewPort);
		m_visualizer3D->setLineWidth(CONTEXT.currentLineWidth, viewID, viewPort);
		m_visualizer3D->setLightMode(viewID, viewPort);
	}
}

void PCLDisplayTools::drawImage(CC_DRAW_CONTEXT & CONTEXT, ccImage * image)
{
	if (!m_visualizer2D) return;

#if 0
	std::string viewID = CVTools::fromQString(CONTEXT.viewID);
	bool firstShow = !m_visualizer2D->contains(viewID);

	if (image->isRedraw() || firstShow)
	{
		m_visualizer2D->showRGBImage(image->data().bits(), image->getW(), image->getH(), viewID, image->getOpacity());
	}
	m_visualizer2D->changeOpacity(viewID, image->getOpacity());
#else
	CVLog::Warning(QString("Image showing has not been supported!"));
#endif
}

bool PCLDisplayTools::updateEntityColor(CC_DRAW_CONTEXT & CONTEXT, ccHObject * ent)
{
#ifdef _DEBUG
	CVTools::TimeStart();
#endif // _DEBUG
	ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(ent);
	if (!cloud)
	{
		return false;
	}

	std::string viewID = CVTools::fromQString(CONTEXT.viewID);
	vtkActor* modelActor = m_visualizer3D->getActorById(viewID);
	if (!modelActor)
	{
		return false;
	}

	// Get the current poly data
	vtkSmartPointer<vtkPolyData> polydata = reinterpret_cast<vtkPolyDataMapper*>(modelActor->GetMapper())->GetInput();
	if (!polydata)
		return (false);

	// Get the colors from the handler
	bool has_colors = false;
	double minmax[2];
	vtkSmartPointer<vtkDataArray> scalars;
	cc2smReader converter(cloud, true);
	unsigned old_points_num = static_cast<unsigned>(polydata->GetNumberOfPoints());
	unsigned new_points_num = converter.getvisibilityNum();
	if (old_points_num != new_points_num)
	{
		return false;
	}

	if (!CONTEXT.drawParam.showColors && !CONTEXT.drawParam.showSF)
	{
		return false;
	}

	if (converter.getvtkScalars(scalars, CONTEXT.drawParam.showSF))
	{
		// Update the data
		polydata->GetPointData()->SetScalars(scalars);
		scalars->GetRange(minmax);
		has_colors = true;
	}

	if (has_colors)
	{
#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
		modelActor->GetMapper()->ImmediateModeRenderingOff();
#endif
		modelActor->GetMapper()->SetScalarRange(minmax);
		// Update the mapper
#if VTK_MAJOR_VERSION < 6
		reinterpret_cast<vtkPolyDataMapper*>(modelActor->GetMapper())->SetInput(polydata);
#else
		reinterpret_cast<vtkPolyDataMapper*> (modelActor->GetMapper())->SetInputData(polydata);
#endif
	}

#ifdef _DEBUG
	CVLog::Print(QString("updateEntityColor: finish cost %1 s").arg(CVTools::TimeOff()));
#endif // _DEBUG
	
	return (true);
}

void PCLDisplayTools::draw(CC_DRAW_CONTEXT& CONTEXT, const ccHObject* obj)
{
	if (obj->isA(CV_TYPES::POINT_CLOUD))
	{
		//the cloud to draw
		ccPointCloud* ecvCloud = ccHObjectCaster::ToPointCloud(const_cast<ccHObject *>(obj));
		if (!ecvCloud) return;

		drawPointCloud(CONTEXT, ecvCloud);
	}
	else if (obj->isKindOf(CV_TYPES::MESH))
	{
		//the mesh to draw
		ccMesh* tempMesh = ccHObjectCaster::ToMesh(const_cast<ccHObject *>(obj));
		if (!tempMesh) return;
		drawMesh(CONTEXT, tempMesh);
	}
	else if (obj->isA(CV_TYPES::POLY_LINE))
	{
		//the polyline to draw
		ccPolyline* tempPolyline = ccHObjectCaster::ToPolyline(const_cast<ccHObject *>(obj));
		if (!tempPolyline) return;
		drawPolygon(CONTEXT, tempPolyline);
	}
	else if (obj->isA(CV_TYPES::IMAGE))
	{
		// the image to draw
		ccImage* image = ccHObjectCaster::ToImage(const_cast<ccHObject *>(obj));
		if (!image) return;
		drawImage(CONTEXT, image);
	}
	else
	{
		return;
	}

	if (m_visualizer3D)
	{
		m_visualizer3D->synchronizeGeometryBounds();
	}
}

bool PCLDisplayTools::checkEntityNeedUpdate(std::string& viewID, const ccHObject * obj)
{
	bool firstShow = !m_visualizer3D->contains(viewID);
	if (firstShow) return true;

	ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(const_cast<ccHObject*>(obj));
	if (!cloud)
	{
		return true;
	}

	vtkActor* modelActor = m_visualizer3D->getActorById(viewID);
	if (!modelActor)
	{
		return true;
	}

	// Get the current poly data
	vtkSmartPointer<vtkPolyData> polydata = reinterpret_cast<vtkPolyDataMapper*>(modelActor->GetMapper())->GetInput();
	if (!polydata)
	{
		return true;
	}

	cc2smReader converter(cloud);
	unsigned old_points_num = static_cast<unsigned>(polydata->GetNumberOfPoints());
	unsigned new_points_num = converter.getvisibilityNum();
	if (old_points_num != new_points_num)
	{
		return true;
	}

	double bounds[6];
	polydata->GetBounds(bounds);
	CCVector3 minCorner(bounds[0], bounds[2], bounds[4]);
	CCVector3 maxCorner(bounds[1], bounds[3], bounds[5]);
	ccBBox oldBBox(minCorner, maxCorner);
	ccBBox newBBox = cloud->getOwnBB();

	if (abs((oldBBox.minCorner() - newBBox.minCorner()).normd()) > EPSILON_VALUE)
	{
		return true;
	}

	if (abs((oldBBox.maxCorner() - newBBox.maxCorner()).normd()) > EPSILON_VALUE)
	{
		return true;
	}

	return false;
}

void PCLDisplayTools::drawBBox(CC_DRAW_CONTEXT& context, const ccBBox * bbox)
{
	ecvColor::Rgbf colf = ecvTools::TransFormRGB(context.bbDefaultCol);
	int viewPort = context.defaultViewPort;
	if (m_visualizer3D)
	{
		std::string bboxID = CVTools::fromQString(context.viewID);
		if (!m_visualizer3D->contains(bboxID))
		{
			m_visualizer3D->addCube(
				bbox->minCorner().x,
				bbox->maxCorner().x,
				bbox->minCorner().y,
				bbox->maxCorner().y,
				bbox->minCorner().z,
				bbox->maxCorner().z,
				colf.r, colf.g, colf.b, bboxID, viewPort);

			//m_visualizer3D->setMeshRenderingMode(context.meshRenderingMode, bboxID, viewPort);
			m_visualizer3D->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
				pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, bboxID, viewPort);
			m_visualizer3D->setLineWidth(context.defaultLineWidth, bboxID, viewPort);
			m_visualizer3D->setLightMode(bboxID, viewPort);
		}
	}
}

void PCLDisplayTools::drawOrientedBBox(CC_DRAW_CONTEXT & context, const ecvOrientedBBox * obb)
{
	int viewPort = context.defaultViewPort;
	if (m_visualizer3D)
	{
		std::string bboxID = CVTools::fromQString(context.viewID);
		if (!m_visualizer3D->contains(bboxID))
		{
			const Eigen::Matrix3d& rotation_OBB = obb->getRotation();
			Eigen::Matrix3f rotation = rotation_OBB.cast<float>();
			Eigen::Quaternionf quat(rotation);
			const Eigen::Vector3d& position_OBB = obb->getPosition();
			Eigen::Vector3f position(position_OBB(0), position_OBB(1), position_OBB(2));
			const Eigen::Vector3d& extent = obb->getExtent();
			const Eigen::Vector3d& color = obb->getColor();

			m_visualizer3D->addOrientedCube(
				position, quat, extent(0), extent(1), extent(2),
				color(0), color(1), color(2), bboxID, viewPort);

			//m_visualizer3D->setMeshRenderingMode(CONTEXT.meshRenderingMode, bboxID, viewPort);
			m_visualizer3D->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
				pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, bboxID, viewPort);
			m_visualizer3D->setLineWidth(context.defaultLineWidth, bboxID, viewPort);
			m_visualizer3D->setLightMode(bboxID, viewPort);
		}
	}
}

void PCLDisplayTools::toggleOrientationMarker(bool state)
{
	if (state) {
		m_visualizer3D->showPclMarkerAxes(m_visualizer3D->getRenderWindowInteractor());
	}
	else
	{
		m_visualizer3D->hidePclMarkerAxes();
	}
}

void PCLDisplayTools::removeEntities(CC_DRAW_CONTEXT& CONTEXT)
{
	if (CONTEXT.removeEntityType == ENTITY_TYPE::ECV_IMAGE ||
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_LINES_2D ||
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_CIRCLE_2D ||
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_RECTANGLE_2D ||
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_TRIANGLE_2D ||
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_MARK_POINT)
	{
		if (!m_visualizer2D) return;
		std::string viewId = CVTools::fromQString(CONTEXT.removeViewID);
		if (m_visualizer2D->contains(viewId))
		{
			m_visualizer2D->removeLayer(viewId);
		}
	}
	else
	{
		if (CONTEXT.removeEntityType == ENTITY_TYPE::ECV_TEXT2D ||
			CONTEXT.removeEntityType == ENTITY_TYPE::ECV_POLYLINE_2D)
		{
			if (m_visualizer2D)
			{
				std::string viewId = CVTools::fromQString(CONTEXT.removeViewID);
				if (m_visualizer2D->contains(viewId))
				{
					m_visualizer2D->removeLayer(viewId);
				}
			}
		}

		if (m_visualizer3D && m_visualizer3D->removeEntities(CONTEXT))
		{
			m_visualizer3D->synchronizeGeometryBounds();
		}
	}
}

bool PCLDisplayTools::hideShowEntities(CC_DRAW_CONTEXT & CONTEXT)
{
	std::string viewId = CVTools::fromQString(CONTEXT.viewID);

	if (CONTEXT.hideShowEntityType == ENTITY_TYPE::ECV_IMAGE ||
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_LINES_2D ||
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_CIRCLE_2D || 
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_TRIANGLE_2D ||
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_RECTANGLE_2D ||
		CONTEXT.removeEntityType == ENTITY_TYPE::ECV_MARK_POINT)
	{
		if (!m_visualizer2D || !m_visualizer2D->contains(viewId)) return false;

		m_visualizer2D->hideShowActors(CONTEXT.visible, viewId);
		return true;
	}
	else if (CONTEXT.removeEntityType == ENTITY_TYPE::ECV_CAPTION)
	{
		if (m_visualizer3D->containWidget(viewId))
		{
			m_visualizer3D->hideShowWidgets(CONTEXT.visible, viewId, CONTEXT.defaultViewPort);
		}
	}
	else
	{
		if (CONTEXT.removeEntityType == ENTITY_TYPE::ECV_TEXT2D ||
			CONTEXT.removeEntityType == ENTITY_TYPE::ECV_POLYLINE_2D)
		{
			if (m_visualizer2D && m_visualizer2D->contains(viewId))
			{
				m_visualizer2D->hideShowActors(CONTEXT.visible, viewId);
			}
		}

		if (!m_visualizer3D->contains(viewId))
		{
			return false;
		}


		m_visualizer3D->hideShowActors(CONTEXT.visible, viewId, CONTEXT.defaultViewPort);

		// for normals case
		std::string normalViewId = CVTools::fromQString(CONTEXT.viewID + "-normal");
		if (m_visualizer3D->contains(normalViewId))
		{
			m_visualizer3D->hideShowActors(CONTEXT.visible, normalViewId, CONTEXT.defaultViewPort);
		}

		m_visualizer3D->synchronizeGeometryBounds();
	}
	
	return true;
}

void PCLDisplayTools::drawWidgets(const WIDGETS_PARAMETER & param)
{
	ccHObject * entity = param.entity;
	int viewPort = param.viewPort;
	std::string viewID = CVTools::fromQString(param.viewID);
	switch (param.type)
	{
	case WIDGETS_TYPE::WIDGET_COORDINATE:
		break;
	case WIDGETS_TYPE::WIDGET_BBOX:
		break;
	case WIDGETS_TYPE::WIDGET_T2D:
		if (m_visualizer2D)
		{
			std::string text = CVTools::fromQString(param.text);
			m_visualizer2D->addText(
				param.rect.x(), param.rect.y(),
				text, param.color.r, param.color.g, param.color.b,
				viewID, param.color.a, param.fontSize);
		}
		else
		{
			CC_DRAW_CONTEXT context = param.context;
			ecvDisplayTools::GetContext(context);
			ecvTextParam tParam;
			tParam.display3D = false;
			tParam.font = ecvDisplayTools::GetLabelDisplayFont();
			tParam.opacity = param.color.a;
			tParam.text = param.text;
			tParam.textPos = CCVector3d(param.rect.x(), param.rect.y(), 0.0);
			context.textDefaultCol = ecvColor::FromRgbf(param.color);
			context.textParam = tParam;
			context.viewID = tParam.text;
			m_visualizer3D->displayText(context);
		}

	case WIDGETS_TYPE::WIDGET_LINE_3D:
		if (param.lineWidget.valid && !m_visualizer3D->contains(viewID))
		{
			PointT lineSt;
			lineSt.x = param.lineWidget.lineSt.x;
			lineSt.y = param.lineWidget.lineSt.y;
			lineSt.z = param.lineWidget.lineSt.z;
			PointT lineEd;
			lineEd.x = param.lineWidget.lineEd.x;
			lineEd.y = param.lineWidget.lineEd.y;
			lineEd.z = param.lineWidget.lineEd.z;
			unsigned char lineWidth = (unsigned char)param.lineWidget.lineWidth;
			ecvColor::Rgbf lineColor = ecvTools::TransFormRGB(param.lineWidget.lineColor);
			m_visualizer3D->addLine(
				lineSt, lineEd, lineColor.r, lineColor.g, lineColor.b, viewID, viewPort);
			m_visualizer3D->setLineWidth(lineWidth, viewID, viewPort);
		}
		break;
	case WIDGETS_TYPE::WIDGET_SPHERE:
		if (!m_visualizer3D->contains(viewID)) {
			PointT center;
			center.x = param.center.x;
			center.y = param.center.y;
			center.z = param.center.z;
			m_visualizer3D->addSphere(center, param.radius, 
				param.color.r, param.color.g, param.color.b, viewID, viewPort);
		}
		break;

	case WIDGETS_TYPE::WIDGET_SCALAR_BAR:
		if (!m_visualizer3D->updateScalarBar(param.context))
		{
			m_visualizer3D->addScalarBar(param.context);
		}
		break;
	case WIDGETS_TYPE::WIDGET_CAPTION:
		if (!m_visualizer3D->updateCaption(
			CVTools::fromQString(param.text), 
			param.pos, param.center,
			param.color.r, param.color.g, param.color.b, 
			param.color.a, param.fontSize,
			viewID, viewPort)) 
		{
			m_visualizer3D->addCaption(
				CVTools::fromQString(param.text), 
				param.pos, param.center,
				param.color.r, param.color.g, param.color.b,
				param.color.a, param.fontSize, viewID, param.handleEnabled, viewPort);
		}
		break;
	case WIDGETS_TYPE::WIDGET_LINE_2D:
		if (m_visualizer2D)
		{
			m_visualizer2D->addLine(
				param.p1.x(),
				param.p1.y(),
				param.p2.x(),
				param.p2.y(),
				param.color.r,
				param.color.g, 
				param.color.b,
				viewID,
				param.color.a);
		}
		break;
	case WIDGETS_TYPE::WIDGET_POLYLINE_2D:
		if (m_visualizer2D)
		{
			if (param.entity && param.entity->isKindOf(CV_TYPES::POLY_LINE))
			{
				ccPolyline* poly = ccHObjectCaster::ToPolyline(param.entity);
				if (poly->size() <= 1)
				{
					return;
				}

				if (!poly->is2DMode())
				{
					CVLog::Warning("[PCLDisplayTools::drawWidgets] draw mode is incompatible with entity mode!");
					return;
				}

				viewID = CVTools::fromQString(QString::number(poly->getUniqueID()));

				ecvColor::Rgbf color =  ecvColor::FromRgb(ecvColor::green);
				if (poly->isColorOverriden())
				{
					color = ecvColor::FromRgb(poly->getTempColor());
				}
				else if (poly->colorsShown())
				{
					color = ecvColor::FromRgb(poly->getColor());
				}

				for (unsigned i = 1; i < poly->size(); ++i)
				{
					const CCVector3 *p1 = poly->getPoint(i-1);
					const CCVector3 *p2 = poly->getPoint(i);

					m_visualizer2D->addLine(
						static_cast<int>(p1->x), 
						static_cast<int>(p1->y),
						static_cast<int>(p2->x), 
						static_cast<int>(p2->y),
						color.r,
						color.g,
						color.b,
						viewID,
						param.opacity);
				}

				if (poly->isClosed())
				{
					m_visualizer2D->addLine(
						static_cast<int>(poly->getPoint(poly->size() - 1)->x),
						static_cast<int>(poly->getPoint(poly->size() - 1)->y),
						static_cast<int>(poly->getPoint(0)->x),
						static_cast<int>(poly->getPoint(0)->y),
						color.r,
						color.g,
						color.b,
						viewID,
						param.opacity);
				}
			}
		}
		else
		{
			if (param.entity && param.entity->isKindOf(CV_TYPES::POLY_LINE))
			{
				ccPolyline* poly = ccHObjectCaster::ToPolyline(param.entity);
				if (poly->size() <= 1)
				{
					return;
				}

				ecvDisplayTools::DrawWidgets(WIDGETS_PARAMETER(poly, WIDGETS_TYPE::WIDGET_POLYLINE), true);
			}

		}
		break;
	case WIDGETS_TYPE::WIDGET_TRIANGLE_2D:
		if (m_visualizer2D)
		{
			// edge 1
			m_visualizer2D->addLine(
				param.p1.x(),
				param.p1.y(),
				param.p2.x(),
				param.p2.y(),
				param.color.r,
				param.color.g,
				param.color.b,
				viewID,
				param.color.a);

			// edge 2
			m_visualizer2D->addLine(
				param.p2.x(),
				param.p2.y(),
				param.p3.x(),
				param.p3.y(),
				param.color.r,
				param.color.g,
				param.color.b,
				viewID,
				param.color.a);
			if (param.p4.x() >= 0 && param.p4.y() >= 0)
			{
				// edge 3
				m_visualizer2D->addLine(
					param.p3.x(),
					param.p3.y(),
					param.p4.x(),
					param.p4.y(),
					param.color.r,
					param.color.g,
					param.color.b,
					viewID,
					param.color.a);
				// edge 4
				m_visualizer2D->addLine(
					param.p4.x(),
					param.p4.y(),
					param.p1.x(),
					param.p1.y(),
					param.color.r,
					param.color.g,
					param.color.b,
					viewID,
					param.color.a);
			}
			else
			{
				// edge 3
				m_visualizer2D->addLine(
					param.p3.x(),
					param.p3.y(),
					param.p1.x(),
					param.p1.y(),
					param.color.r,
					param.color.g,
					param.color.b,
					viewID,
					param.color.a);
			}
			
		}
		break;
	case WIDGETS_TYPE::WIDGET_POINTS_2D:
		if (m_visualizer2D)
		{
			pcl::visualization::Vector3ub color = 
				pcl::visualization::Vector3ub(param.color.r, param.color.g, param.color.b);
			m_visualizer2D->markPoint(
				param.rect.x(),
				param.rect.y(),
				color,
				color,
				param.radius,
				viewID,
				param.color.a);
		}
		break;
	case WIDGETS_TYPE::WIDGET_RECTANGLE_2D:
		if (m_visualizer2D)
		{
			int minX = std::max(param.rect.x(), 0);
			int maxX = std::min(minX + param.rect.width(), m_visualizer2D->getSize()[0]);
			int minY = std::max(param.rect.y(), 0);
			int maxY = std::min(minY + param.rect.height(), m_visualizer2D->getSize()[1]);

			if (param.filled)
			{
				m_visualizer2D->addFilledRectangle(
					minX, maxX,
					minY, maxY,
					param.color.r,
					param.color.g,
					param.color.b,
					viewID,
					param.color.a);
			}
			else
			{
				m_visualizer2D->addRectangle(
					minX, maxX,
					minY, maxY,
					param.color.r,
					param.color.g,
					param.color.b,
					viewID,
					param.color.a);
			}
		}
		break;
	case WIDGETS_TYPE::WIDGET_CIRCLE_2D:
		if (m_visualizer2D)
		{
			m_visualizer2D->addCircle(
				param.rect.x(),
				param.rect.y(),
				param.radius,
				param.color.r,
				param.color.g,
				param.color.b,
				viewID,
				param.color.a);
		}
		break;
	case WIDGETS_TYPE::WIDGET_IMAGE:
		if (m_visualizer2D)
		{
			if (param.image.isNull()) return;

			m_visualizer2D->addRGBImage(
				param.image.bits(), 
				param.rect.x(),
				param.rect.y(),
				param.image.width(), 
				param.image.height(), 
				viewID, 
				param.opacity);
		}
		break;
	default:
		break;
	}
}

void PCLDisplayTools::displayText(const CC_DRAW_CONTEXT& CONTEXT)
{
	if (m_visualizer2D)
	{
		ecvTextParam textParam = CONTEXT.textParam;
		std::string viewID = CVTools::fromQString(CONTEXT.viewID);
		std::string text = CVTools::fromQString(textParam.text);

		ecvColor::Rgbf textColor = ecvTools::TransFormRGB(CONTEXT.textDefaultCol);
		{
			m_visualizer2D->addText(textParam.textPos.x, textParam.textPos.y, 
				text, textColor.r, textColor.g, textColor.b, 
				viewID, textParam.opacity, textParam.font.pointSize(), 
				textParam.font.bold());
		}
	}
	else
	{
		m_visualizer3D->displayText(CONTEXT);
	}
}

void PCLDisplayTools::toggle2Dviewer(bool state)
{
	if (m_visualizer2D)
	{
		m_visualizer2D->enable2Dviewer(state);
	}
}

QString PCLDisplayTools::pick2DLabel(int x, int y)
{
	if (m_visualizer2D)
	{
		return m_visualizer2D->pickItem(x, y).c_str();
	}

	return QString();
}

QString PCLDisplayTools::pick3DItem(int x, int y)
{
	if (m_visualizer3D)
	{
		return m_visualizer3D->pickItem(x, y).c_str();
	}

	return QString();
}

double PCLDisplayTools::getParallelScale(int viewPort)
{
	if (m_visualizer3D)
	{
		return m_visualizer3D->getParallelScale() * CV_DEG_TO_RAD;
	}
	
	return -1;
}

void PCLDisplayTools::getProjectionMatrix(double * projArray, int viewPort)
{
	Eigen::Matrix4d projMat;
	m_visualizer3D->getCamera(viewPort).computeProjectionMatrix(projMat);
	//m_visualizer3D->getProjectionTransformMatrix(projMat);
	double *tempArray = projMat.data();
	projArray[0] = tempArray[0];
	projArray[1] = tempArray[1];
	projArray[2] = tempArray[2];
	projArray[3] = tempArray[3];
	projArray[4] = tempArray[4];
	projArray[5] = tempArray[5];
	projArray[6] = tempArray[6];
	projArray[7] = tempArray[7];
	projArray[8] = tempArray[8];
	projArray[9] = tempArray[9];
	projArray[10] = tempArray[10];
	projArray[11] = tempArray[11];
	projArray[12] = tempArray[12];
	projArray[13] = tempArray[13];
	projArray[14] = tempArray[14];
	projArray[15] = tempArray[15];
}

void PCLDisplayTools::getViewMatrix(double * ViewArray, int viewPort)
{
	Eigen::Matrix4d viewMat;
	m_visualizer3D->getCamera(viewPort).computeViewMatrix(viewMat);
	//m_visualizer3D->getModelViewTransformMatrix(viewMat);
	double *tempArray = viewMat.data();
	ViewArray[0] = tempArray[0];
	ViewArray[1] = tempArray[1];
	ViewArray[2] = tempArray[2];
	ViewArray[3] = tempArray[3];
	ViewArray[4] = tempArray[4];
	ViewArray[5] = tempArray[5];
	ViewArray[6] = tempArray[6];
	ViewArray[7] = tempArray[7];
	ViewArray[8] = tempArray[8];
	ViewArray[9] = tempArray[9];
	ViewArray[10] = tempArray[10];
	ViewArray[11] = tempArray[11];
	ViewArray[12] = tempArray[12];
	ViewArray[13] = tempArray[13];
	ViewArray[14] = tempArray[14];
	ViewArray[15] = tempArray[15];
}

void PCLDisplayTools::setViewMatrix(double* viewArray, int viewPort/* = 0*/)
{
	//vtkSmartPointer<vtkCamera> cam = PCLVis::GetVtkCamera();
	//vtkSmartPointer<vtkTransform> trans = vtkSmartPointer<vtkTransform>::New();
	//trans->SetMatrix(viewArray);
	//cam->SetUserViewTransform();
	CVLog::Warning("[PCLDisplayTools::setViewMatrix] has been deprecated, and do nothing!");
}

void PCLDisplayTools::changeEntityProperties(PROPERTY_PARAM & param)
{
	std::string viewId = CVTools::fromQString(param.viewId);
	int viewPort = param.viewPort;
	switch (param.property)
	{
	case PROPERTY_MODE::ECV_POINTSSIZE_PROPERTY:
	{
		m_visualizer3D->setPointSize(param.pointSize, viewId, viewPort);
	}
	break;
	case PROPERTY_MODE::ECV_LINEWITH_PROPERTY:
	{
		m_visualizer3D->setLineWidth(param.lineWidth, viewId, viewPort);
	}
	break;
	case PROPERTY_MODE::ECV_COLOR_PROPERTY:
	{
		ecvColor::Rgbf colf = ecvTools::TransFormRGB(param.color);
		switch (param.entityType)
		{
		case ENTITY_TYPE::ECV_POINT_CLOUD:
		case ENTITY_TYPE::ECV_MESH:
		{
			m_visualizer3D->setPointCloudUniqueColor(colf.r, colf.g, colf.b, viewId, viewPort);
		}
		break;
		case ENTITY_TYPE::ECV_SHAPE:
		case ENTITY_TYPE::ECV_LINES_3D:
		{
			m_visualizer3D->setShapeUniqueColor(colf.r, colf.g, colf.b, viewId, viewPort);
		}
		break;
		default:
			break;
		}
	}
	break;
	case PROPERTY_MODE::ECV_OPACITY_PROPERTY:
	{
		switch (param.entityType)
		{
		case ENTITY_TYPE::ECV_POINT_CLOUD:
		case ENTITY_TYPE::ECV_MESH:
		{
			m_visualizer3D->setPointCloudOpacity(param.opacity, viewId, viewPort);
		}
		break;
		case ENTITY_TYPE::ECV_SHAPE:
		case ENTITY_TYPE::ECV_LINES_3D:
		{
			m_visualizer3D->setShapeOpacity(param.opacity, viewId, viewPort);
		}
		break;
		default:
			break;

		}
	}
	break;

	default:
		break;

	}
}

void PCLDisplayTools::transformCameraView(const ccGLMatrixd & viewMat)
{
	getQVtkWidget()->transformCameraView(viewMat.data());
}

void PCLDisplayTools::transformCameraProjection(const ccGLMatrixd & projMat)
{
	getQVtkWidget()->transformCameraProjection(projMat.data());
}
