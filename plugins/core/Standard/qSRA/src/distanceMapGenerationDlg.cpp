//##########################################################################
//#                                                                        #
//#                      CLOUDVIEWER  PLUGIN: qSRA                         #
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
//#                           COPYRIGHT: EDF                               #
//#                                                                        #
//##########################################################################

#include "distanceMapGenerationDlg.h"

//local
#include "ccSymbolCloud.h"
#include "ccMapWindow.h"
#include "dxfProfilesExporter.h"
#include "dxfProfilesExportDlg.h"

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

//qCC
//#include <ecvCommon.h>
#include <ecvColorScaleSelector.h>
#include <ecvColorScalesManager.h>
#include <ecvColorScaleEditorDlg.h>
#include <ecvRenderToFileDlg.h>

//common
#include <ecvQtHelpers.h>

// ECV_DB_LIB
#include <ecvFileUtils.h>
#include <ecvDisplayTools.h>
#include <ecvPointCloud.h>
#include <ecvPlane.h>
#include <ecvScalarField.h>
#include <ecvPolyline.h>

//Qt
#include <QMainWindow>
#include <QHBoxLayout>
#include <QSettings>
#include <QFileDialog>
#include <QFileInfo>
#include <QFile>
#include <QTextStream>
#include <QProgressDialog>
#include <QColorDialog>
#include <QFontMetrics>
#include <QLocale>

//system
#include <assert.h>

//default names
static const char XLABEL_CLOUD_NAME[] = "X_Labels";
static const char YLABEL_CLOUD_NAME[] = "Y_Labels";
static const double DEFAULT_LABEL_MARGIN = 20.0; //half of this in fact (we use the 'symbol' size)

static double ConvertAngleFromRad(double angle_rad, DistanceMapGenerationDlg::ANGULAR_UNIT destUnit)
{
	switch(destUnit)
	{
	case DistanceMapGenerationDlg::ANG_DEG: //degrees
        return cloudViewer::RadiansToDegrees(angle_rad);
	case DistanceMapGenerationDlg::ANG_RAD: //radians
		return angle_rad;
	case DistanceMapGenerationDlg::ANG_GRAD: //grades
		return angle_rad / M_PI * 200.0;
	default:
		assert(false);
	}

	return 0.0;
}

static double ConvertAngleToRad(double angle, DistanceMapGenerationDlg::ANGULAR_UNIT srcUnit)
{
	switch(srcUnit)
	{
	case DistanceMapGenerationDlg::ANG_DEG: //degrees
        return cloudViewer::DegreesToRadians(angle);
	case DistanceMapGenerationDlg::ANG_RAD: //radians
		return angle;
	case DistanceMapGenerationDlg::ANG_GRAD: //grades
		return angle / 200.0 * M_PI;
	default:
		assert(false);
	}

	return 0.0;
}

DistanceMapGenerationDlg::DistanceMapGenerationDlg(ccPointCloud* cloud, ccScalarField* sf, ccPolyline* polyline, ecvMainAppInterface* app/*=0*/)
	: QDialog(app ? app->getMainWindow() : 0)
	, m_app(app)
	, m_cloud(cloud)
	, m_profile(polyline)
	, m_sf(sf)
	, m_map(0)
	, m_angularUnits(ANG_GRAD)
	, m_window(0)
	, m_colorScaleSelector(0)
	, m_xLabels(0)
	, m_yLabels(0)
	, m_gridColor(Qt::gray)
	, m_symbolColor(Qt::black)
{
	setupUi(this);

	assert(m_cloud && m_sf && m_profile);

	//add color ramp selector widget (before calling initFromPersistentSettings!)
	if (m_sf)
	{
		//create selector widget
		m_colorScaleSelector = new ccColorScaleSelector(m_app->getColorScalesManager(),this,QString::fromUtf8(":/CC/plugin/qSRA/gearIcon.png"));
		m_colorScaleSelector->init();
		m_colorScaleSelector->setSelectedScale(ccColorScalesManager::GetDefaultScale()->getUuid());
		connect(m_colorScaleSelector, SIGNAL(colorScaleSelected(int)), this, SLOT(colorScaleChanged(int)));
		connect(m_colorScaleSelector, SIGNAL(colorScaleEditorSummoned()), this, SLOT(spawnColorScaleEditor()));
		//add selector to group's layout
		if (!colorRampGroupBox->layout())
			colorRampGroupBox->setLayout(new QHBoxLayout());
		colorRampGroupBox->layout()->addWidget(m_colorScaleSelector);
		colorScaleStepsSpinBox->setRange(ccColorScale::MIN_STEPS,ccColorScale::MAX_STEPS);
	}

	//init parameters from persistent settings
	initFromPersistentSettings();

	if (m_sf)
	{
		//we apply the cloud current color scale ONLY if it is not a default one
		//(otherwise we keep the default dialog's one)
		const ccColorScale::Shared& scale = m_sf->getColorScale();
		if (scale && !scale->isLocked())
			m_colorScaleSelector->setSelectedScale(scale->getUuid());
	}

	//profile meta-data
	DistanceMapGenerationTool::ProfileMetaData profileDesc;
	bool validProfile = false;

	//set default dialog values with polyline & cloud information
	if (m_profile)
	{
		validProfile = DistanceMapGenerationTool::GetPoylineMetaData(m_profile, profileDesc);
		if (validProfile)
		{
			//update the 'Generatrix' tab
			{
				axisDimComboBox->setCurrentIndex(profileDesc.revolDim);

				xOriginDoubleSpinBox->setValue(profileDesc.origin.x);
				yOriginDoubleSpinBox->setValue(profileDesc.origin.y);
				zOriginDoubleSpinBox->setValue(profileDesc.origin.z);
			}

			updateMinAndMaxLimits();
		}
		else
		{
			if (m_app)
				m_app->dispToConsole(QString("Invalid profile: can't generate a proper map!"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		}
	}

	//compute min and max height of the points
	//we will 'lock" the max height value with that information
	if (m_cloud)
	{
		ccBBox bbox = m_cloud->getOwnBB();
		PointCoordinateType hMin = 0, hMax = 0;
		if (bbox.isValid())
		{
			hMin = bbox.minCorner().u[profileDesc.revolDim];
			hMax = bbox.maxCorner().u[profileDesc.revolDim];
		}

		if (hMax - hMin <= 0)
		{
			if (m_app)
				m_app->dispToConsole(QString("Cloud is flat: can't generate a proper map!"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		}
		else
		{
			hStepDoubleSpinBox->setMaximum(hMax-hMin);
		}
	}

	//add window
	{
		m_window = new ccMapWindow();
		ecvGui::ParamStruct params = ecvDisplayTools::GetDisplayParameters();
		params.backgroundCol = ecvColor::white;
		params.textDefaultCol = ecvColor::black;
		params.drawBackgroundGradient = false;
		params.colorScaleShowHistogram = false;
		params.colorScaleRampWidth = 30;
		params.decimateMeshOnMove = false;
		params.displayCross = false;
		params.colorScaleUseShader = false;
		ecvDisplayTools::SetDisplayParameters(params);
		ecvDisplayTools::SetPerspectiveState(false,true);
		ecvDisplayTools::SetInteractionMode(
			ecvDisplayTools::INTERACT_PAN | 
			ecvDisplayTools::INTERACT_CLICKABLE_ITEMS |
			ecvDisplayTools::INTERACT_ZOOM_CAMERA);
		ecvDisplayTools::DisplayOverlayEntities(false);

		m_window->showSF(displayColorScaleCheckBox->isChecked());
		//add window to the right side layout
		mapFrame->setLayout(new QHBoxLayout());
#ifdef CV_GL_WINDOW_USE_QWINDOW
		mapFrame->layout()->addWidget(QWidget::createWindowContainer(m_window));
#else
		mapFrame->layout()->addWidget(m_window);
#endif
		precisionSpinBox->setValue(params.displayedNumPrecision);
	}

	//create labels "clouds" (empty)
	{
		m_xLabels = new ccSymbolCloud(XLABEL_CLOUD_NAME);
		m_xLabels->showSymbols(false);
		m_xLabels->setSymbolSize(DEFAULT_LABEL_MARGIN);
		m_xLabels->setLabelAlignmentFlags(ecvDisplayTools::ALIGN_HMIDDLE | ecvDisplayTools::ALIGN_VBOTTOM);
		ecvDisplayTools::AddToOwnDB(m_xLabels,false);
		m_yLabels = new ccSymbolCloud(YLABEL_CLOUD_NAME);
		m_yLabels->showSymbols(false);
		m_yLabels->setSymbolSize(DEFAULT_LABEL_MARGIN);
		m_yLabels->setLabelAlignmentFlags(ecvDisplayTools::ALIGN_HRIGHT | ecvDisplayTools::ALIGN_VMIDDLE);
		ecvDisplayTools::AddToOwnDB(m_yLabels,false);
	}

	connect(projectionComboBox,				SIGNAL(currentIndexChanged(int)),	this,	SLOT(projectionModeChanged(int)));
	connect(angularUnitComboBox,			SIGNAL(currentIndexChanged(int)),	this,	SLOT(angularUnitChanged(int)));
	connect(xStepDoubleSpinBox,				SIGNAL(valueChanged(double)),		this,	SLOT(updateGridSteps()));
	connect(hStepDoubleSpinBox,				SIGNAL(valueChanged(double)),		this,	SLOT(updateGridSteps()));
	connect(latStepDoubleSpinBox,			SIGNAL(valueChanged(double)),		this,	SLOT(updateGridSteps()));
	connect(xMinDoubleSpinBox,				SIGNAL(valueChanged(double)),		this,	SLOT(updateGridSteps()));
	connect(xMaxDoubleSpinBox,				SIGNAL(valueChanged(double)),		this,	SLOT(updateGridSteps()));
	connect(hMinDoubleSpinBox,				SIGNAL(valueChanged(double)),		this,	SLOT(updateGridSteps()));
	connect(hMaxDoubleSpinBox,				SIGNAL(valueChanged(double)),		this,	SLOT(updateGridSteps()));
	connect(latMinDoubleSpinBox,			SIGNAL(valueChanged(double)),		this,	SLOT(updateGridSteps()));
	connect(latMaxDoubleSpinBox,			SIGNAL(valueChanged(double)),		this,	SLOT(updateGridSteps()));
	connect(axisDimComboBox,				SIGNAL(currentIndexChanged(int)),	this,	SLOT(updateProfileRevolDim(int)));
	connect(xOriginDoubleSpinBox,			SIGNAL(valueChanged(double)),		this,	SLOT(updateProfileOrigin()));
	connect(yOriginDoubleSpinBox,			SIGNAL(valueChanged(double)),		this,	SLOT(updateProfileOrigin()));
	connect(zOriginDoubleSpinBox,			SIGNAL(valueChanged(double)),		this,	SLOT(updateProfileOrigin()));
	connect(baseRadiusDoubleSpinBox,		SIGNAL(valueChanged(double)),		this,	SLOT(baseRadiusChanged(double)));
	connect(heightUnitLineEdit,				SIGNAL(editingFinished()),			this,	SLOT(updateHeightUnits()));
	connect(exportCloudPushButton,			SIGNAL(clicked()),					this,	SLOT(exportMapAsCloud()));
	connect(exportMeshPushButton,			SIGNAL(clicked()),					this,	SLOT(exportMapAsMesh()));
	connect(exportMatrixPushButton,			SIGNAL(clicked()),					this,	SLOT(exportMapAsGrid()));
	connect(exportImagePushButton,			SIGNAL(clicked()),					this,	SLOT(exportMapAsImage()));
	connect(loadLabelsPushButton,			SIGNAL(clicked()),					this,	SLOT(loadOverlaySymbols()));
	connect(clearLabelsPushButton,			SIGNAL(clicked()),					this,	SLOT(clearOverlaySymbols()));
	connect(symbolSizeSpinBox,				SIGNAL(valueChanged(int)),			this,	SLOT(overlaySymbolsSizeChanged(int)));
	connect(fontSizeSpinBox,				SIGNAL(valueChanged(int)),			this,	SLOT(labelFontSizeChanged(int)));
	connect(precisionSpinBox,				SIGNAL(valueChanged(int)),			this,	SLOT(labelPrecisionChanged(int)));
	
	connect(colorScaleStepsSpinBox,			SIGNAL(valueChanged(int)),			this,	SLOT(colorRampStepsChanged(int)));
	connect(overlayGridGroupBox,			SIGNAL(toggled(bool)),				this,	SLOT(toggleOverlayGrid(bool)));
	connect(scaleXStepDoubleSpinBox,		SIGNAL(editingFinished()),			this,	SLOT(updateOverlayGrid()));
	connect(scaleHStepDoubleSpinBox,		SIGNAL(editingFinished()),			this,	SLOT(updateOverlayGrid()));
	connect(scaleLatStepDoubleSpinBox,		SIGNAL(editingFinished()),			this,	SLOT(updateOverlayGrid()));
	connect(xScaleCheckBox,					SIGNAL(clicked()),					this,	SLOT(updateOverlayGrid()));
	connect(yScaleCheckBox,					SIGNAL(clicked()),					this,	SLOT(updateOverlayGrid()));
	connect(gridColorButton,				SIGNAL(clicked()),					this,	SLOT(changeGridColor()));
	connect(symbolColorButton,				SIGNAL(clicked()),					this,	SLOT(changeSymbolColor()));
	connect(displayColorScaleCheckBox,		SIGNAL(toggled(bool)),				this,	SLOT(toggleColorScaleDisplay(bool)));
	connect(updateVolumesPushButton,		SIGNAL(clicked()),					this,	SLOT(updateVolumes()));

	//DXF profiles export button is only visible/connected if DXF support is enabled!
	if (DxfProfilesExporter::IsEnabled())
	{
		connect(exportImageDXFButton,			SIGNAL(clicked()),					this,	SLOT(exportProfilesAsDXF()));
	}
	else
	{
		exportImageDXFButton->hide();
	}

	//button box
	{
		QPushButton* applyButton = buttonBox->button(QDialogButtonBox::Apply);
		QPushButton* closeButton = buttonBox->button(QDialogButtonBox::Close);
		connect(applyButton,				SIGNAL(clicked()),					this,	SLOT(update()));
		connect(closeButton,				SIGNAL(clicked()),					this,	SLOT(accept()));
	}

	angularUnitChanged(m_angularUnits); //just to be sure
	baseRadiusChanged(0);
	overlaySymbolsColorChanged();
	overlayGridColorChanged();
	labelFontSizeChanged(-1);
	projectionModeChanged(-1);
}

DistanceMapGenerationDlg::~DistanceMapGenerationDlg()
{
}

void DistanceMapGenerationDlg::updateMinAndMaxLimits()
{
	if (m_cloud && m_profile)
	{
		DistanceMapGenerationTool::ProfileMetaData profileDesc;
		if (DistanceMapGenerationTool::GetPoylineMetaData(m_profile, profileDesc))
		{
			//compute mean 'radius'
			//as well as min and max 'height'
			double baseRadius = 0.0;
			double minHeight = 0.0;
			double maxHeight = 0.0;
			for (unsigned i = 0; i<m_profile->size(); ++i)
			{
				const CCVector3* P = m_profile->getPoint(i);
				double radius = P->x;
				double height = P->y + profileDesc.heightShift;
				baseRadius += radius;

				if (i != 0)
				{
					if (height < minHeight)
						minHeight = height;
					else if (height > maxHeight)
						maxHeight = height;
				}
				else
				{
					minHeight = maxHeight = height;
				}
			}

			//set default 'base radius'
			if (m_profile->size() != 0)
				baseRadius /= m_profile->size();
			if (baseRadius == 0.0)
				baseRadius = 1.0;

			baseRadiusDoubleSpinBox->blockSignals(true);
			baseRadiusDoubleSpinBox->setValue(baseRadius);
			baseRadiusDoubleSpinBox->blockSignals(false);

			//set default min and max height
			hMinDoubleSpinBox->blockSignals(true);
			hMinDoubleSpinBox->setValue(minHeight);
			hMinDoubleSpinBox->blockSignals(false);

			hMaxDoubleSpinBox->blockSignals(true);
			hMaxDoubleSpinBox->setValue(maxHeight);
			hMaxDoubleSpinBox->blockSignals(false);

			//do the same for the latitude

			//compute transformation from the cloud to the surface (of revolution)
			ccGLMatrix cloudToSurfaceOrigin = profileDesc.computeCloudToSurfaceOriginTrans();

			double minLat_rad = 0.0, maxLat_rad = 0.0;
			if (DistanceMapGenerationTool::ComputeMinAndMaxLatitude_rad(m_cloud,
				minLat_rad,
				maxLat_rad,
				cloudToSurfaceOrigin,
				static_cast<unsigned char>(profileDesc.revolDim)))
			{
				latMinDoubleSpinBox->blockSignals(true);
				latMinDoubleSpinBox->setValue(ConvertAngleFromRad(minLat_rad, m_angularUnits));
				latMinDoubleSpinBox->blockSignals(false);
				
				latMaxDoubleSpinBox->blockSignals(true);
				latMaxDoubleSpinBox->setValue(ConvertAngleFromRad(maxLat_rad, m_angularUnits));
				latMaxDoubleSpinBox->blockSignals(false);
			}
		}
	}
}

void DistanceMapGenerationDlg::projectionModeChanged(int)
{
	//reset eveything, etc.
	ProjectionMode mode = getProjectionMode();

	clearView();

	//conical mode only
	latLabel->setVisible(mode == PROJ_CONICAL);
	latMinDoubleSpinBox->setVisible(mode == PROJ_CONICAL);
	latMaxDoubleSpinBox->setVisible(mode == PROJ_CONICAL);
	latStepDoubleSpinBox->setVisible(mode == PROJ_CONICAL);
	latStepLabel->setVisible(mode == PROJ_CONICAL);
	scaleLatStepLabel->setVisible(mode == PROJ_CONICAL);
	scaleLatStepDoubleSpinBox->setVisible(mode == PROJ_CONICAL);
	spanRatioFrame->setVisible(mode == PROJ_CONICAL);

	//cylindrical mode only
	yLabel->setVisible(mode == PROJ_CYLINDRICAL);
	hMinDoubleSpinBox->setVisible(mode == PROJ_CYLINDRICAL);
	hMaxDoubleSpinBox->setVisible(mode == PROJ_CYLINDRICAL);
	exportImageDXFButton->setVisible(mode == PROJ_CYLINDRICAL);
	heightStepLabel->setVisible(mode == PROJ_CYLINDRICAL);
	hStepDoubleSpinBox->setVisible(mode == PROJ_CYLINDRICAL);
	scaleHeightStepLabel->setVisible(mode == PROJ_CYLINDRICAL);
	scaleHStepDoubleSpinBox->setVisible(mode == PROJ_CYLINDRICAL);
	heightUnitLineEdit->setVisible(mode == PROJ_CYLINDRICAL);
	yScaleCheckBox->setVisible(mode == PROJ_CYLINDRICAL);
	
	baseRadiusChanged(0);
	updateGridSteps();

	if (m_map)
		update();
}

DistanceMapGenerationDlg::ProjectionMode DistanceMapGenerationDlg::getProjectionMode() const
{
	switch(projectionComboBox->currentIndex())
	{
	case 0:
		return PROJ_CYLINDRICAL;
	case 1:
		return PROJ_CONICAL;
	default:
		assert(false);
	}
	
	return PROJ_CYLINDRICAL;
}

DistanceMapGenerationDlg::ANGULAR_UNIT DistanceMapGenerationDlg::getAngularUnit() const
{
	switch(m_angularUnits)
	{
	case 0: //degrees
		return ANG_DEG;
	case 1: //radians
		return ANG_RAD;
	case 2: //grades
		return ANG_GRAD;
	default:
		assert(false);
	}

	return ANG_DEG;
}

QString DistanceMapGenerationDlg::getAngularUnitString() const
{
	switch(m_angularUnits)
	{
	case 0: //degrees
		return "deg";
	case 1: //radians
		return "rad";
	case 2: //grades
		return "grad";
	default:
		assert(false);
	}

	return "none";
}

QString DistanceMapGenerationDlg::getCondensedAngularUnitString() const
{
	switch(m_angularUnits)
	{
	case 0: //degrees
		return QChar(0x00B0);
	case 1: //radians
		return "rd";
	case 2: //grades
		return "gr";
	default:
		assert(false);
	}

	return "none";
}


double DistanceMapGenerationDlg::getSpinboxAngularValue(QDoubleSpinBox* spinBox,
														DistanceMapGenerationDlg::ANGULAR_UNIT destUnit) const
{
	//no conversion necessary?
	if (m_angularUnits == destUnit)
		return spinBox->value();

	//otherwise we convert to radians first
	double angle_rad = ConvertAngleToRad(spinBox->value(), m_angularUnits);
	//then to the destination value
	return ConvertAngleFromRad(angle_rad, destUnit);
}

void DistanceMapGenerationDlg::updateZoom(ccBBox& box)
{
	if (!m_window || !box.isValid())
		return;

	//equivalent to 'ccGLWindow::updateConstellationCenterAndZoom' but we take aspect ratio into account

	//we get the bounding-box diagonal length
	PointCoordinateType bbDiag = box.getDiagNorm();
    if (cloudViewer::GreaterThanEpsilon(bbDiag))
	{
		bool sfDisplayed = m_window->getAssociatedScalarField() && m_window->sfShown();
		bool yLabelDisplayed = m_yLabels && m_yLabels->isVisible() && m_yLabels->size();
		float centerPos = 0.5f;

		//we compute the pixel size (in world coordinates)
		{
			ecvViewportParameters params = ecvDisplayTools::GetViewportParameters();
			params.zoom = 1.0f;

			int screenWidth = ecvDisplayTools::GlWidth();
			int scaleWidth = 0;
			int labelsWidth = 0;
			if (sfDisplayed)
			{
				scaleWidth = ecvDisplayTools::GetDisplayParameters().colorScaleRampWidth + QFontMetrics(ecvDisplayTools::GetTextDisplayFont()).horizontalAdvance("123.456789");
				//scaleWidth = ecvDisplayTools::GetDisplayParameters().colorScaleRampWidth + QFontMetrics(ecvDisplayTools::GetTextDisplayFont()).width("123.456789");
			}
			if (yLabelDisplayed)
			{
				//find the largest label width
				QFont labelFont = ecvDisplayTools::GetTextDisplayFont();
				labelFont.setPointSize(m_yLabels->getFontSize());
				QFontMetrics fm(labelFont);
				int maxWidth = 0;
				for (unsigned i=0; i<m_yLabels->size(); ++i)
				{
					QString label = m_yLabels->getLabel(i);
					if (!label.isNull())
					{
						//int width = fm.width(label);
						int width = fm.horizontalAdvance(label);
						maxWidth = std::max(maxWidth,width);
					}
				}
				labelsWidth = maxWidth;
			}

			//available room for the map
			int mapWidth = std::max(1, screenWidth - scaleWidth - labelsWidth);

			//we zoom so that the map takes all the room left
			float mapPart = static_cast<float>(mapWidth) / static_cast<float>(screenWidth);
			params.zoom *= mapPart;

			//we must also center the camera on the right position so that the map
			//appears in between the scale and the color ramp
			float mapStart = static_cast<float>(labelsWidth) / static_cast<float>(screenWidth);
			centerPos = (0.5f - mapStart) / mapPart;

			//update pixel size accordingly
            float screenHeight = ecvDisplayTools::GlHeight() * params.cameraAspectRatio;
			params.pixelSize = static_cast<float>(std::max(box.getDiagVec().x / mapWidth, box.getDiagVec().y / screenHeight));
			ecvDisplayTools::SetViewportParameters(params);
		}

		//we set the pivot point on the box center
		CCVector3 P = box.getCenter();
		if (centerPos != 0.5f) //if we don't look exactly at the center of the map
			P.x = box.minCorner().x * (1.0f - centerPos) + box.maxCorner().x * centerPos;
		ecvDisplayTools::SetPivotPoint(CCVector3d::fromArray(P.u));
		ecvDisplayTools::SetCameraPos(CCVector3d::fromArray(P.u));

		ecvDisplayTools::InvalidateViewport();
		ecvDisplayTools::InvalidateVisualization();
		ecvDisplayTools::Deprecate3DLayer();
	}

	ecvDisplayTools::RedrawDisplay();
}

void DistanceMapGenerationDlg::clearView()
{
	if (!m_window)
		return;

	//remove existing sf
	m_window->setAssociatedScalarField(0);
	
	//remove existing map (or maps?)
	ccHObject::Container maps;
	ecvDisplayTools::GetOwnDB()->filterChildren(maps, false, CV_TYPES::MESH);
	for (size_t i = 0; i < maps.size(); ++i)
	{
		ecvDisplayTools::RemoveFromOwnDB(maps[i]);
	}

	//remove any polylines
	{
		ccHObject::Container polylines;
		ecvDisplayTools::GetOwnDB()->filterChildren(polylines, false, CV_TYPES::POLY_LINE);
		for (size_t i = 0; i < polylines.size(); ++i)
		{
			ecvDisplayTools::RemoveFromOwnDB(polylines[i]);
		}
	}
	m_xLabels->setVisible(false);
	m_yLabels->setVisible(false);
}

void DistanceMapGenerationDlg::update()
{
	if (m_map)
	{
		if (getProjectionMode() == PROJ_CONICAL)
		{
			//we must check that the projection parameter have not changed!
			//Otherwise the symbols will be misplaced...
			double yMin, yMax, yStep;
			getGridYValues(yMin, yMax, yStep, ANG_RAD);
			if (	!m_map->conical
				||	m_map->yMin != yMin
				||	m_map->yMax != yMax
				||	m_map->conicalSpanRatio != conicSpanRatioDoubleSpinBox->value())
			{
				clearOverlaySymbols();
			}
		}
		else if (m_map->conical)
		{
			//we can't keep the symbols when switching the projection mode
			clearOverlaySymbols();
		}
	}

	//release memory
	m_map.clear();

	//clear 3D view
	clearView();

	//update map
	m_map = updateMap();
	//and GUI
	exportGroupBox->setEnabled(m_map != 0);

	//auto update volumes
	updateVolumes();

	if (m_map && m_window)
	{
		ccMesh* mapMesh = 0;

		ProjectionMode mode = getProjectionMode();
		if (mode == PROJ_CYLINDRICAL)
		{
			//by default we map an image on a plane
			double dx = static_cast<double>(m_map->xSteps) * m_map->xStep;
			double dy = static_cast<double>(m_map->ySteps) * m_map->yStep;
			ccGLMatrix transMat;
			transMat.setTranslation(CCVector3(	static_cast<PointCoordinateType>(dx / 2 + m_map->xMin),
												static_cast<PointCoordinateType>(dy / 2 + m_map->yMin),
												0));
			ccPlane* mapPlane = new ccPlane(	static_cast<PointCoordinateType>(dx),
												static_cast<PointCoordinateType>(dy),
												&transMat,
												"map");
			mapMesh = static_cast<ccMesh*>(mapPlane);
		}
		else //if (mode == PROJ_CONICAL) //conical projection
		{
			//no choice, we create a mesh
			bool ccw = ccwCheckBox->isChecked();
			m_map->conicalSpanRatio = conicSpanRatioDoubleSpinBox->value();
			mapMesh = DistanceMapGenerationTool::ConvertConicalMapToMesh(m_map, ccw);
		}

		if (mapMesh)
		{
			mapMesh->setVisible(true);
			mapMesh->showNormals(false);
			ecvDisplayTools::AddToOwnDB(mapMesh,false);

			updateMapTexture();

			//add a virtual scalar field for color ramp display
			ccScalarField* sf = new ccScalarField();
			{
				sf->reserve(2);
				ScalarType smin = static_cast<ScalarType>(m_map->minVal);
				ScalarType smax = static_cast<ScalarType>(m_map->maxVal);
				sf->addElement(smin);
				sf->addElement(smax);
				sf->computeMinAndMax();
			}
			//selected color scale
			ccColorScale::Shared colorScale = ccColorScalesManager::GetDefaultScale();
			if (m_colorScaleSelector)
				colorScale = m_colorScaleSelector->getSelectedScale();
			sf->setColorScale(colorScale);
			sf->setColorRampSteps(colorScaleStepsSpinBox->value());
			m_window->setAssociatedScalarField(sf);
		}
		else
		{
			m_app->dispToConsole(QString("Not enough memory to display the map!"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		}
	}

	//update sf names, etc.
	updateHeightUnits(); //already call 'updateOverlayGrid'!
	//force grid update if necessary
	//updateOverlayGrid();
	//update zoom
	ccBBox box = m_window ? ecvDisplayTools::GetOwnDB()->getDisplayBB_recursive(false) : ccBBox();
	updateZoom(box);

	saveToPersistentSettings();
}

void DistanceMapGenerationDlg::updateHeightUnits()
{
	scaleHStepDoubleSpinBox->setSuffix(QString(" ")+heightUnitLineEdit->text());

	if (m_window && m_window->getAssociatedScalarField())
	{
		m_window->getAssociatedScalarField()->setName(qPrintable(QString("Distance (%1)").arg(getHeightUnitString())));
	}

	updateOverlayGrid();
}

void DistanceMapGenerationDlg::updateMapTexture()
{
	if (!m_map || !m_colorScaleSelector || !m_window)
		return;

	ccHObject::Container texturedEntities;

	ProjectionMode mode = getProjectionMode();
	if (mode == PROJ_CYLINDRICAL)
	{
		//cylindrical projection: look for a plane
		if (ecvDisplayTools::GetOwnDB()->filterChildren(texturedEntities,false,CV_TYPES::PLANE) == 0)
			return;
	}
	else if (mode == PROJ_CONICAL)
	{
		//conical projection: look for a standard mesh
		if (ecvDisplayTools::GetOwnDB()->filterChildren(texturedEntities,false,CV_TYPES::MESH) == 0)
			return;
	}

	//spawn "update" dialog
	QProgressDialog progressDlg(QString("Updating..."),0,0,0,0,Qt::Popup);
	progressDlg.setMinimumDuration(0);
	progressDlg.setModal(true);
	progressDlg.show();
	QApplication::processEvents();

	//current color scale
	ccColorScale::Shared colorScale = m_colorScaleSelector->getSelectedScale();
	if (!colorScale)
	{
		if (m_app)
			m_app->dispToConsole(QString("No color scale chosen!"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}

	//create new texture QImage
	QImage mapImage = DistanceMapGenerationTool::ConvertMapToImage(m_map, colorScale, colorScaleStepsSpinBox->value());
	if (mapImage.isNull())
	{
		if (m_app)
			m_app->dispToConsole(QString("Failed to create map texture! Not enough memory?"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}

	for (size_t i=0; i<texturedEntities.size(); ++i)
	{
		//we release the old texture!
		//texturedEntities[i]->setDisplay(0);
		//texturedEntities[i]->setDisplay(m_window);
		
		//set new image as texture
		if (mode == PROJ_CYLINDRICAL && texturedEntities[i]->isA(CV_TYPES::PLANE))
		{
			if (!static_cast<ccPlane*>(texturedEntities[i])->setAsTexture(mapImage))
			{
				if (m_app)
					m_app->dispToConsole(QString("Not enough memory to update the map!"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
				return;
			}
		}
		if (mode == PROJ_CONICAL && texturedEntities[i]->isA(CV_TYPES::MESH))
		{
			ccMesh* mesh = static_cast<ccMesh*>(texturedEntities[i]);
			//set material
			//ccMaterialSet* materialSet = const_cast<ccMaterialSet*>(mesh->getMaterialSet());
			//assert(materialSet);
			//remove old material (if any)
			//materialSet->clear();
			//add new material
			//{
			//	ccMaterial::Shared material(new ccMaterial("texture"));
			//	material->setTexture(mapImage,QString(),false);
			//	materialSet->addMaterial(material);
			//}
		}
	}

	ecvDisplayTools::RedrawDisplay();
}

void DistanceMapGenerationDlg::colorScaleChanged(int)
{
	if (!m_window || !m_colorScaleSelector)
		return;

	ccScalarField* sf = m_window->getAssociatedScalarField();
	if (sf)
	{
		ccColorScale::Shared colorScale = m_colorScaleSelector->getSelectedScale();
		unsigned steps = static_cast<unsigned>(colorScaleStepsSpinBox->value());

		sf->setColorScale(colorScale);
		sf->setColorRampSteps(steps);

		ecvDisplayTools::RedrawDisplay();
	}

	//same thing with textures
	updateMapTexture();
}

void DistanceMapGenerationDlg::spawnColorScaleEditor()
{
	if (!m_app || !m_app->getColorScalesManager())
		return;

	ccColorScale::Shared colorScale = (m_colorScaleSelector ? m_colorScaleSelector->getSelectedScale() : m_app->getColorScalesManager()->getDefaultScale(ccColorScalesManager::BGYR));
	ccColorScaleEditorDialog cseDlg(m_app->getColorScalesManager(),m_app,colorScale,m_app->getMainWindow());
	if (cseDlg.exec())
	{
		colorScale = cseDlg.getActiveScale();
		if (colorScale && m_colorScaleSelector)
		{
			m_colorScaleSelector->init(); //in fact it's a 're-init'
			m_colorScaleSelector->setSelectedScale(colorScale->getUuid());
		}
		
		//save current scale manager state to persistent settings
		m_app->getColorScalesManager()->toPersistentSettings();
	}
}

//helper
static void SetSpinBoxValues(	QDoubleSpinBox* spinBox,
								int decimals,
								double minVal,
								double maxVal,
								double step,
								double value)
{
	if (spinBox)
	{
		spinBox->setDecimals(decimals);
		spinBox->setRange(minVal,maxVal);
		spinBox->setSingleStep(step);
		spinBox->setValue(value);
	}
}

void DistanceMapGenerationDlg::angularUnitChanged(int index)
{
	//backup previous value
	double xStep_rad		= getSpinboxAngularValue(xStepDoubleSpinBox,		ANG_RAD);
	double xMin_rad			= getSpinboxAngularValue(xMinDoubleSpinBox,			ANG_RAD);
	double xMax_rad			= getSpinboxAngularValue(xMaxDoubleSpinBox,			ANG_RAD);
	double scaleXStep_rad	= getSpinboxAngularValue(scaleXStepDoubleSpinBox,	ANG_RAD);

	//same for latitude-related spinboxes
	double latStep_rad		= getSpinboxAngularValue(latStepDoubleSpinBox,		ANG_RAD);
	double latMin_rad		= getSpinboxAngularValue(latMinDoubleSpinBox,		ANG_RAD);
	double latMax_rad		= getSpinboxAngularValue(latMaxDoubleSpinBox,		ANG_RAD);
	double scaleLatStep_rad	= getSpinboxAngularValue(scaleLatStepDoubleSpinBox,	ANG_RAD);

	switch(index)
	{
	case 0: //degrees
		{
		m_angularUnits = ANG_DEG;

        SetSpinBoxValues(xStepDoubleSpinBox,		2, 0.01, 360.0, 0.1,    cloudViewer::RadiansToDegrees( xStep_rad )       );
        SetSpinBoxValues(scaleXStepDoubleSpinBox,	2, 0.01, 360.0, 5.0,    cloudViewer::RadiansToDegrees( scaleXStep_rad )  );
        SetSpinBoxValues(xMinDoubleSpinBox,			2,  0.0, 360.0, 5.0,    cloudViewer::RadiansToDegrees( xMin_rad )        );
        SetSpinBoxValues(xMaxDoubleSpinBox,			2,  0.0, 360.0, 5.0,    cloudViewer::RadiansToDegrees( xMax_rad )        );

        SetSpinBoxValues(latStepDoubleSpinBox,		2,   0.01, 89.99, 1.0,  cloudViewer::RadiansToDegrees( latStep_rad )     );
        SetSpinBoxValues(scaleLatStepDoubleSpinBox,	2,   0.01, 89.99, 1.0,  cloudViewer::RadiansToDegrees( scaleLatStep_rad ));
        SetSpinBoxValues(latMinDoubleSpinBox,		2, -89.99, 89.99, 1.0,  cloudViewer::RadiansToDegrees( latMin_rad )      );
        SetSpinBoxValues(latMaxDoubleSpinBox,		2, -89.99, 89.99, 1.0,  cloudViewer::RadiansToDegrees( latMax_rad)       );

		xMaxDoubleSpinBox->setMaximum(360.0);
		xMaxDoubleSpinBox->setValue(360.0);
		break;
		}

	case 1: //radians
		{
		m_angularUnits = ANG_RAD;

		double PIx2 = 2.0*M_PI;
		SetSpinBoxValues(xStepDoubleSpinBox,		4, 0.0001, PIx2,	0.1, xStep_rad);
		SetSpinBoxValues(scaleXStepDoubleSpinBox,	4, 0.0001, PIx2,	0.5, scaleXStep_rad);
		SetSpinBoxValues(xMinDoubleSpinBox,			4,    0.0, PIx2,	0.5, xMin_rad);
		SetSpinBoxValues(xMaxDoubleSpinBox,			4,    0.0, PIx2,	0.5, xMax_rad);

		double PIdiv2 = M_PI/2.0-0.0001;
		SetSpinBoxValues(scaleLatStepDoubleSpinBox,	4,  0.0001, PIdiv2, 0.3, scaleLatStep_rad);
		SetSpinBoxValues(latStepDoubleSpinBox,		4,  0.0001, PIdiv2, 0.3, latStep_rad);
		SetSpinBoxValues(latMinDoubleSpinBox,		4, -PIdiv2, PIdiv2, 0.3, latMin_rad);
		SetSpinBoxValues(latMaxDoubleSpinBox,		4, -PIdiv2, PIdiv2, 0.3, latMax_rad);

		xMaxDoubleSpinBox->setMaximum(PIx2);
		xMaxDoubleSpinBox->setValue(PIx2);
		break;
		}
	
	case 2: //grades
		{		
		m_angularUnits = ANG_GRAD;

		SetSpinBoxValues(xStepDoubleSpinBox,		2, 0.01, 400.0, 0.1, xStep_rad			* 200.0 / M_PI);
		SetSpinBoxValues(scaleXStepDoubleSpinBox,	2, 0.01, 400.0, 5.0, scaleXStep_rad		* 200.0 / M_PI);
		SetSpinBoxValues(xMinDoubleSpinBox,			2,  0.0, 400.0, 5.0, xMin_rad			* 200.0 / M_PI);
		SetSpinBoxValues(xMaxDoubleSpinBox,			2,  0.0, 400.0, 5.0, xMax_rad			* 200.0 / M_PI);

		SetSpinBoxValues(scaleLatStepDoubleSpinBox,	2,   0.01, 99.99, 1.0, scaleLatStep_rad	* 200.0 / M_PI);
		SetSpinBoxValues(latStepDoubleSpinBox,		2,   0.01, 99.99, 1.0, latStep_rad		* 200.0 / M_PI);
		SetSpinBoxValues(latMinDoubleSpinBox,		2, -99.99, 99.99, 1.0, latMin_rad		* 200.0 / M_PI);
		SetSpinBoxValues(latMaxDoubleSpinBox,		2, -99.99, 99.99, 1.0, latMax_rad		* 200.0 / M_PI);

		xMaxDoubleSpinBox->setMaximum(400.0);
		xMaxDoubleSpinBox->setValue(400.0);
		break;
		}

	default: //shouldn't happen!
		assert(false);
	}

	//update spinboxes suffix
	{
		QString suffix = QString(" ")+getAngularUnitString();
		scaleXStepDoubleSpinBox->setSuffix(suffix);
		latMinDoubleSpinBox->setSuffix(suffix);
		latMaxDoubleSpinBox->setSuffix(suffix);
		scaleLatStepDoubleSpinBox->setSuffix(suffix);
	}
	
	updateOverlayGrid();
}

void DistanceMapGenerationDlg::getGridXValues(double& minX, double& maxX, double& step, ANGULAR_UNIT unit/*=ANG_RAD*/) const
{
	minX = getSpinboxAngularValue(xMinDoubleSpinBox, unit);
	maxX = getSpinboxAngularValue(xMaxDoubleSpinBox, unit);
	step = getSpinboxAngularValue(xStepDoubleSpinBox, unit);
}

void DistanceMapGenerationDlg::getGridYValues(double& minY, double& maxY, double& step, ANGULAR_UNIT unit/*=ANG_RAD*/) const
{
	switch (getProjectionMode())
	{
	case PROJ_CYLINDRICAL:
		minY = hMinDoubleSpinBox->value();
		maxY = hMaxDoubleSpinBox->value();
		step = hStepDoubleSpinBox->value();
		break;
	case PROJ_CONICAL:
		minY = getSpinboxAngularValue(latMinDoubleSpinBox, unit);
		maxY = getSpinboxAngularValue(latMaxDoubleSpinBox, unit);
		step = getSpinboxAngularValue(latStepDoubleSpinBox, unit);
		break;
	default:
		assert(false);
		break;
	}
}

double DistanceMapGenerationDlg::getScaleYStep(ANGULAR_UNIT unit/*=ANG_RAD*/) const
{
	if (getProjectionMode() == PROJ_CYLINDRICAL)
		return scaleHStepDoubleSpinBox->value();
	else
		return getSpinboxAngularValue(scaleLatStepDoubleSpinBox, unit);
}

void DistanceMapGenerationDlg::updateProfileRevolDim(int dim)
{
	if (!m_profile)
	{
		assert(false);
		return;
	}

	//update projection dimension
	assert(dim >= 0 && dim < 3);
	DistanceMapGenerationTool::SetPoylineRevolDim(m_profile, dim);
}

void DistanceMapGenerationDlg::updateProfileOrigin()
{
	if (!m_profile)
	{
		assert(false);
		return;
	}

	DistanceMapGenerationTool::ProfileMetaData profileDesc;
	DistanceMapGenerationTool::GetPoylineMetaData(m_profile, profileDesc);

	//update origin
	CCVector3 origin(	static_cast<PointCoordinateType>(xOriginDoubleSpinBox->value()),
						static_cast<PointCoordinateType>(yOriginDoubleSpinBox->value()),
						static_cast<PointCoordinateType>(zOriginDoubleSpinBox->value()) );

	//DGM: we must compensate for the change of shift along the revolution axis!
	double dShift = origin.u[profileDesc.revolDim] - profileDesc.origin.u[profileDesc.revolDim];
	profileDesc.heightShift -= dShift;
	
	DistanceMapGenerationTool::SetPoylineOrigin(m_profile, origin);
	DistanceMapGenerationTool::SetPolylineHeightShift(m_profile, profileDesc.heightShift);

	if (dShift != 0)
	{
		clearOverlaySymbols(); //symbols placement depend on the origin position along the revolution axis
	}
	updateMinAndMaxLimits();
}

void DistanceMapGenerationDlg::updateGridSteps()
{
	//angular step
	QString xStepsStr;
	{
		double minX, maxX, step;
		getGridXValues(minX, maxX, step, m_angularUnits);
		xStepsStr = (step > 0 ? QString::number(ceil(std::max(maxX-minX,0.0)/ step)) : "inf");
	}

	//Y step
	QString yStepsStr;
	{
		double minY, maxY, step;
		getGridYValues(minY, maxY, step, m_angularUnits);
		yStepsStr = (step > 0 ? QString::number(ceil(std::max(maxY-minY,0.0)/ step)) : "inf");
	}

	gridSizeLabel->setText(QString("%1 x %2").arg(xStepsStr).arg(yStepsStr));
}

double DistanceMapGenerationDlg::getBaseRadius() const
{
	return getProjectionMode() == PROJ_CONICAL ? 1.0 : baseRadiusDoubleSpinBox->value();
}

void DistanceMapGenerationDlg::baseRadiusChanged(double)
{
	if (!m_window)
		return;

	ecvViewportParameters params = ecvDisplayTools::GetViewportParameters();
    params.cameraAspectRatio = static_cast<float>( getBaseRadius() );
	ecvDisplayTools::SetViewportParameters(params);
	ecvDisplayTools::RedrawDisplay();
}

QString DistanceMapGenerationDlg::getHeightUnitString() const
{
	return heightUnitLineEdit->text();
}

DistanceMapGenerationTool::FillStrategyType DistanceMapGenerationDlg::getFillingStrategy() const
{
	switch(fillingStrategyComboxBox->currentIndex())
	{
	case 0:
		return DistanceMapGenerationTool::FILL_STRAT_MIN_DIST;
	case 1:
		return DistanceMapGenerationTool::FILL_STRAT_AVG_DIST;
	case 2:
		return DistanceMapGenerationTool::FILL_STRAT_MAX_DIST;
	default:
		return DistanceMapGenerationTool::INVALID_STRATEGY_TYPE;
	}
	return DistanceMapGenerationTool::INVALID_STRATEGY_TYPE;
}

DistanceMapGenerationTool::EmptyCellFillOption DistanceMapGenerationDlg::getEmptyCellFillingOption() const
{
	switch(emptyCellsComboBox->currentIndex())
	{
	case 0:
		return DistanceMapGenerationTool::LEAVE_EMPTY;
	case 1:
		return DistanceMapGenerationTool::FILL_WITH_ZERO;
	case 2:
		return DistanceMapGenerationTool::FILL_INTERPOLATE;
	default:
		return DistanceMapGenerationTool::LEAVE_EMPTY;
	}
	return DistanceMapGenerationTool::LEAVE_EMPTY;
}

QSharedPointer<DistanceMapGenerationTool::Map> DistanceMapGenerationDlg::updateMap()
{
	if (!m_cloud || !m_sf || !m_profile)
	{
		assert(false);
		return QSharedPointer<DistanceMapGenerationTool::Map>(0);
	}

	//profile parameters
	DistanceMapGenerationTool::ProfileMetaData profileDesc;
	if (!DistanceMapGenerationTool::GetPoylineMetaData(m_profile, profileDesc))
	{
		assert(false);
		return QSharedPointer<DistanceMapGenerationTool::Map>(0);
	}
	
	//compute transformation from cloud to the surface (of revolution)
	ccGLMatrix cloudToSurface = profileDesc.computeCloudToSurfaceOriginTrans();

	//steps
	double angStep_rad = getSpinboxAngularValue(xStepDoubleSpinBox, ANG_RAD);
	//CW (clockwise) or CCW (counterclockwise)
	bool ccw = ccwCheckBox->isChecked();

	//Y values
	double yMin, yMax, yStep;
	getGridYValues(yMin, yMax, yStep, ANG_RAD);
	
	//generate map
	return DistanceMapGenerationTool::CreateMap(m_cloud,
		m_sf,
		cloudToSurface,
		profileDesc.revolDim,
		angStep_rad,
		yStep,
		yMin,
		yMax,
		getProjectionMode() == PROJ_CONICAL,
		ccw,
		getFillingStrategy(),
		getEmptyCellFillingOption(),
		m_app );
}

void DistanceMapGenerationDlg::exportMapAsCloud()
{
	if (!m_map)
	{
		if (m_app)
			m_app->dispToConsole(QString("Invalid map! Try to refresh it?"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}
	if (!m_profile)
	{
		if (m_app)
			m_app->dispToConsole(QString("Invalid profile?!"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}

	double baseRadius = getBaseRadius();

	ccPointCloud* cloud = DistanceMapGenerationTool::ConvertMapToCloud(m_map,m_profile,baseRadius);
	if (m_colorScaleSelector)
		cloud->getCurrentDisplayedScalarField()->setColorScale(m_colorScaleSelector->getSelectedScale());
	cloud->setName(m_cloud->getName()+QString(".map(%1,%2)").arg(m_map->xSteps).arg(m_map->ySteps));

	if (cloud && m_app)
		m_app->addToDB(cloud);
}

void DistanceMapGenerationDlg::exportMapAsMesh()
{
	if (!m_profile || !m_colorScaleSelector)
	{
		assert(false);
		return;
	}
	
	if (!m_map)
	{
		if (m_app)
			m_app->dispToConsole(QString("Invalid map! Try to refresh it?"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}

	//profile parameters
	DistanceMapGenerationTool::ProfileMetaData profileDesc;
	if (!DistanceMapGenerationTool::GetPoylineMetaData(m_profile, profileDesc))
	{
		assert(false);
		return;
	}

	ccColorScale::Shared colorScale = m_colorScaleSelector->getSelectedScale();

	//create new texture QImage
	QImage mapImage = DistanceMapGenerationTool::ConvertMapToImage(m_map, colorScale, colorScaleStepsSpinBox->value());
	if (mapImage.isNull())
	{
		if (m_app)
			m_app->dispToConsole(QString("Failed to generate mesh texture! Not enough memory?"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}

	//compute transformation from cloud to the profile (origin)
	ccGLMatrix cloudToProfile = profileDesc.computeCloudToProfileOriginTrans();
	ccMesh* mesh = 
		DistanceMapGenerationTool::ConvertProfileToMesh(
		m_profile, cloudToProfile, 
		m_map->counterclockwise,
		m_map->xSteps, mapImage);

	if (mesh)
	{
		//mesh->setDisplay_recursive(m_cloud->getDisplay());
		mesh->setName(m_cloud->getName()+QString(".map(%1,%2)").arg(m_map->xSteps).arg(m_map->ySteps));
		if (m_app)
			m_app->addToDB(mesh);
	}
	else
	{
		if (m_app)
			m_app->dispToConsole(QString("Failed to generate mesh! Not enough memory?"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
	}
}

void DistanceMapGenerationDlg::exportMapAsGrid()
{
	if (!m_map)
	{
		if (m_app)
			m_app->dispToConsole(QString("Invalid map! Try to refresh it?"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}

	//persistent settings (default export path)
	QSettings settings;
	settings.beginGroup("qSRA");
	QString path = settings.value("exportPath", ecvFileUtils::defaultDocPath()).toString();

	QString filter("Grid file (*.csv)");

	//open file saving dialog
	QString filename = QFileDialog::getSaveFileName(0,"Select output file",path,filter);
	if (filename.isEmpty())
		return;

	//save current export path to persistent settings
	settings.setValue("exportPath",QFileInfo(filename).absolutePath());

	QString xUnit = getAngularUnitString();
	double xConversionFactor = ConvertAngleFromRad(1.0,m_angularUnits);
	QString yUnit = getHeightUnitString();
	double yConversionFactor = 1.0;

	if (DistanceMapGenerationTool::SaveMapAsCSVMatrix(m_map, filename, xUnit, yUnit, xConversionFactor, yConversionFactor, m_app))
	{
		if (m_app)
			m_app->dispToConsole(QString("File '%1' saved successfully").arg(filename),ecvMainAppInterface::STD_CONSOLE_MESSAGE);
	}
}

void DistanceMapGenerationDlg::exportMapAsImage()
{
	if (!m_window)
		return;

	ccRenderToFileDlg rtfDlg(ecvDisplayTools::GlWidth(), ecvDisplayTools::GlHeight(), m_app->getMainWindow());
	rtfDlg.hideOptions();

	if (rtfDlg.exec())
	{
		QApplication::processEvents();
		ecvDisplayTools::RenderToFile(
			rtfDlg.getFilename(), 
			rtfDlg.getZoom(), 
			rtfDlg.dontScalePoints(), 
			rtfDlg.renderOverlayItems());
	}
}

void DistanceMapGenerationDlg::exportProfilesAsDXF()
{
	if (!m_map || !m_profile)
		return;

	DxfProfilesExportDlg dpeDlg(this);

	if (!dpeDlg.exec())
		return;

	//profile meta-data (we only need the height shift)
	PointCoordinateType heightShift = 0;
	DistanceMapGenerationTool::GetPolylineHeightShift(m_profile, heightShift);

	DxfProfilesExporter::Parameters params;
	params.legendTheoProfileTitle = dpeDlg.theoNameLineEdit->text();
	params.legendRealProfileTitle = dpeDlg.realNameLineEdit->text();
	params.scaledDevUnits = dpeDlg.scaledDevUnitsLineEdit->text();
	params.devLabelMultCoef = dpeDlg.devValuesScaleDoubleSpinBox->value();
	params.devMagnifyCoef = dpeDlg.magnifyCoefSpinBox->value();
	params.precision = dpeDlg.precisionSpinBox->value();

	/*** vertical profiles ***/

	int angularStepCount = dpeDlg.angularStepsSpinBox->value();
	assert(angularStepCount >= 1);
	//we take the same steps as the overlay grid for labels
	QString vertFilename = dpeDlg.getVertFilename();
	if (!vertFilename.isNull())
	{
		//generate profiles titles
		params.profileTitles.clear();
		QString vertProfileBaseTitle = dpeDlg.vertTitleLineEdit->text();
		for (int i=0; i<angularStepCount; ++i)
		{
			double angle_rad = static_cast<double>(i) * 2.0 * M_PI / angularStepCount;
			double angle_cur = ConvertAngleFromRad(angle_rad, m_angularUnits);
			params.profileTitles << QString("%1 - %2 %3").arg(vertProfileBaseTitle).arg(angle_cur).arg(getAngularUnitString());
		}

		double heightStep = getScaleYStep();
		if (!DxfProfilesExporter::SaveVerticalProfiles(	m_map,
														m_profile,
														vertFilename,
														angularStepCount,
														heightStep,
														heightShift,
														params,
														m_app))
		{
			if (m_app)
				m_app->dispToConsole(QString("Failed to save file '%1'!").arg(vertFilename),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
			return;
		}
		else
		{
			if (m_app)
				m_app->dispToConsole(QString("File '%1' saved successfully").arg(vertFilename),ecvMainAppInterface::STD_CONSOLE_MESSAGE);
		}
	}

	/*** horizontal profiles ***/

	QString horizFilename = dpeDlg.getHorizFilename();
	int heightStepCount = dpeDlg.heightStepsSpinBox->value();
	assert(heightStepCount >= 1);
	//we take the same steps as the overlay grid for labels
	if (!horizFilename.isNull())
	{
		//generate profiles titles
		params.profileTitles.clear();
		QString horizProfileBaseTitle = dpeDlg.horizTitleLineEdit->text();
		params.profileTitles << QString("%1 - %2 ").arg(horizProfileBaseTitle).arg("%1") + getHeightUnitString();

		double angleStep_rad = getSpinboxAngularValue(scaleXStepDoubleSpinBox,ANG_RAD);
		if (!DxfProfilesExporter::SaveHorizontalProfiles(	m_map,
															m_profile,
															horizFilename,
															heightStepCount,
															heightShift,
															angleStep_rad,
															ConvertAngleFromRad(1.0,m_angularUnits),
															getCondensedAngularUnitString(),
															params,
															m_app))
		{
			if (m_app)
				m_app->dispToConsole(QString("Failed to save file '%1'!").arg(horizFilename),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
			return;
		}
		else
		{
			if (m_app)
				m_app->dispToConsole(QString("File '%1' saved successfully").arg(horizFilename),ecvMainAppInterface::STD_CONSOLE_MESSAGE);
		}
	}
}

void DistanceMapGenerationDlg::loadOverlaySymbols()
{
	//need a valid map
	if (!m_map)
	{
		if (m_app)
			m_app->dispToConsole(QString("Generate a valid map first!"), ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}

	//profile parameters
	DistanceMapGenerationTool::ProfileMetaData profileDesc;
	if (!DistanceMapGenerationTool::GetPoylineMetaData(m_profile, profileDesc))
	{
		assert(false);
		return;
	}

	//persistent settings (default import path)
	QSettings settings;
	settings.beginGroup("qSRA");
	QString path = settings.value("importPath", ecvFileUtils::defaultDocPath()).toString();

	QString filter("Symbols (*.txt)");

	//open file loading dialog
	QString filename = QFileDialog::getOpenFileName(0, "Select symbols file", path, filter);
	if (filename.isEmpty())
		return;

	QFileInfo fileInfo(filename);
	if (!fileInfo.exists()) //?!
	{
		if (m_app)
			m_app->dispToConsole(QString("Failed to find symbol file '%1'?!").arg(filename), ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		return;
	}

	//save current impoort path to persistent settings
	settings.setValue("importPath", fileInfo.absolutePath());

	ccSymbolCloud* symbolCloud = 0;
	//try to load the file (as a "symbol" point cloud)
	{
		QFile file(filename);
		assert(file.exists());
		if (!file.open(QFile::ReadOnly))
		{
			if (m_app)
				m_app->dispToConsole(QString("Failed to open symbol file '%1'!").arg(filename), ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
			return;
		}

		symbolCloud = new ccSymbolCloud(fileInfo.baseName());

		QTextStream stream(&file);
		QString currentLine = stream.readLine();
		bool error = false;
		while (!currentLine.isNull())
		{
			QStringList tokens = currentLine.split(QRegExp("\\s+"), QString::SkipEmptyParts);
			if (tokens.size() == 4)
			{
				bool okX, okY, okZ;
				CCVector3 P(static_cast<PointCoordinateType>(tokens[1].toDouble(&okX)),
							static_cast<PointCoordinateType>(tokens[2].toDouble(&okY)),
							static_cast<PointCoordinateType>(tokens[3].toDouble(&okZ)));

				if (!okX || !okY || !okZ)
				{
					error = true;
					break;
				}

				QString label = tokens[0];
				if (symbolCloud->size() == symbolCloud->capacity())
				{
					if (!symbolCloud->reserveThePointsTable(symbolCloud->size() + 64) || !symbolCloud->reserveLabelArray(symbolCloud->size() + 64))
					{
						if (m_app)
							m_app->dispToConsole(QString("Not enough memory!"), ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
						error = true;
						break;
					}
				}

				//DGM: warning, for historical reasons height values are expressed relative to the profile origin!
				P.u[profileDesc.revolDim] += profileDesc.origin.u[profileDesc.revolDim];

				symbolCloud->addPoint(P);
				symbolCloud->addLabel(label);
			}

			//next line
			currentLine = stream.readLine();
		}

		if (symbolCloud->size() == 0)
		{
			delete symbolCloud;
			symbolCloud = 0;
		}
		else
		{
			symbolCloud->shrinkToFit();
		}

		if (error)
		{
			delete symbolCloud;
			symbolCloud = nullptr;
			
			if (m_app)
				m_app->dispToConsole(QString("An error occurred while loading the file! Result may be incomplete"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
			return;
		}
	}

	if (symbolCloud)
	{
		//unroll the symbol cloud the same way as the input cloud
		if (m_window)
		{
			//compute transformation from cloud to the surface (of revolution)
			ccGLMatrix cloudToSurface = profileDesc.computeCloudToSurfaceOriginTrans();
			//CW (clockwise) or CCW (counterclockwise)
			bool ccw = ccwCheckBox->isChecked();

			if (getProjectionMode() == PROJ_CYLINDRICAL)
			{
				DistanceMapGenerationTool::ConvertCloudToCylindrical(	symbolCloud,
																		cloudToSurface,
																		profileDesc.revolDim,
																		ccw);
			}
			else /*if (getProjectionMode() == PROJ_CONICAL)*/
			{
				double conicalSpanRatio = conicSpanRatioDoubleSpinBox->value();
				DistanceMapGenerationTool::ConvertCloudToConical(	symbolCloud,
																	cloudToSurface,
																	profileDesc.revolDim,
																	m_map->yMin,
																	m_map->yMax,
																	conicalSpanRatio,
																	ccw);
			}
		}
		symbolCloud->setSymbolSize(static_cast<double>(symbolSizeSpinBox->value()));
		symbolCloud->setFontSize(fontSizeSpinBox->value());
		symbolCloud->setVisible(true);
		//symbolCloud->setDisplay(m_window);
		ecvColor::Rgb rgb(	static_cast<ColorCompType>(m_symbolColor.red()),
							static_cast<ColorCompType>(m_symbolColor.green()),
							static_cast<ColorCompType>(m_symbolColor.blue()) );
		symbolCloud->setTempColor(rgb, true);
		if (m_window != nullptr)
		{
			ecvDisplayTools::AddToOwnDB(symbolCloud, false);
			ecvDisplayTools::RedrawDisplay();
		}

		clearLabelsPushButton->setEnabled(true);
		clearLabelsPushButton->setText(QString("Clear (%1)").arg(symbolCloud->size()));
	}
	else
	{
		assert(false);
		delete symbolCloud;
		symbolCloud = 0;
	}
}

void DistanceMapGenerationDlg::clearOverlaySymbols()
{
	if (!m_window)
		return;

	ccHObject::Container clouds;
	ecvDisplayTools::GetOwnDB()->filterChildren(clouds,false,CV_TYPES::POINT_CLOUD);
	
	for (size_t i=0; i<clouds.size(); ++i)
		if (clouds[i] != m_xLabels && clouds[i] != m_yLabels)
			ecvDisplayTools::RemoveFromOwnDB(clouds[i]);
	
	clearLabelsPushButton->setEnabled(false);
	clearLabelsPushButton->setText("Clear");
	ecvDisplayTools::RedrawDisplay();
}

void DistanceMapGenerationDlg::overlaySymbolsSizeChanged(int size)
{
	if (!m_window)
		return;

	double symbolSize = (double)symbolSizeSpinBox->value();

	ccHObject* db = ecvDisplayTools::GetOwnDB();
	for (unsigned i=0; i<db->getChildrenNumber(); ++i)
	{
		ccHObject* child = db->getChild(i);
		if (child->isA(CV_TYPES::POINT_CLOUD)
			&& child != m_xLabels && child != m_yLabels) //don't modify the X an Y label clouds!
		{
			static_cast<ccSymbolCloud*>(child)->setSymbolSize(symbolSize);
		}
	}
	ecvDisplayTools::RedrawDisplay();
}

void DistanceMapGenerationDlg::overlaySymbolsColorChanged()
{
	ccQtHelpers::SetButtonColor(symbolColorButton,m_symbolColor);

	if (!m_window)
		return;

	ecvColor::Rgb rgb(	static_cast<ColorCompType>(m_symbolColor.red()),
						static_cast<ColorCompType>(m_symbolColor.green()),
						static_cast<ColorCompType>(m_symbolColor.blue()) );

	ccHObject* db = ecvDisplayTools::GetOwnDB();
	for (unsigned i=0; i<db->getChildrenNumber(); ++i)
	{
		ccHObject* child = db->getChild(i);
		if (child->isA(CV_TYPES::POINT_CLOUD)
			&& child != m_xLabels && child != m_yLabels) //don't modify the X an Y label clouds!
		{
			child->setTempColor(rgb,true);
		}
	}

	ecvDisplayTools::RedrawDisplay();
}

void DistanceMapGenerationDlg::overlayGridColorChanged()
{
	ccQtHelpers::SetButtonColor(gridColorButton,m_gridColor);

	if (!m_window)
		return;

	ecvColor::Rgb rgb(	static_cast<ColorCompType>(m_gridColor.red()),
						static_cast<ColorCompType>(m_gridColor.green()),
						static_cast<ColorCompType>(m_gridColor.blue()) );

	ccHObject* db = ecvDisplayTools::GetOwnDB();
	for (unsigned i=0; i<db->getChildrenNumber(); ++i)
	{
		ccHObject* child = db->getChild(i);
		if (child->isA(CV_TYPES::POLY_LINE))
		{
			static_cast<ccPolyline*>(child)->setColor(rgb);
		}
	}

	m_xLabels->setTempColor(rgb,true);
	m_yLabels->setTempColor(rgb,true);

	ecvDisplayTools::RedrawDisplay();
}

void DistanceMapGenerationDlg::labelFontSizeChanged(int)
{
	if (!m_window)
		return;

	int fontSize = fontSizeSpinBox->value();

	ccHObject* db = ecvDisplayTools::GetOwnDB();
	for (unsigned i=0; i<db->getChildrenNumber(); ++i)
	{
		ccHObject* child = db->getChild(i);
		if (child->isA(CV_TYPES::POINT_CLOUD))
		{
			static_cast<ccSymbolCloud*>(child)->setFontSize(fontSize);
		}
	}
	
	//update window font-size
	ecvGui::ParamStruct params = ecvDisplayTools::GetDisplayParameters();
	params.defaultFontSize = fontSize;

	ecvDisplayTools::SetDisplayParameters(params);
	ecvDisplayTools::RedrawDisplay();
}

void DistanceMapGenerationDlg::labelPrecisionChanged(int prec)
{
	if (!m_window)
		return;

	//update numerical precision
	ecvGui::ParamStruct params = ecvDisplayTools::GetDisplayParameters();
	params.displayedNumPrecision = prec;
	ecvDisplayTools::SetDisplayParameters(params);

	ecvDisplayTools::RedrawDisplay();
}

void DistanceMapGenerationDlg::colorRampStepsChanged(int)
{
	colorScaleChanged(-1); //dummy index, not used
}

void DistanceMapGenerationDlg::updateOverlayGrid()
{
	toggleOverlayGrid(overlayGridGroupBox->isChecked());
}

void DistanceMapGenerationDlg::toggleOverlayGrid(bool state)
{
	if (!m_window)
		return;

	assert(m_xLabels && m_yLabels);
	if (!m_xLabels || !m_yLabels)
		return;

	//remove any polylines
	{
		ccHObject::Container polylines;
		ecvDisplayTools::GetOwnDB()->filterChildren(polylines,false,CV_TYPES::POLY_LINE);
		for (size_t i=0; i<polylines.size(); ++i)
			ecvDisplayTools::RemoveFromOwnDB(polylines[i]);
	}
	//and labels
	m_xLabels->clear();
	m_yLabels->clear();
	m_xLabels->setVisible(state && xScaleCheckBox->isChecked());
	m_yLabels->setVisible(state && yScaleCheckBox->isChecked());

	if (state && m_map) //on
	{
		ecvColor::Rgb rgb(	static_cast<ColorCompType>(m_gridColor.red()),
							static_cast<ColorCompType>(m_gridColor.green()),
							static_cast<ColorCompType>(m_gridColor.blue()) );

		//we reconstruct the grid and the corresponding labels
		double xMin_rad, xMax_rad, xStep_rad;
		getGridXValues(xMin_rad, xMax_rad, xStep_rad, ANG_RAD);
		double scaleXStep_rad = getSpinboxAngularValue(scaleXStepDoubleSpinBox,ANG_RAD);

		double yMin, yMax, yStep;
		getGridYValues(yMin, yMax, yStep, ANG_RAD);
		double scaleYStep = getScaleYStep(ANG_RAD);

		if (scaleXStep_rad == 0 || scaleYStep == 0)
		{
			if (m_app)
				m_app->dispToConsole(QString("Internal error: invalid step values?!"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
			ecvDisplayTools::RedrawDisplay();
			return;
		}

		unsigned xStepCount = static_cast<unsigned>( ceil( std::max(xMax_rad-xMin_rad,0.0) / scaleXStep_rad) );
		unsigned yStepCount = static_cast<unsigned>( ceil( std::max(yMax-yMin,0.0) / scaleYStep) );

		//correct 'xMax' and 'yMax'
		xMax_rad = xMin_rad + static_cast<double>(xStepCount) * scaleXStep_rad;
		yMax = yMin + static_cast<double>(yStepCount) * scaleYStep;

		//projection mode
		ProjectionMode mode = getProjectionMode();
		double nProj = 1.0;
		if (mode == PROJ_CONICAL)
		{
			double conicalSpanRatio = conicSpanRatioDoubleSpinBox->value();
			nProj = DistanceMapGenerationTool::ConicalProjectN(m_map->yMin,m_map->yMax) * conicalSpanRatio;
		}
		bool ccw = ccwCheckBox->isChecked();

		//create vertical polylines
		{
			QString angularUnitsStr = getCondensedAngularUnitString();
			if (m_xLabels->isVisible())
			{
				if (!m_xLabels->reserve(xStepCount+1) || !m_xLabels->reserveLabelArray(xStepCount+1))
				{
					if (m_app)
						m_app->dispToConsole(QString("Not engouh memory to display the 'X' scale?!"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
					m_xLabels->clear();
					m_xLabels->setVisible(false);
				}
				m_xLabels->setTempColor(rgb,true);
			}
			for (unsigned i=0; i<=xStepCount; ++i)
			{
				double angle_rad = xMin_rad + static_cast<double>(i) * scaleXStep_rad;
				
				CCVector3 Pbottom(	static_cast<PointCoordinateType>(angle_rad),
									static_cast<PointCoordinateType>(yMin),
									0);
				CCVector3 Pup(	static_cast<PointCoordinateType>(angle_rad),
								static_cast<PointCoordinateType>(yMax),
								0);

				if (mode == PROJ_CONICAL)
				{
					//vertical lines remain "straight" lines after Conical projection
					Pbottom = DistanceMapGenerationTool::ProjectPointOnCone(Pbottom.x,Pbottom.y,m_map->yMin,nProj,ccw);
					Pup = DistanceMapGenerationTool::ProjectPointOnCone(Pup.x,Pup.y,m_map->yMin,nProj,ccw);
				}
				Pbottom.z = 1.0;
				Pup.z = 1.0;

				//polyline
				ccPointCloud* vertices = new ccPointCloud(/*QString("Angle %1").arg(static_cast<int>(cloudViewer::RadiansToDegrees(angle_rad)))*/);
				vertices->reserve(2);
				vertices->addPoint(Pbottom);
				vertices->addPoint(Pup);
				ccPolyline* poly = new ccPolyline(vertices);
				poly->addPointIndex(0,2);
				poly->addChild(vertices);
				vertices->setEnabled(false);
				poly->setColor(rgb);
				poly->showColors(true);
				poly->setVisible(true);
				poly->set2DMode(false);
				ecvDisplayTools::AddToOwnDB(poly,false);

				if (m_xLabels->isVisible())
				{
					m_xLabels->addPoint(Pbottom);
					m_xLabels->addLabel(QString("%1%2").arg(ConvertAngleFromRad(angle_rad,m_angularUnits),0,'f',m_angularUnits == ANG_RAD ? 2 : 0).arg(angularUnitsStr));
				}
			}
		}

		//create horizontal polylines
		{
			if (m_yLabels->isVisible())
			{
				if (!m_yLabels->reserve(yStepCount+1) || !m_yLabels->reserveLabelArray(yStepCount+1))
				{
					if (m_app)
						m_app->dispToConsole(QString("Not enough memory to display the 'Y' scale?!"),ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
					m_yLabels->clear();
					m_yLabels->setVisible(false);
				}
				m_yLabels->setTempColor(rgb,true);
			}
			for (unsigned i=0; i<=yStepCount; ++i)
			{
				double y = yMin + static_cast<double>(i) * scaleYStep;
				
				//polyline
				ccPointCloud* vertices = new ccPointCloud(/*QString("Height %1").arg(height)*/);

				if (mode == PROJ_CONICAL)
				{
					//horizontal lines become "curved" lines after Conical projection!
					const unsigned polySteps = 100;
					if (vertices->reserve(polySteps+1))
						for (unsigned j=0; j<=polySteps; ++j)
						{
							double angle_rad = xMin_rad + static_cast<double>(j)/static_cast<double>(polySteps) * (xMax_rad-xMin_rad);
							CCVector3 P = DistanceMapGenerationTool::ProjectPointOnCone(angle_rad,y,m_map->yMin,nProj,ccw);
							P.z = 1.0;
							vertices->addPoint(P);
						}
				}
				else
				{
					CCVector3 Pleft(	static_cast<PointCoordinateType>(xMin_rad),
										static_cast<PointCoordinateType>(y),
										PC_ONE);
					CCVector3 Pright(	static_cast<PointCoordinateType>(xMax_rad),
										static_cast<PointCoordinateType>(y),
										PC_ONE);
					vertices->reserve(2);
					vertices->addPoint(Pleft);
					vertices->addPoint(Pright);
				}
				
				unsigned vertCount = vertices->size();
				if (vertCount)
				{
					ccPolyline* poly = new ccPolyline(vertices);
					poly->addPointIndex(0,vertices->size());
					poly->addChild(vertices);
					vertices->setEnabled(false);
					poly->setColor(rgb);
					poly->showColors(true);
					poly->setVisible(true);
					poly->set2DMode(false);
					ecvDisplayTools::AddToOwnDB(poly,false);
				}
				else
				{
					delete vertices;
					vertices = 0;
				}

				if (mode != PROJ_CONICAL && m_yLabels->isVisible())
				{
					//cylindrical 'mode' labels
					CCVector3 Pleft(static_cast<PointCoordinateType>(xMin_rad),
									static_cast<PointCoordinateType>(y),
									PC_ONE);
					m_yLabels->addPoint(Pleft);
					m_yLabels->addLabel(QString("%1 %2").arg(y).arg(getHeightUnitString()));
				}
			}
		}
	}

	ecvDisplayTools::RedrawDisplay();
}

void DistanceMapGenerationDlg::changeGridColor()
{
	QColor newCol = QColorDialog::getColor(m_gridColor, this);
	if (!newCol.isValid())
		return;

	m_gridColor = newCol;

	overlayGridColorChanged();
}

void DistanceMapGenerationDlg::changeSymbolColor()
{
	QColor newCol = QColorDialog::getColor(m_symbolColor, this);
	if (!newCol.isValid())
		return;

	m_symbolColor = newCol;

	overlaySymbolsColorChanged();
}

void DistanceMapGenerationDlg::toggleColorScaleDisplay(bool state)
{
	if (m_window)
	{
		m_window->showSF(state);
		ecvDisplayTools::RedrawDisplay();
	}
}

void DistanceMapGenerationDlg::updateVolumes()
{
	if (getProjectionMode() == PROJ_CONICAL)
	{
		volumeTextEdit->setText("Cylindrical projection mode only!");
		return;
	}

	if (m_map && m_profile)
	{
		DistanceMapGenerationTool::Measures surfaces;
		DistanceMapGenerationTool::Measures volumes;
		if (DistanceMapGenerationTool::ComputeSurfacesAndVolumes(m_map,m_profile,surfaces,volumes))
		{
			QLocale locale(QLocale::English);
			QString text;
			text.append(QString("[Theoretical]\n"));
			text.append(QString("surface = %1\n").arg(locale.toString(surfaces.theoretical)));
			text.append(QString("volume = %1\n").arg(locale.toString(volumes.theoretical)));
			text.append(QString("\n"));
			text.append(QString("[Actual]\n"));
			text.append(QString("Surface: %1\n").arg(locale.toString(surfaces.total)));
			text.append(QString("Volume: %1\n").arg(locale.toString(volumes.total)));
			text.append(QString("\n"));
			text.append(QString("Positive (deviations) surface:\n%1\n").arg(locale.toString(surfaces.positive)));
			text.append(QString("Negative (deviations) surface:\n%1\n").arg(locale.toString(surfaces.negative)));
			text.append(QString("\n"));
			text.append(QString("Positive volume (gain of matter):\n%1\n").arg(locale.toString(volumes.positive)));
			text.append(QString("Negative volume (loss of matter):\n%1\n").arg(locale.toString(volumes.negative)));
			text.append(QString("Sum:\n%1\n").arg(locale.toString(volumes.positive+volumes.negative)));
			volumeTextEdit->setText(text);
		}
		else
		{
			volumeTextEdit->setText("Volume(s) computation failed!");
		}
	}
	else
	{
		if (!m_map)
			volumeTextEdit->setText("No map!");
		else
			volumeTextEdit->setText("No profile defined!");
	}
}

void DistanceMapGenerationDlg::initFromPersistentSettings()
{
	QSettings settings;
	settings.beginGroup("DistanceMapGenerationDialog");

	//read parameters
	double conicSpanRatio		= settings.value("conicSpanRatio",		conicSpanRatioDoubleSpinBox->value()).toDouble();
	int angularUnit				= settings.value("angularUnit",			angularUnitComboBox->currentIndex()).toInt();
	QString heightUnit			= settings.value("heightUnit",			heightUnitLineEdit->text()).toString();
	double angularStep			= settings.value("angularStep",			xStepDoubleSpinBox->value()).toDouble();
	double heightStep			= settings.value("heightStep",			hStepDoubleSpinBox->value()).toDouble();
	double latitudeStep			= settings.value("latitudeStep",		latStepDoubleSpinBox->value()).toDouble();
	double scaleAngularStep		= settings.value("scaleAngularStep",	scaleXStepDoubleSpinBox->value()).toDouble();
	double scaleHeightStep		= settings.value("scaleHeightStep",		scaleHStepDoubleSpinBox->value()).toDouble();
	double scaleLatitudeStep	= settings.value("scaleLatitudeStep",	scaleLatStepDoubleSpinBox->value()).toDouble();
	bool ccw					= settings.value("CCW",					ccwCheckBox->isChecked()).toBool();
	int fillStrategy			= settings.value("fillStrategy",		fillingStrategyComboxBox->currentIndex()).toBool();
	int emptyCells				= settings.value("emptyCells",			emptyCellsComboBox->currentIndex()).toInt();
	bool showOverlayGrid		= settings.value("showOverlayGrid",		overlayGridGroupBox->isChecked()).toBool();
	bool showXScale				= settings.value("showXScale",			xScaleCheckBox->isChecked()).toBool();
	bool showYScale				= settings.value("showYScale",			yScaleCheckBox->isChecked()).toBool();
	bool showColorScale			= settings.value("showColorScale",		displayColorScaleCheckBox->isChecked()).toBool();
	QString uuid				= settings.value("colorScale",			QString()).toString();
	int colorScaleSteps			= settings.value("colorScaleSteps",		colorScaleStepsSpinBox->value()).toInt();
	int symbolSize				= settings.value("symbolSize",			symbolSizeSpinBox->value()).toInt();
	int fontSize				= settings.value("fontSize",			fontSizeSpinBox->value()).toInt();
	
	//apply parameters
	conicSpanRatioDoubleSpinBox->setValue(conicSpanRatio);
	angularUnitComboBox->setCurrentIndex(angularUnit);
	angularUnitChanged(angularUnit); //force update
	heightUnitLineEdit->setText(heightUnit);
	updateHeightUnits(); //force update
	xStepDoubleSpinBox->setValue(angularStep);
	hStepDoubleSpinBox->setValue(heightStep);
	latStepDoubleSpinBox->setValue(latitudeStep);
	scaleXStepDoubleSpinBox->setValue(scaleAngularStep);
	scaleHStepDoubleSpinBox->setValue(scaleHeightStep);
	scaleLatStepDoubleSpinBox->setValue(scaleLatitudeStep);
	ccwCheckBox->setChecked(ccw);
	fillingStrategyComboxBox->setCurrentIndex(fillStrategy);
	emptyCellsComboBox->setCurrentIndex(emptyCells);
	overlayGridGroupBox->setChecked(showOverlayGrid);
	xScaleCheckBox->setChecked(showXScale);
	yScaleCheckBox->setChecked(showYScale);
	displayColorScaleCheckBox->setChecked(showColorScale);
	if (m_colorScaleSelector && !uuid.isNull())
		m_colorScaleSelector->setSelectedScale(uuid);
	colorScaleStepsSpinBox->setValue(colorScaleSteps);
	symbolSizeSpinBox->setValue(symbolSize);
	fontSizeSpinBox->setValue(fontSize);

	settings.endGroup();
}

void DistanceMapGenerationDlg::saveToPersistentSettings()
{
	QSettings settings;
	settings.beginGroup("DistanceMapGenerationDialog");

	//write parameters
	settings.setValue("conicSpanRatio",		conicSpanRatioDoubleSpinBox->value());
	settings.setValue("angularUnit",		angularUnitComboBox->currentIndex());
	settings.setValue("heightUnit",			heightUnitLineEdit->text());
	settings.setValue("angularStep",		xStepDoubleSpinBox->value());
	settings.setValue("heightStep",			hStepDoubleSpinBox->value());
	settings.setValue("latitudeStep",		latStepDoubleSpinBox->value());
	settings.setValue("scaleAngularStep",	scaleXStepDoubleSpinBox->value());
	settings.setValue("scaleHeightStep",	scaleHStepDoubleSpinBox->value());
	settings.setValue("scaleLatitudeStep",	scaleLatStepDoubleSpinBox->value());
	settings.setValue("CCW",				ccwCheckBox->isChecked());
	settings.setValue("fillStrategy",		fillingStrategyComboxBox->currentIndex());
	settings.setValue("emptyCells",			emptyCellsComboBox->currentIndex());
	settings.setValue("showOverlayGrid",	overlayGridGroupBox->isChecked());
	settings.setValue("showXScale",			xScaleCheckBox->isChecked());
	settings.setValue("showYScale",			yScaleCheckBox->isChecked());
	settings.setValue("showColorScale",		displayColorScaleCheckBox->isChecked());
	if (m_colorScaleSelector)
	{
		ccColorScale::Shared colorScale = m_colorScaleSelector->getSelectedScale();
		if (colorScale)
			settings.setValue("colorScale",			colorScale->getUuid());
	}
	settings.setValue("colorScaleSteps",	colorScaleStepsSpinBox->value());
	settings.setValue("symbolSize",			symbolSizeSpinBox->value());
	settings.setValue("fontSize",			fontSizeSpinBox->value());

	
	settings.endGroup();
}
