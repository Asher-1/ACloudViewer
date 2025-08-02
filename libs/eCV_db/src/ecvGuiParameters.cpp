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

#include "ecvGuiParameters.h"

// LOCAL
#include "ecvBasicTypes.h"
#include "ecvSingleton.h"
#include "ecvDisplayTools.h"

//Qt
#include <QSettings>

//System
#include <string.h>

//! Unique instance of ecvGui
static ecvSingleton<ecvGui> s_gui;

const ecvGui::ParamStruct& ecvGui::Parameters()
{
	if (!s_gui.instance)
	{
		s_gui.instance = new ecvGui();
		s_gui.instance->params.fromPersistentSettings();
	}

	return s_gui.instance->params;
}

void ecvGui::UpdateParameters() {
	if (s_gui.instance) {
		s_gui.instance->params.initFontSizesIfNeeded();
	}
}

void ecvGui::ReleaseInstance()
{
	s_gui.release();
}

void ecvGui::Set(const ParamStruct& params)
{
	if (!s_gui.instance)
	{
		s_gui.instance = new ecvGui();
		s_gui.instance->params.fromPersistentSettings();
	}

	s_gui.instance->params = params;
}

ecvGui::ParamStruct::ParamStruct()
{
	reset();
}

void ecvGui::ParamStruct::reset()
{
	lightAmbientColor	= ecvColor::night;
	lightSpecularColor	= ecvColor::darker;
	lightDiffuseColor	= ecvColor::bright;
	meshFrontDiff		= ecvColor::defaultMeshFrontDiff;
	meshBackDiff		= ecvColor::defaultMeshBackDiff;
	meshSpecular		= ecvColor::middle;
	pointsDefaultCol	= ecvColor::defaultColor;
	textDefaultCol		= ecvColor::defaultColor;
	backgroundCol		= ecvColor::defaultBkgColor;
	labelBackgroundCol	= ecvColor::defaultLabelBkgColor;
	labelMarkerCol		= ecvColor::defaultLabelMarkerColor;
	bbDefaultCol		= ecvColor::yellow;

	lightDoubleSided			= true;
	drawBackgroundGradient		        = true;
	drawRoundedPoints			= false;
	decimateMeshOnMove			= true;
	minLoDMeshSize				= 2500000;
	decimateCloudOnMove			= true;
	minLoDCloudSize				= 10000000;
	useVBOs					= true;
	displayCross				= true;

	labelMarkerSize				= 5;

	colorScaleShowHistogram		        = true;
	colorScaleUseShader			= false;
	colorScaleShaderSupported	        = false;
	colorScaleRampWidth			= 50;

#ifdef Q_OS_MAC
	defaultFontSize				= 12;
	labelFontSize					= 10;
#else
	defaultFontSize				= 10;
	labelFontSize					= 8;
#endif
	
	displayedNumPrecision		        = 6;
	labelOpacity				= 75;

	zoomSpeed				= 1.0;

	autoComputeOctree			= ASK_USER;
}

void ecvGui::ParamStruct::initFontSizesIfNeeded()
{
	// 只有在QApplication已初始化后才调用
	defaultFontSize = ecvDisplayTools::GetOptimizedFontSize(12);
	labelFontSize = ecvDisplayTools::GetOptimizedFontSize(10);
}

static int c_fColorArraySize  = sizeof(float) * 4;
static int c_ubColorArraySize = sizeof(unsigned char) * 3;

void ecvGui::ParamStruct::fromPersistentSettings()
{
	QSettings settings;
	settings.beginGroup("OpenGL");
	lightAmbientColor	= ecvColor::Rgbaf (reinterpret_cast<float*>        (settings.value("lightAmbientColor",		QByteArray::fromRawData((const char*)ecvColor::darkest.rgba,					c_fColorArraySize )).toByteArray().data()));
	lightSpecularColor	= ecvColor::Rgbaf (reinterpret_cast<float*>        (settings.value("lightSpecularColor",		QByteArray::fromRawData((const char*)ecvColor::darker.rgba,					c_fColorArraySize )).toByteArray().data()));
	lightDiffuseColor	= ecvColor::Rgbaf (reinterpret_cast<float*>        (settings.value("lightDiffuseColor",		QByteArray::fromRawData((const char*)ecvColor::bright.rgba,					c_fColorArraySize )).toByteArray().data()));
	meshFrontDiff		= ecvColor::Rgbaf (reinterpret_cast<float*>        (settings.value("meshFrontDiff",			QByteArray::fromRawData((const char*)ecvColor::defaultMeshFrontDiff.rgba,	c_fColorArraySize )).toByteArray().data()));
	meshBackDiff		= ecvColor::Rgbaf (reinterpret_cast<float*>        (settings.value("meshBackDiff",			QByteArray::fromRawData((const char*)ecvColor::defaultMeshBackDiff.rgba,		c_fColorArraySize )).toByteArray().data()));
	meshSpecular		= ecvColor::Rgbaf (reinterpret_cast<float*>        (settings.value("meshSpecular",			QByteArray::fromRawData((const char*)ecvColor::middle.rgba,					c_fColorArraySize )).toByteArray().data()));
	pointsDefaultCol	= ecvColor::Rgbaub(reinterpret_cast<unsigned char*>(settings.value("pointsDefaultColor",		QByteArray::fromRawData((const char*)ecvColor::defaultColor.rgb,				c_ubColorArraySize)).toByteArray().data()));
	textDefaultCol		= ecvColor::Rgbaub(reinterpret_cast<unsigned char*>(settings.value("textDefaultColor",		QByteArray::fromRawData((const char*)ecvColor::defaultColor.rgb,				c_ubColorArraySize)).toByteArray().data()));
	backgroundCol		= ecvColor::Rgbaub(reinterpret_cast<unsigned char*>(settings.value("backgroundColor",		QByteArray::fromRawData((const char*)ecvColor::defaultBkgColor.rgb,			c_ubColorArraySize)).toByteArray().data()));
	labelBackgroundCol	= ecvColor::Rgbaub(reinterpret_cast<unsigned char*>(settings.value("labelBackgroundColor",	QByteArray::fromRawData((const char*)ecvColor::defaultLabelBkgColor.rgb,		c_ubColorArraySize)).toByteArray().data()));
	labelMarkerCol		= ecvColor::Rgbaub(reinterpret_cast<unsigned char*>(settings.value("labelMarkerColor",		QByteArray::fromRawData((const char*)ecvColor::defaultLabelMarkerColor.rgb,	c_ubColorArraySize)).toByteArray().data()));
	bbDefaultCol		= ecvColor::Rgbaub(reinterpret_cast<unsigned char*>(settings.value("bbDefaultColor",			QByteArray::fromRawData((const char*)ecvColor::yellow.rgb,					c_ubColorArraySize)).toByteArray().data()));

	lightDoubleSided			=                                      settings.value("lightDoubleSided",        true ).toBool();
	drawBackgroundGradient		        =                                      settings.value("backgroundGradient",      true ).toBool();
	drawRoundedPoints			=                                      settings.value("drawRoundedPoints",       false).toBool();
	decimateMeshOnMove			=                                      settings.value("meshDecimation",          true ).toBool();
	minLoDMeshSize				=                                      settings.value("minLoDMeshSize",       2500000 ).toUInt();
	decimateCloudOnMove			=                                      settings.value("cloudDecimation",         true ).toBool();
	minLoDCloudSize				=                                      settings.value("minLoDCloudSize",     10000000 ).toUInt();
	useVBOs				        =                                      settings.value("useVBOs",                 true ).toBool();
	displayCross				=                                      settings.value("crossDisplayed",          true ).toBool();
	labelMarkerSize				= static_cast<unsigned>(std::max(0,    settings.value("labelMarkerSize",         5    ).toInt()));
	colorScaleShowHistogram		        =                                      settings.value("colorScaleShowHistogram", true ).toBool();
	colorScaleUseShader			=                                      settings.value("colorScaleUseShader",     false).toBool();
	//colorScaleShaderSupported	= not saved
	colorScaleRampWidth			= static_cast<unsigned>(std::max(0,    settings.value("colorScaleRampWidth",      50  ).toInt()));
	defaultFontSize				= static_cast<unsigned>(std::max(0,    settings.value("defaultFontSize",          10  ).toInt()));
	labelFontSize				= static_cast<unsigned>(std::max(0,    settings.value("labelFontSize",            8   ).toInt()));
	displayedNumPrecision		        = static_cast<unsigned>(std::max(0,    settings.value("displayedNumPrecision",    6   ).toInt()));
	labelOpacity				= static_cast<unsigned>(std::max(0,    settings.value("labelOpacity",             75  ).toInt()));
	zoomSpeed				=                                      settings.value("zoomSpeed",                1.0 ).toDouble();
	autoComputeOctree			= static_cast<ComputeOctreeForPicking>(settings.value("autoComputeOctree",   ASK_USER ).toInt());

	settings.endGroup();
}

void ecvGui::ParamStruct::toPersistentSettings() const
{
	QSettings settings;
	settings.beginGroup("OpenGL");

	settings.setValue("lightDiffuseColor",        QByteArray((const char*)lightDiffuseColor.rgba,  c_fColorArraySize ));
	settings.setValue("lightAmbientColor",        QByteArray((const char*)lightAmbientColor.rgba,  c_fColorArraySize ));
	settings.setValue("lightSpecularColor",       QByteArray((const char*)lightSpecularColor.rgba, c_fColorArraySize ));
	settings.setValue("meshFrontDiff",            QByteArray((const char*)meshFrontDiff.rgba,      c_fColorArraySize ));
	settings.setValue("meshBackDiff",             QByteArray((const char*)meshBackDiff.rgba,       c_fColorArraySize ));
	settings.setValue("meshSpecular",             QByteArray((const char*)meshSpecular.rgba,       c_fColorArraySize ));
	settings.setValue("pointsDefaultColor",       QByteArray((const char*)pointsDefaultCol.rgb,    c_ubColorArraySize));
	settings.setValue("textDefaultColor",         QByteArray((const char*)textDefaultCol.rgb,      c_ubColorArraySize));
	settings.setValue("backgroundColor",          QByteArray((const char*)backgroundCol.rgb,       c_ubColorArraySize));
	settings.setValue("labelBackgroundColor",     QByteArray((const char*)labelBackgroundCol.rgb,  c_ubColorArraySize));
	settings.setValue("labelMarkerColor",         QByteArray((const char*)labelMarkerCol.rgb,      c_ubColorArraySize));
	settings.setValue("bbDefaultColor",           QByteArray((const char*)bbDefaultCol.rgb,        c_ubColorArraySize));
	settings.setValue("backgroundGradient",       drawBackgroundGradient);
	settings.setValue("drawRoundedPoints",        drawRoundedPoints);
	settings.setValue("meshDecimation",           decimateMeshOnMove);
	settings.setValue("minLoDMeshSize",	          minLoDMeshSize);
	settings.setValue("cloudDecimation",          decimateCloudOnMove);
	settings.setValue("minLoDCloudSize",	      minLoDCloudSize);
	settings.setValue("useVBOs",                  useVBOs);
	settings.setValue("crossDisplayed",           displayCross);
	settings.setValue("labelMarkerSize",          labelMarkerSize);
	settings.setValue("colorScaleShowHistogram",  colorScaleShowHistogram);
	settings.setValue("colorScaleUseShader",      colorScaleUseShader);
	//settings.setValue("colorScaleShaderSupported", not saved);
	settings.setValue("colorScaleRampWidth",      colorScaleRampWidth);
	settings.setValue("defaultFontSize",          defaultFontSize);
	settings.setValue("labelFontSize",            labelFontSize);
	settings.setValue("displayedNumPrecision",    displayedNumPrecision);
	settings.setValue("labelOpacity",             labelOpacity);
	settings.setValue("zoomSpeed",                zoomSpeed);
	settings.setValue("autoComputeOctree",        static_cast<int>(autoComputeOctree));

	settings.endGroup();
}

bool ecvGui::ParamStruct::isInPersistentSettings(QString paramName) const
{
	QSettings settings;
	settings.beginGroup("OpenGL");
	return settings.contains(paramName);
}
