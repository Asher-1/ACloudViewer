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
#include "ecv2DViewportLabel.h"
#include "ecvDisplayTools.h"

// CV_CORE_LIB
#include <CVConst.h>

//Qt
#include <QDataStream>
#include <QFontMetrics>

//system
#include <string.h>

cc2DViewportLabel::cc2DViewportLabel(QString name/*=QString()*/)
	: cc2DViewportObject(name)
{
	//label rectangle
	memset(m_roi,0,sizeof(float)*4);
	setVisible(false);
}

void cc2DViewportLabel::setRoi(const float* roi)
{
	memcpy(m_roi,roi,sizeof(float)*4);
}

bool cc2DViewportLabel::toFile_MeOnly(QFile& out) const
{
	if (!cc2DViewportObject::toFile_MeOnly(out))
		return false;

	//ROI (dataVersion>=21)
	QDataStream outStream(&out);
	for (int i=0; i<4; ++i)
		outStream << m_roi[i];

	return true;
}

bool cc2DViewportLabel::fromFile_MeOnly(QFile& in, short dataVersion, int flags)
{
	if (!cc2DViewportObject::fromFile_MeOnly(in, dataVersion, flags))
		return false;

	if (dataVersion < 21)
		return false;

	//ROI (dataVersion>=21)
	QDataStream inStream(&in);
	for (int i=0; i<4; ++i)
		inStream >> m_roi[i];

	return true;
}


void cc2DViewportLabel::clear2Dviews()
{
	if (!ecvDisplayTools::GetCurrentScreen()) return;

	ecvDisplayTools::RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_TRIANGLE_2D, QString::number(getUniqueID())));
}

void cc2DViewportLabel::updateLabel()
{
	CC_DRAW_CONTEXT context;
	ecvDisplayTools::GetContext(context);
	update2DLabelView(context, true);
}

void cc2DViewportLabel::update2DLabelView(
	CC_DRAW_CONTEXT& context,
	bool updateScreen /* = true */)
{
	context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
	drawMeOnly(context);
	if (updateScreen)
	{
		ecvDisplayTools::UpdateScreen();
	}
}

void cc2DViewportLabel::drawMeOnly(CC_DRAW_CONTEXT& context)
{
	//2D foreground only
	if (!MACRO_Foreground(context) || !MACRO_Draw2D(context))
		return;
	
	if (!ecvDisplayTools::GetCurrentScreen())
		return;

	// clear history
	clear2Dviews();
	if (!isVisible() || !isEnabled())
	{
		return;
	}
	
	//test viewport parameters
	const ecvViewportParameters& params =
		ecvDisplayTools::GetViewportParameters();

	//general parameters
	if (	params.perspectiveView != m_params.perspectiveView
		||	params.objectCenteredView != m_params.objectCenteredView
		||	params.pixelSize != m_params.pixelSize)
	{
		return;
	}

    //test base view matrix
    for (unsigned i = 0; i < 12; ++i)
    {
        if ( cloudViewer::GreaterThanEpsilon( fabs(params.viewMat.data()[i] - m_params.viewMat.data()[i]) ) )
        {
            return;
        }
    }


	if (m_params.perspectiveView)
	{
		if (params.fov_deg != m_params.fov_deg || 
			params.cameraAspectRatio != m_params.cameraAspectRatio)
			return;

        if (cloudViewer::GreaterThanEpsilon( (params.getPivotPoint() - m_params.getPivotPoint()).norm() )
            || cloudViewer::GreaterThanEpsilon( (params.getCameraCenter() - m_params.getCameraCenter()).norm() ))
        {
            return;
        }
	}

	float relativeZoom = 1.0f;
	float dx = 0, dy = 0;
	if (!m_params.perspectiveView) //ortho mode
	{
		//Screen pan & pivot compensation
		float totalZoom = m_params.zoom / m_params.pixelSize;
		float winTotalZoom = params.zoom / params.pixelSize;
		relativeZoom = winTotalZoom / totalZoom;

        CCVector3d dC = m_params.getCameraCenter() - params.getCameraCenter();

        CCVector3d P = m_params.getPivotPoint() - params.getPivotPoint();
		m_params.viewMat.apply(P);

		dx = static_cast<float>(dC.x + P.x);
		dy = static_cast<float>(dC.y + P.y);

		dx *= winTotalZoom;
		dy *= winTotalZoom;
	}

	const ecvColor::Rgb* defaultColor = m_selected ? &ecvColor::red : &context.textDefaultCol;
	
	WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_TRIANGLE_2D, QString::number(getUniqueID()));
	const ecvColor::Rgbf& tempColor = ecvColor::FromRgb(*defaultColor);
	param.color.r = tempColor.r;
	param.color.g = tempColor.g;
	param.color.b = tempColor.b;
	param.color.a = 1.0f;
	param.p1 = QPoint(dx + m_roi[0] * relativeZoom, dy + m_roi[1] * relativeZoom);
	param.p2 = QPoint(dx + m_roi[2] * relativeZoom, dy + m_roi[1] * relativeZoom);
	param.p3 = QPoint(dx + m_roi[2] * relativeZoom, dy + m_roi[3] * relativeZoom);
	param.p4 = QPoint(dx + m_roi[0] * relativeZoom, dy + m_roi[3] * relativeZoom);
	ecvDisplayTools::DrawWidgets(param, false);

	//title
	QString title(getName());
	if (!title.isEmpty())
	{
		// takes rendering zoom into account!
		QFont titleFont(ecvDisplayTools::GetTextDisplayFont());
		titleFont.setBold(true);
		QFontMetrics titleFontMetrics(titleFont);
		int titleHeight = titleFontMetrics.height();

		int xStart = static_cast<int>(dx + std::min<float>(m_roi[0], m_roi[2]) * relativeZoom);
		int yStart = static_cast<int>(dy + std::min<float>(m_roi[1], m_roi[3]) * relativeZoom);

		ecvDisplayTools::DisplayText(title, xStart, yStart - 5 - titleHeight,
			ecvDisplayTools::ALIGN_DEFAULT, 0, defaultColor->rgb, &titleFont, QString::number(getUniqueID()));
	}
}
