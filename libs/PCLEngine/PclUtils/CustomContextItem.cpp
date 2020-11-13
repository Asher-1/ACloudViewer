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
//#                         COPYRIGHT: DAHAI LU                            #
//#                                                                        #
//##########################################################################
//

#include "PclUtils/CustomContextItem.h"

#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkContext2D.h>
#include <vtkImageData.h>
#include <vtkPen.h>
#include <vtkBrush.h>
#include <vtkTextProperty.h>

namespace PclUtils
{
	namespace context_items
	{
		vtkStandardNewMacro(Point);
		vtkStandardNewMacro(Line);
		vtkStandardNewMacro(Circle);
		vtkStandardNewMacro(Disk);
		vtkStandardNewMacro(Rectangle);
		vtkStandardNewMacro(FilledRectangle);
		vtkStandardNewMacro(Points);
		vtkStandardNewMacro(Polygon);
		vtkStandardNewMacro(Text);
		vtkStandardNewMacro(Markers);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
void
PclUtils::context_items::Point::set(float x, float y)
{
	params.resize(2);
	params[0] = x; params[1] = y;
}

///////////////////////////////////////////////////////////////////////////////////////////
void
PclUtils::context_items::Circle::set(float x, float y, float radius)
{
	params.resize(4);
	params[0] = x; params[1] = y; params[2] = radius; params[3] = radius - 1;
}

///////////////////////////////////////////////////////////////////////////////////////////
void
PclUtils::context_items::Rectangle::set(float x, float y, float w, float h)
{
	params.resize(4);
	params[0] = x; params[1] = y; params[2] = w; params[3] = h;
}

///////////////////////////////////////////////////////////////////////////////////////////
void
PclUtils::context_items::Line::set(float start_x, float start_y, float end_x, float end_y)
{
	params.resize(4);
	params[0] = start_x; params[1] = start_y; params[2] = end_x; params[3] = end_y;
}

///////////////////////////////////////////////////////////////////////////////////////////
void
PclUtils::context_items::Text::set(float x, float y, const std::string& _text, int fontSize)
{
	params.resize(3);
	params[0] = x; 
	params[1] = y;
	params[2] = fontSize;
	text = _text;
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::Circle::Paint(vtkContext2D *painter)
{
	painter->GetBrush()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->GetPen()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->DrawWedge(params[0], params[1], params[2], params[3], 0.0, 360.0);
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::Disk::Paint(vtkContext2D *painter)
{
	painter->GetBrush()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->GetPen()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->DrawEllipse(params[0], params[1], params[2], params[2]);
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::Rectangle::Paint(vtkContext2D *painter)
{
	painter->GetPen()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	float p[] =
	{
	  params[0], params[1],
	  params[2], params[1],
	  params[2], params[3],
	  params[0], params[3],
	  params[0], params[1]
	};

	painter->DrawPoly(p, 5);
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::FilledRectangle::Paint(vtkContext2D *painter)
{
	painter->GetBrush()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->GetPen()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->DrawRect(params[0], params[1], params[2], params[3]);
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::Line::Paint(vtkContext2D *painter)
{
	painter->GetPen()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->DrawLine(params[0], params[1], params[2], params[3]);
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::Polygon::Paint(vtkContext2D *painter)
{
	painter->GetBrush()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->GetPen()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->DrawPolygon(&params[0], static_cast<int> (params.size() / 2));
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::Point::Paint(vtkContext2D *painter)
{
	painter->GetPen()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->DrawPoint(params[0], params[1]);
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::Points::Paint(vtkContext2D *painter)
{
	painter->GetPen()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->DrawPoints(&params[0], static_cast<int> (params.size() / 2));
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::Text::Paint(vtkContext2D *painter)
{
	vtkTextProperty *text_property = painter->GetTextProp();
	text_property->SetColor(255.0 * colors[0], 255.0 * colors[1], 255.0 * colors[2]);
	text_property->SetOpacity(GetOpacity());
	text_property->SetFontFamilyToArial();
	text_property->SetFontSize(static_cast<int>(params[2]));
	text_property->SetJustificationToLeft();
	text_property->BoldOff();
	text_property->ShadowOff();
	painter->DrawString(params[0], params[1], text.c_str());
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////
void
PclUtils::context_items::Markers::setPointColors(unsigned char r, unsigned char g, unsigned char b)
{
	point_colors[0] = r; point_colors[1] = g; point_colors[2] = b;
}

///////////////////////////////////////////////////////////////////////////////////////////
void
PclUtils::context_items::Markers::setPointColors(unsigned char rgb[3])
{
	memcpy(point_colors, rgb, 3 * sizeof(unsigned char));
}

///////////////////////////////////////////////////////////////////////////////////////////
bool
PclUtils::context_items::Markers::Paint(vtkContext2D *painter)
{
	int nb_points(params.size() / 2);
	if (size <= 0)
		size = 2.3 * painter->GetPen()->GetWidth();

	painter->GetPen()->SetWidth(size);
	painter->GetPen()->SetColor(colors[0], colors[1], colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->DrawPointSprites(0, &params[0], nb_points);
	painter->GetPen()->SetWidth(1);
	painter->GetPen()->SetColor(point_colors[0], point_colors[1], point_colors[2], static_cast<unsigned char> ((255.0 * GetOpacity())));
	painter->DrawPointSprites(0, &params[0], nb_points);
	return (true);
}
