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

#ifndef qPCL_CUSTOM_CONTEXT_ITEM_H
#define qPCL_CUSTOM_CONTEXT_ITEM_H

//Local
#include "../qPCL.h"
#include "PCLCloud.h"

//PCL
#include <pcl/visualization/vtk/pcl_context_item.h>

namespace PclUtils
{
	namespace context_items
	{
		struct Point : public pcl::visualization::PCLContextItem
		{
			vtkTypeMacro(Point, PCLContextItem);
			static Point *New();
			virtual bool Paint(vtkContext2D *painter);
			virtual void set(float _x, float _y);
		};

		struct Line : public pcl::visualization::PCLContextItem
		{
			vtkTypeMacro(Line, PCLContextItem);
			static Line *New();
			virtual bool Paint(vtkContext2D *painter);
			virtual void set(float _x_1, float _y_1, float _x_2, float _y_2);
		};

		struct Circle : public pcl::visualization::PCLContextItem
		{
			vtkTypeMacro(Circle, PCLContextItem);
			static Circle *New();
			virtual bool Paint(vtkContext2D *painter);
			virtual void set(float _x, float _y, float _r);
		};

		struct Disk : public Circle
		{
			vtkTypeMacro(Disk, Circle);
			static Disk *New();
			virtual bool Paint(vtkContext2D *painter);
		};

		struct Rectangle : public pcl::visualization::PCLContextItem
		{
			vtkTypeMacro(Rectangle, Point);
			static Rectangle *New();
			virtual bool Paint(vtkContext2D *painter);
			virtual void set(float _x, float _y, float _w, float _h);
		};

		struct FilledRectangle : public Rectangle
		{
			vtkTypeMacro(FilledRectangle, Rectangle);
			static FilledRectangle *New();
			virtual bool Paint(vtkContext2D *painter);
		};

		struct Points : public pcl::visualization::PCLContextItem
		{
			vtkTypeMacro(Points, PCLContextItem);
			static Points *New();
			virtual bool Paint(vtkContext2D *painter);
			void set(const std::vector<float>& _xy) { params = _xy; }
		};

		struct Polygon : public Points
		{
			vtkTypeMacro(Polygon, Points);
			static Polygon *New();
			virtual bool Paint(vtkContext2D *painter);
		};

		struct Text : public pcl::visualization::PCLContextItem
		{
			vtkTypeMacro(Text, PCLContextItem);
			static Text *New();
			virtual bool Paint(vtkContext2D *painter);
			virtual void set(float x, float y, const std::string& _text, int fontSize = 10);
			std::string text;
		};

		struct Markers : public Points
		{
			vtkTypeMacro(Markers, Points);
			static Markers *New();
			virtual bool Paint(vtkContext2D *painter);
			void setSize(float _size) { size = _size; }
			void setPointColors(unsigned char r, unsigned char g, unsigned char b);
			void setPointColors(unsigned char rgb[3]);
			float size;
			unsigned char point_colors[3];
		};
	}

}

#endif // qPCL_CUSTOM_CONTEXT_ITEM_H