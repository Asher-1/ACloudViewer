#include "PclCloudLUT.h"
#include <pcl/point_types.h>
//#include <pcl/common/colors.h>
#include <ecvColorTypes.h>

/// Lookup table
static const unsigned char CloudLUT[] =
{
  255 , 255, 255 , // default
  255,  0,   0   , // highted
  0  ,  0,   255   // ground

};


/// Number of colors in Glasbey lookup table
static const unsigned int CloudLUT_SIZE = sizeof (CloudLUT) / (sizeof (CloudLUT[0]) * 3);

pcl::RGB
PclCloudLUT::at (int color_id)
{
  //assert (color_id < CloudLUT_SIZE);
	pcl::RGB color;
	if (color_id == -1 || color_id == -2)
	{
		color_id += 3;
		color.r = CloudLUT[color_id * 3 + 0];
		color.g = CloudLUT[color_id * 3 + 1];
		color.b = CloudLUT[color_id * 3 + 2];
	}
	else if (color_id == 0)
	{
		color.r = CloudLUT[color_id * 3 + 0];
		color.g = CloudLUT[color_id * 3 + 1];
		color.b = CloudLUT[color_id * 3 + 2];
	}
	else if (color_id > 0)
	{
		//color = pcl::GlasbeyLUT::at(color_id + 20);
        ecvColor::Rgb col = ecvColor::LookUpTable::at(color_id);
		color.r = col.r;
		color.g = col.g;
		color.b = col.b;
		//color.r = COLOR_LUT[color_id * 3 + 0];
		//color.g = COLOR_LUT[color_id * 3 + 1];
		//color.b = COLOR_LUT[color_id * 3 + 2];
	}

	return (color);
}

size_t
PclCloudLUT::size ()
{
	return CloudLUT_SIZE;
}

const unsigned char*
PclCloudLUT::data ()
{
	return CloudLUT;
}
