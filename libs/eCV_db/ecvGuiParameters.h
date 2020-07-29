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

#ifndef GUI_PARAMETERS_HEADER
#define GUI_PARAMETERS_HEADER

/***************************************************
				GUI parameters
***************************************************/

//! This class manages some persistent parameters (mostly for display)
/** Values of persistent parameters are stored by the system
	(either in the registry or in a separate file depending on the OS).
**/

// LOCAL
#include <ecvColorTypes.h>

//Qt
#include <QString>

class ECV_DB_LIB_API ecvGui
{
public:

	//! GUI parameters
	struct ECV_DB_LIB_API ParamStruct
	{
		//! Light diffuse color (RGBA)
		ecvColor::Rgbaf lightDiffuseColor;
		//! Light ambient color (RGBA)
		ecvColor::Rgbaf lightAmbientColor;
		//! Light specular color (RGBA)
		ecvColor::Rgbaf lightSpecularColor;
		//! Double sided light
		bool lightDoubleSided;

		//! Default mesh diffuse color (front)
		ecvColor::Rgbaf meshFrontDiff;
		//! Default mesh diffuse color (back)
		ecvColor::Rgbaf meshBackDiff;
		//! Default mesh specular color
		ecvColor::Rgbaf meshSpecular;

		//! Default text color
		ecvColor::Rgbub textDefaultCol;
		//! Default 3D points color
		ecvColor::Rgbub pointsDefaultCol;
		//! Background color
		ecvColor::Rgbub backgroundCol;
		//! Labels background color
		ecvColor::Rgbub labelBackgroundCol;
		//! Labels marker color
		ecvColor::Rgbub labelMarkerCol;
		//! Bounding-boxes color
		ecvColor::Rgbub bbDefaultCol;
		
		//! Use background gradient
		bool drawBackgroundGradient;
		//! Decimate meshes when moved
		bool decimateMeshOnMove;
		//! Min mesh size for decimation
		unsigned minLoDMeshSize;
		//! Decimate clouds when moved
		bool decimateCloudOnMove;
		//! Min cloud size for decimation
		unsigned minLoDCloudSize;
		//! Display cross in the middle of the screen
		bool displayCross;
		//! Whether to use VBOs for faster display
		bool useVBOs;

		//! Label marker size
		unsigned labelMarkerSize;

		//! Color scale option: show histogram next to color ramp
		bool colorScaleShowHistogram;
		//! Whether to use shader for color scale display (if available) or not
		bool colorScaleUseShader;
		//! Whether shader for color scale display is available or not
		bool colorScaleShaderSupported;
		//! Color scale ramp width (for display)
		unsigned colorScaleRampWidth;

		//! Default displayed font size
		unsigned defaultFontSize;
		//! Label font size
		unsigned labelFontSize;
		//! Displayed numbers precision
		unsigned displayedNumPrecision;
		//! Labels background opcaity
		unsigned labelOpacity;

		//! Zoom speed (1.0 by default)
		double zoomSpeed;

		//! Octree computation (for picking) behaviors
		enum ComputeOctreeForPicking { ALWAYS = 0, ASK_USER = 1, NEVER = 2 };

		//! Octree computation (for picking) behavior
		ComputeOctreeForPicking autoComputeOctree;

		//! Whether to draw rounded points (slower) or not
		bool drawRoundedPoints;

		//! Default constructor
		ParamStruct();

		//! Resets parameters to default values
		void reset();

		//! Loads from persistent DB
		void fromPersistentSettings();

		//! Saves to persistent DB
		void toPersistentSettings() const;

		//! Returns whether a given parameter is already defined in persistent settings or not
		/** \param paramName the corresponding attribute name
		**/
		bool isInPersistentSettings(QString paramName) const;
	};

	//! Returns the stored values of each parameter.
	static const ParamStruct& Parameters();

	//! Sets GUI parameters
	static void Set(const ParamStruct& params);

	//! Release unique instance (if any)
	static void ReleaseInstance();

protected:

	//! Parameters set
	ParamStruct params;

};

#endif // GUI_PARAMETERS_HEADER
