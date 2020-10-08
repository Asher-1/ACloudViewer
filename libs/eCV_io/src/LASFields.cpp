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
//#                       COPYRIGHT: CLOUDVIEWER  project                  #
//#                                                                        #
//##########################################################################

#include "LASFields.h"

//qCC_db
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

LasField::LasField(	LAS_FIELDS fieldType/*=LAS_INVALID*/,
					double defaultVal/*=0*/,
					double min/*=0.0*/,
					double max/*=-1.0*/,
					uint8_t _minPointFormat/* = 0*/)
	: type(fieldType)
	, sf(nullptr)
	, firstValue(0.0)
	, minValue(min)
	, maxValue(max)
	, defaultValue(defaultVal)
	, minPointFormat(_minPointFormat)
{}

bool LasField::GetLASFields(ccPointCloud* cloud, std::vector<LasField>& fieldsToSave, uint8_t minPointFormat)
{
	try
	{
		//official LAS fields
		std::vector<LasField> lasFields;
		lasFields.reserve(14);
		{
			lasFields.emplace_back(LAS_CLASSIFICATION, 0, 0, 255, 0); //unsigned char: between 0 and 255
			lasFields.emplace_back(LAS_CLASSIF_VALUE, 0, 0, 31, 0); //5 bits: between 0 and 31
			lasFields.emplace_back(LAS_CLASSIF_SYNTHETIC, 0, 0, 1, 0); //1 bit: 0 or 1
			lasFields.emplace_back(LAS_CLASSIF_KEYPOINT, 0, 0, 1, 0); //1 bit: 0 or 1
			lasFields.emplace_back(LAS_CLASSIF_WITHHELD, 0, 0, 1, 0); //1 bit: 0 or 1
			lasFields.emplace_back(LAS_CLASSIF_OVERLAP, 0, 0, 1, 6); //1 bit: 0 or 1
			lasFields.emplace_back(LAS_INTENSITY, 0, 0, 65535, 0); //16 bits: between 0 and 65536
			lasFields.emplace_back(LAS_TIME, 0, 0, -1.0, 1); //8 bytes (double)
			lasFields.emplace_back(LAS_RETURN_NUMBER, 1, 1, 7, 0); //3 bits: between 1 and 7
			lasFields.emplace_back(LAS_NUMBER_OF_RETURNS, 1, 1, 7, 0); //3 bits: between 1 and 7
			lasFields.emplace_back(LAS_SCAN_DIRECTION, 0, 0, 1, 0); //1 bit: 0 or 1
			lasFields.emplace_back(LAS_FLIGHT_LINE_EDGE, 0, 0, 1, 0); //1 bit: 0 or 1
			lasFields.emplace_back(LAS_SCAN_ANGLE_RANK, 0, -90, 90, 0); //signed char: between -90 and +90
			lasFields.emplace_back(LAS_USER_DATA, 0, 0, 255, 0); //unsigned char: between 0 and 255
			lasFields.emplace_back(LAS_POINT_SOURCE_ID, 0, 0, 65535, 0); //16 bits: between 0 and 65536
		}

		//we are going to check now the existing cloud SFs
		for (unsigned i = 0; i < cloud->getNumberOfScalarFields(); ++i)
		{
			ccScalarField* sf = static_cast<ccScalarField*>(cloud->getScalarField(i));
			//find an equivalent in official LAS fields
			QString sfName = QString(sf->getName()).toUpper();
			bool outBounds = false;
			for (size_t j = 0; j < lasFields.size(); ++j)
			{
				//if the name matches
				if (sfName == lasFields[j].getName().toUpper())
				{
					//check bounds
					double sfMin = sf->getGlobalShift() + sf->getMax();
					double sfMax = sf->getGlobalShift() + sf->getMax();
					if (sfMin < lasFields[j].minValue || (lasFields[j].maxValue != -1.0 && sfMax > lasFields[j].maxValue)) //outbounds?
					{
						CVLog::Warning(QString("[LAS] Found a '%1' scalar field, but its values outbound LAS specifications (%2-%3)...").arg(sf->getName()).arg(lasFields[j].minValue).arg(lasFields[j].maxValue));
						outBounds = true;
					}
					else
					{
						//we add the SF to the list of saved fields
						fieldsToSave.push_back(lasFields[j]);
						fieldsToSave.back().sf = sf;

						minPointFormat = std::max(minPointFormat, fieldsToSave.back().minPointFormat);
					}
					break;
				}
			}
		}
	}
	catch (const std::bad_alloc&)
	{
		CVLog::Warning("[LasField::GetLASFields] Not enough memory");
		return false;
	}

	return true;
}

unsigned LasField::GetFormatRecordLength(uint8_t pointFormat)
{
	switch (pointFormat)
	{
	case 0:
		return 20;              //0 - base
	case 1:
		return 20 + 8;          //1 - base + GPS
	case 2:
		return 20 + 6;          //2 - base + RGB
	case 3:
		return 20 + 8 + 6;      //3 - base + GPS + RGB
	case 4:
		return 20 + 8 + 29;     //4 - base + GPS + FWF
	case 5:
		return 20 + 8 + 6 + 29; //5 - base + GPS + FWF + RGB
	case 6:
		return 30;              //6  - base (GPS included)
	case 7:
		return 30 + 6;          //7  - base + RGB
	case 8:
		return 30 + 6 + 2;      //8  - base + RGB + NIR (not used)
	case 9:
		return 30 + 29;         //9  - base + FWF
	case 10:
		return 30 + 6 + 2 + 29; //10 - base + RGB + NIR + FWF
	default:
		assert(false);
		return 0;
	}
}

uint8_t LasField::UpdateMinPointFormat(uint8_t minPointFormat, bool withRGB, bool withFWF, bool allowLegacyFormats/* = true*/)
{
	//can we keep the (short) legacy formats?
	if (allowLegacyFormats && minPointFormat < 6)
	{
		//LAS formats:
		//0 - base
		//1 - base + GPS TIME
		//2 - base + RGB
		//3 - base + GPS + RGB
		//4 - base + GPS + FWF
		//5 - base + GPS + FWF + RGB

		if (withFWF)
		{
			//0, 1 --> 4
			minPointFormat = std::max(minPointFormat, (uint8_t)4);
		}

		if (withRGB)
		{
			if (minPointFormat < 2)
			{
				//0 --> 2
				//1 --> 3
				minPointFormat += 2;
			}
			else if (minPointFormat == 4)
			{
				//4 --> 5
				minPointFormat = 5;
			}
			//else the format already has colors
		}
	}
	else //we'll use extended versions (up to 15 returns, up to 256 classes for classification, higher precision scan angle)
	{
		//new LAS formats:
		//6  - base (GPS included)
		//7  - base + RGB
		//8  - base + RGB + NIR (not used)
		//9  - base + FWF
		//10 - base + FWF + RGB + NIR
		assert(minPointFormat <= 6); //in effect, standard LAS fields will only require version 6 at most

		minPointFormat = std::max(minPointFormat, (uint8_t)6);
		//FWF data?
		if (withFWF)
		{
			//6 --> 9
			minPointFormat = std::max(minPointFormat, (uint8_t)9);
		}
		//Colors?
		if (withRGB)
		{
			if (minPointFormat == 6)
			{
				//6 --> 7
				minPointFormat = 7;
			}
			else if (minPointFormat == 9)
			{
				//9 --> 10
				minPointFormat = 10;
			}
		}
	}

	return minPointFormat;
}

QString LasField::SanitizeString(QString str)
{
	QString sanitizedStr;
	if (str.size() > 32)
	{
		sanitizedStr = str.left(32);
	}
	else
	{
		sanitizedStr = str;
	}
	sanitizedStr.replace('=', '_');

	return sanitizedStr;
}
