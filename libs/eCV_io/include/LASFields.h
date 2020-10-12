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

#ifndef ECV_LAS_FIELDS_HEADER
#define ECV_LAS_FIELDS_HEADER

//Local
#include "eCV_io.h"

//Qt
#include <QSharedPointer>

//System
#include <vector>

class ccPointCloud;
class ccScalarField;

static const char LAS_SCALE_X_META_DATA[] = "LAS.scale.x";
static const char LAS_SCALE_Y_META_DATA[] = "LAS.scale.y";
static const char LAS_SCALE_Z_META_DATA[] = "LAS.scale.z";
static const char LAS_OFFSET_X_META_DATA[] = "LAS.offset.x";
static const char LAS_OFFSET_Y_META_DATA[] = "LAS.offset.y";
static const char LAS_OFFSET_Z_META_DATA[] = "LAS.offset.z";
static const char LAS_VERSION_MAJOR_META_DATA[] = "LAS.version.major";
static const char LAS_VERSION_MINOR_META_DATA[] = "LAS.version.minor";
static const char LAS_POINT_FORMAT_META_DATA[] = "LAS.point_format";
static const char LAS_GLOBAL_ENCODING_META_DATA[] = "LAS.global_encoding";
static const char LAS_PROJECT_UUID_META_DATA[] = "LAS.project_uuid";

enum LAS_FIELDS {
	LAS_X = 0,
	LAS_Y = 1,
	LAS_Z = 2,
	LAS_INTENSITY = 3,
	LAS_RETURN_NUMBER = 4,
	LAS_NUMBER_OF_RETURNS = 5,
	LAS_SCAN_DIRECTION = 6,
	LAS_FLIGHT_LINE_EDGE = 7,
	LAS_CLASSIFICATION = 8,
	LAS_SCAN_ANGLE_RANK = 9,
	LAS_USER_DATA = 10,
	LAS_POINT_SOURCE_ID = 11,
	LAS_RED = 12,
	LAS_GREEN = 13,
	LAS_BLUE = 14,
	LAS_TIME = 15,
	LAS_EXTRA = 16,
	//Sub fields
	LAS_CLASSIF_VALUE = 17,
	LAS_CLASSIF_SYNTHETIC = 18,
	LAS_CLASSIF_KEYPOINT = 19,
	LAS_CLASSIF_WITHHELD = 20,
	LAS_CLASSIF_OVERLAP = 21,
	//Invald flag
	LAS_INVALID = 255
};

const char LAS_FIELD_NAMES[][28] = {"X",
									"Y",
									"Z",
									"Intensity",
									"ReturnNumber",
									"NumberOfReturns",
									"ScanDirectionFlag",
									"EdgeOfFlightLine",
									"Classification",
									"ScanAngleRank",
									"UserData",
									"PointSourceId",
									"Red",
									"Green",
									"Blue",
									"GpsTime",
									"extra",
									"[Classif] Value",
									"[Classif] Synthetic flag",
									"[Classif] Key-point flag",
									"[Classif] Withheld flag",
									"[Classif] Overlap flag",
};

//! LAS field descriptor
struct ECV_IO_LIB_API LasField
{
	//! Shared type
	typedef QSharedPointer<LasField> Shared;

	//! Default constructor
	LasField(LAS_FIELDS fieldType = LAS_INVALID, 
		double defaultVal = 0, double min = 0.0, double max = -1.0,
		uint8_t _minPointFormat = 0);

	//! Returns official field name
	virtual inline QString getName() const { return type < LAS_INVALID ? QString(LAS_FIELD_NAMES[type]) : QString(); }

	//! Returns the (compliant) LAS fields in a point cloud
	static bool GetLASFields(ccPointCloud* cloud, std::vector<LasField>& fieldsToSave, uint8_t minPointFormat);

	static unsigned GetFormatRecordLength(uint8_t pointFormat);

	static uint8_t VersionMinorForPointFormat(uint8_t pointFormat) {
		return pointFormat >= 6 ? 4 : 2;
	}

	static uint8_t UpdateMinPointFormat(uint8_t minPointFormat, bool withRGB, 
										bool withFWF, bool allowLegacyFormats = true);

	static QString SanitizeString(QString str);


	LAS_FIELDS type;
	ccScalarField* sf;
	double firstValue;
	double minValue;
	double maxValue;
	double defaultValue;
	uint8_t minPointFormat;
};

#endif // ECV_LAS_FIELDS_HEADER
