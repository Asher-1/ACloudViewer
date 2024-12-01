#pragma once

//##########################################################################
//#                                                                        #
//#                     ACLOUDVIEWER PLUGIN: q3DMASC                       #
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
//#                 COPYRIGHT: Dimitri Lague / CNRS / UEB                  #
//#                                                                        #
//##########################################################################


//Local
#include "CorePoints.h"
#include "ScalarFieldCollector.h"
#include "ScalarFieldWrappers.h"

//Qt
#include <QString>

//system
#include <assert.h>

class ccPointCloud;

namespace CCLib
{
	class ScalarField;
};

namespace masc
{
	//! Generic feature descriptor
	struct Feature
	{
	public: //shortcuts

		//!Shared type
		typedef QSharedPointer<Feature> Shared;

		//! Set of features
		typedef std::vector<Shared> Set;

	public: //enumerators

		//! Feature type
		enum class Type
		{
			PointFeature,			/*!< Point features (scalar field, etc.) */
			NeighborhoodFeature,	/*!< Neighborhood based features for a given scale */
			ContextBasedFeature,	/*!< Contextual based features */
			DualCloudFeature,		/*!< Dual Cloud features: requires 2 point clouds */
			Invalid					/*!< Invalid feature */
		};

		enum Stat
		{
			NO_STAT,
			MEAN,
			MODE, //number with the highest frequency
			MEDIAN,
			STD,
			RANGE,
			SKEW //(SKEW = (MEAN - MODE)/STD)
		};

		static QString StatToString(Stat stat)
		{
			switch (stat)
			{
			case MEAN:
				return "MEAN";
			case MODE:
				return "MODE";
			case MEDIAN:
				return "MEDIAN";
			case STD:
				return "STD";
			case RANGE:
				return "RANGE";
			case SKEW:
				return "SKEW";
			default:
				break;
			};
			return QString();
		}

		enum Operation
		{
			NO_OPERATION, MINUS, PLUS, DIVIDE, MULTIPLY
		};

		static QString OpToString(Operation op)
		{
			switch (op)
			{
			case MINUS:
				return "MINUS";
			case PLUS:
				return "PLUS";
			case DIVIDE:
				return "DIVIDE";
			case MULTIPLY:
				return "MULTIPLY";
			default:
				break;
			};
			return QString();
		}

		//! Sources of values for this feature
		struct Source
		{
			using Set = std::vector<Source>;

			//! Sources types
			enum Type
			{
				ScalarField = 0,
				DimX,
				DimY,
				DimZ,
				Red,
				Green,
				Blue
			};

			Source(Type t = ScalarField, QString n = QString())
				: type(t)
				, name(n)
			{}
			
			Type type;
			QString name;
		};

		//! Extracts the set of 'sources' from a set of features
		static bool ExtractSources(const Set& features, Source::Set& sources);

		//! Saves a set of 'sources' to a file
		static bool SaveSources(const Source::Set& sources, QString filename);

		//! Loads a set of 'sources' from a file
		static bool LoadSources(Source::Set& sources, QString filename);

	public: //methods

		//! Default constructor
		Feature(double p_scale = std::numeric_limits<double>::quiet_NaN(), Source::Type p_source = Source::ScalarField, QString p_sourceName = QString())
			: scale(p_scale)
			, cloud1(nullptr)
			, cloud2(nullptr)
			, source(p_source, p_sourceName)
			, stat(NO_STAT)
			, op(NO_OPERATION)
			, sf1WasAlreadyExisting(false)
			, sf2WasAlreadyExisting(false)
		{}

		//! Destructor
		virtual ~Feature() {}

		//! Returns the type (must be reimplemented by child struct)
		virtual Type getType() const = 0;

		//! Returns the formatted description
		virtual QString toString() const = 0;

		//! Clones this feature
		virtual Feature::Shared clone() const = 0;

		//! Prepares the feature (compute the scalar field, etc.)
        virtual bool prepare(const CorePoints& corePoints, QString& error, cloudViewer::GenericProgressCallback* progressCb = nullptr, SFCollector* generatedScalarFields = nullptr) = 0;

		//! Finishes the feature preparation (update the scalar field, etc.)
		virtual bool finish(const CorePoints& corePoints, QString& error) { /* does nothing by default*/return true; }

		//! Returns whether the feature has an associated scale
		inline bool scaled() const { return std::isfinite(scale); }

		//! Checks the feature definition validity
		virtual bool checkValidity(QString corePointRole, QString &error) const
		{
			unsigned char cloudCount = (cloud1 ? (cloud2 ? 2 : 1) : 0);
			if (cloudCount == 0)
			{
				error = "feature has no associated cloud";
				return false;
			}

			if (stat != NO_STAT && getType() != Type::PointFeature)
			{
				error = "STAT measures can only be defined on Point features";
				return false;
			}

			if (op != NO_OPERATION && cloudCount < 2)
			{
				error = "at least two clouds are required to apply math operations";
				return false;
			}

			return true;
		}

	public: //helpers

		//! Creates (or resets) a scalar field with the given name on the input core points cloud
		static bool CheckSFExistence(ccPointCloud* cloud, const char* resultSFName);

		//! Creates (or resets) a scalar field with the given name on the input core points cloud
		static cloudViewer::ScalarField* PrepareSF(ccPointCloud* cloud, const char* resultSFName, SFCollector* generatedScalarFields/*= nullptr*/, SFCollector::Behavior behavior);

		//! Performs a mathematical operation between two scalars
		static ScalarType PerformMathOp(double s1, double s2, Operation op);

		//! Performs a mathematical operation between two scalar fields (they must have the same size!)
		static bool PerformMathOp(cloudViewer::ScalarField* sf1, const cloudViewer::ScalarField* sf2, Operation op);

		//! Performs a mathematical operation between two scalar fields (they must have the same size!)
		static bool PerformMathOp(const IScalarFieldWrapper& sf1, const IScalarFieldWrapper& sf2, Operation op, cloudViewer::ScalarField* outSF);

	public: //members

		//! Scale (diameter)
		double scale;

		ccPointCloud *cloud1, *cloud2;
		QString cloud1Label, cloud2Label;
	
		Source source; //values source
	
		Stat stat; //only considered if a scale is defined
		Operation op; //only considered if 2 clouds are defined

		bool sf1WasAlreadyExisting;
		bool sf2WasAlreadyExisting; // only considered if a second scalar field may be necessary (PointFeature, NeighborhoodFeature)
	};
}
