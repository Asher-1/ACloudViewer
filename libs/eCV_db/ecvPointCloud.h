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

#ifndef ECV_POINT_CLOUD_HEADER
#define ECV_POINT_CLOUD_HEADER

#ifdef _MSC_VER
//To get rid of the warnings about dominant inheritance
#pragma warning( disable: 4250 )
#endif

//CVLib
#include <PointCloudTpl.h>

//Local
#include "ecvKDTreeSearchParam.h"
#include "ecvColorScale.h"
#include "ecvNormalVectors.h"
#include "ecvWaveform.h"

//Qt
#include <QGLBuffer>

class ccMesh;
class QGLBuffer;
class ccPolyline;
class ccScalarField;
class ecvOrientedBBox;
class ccPointCloudLOD;
class ecvProgressDialog;

namespace cloudViewer
{
	namespace geometry
	{
		class Image;
		class RGBDImage;
		class VoxelGrid;
	}
	namespace camera
	{
		class PinholeCameraIntrinsic;
	}
}

/***************************************************
				ccPointCloud
***************************************************/

//! Max number of points per cloud (point cloud will be chunked above this limit)
#if defined(CV_ENV_32)
const unsigned CC_MAX_NUMBER_OF_POINTS_PER_CLOUD =  128000000;
#else // CV_ENV_64 (but maybe CC_ENV_128 one day ;)
const unsigned CC_MAX_NUMBER_OF_POINTS_PER_CLOUD = 2000000000; //we must keep it below MAX_INT to avoid probable issues ;)
#endif

//! A 3D cloud and its associated features (color, normals, scalar fields, etc.)
/** A point cloud can have multiple features:
	- colors (RGB)
	- normals (compressed)
	- scalar fields
	- an octree strucutre
	- per-point visibility information (to hide/display subsets of points)
	- other children objects (meshes, calibrated pictures, etc.)
**/
class ECV_DB_LIB_API ccPointCloud : public CVLib::PointCloudTpl<ccGenericPointCloud>
{
public:
	//! Base class (shortcut)
	using BaseClass = CVLib::PointCloudTpl<ccGenericPointCloud>;

	//! Default constructor
	/** Creates an empty cloud without any feature. Each of them shoud be
		specifically instantiated/created (once the points have been
		added to this cloud, at least partially).
		\param name cloud name (optional)
	**/
	ccPointCloud(QString name = QString()) throw();
	ccPointCloud(const ccPointCloud& cloud);
	ccPointCloud(
		const std::vector<Eigen::Vector3d> &points, 
		const std::string& name = "cloud");

	//! Default destructor
	~ccPointCloud() override;

	//! Returns class ID
	CV_CLASS_ENUM getClassID() const override { return CV_TYPES::POINT_CLOUD; }

public: //clone, copy, etc.

	//! Creates a new point cloud object from a GenericIndexedCloud
	/** "GenericIndexedCloud" is an extension of GenericCloud (from CVLib)
		which provides a const random accessor to points.
		See CVLib documentation for more information about GenericIndexedCloud.
		As the GenericIndexedCloud interface is very simple, only points are imported.
		Note: throws an 'int' exception in case of error (see CTOR_ERRORS)
		\param cloud a GenericIndexedCloud structure
		\param sourceCloud cloud from which main parameters will be imported (optional)
	**/
	static ccPointCloud* From(const CVLib::GenericIndexedCloud* cloud, const ccGenericPointCloud* sourceCloud = nullptr);

	//! Creates a new point cloud object from a GenericCloud
	/** "GenericCloud" is a very simple and light interface from CVLib. It is
		meant to give access to points coordinates of any cloud (on the
		condition it implements the GenericCloud interface of course).
		See CVLib documentation for more information about GenericClouds.
		As the GenericCloud interface is very simple, only points are imported.
		Note: throws an 'int' exception in case of error (see CTOR_ERRORS)
		\param cloud a GenericCloud structure
		\param sourceCloud cloud from which main parameters will be imported (optional)
	**/
	static ccPointCloud* From(CVLib::GenericCloud* cloud, const ccGenericPointCloud* sourceCloud = nullptr);

	/// \brief Function to select points from \p input ccPointCloud into
	/// \p output ccPointCloud.
	///
	/// Points with indices in \param indices are selected.
	///
	/// \param sourceCloud.
	/// \param indices Indices of points to be selected.
	/// \param invert Set to `True` to invert the selection of indices.
	static ccPointCloud* From(const ccPointCloud* sourceCloud, const std::vector<size_t>& indices, bool invert = false);

	//! Warnings for the partialClone method (bit flags)
	enum CLONE_WARNINGS {	WRN_OUT_OF_MEM_FOR_COLORS		= 1,
							WRN_OUT_OF_MEM_FOR_NORMALS		= 2,
							WRN_OUT_OF_MEM_FOR_SFS			= 4,
							WRN_OUT_OF_MEM_FOR_FWF			= 8
	};

	//! Creates a new point cloud object from a ReferenceCloud (selection)
	/** "Reference clouds" are a set of indexes referring to a real point cloud.
		See CVLib documentation for more information about ReferenceClouds.
		Warning: the ReferenceCloud structure must refer to this cloud.
		\param selection a ReferenceCloud structure (pointing to source)
		\param[out] warnings [optional] to determine if warnings (CTOR_ERRORS) occurred during the duplication process
	**/
	ccPointCloud* partialClone(const CVLib::ReferenceCloud* selection, int* warnings = nullptr) const;

	//! Clones this entity
	/** All the main features of the entity are cloned, except from the octree and
		the points visibility information.
		\param destCloud [optional] the destination cloud can be provided here
		\param ignoreChildren [optional] whether to ignore the cloud's children or not (in which case they will be cloned as well)
		\return a copy of this entity
	**/
	ccPointCloud* cloneThis(ccPointCloud* destCloud = nullptr, bool ignoreChildren = false);

	//inherited from ccGenericPointCloud
	ccGenericPointCloud* clone(ccGenericPointCloud* destCloud = nullptr, bool ignoreChildren = false) override;

	//! Fuses another 3D entity with this one
	/** All the main features of the given entity are added, except from the octree and
		the points visibility information. Those features are deleted on this cloud.
	**/
	ccPointCloud& operator=(const ccPointCloud& cloud);
	const ccPointCloud& operator +=(const ccPointCloud &cloud);
	const ccPointCloud& operator +=(ccPointCloud*);
	ccPointCloud ccPointCloud::operator+(const ccPointCloud &cloud) const {
		return (ccPointCloud(*this) += cloud);
	}

public: //features deletion/clearing

	//! Clears the entity from all its points and features
	/** Display parameters are also reseted to their default values.
	**/
	void clear() override;

	//! Erases the cloud points
	/** Prefer ccPointCloud::clear by default.
		\warning DANGEROUS
	**/
	void unalloactePoints();

	//! Erases the cloud colors
	void unallocateColors();

	//! Erases the cloud normals
	void unallocateNorms();

	//! Notify a modification of color / scalar field display parameters or contents
	inline void colorsHaveChanged() { m_vboManager.updateFlags |= vboSet::UPDATE_COLORS; }
	//! Notify a modification of normals display parameters or contents
	inline void normalsHaveChanged() { m_vboManager.updateFlags |= vboSet::UPDATE_NORMALS; }
	//! Notify a modification of points display parameters or contents
	inline void pointsHaveChanged() { m_vboManager.updateFlags |= vboSet::UPDATE_POINTS; }

public: //features allocation/resize

	//! Reserves memory to store the points coordinates
	/** Before adding points to the cloud (with addPoint())
		be sure to reserve the necessary amount of memory
		with this method. If the number of new elements is
		smaller than the actual one, nothing will happen.
		\param _numberOfPoints number of points to reserve the memory for
		\return true if ok, false if there's not enough memory
	**/
	bool reserveThePointsTable(unsigned _numberOfPoints);

	//! Reserves memory to store the RGB colors
	/** Before adding colors to the cloud (with addRGBColor())
		be sure to reserve the necessary amount of memory
		with this method. This method reserves memory for as
		many colors as the number of points in the cloud
		(effictively stored or reserved).
		\return true if ok, false if there's not enough memory
	**/
	bool reserveTheRGBTable();

	//! Resizes the RGB colors array
	/** If possible, the colors array is resized to fit exactly the number
		of points in the cloud (effictively stored or reserved). If the
		new size is inferior to the actual one, the last elements will be
		deleted. Otherwise, the array is filled with zeros (default behavior)
		or "white" colors (is fillWithWhite).
		WARNING: don't try to "add" any element on a resized array...
		\param fillWithWhite whether to fill new array elements with zeros (false) or white color (true)
		\return true if ok, false if there's not enough memory
	**/
	bool resizeTheRGBTable(bool fillWithWhite = false);

	//! Reserves memory to store the compressed normals
	/** Before adding normals to the cloud (with addNorm())
		be sure to reserve the necessary amount of memory
		with this method. This method reserves memory for as
		many normals as the number of points in the cloud
		(effictively stored or reserved).
		\return true if ok, false if there's not enough memory
	**/
	bool reserveTheNormsTable();

	//! Resizes the compressed normals array
	/** If possible, the normals array is resized to fit exactly the number
		of points in the cloud (effictively stored or reserved). If the
		new size is inferior to the actual one, the last elements will be
		deleted. Otherwise, the array is filled with blank elements.
		WARNING: don't try to "add" any element on a resized array...
		\return true if ok, false if there's not enough memory
	**/
	bool resizeTheNormsTable();

	//! Reserves memory for all the active features
	/** This method is meant to be called before increasing the cloud
		population. Only the already allocated features will be re-reserved.
		\return true if ok, false if there's not enough memory
	**/
	bool reserve(unsigned numberOfPoints) override;

	//! Resizes all the active features arrays
	/** This method is meant to be called after having increased the cloud
		population (if the final number of insterted point is lower than the
		reserved size). Otherwise, it fills all new elements with blank values.
		\return true if ok, false if there's not enough memory
	**/
	bool resize(unsigned numberOfPoints) override;

	//! Removes unused capacity
	inline void shrinkToFit() { if (size() < capacity()) resize(size()); }

public: //scalar-fields management

	//! Returns the currently displayed scalar (or 0 if none)
	ccScalarField* getCurrentDisplayedScalarField() const;
	//! Returns the currently displayed scalar field index (or -1 if none)
	int getCurrentDisplayedScalarFieldIndex() const;
	//! Sets the currently displayed scalar field
	/** Warning: this scalar field will automatically be set as the OUTPUT one!
	**/
	void setCurrentDisplayedScalarField(int index);

	//inherited from base class
	void deleteScalarField(int index) override;
	void deleteAllScalarFields() override;
	int addScalarField(const char* uniqueName) override;

	//! Returns whether color scale should be displayed or not
	bool sfColorScaleShown() const;
	//! Sets whether color scale should be displayed or not
	void showSFColorsScale(bool state);

public: //associated (scan) grid structure

	//! Grid structure
	struct Grid
	{
		//! Shared type
		using Shared = QSharedPointer<Grid>;

		//! Default constructor
		Grid()
			: w(0)
			, h(0)
			, validCount(0)
			, minValidIndex(0)
			, maxValidIndex(0)
		{
			sensorPosition.toIdentity();
		}

		//! Copy constructor
		/** \warning May throw a bad_alloc exception
		**/
		Grid(const Grid& grid)
			: w(grid.w)
			, h(grid.h)
			, validCount(grid.validCount)
			, minValidIndex(grid.minValidIndex)
			, maxValidIndex(grid.minValidIndex)
			, indexes(grid.indexes)
			, colors(grid.colors)
			, sensorPosition(grid.sensorPosition)
		{}

		//! Converts the grid to an RGB image (needs colors)
		QImage toImage() const
		{
			if (colors.size() == w*h)
			{
				QImage image(w, h, QImage::Format_ARGB32);
				for (unsigned j = 0; j < h; ++j)
				{
					for (unsigned i = 0; i < w; ++i)
					{
						const ecvColor::Rgb& col = colors[j*w + i];
						image.setPixel(i, j, qRgb(col.r, col.g, col.b));
					}
				}
				return image;
			}
			else
			{
				return QImage();
			}
		}

		//! Grid width
		unsigned w;
		//! Grid height
		unsigned h;

		//! Number of valid indexes
		unsigned validCount;
		//! Minimum valid index
		unsigned minValidIndex;
		//! Maximum valid index
		unsigned maxValidIndex;

		//! Grid indexes (size: w x h)
		std::vector<int> indexes;
		//! Grid colors (size: w x h, or 0 = no color)
		std::vector<ecvColor::Rgb> colors;

		//! Sensor position (expressed relatively to the cloud points)
		ccGLMatrixd sensorPosition;
	};

	//! Returns the number of associated grids
	size_t gridCount() const { return m_grids.size(); }
	//! Returns an associated grid
	inline Grid::Shared& grid(size_t gridIndex) { return m_grids[gridIndex]; }
	//! Returns an associated grid (const verson)
	inline const Grid::Shared& grid(size_t gridIndex) const { return m_grids[gridIndex]; }
	//! Adds an associated grid
	inline bool addGrid(Grid::Shared grid) { try{ m_grids.push_back(grid); } catch (const std::bad_alloc&) { return false; } return true; }
	//! Remove all associated grids
	inline void removeGrids() { m_grids.resize(0); }

	//! Meshes a scan grid
	/** \warning The mesh vertices will be this cloud instance!
	**/
	ccMesh* triangulateGrid(const Grid& grid, double minTriangleAngle_deg = 0.0) const;

public: //normals computation/orientation

	//! Compute the normals with the associated grid structure(s)
	/** Can also orient the normals in the same run.
	**/
	bool computeNormalsWithGrids(	double minTriangleAngle_deg = 1.0,
									ecvProgressDialog* pDlg = nullptr );

	//! Orient the normals with the associated grid structure(s)
	bool orientNormalsWithGrids(    ecvProgressDialog* pDlg = nullptr );

	//! Normals are forced to point to O
	bool orientNormalsTowardViewPoint( CCVector3 & VP, ecvProgressDialog* pDlg = nullptr);

	//! Compute the normals by approximating the local surface around each point
	bool computeNormalsWithOctree(	CV_LOCAL_MODEL_TYPES model,
									ccNormalVectors::Orientation preferredOrientation,
									PointCoordinateType defaultRadius,
									ecvProgressDialog* pDlg = nullptr );

	//! Orient the normals with a Minimum Spanning Tree
	bool orientNormalsWithMST(		unsigned kNN = 6,
									ecvProgressDialog* pDlg = nullptr );

	//! Orient normals with Fast Marching
	bool orientNormalsWithFM(		unsigned char level,
									ecvProgressDialog* pDlg = nullptr);

public: //waveform (e.g. from airborne scanners)

	//! Returns whether the cloud has associated Full WaveForm data
	bool hasFWF() const;

	//! Returns a proxy on a given waveform
	ccWaveformProxy waveformProxy(unsigned index) const;

	//! Waveform descriptors set
	using FWFDescriptorSet = QMap<uint8_t, WaveformDescriptor>;

	//! Waveform data container
	using FWFDataContainer = std::vector<uint8_t>;
	using SharedFWFDataContainer = QSharedPointer<const FWFDataContainer>;

	//! Gives access to the FWF descriptors
	FWFDescriptorSet& fwfDescriptors() { return m_fwfDescriptors; }
	//! Gives access to the FWF descriptors (const version)
	const FWFDescriptorSet& fwfDescriptors() const { return m_fwfDescriptors; }

	//! Gives access to the associated FWF data
	std::vector<ccWaveform>& waveforms() { return m_fwfWaveforms; }
	//! Gives access to the associated FWF data (const version)
	const std::vector<ccWaveform>& waveforms() const { return m_fwfWaveforms; }

	//! Reserves the FWF table
	bool reserveTheFWFTable();
	//! Resizes the FWF table
	bool resizeTheFWFTable();

	//! Gives access to the associated FWF data container
	SharedFWFDataContainer& fwfData() { return m_fwfData; }
	//! Gives access to the associated FWF data container (const version)
	const SharedFWFDataContainer& fwfData() const { return m_fwfData; }

	//! Compresses the associated FWF data container
	/** As the container is shared, the compressed version will be potentially added to the memory
		resulting in a decrease of the available memory...
	**/
	bool compressFWFData();

	//! Computes the maximum amplitude of all associated waveforms
	bool computeFWFAmplitude(double& minVal, double& maxVal, ecvProgressDialog* pDlg = nullptr) const;

	//! Clears all associated FWF data
	void clearFWFData();

public: //other methods

	//! Returns the cloud gravity center
	/** \return gravity center
	**/
	CCVector3 computeGravityCenter();

	//inherited from base class
	void invalidateBoundingBox() override;

	//inherited from ccHObject
	void getDrawingParameters(glDrawParams& params) const override;
	unsigned getUniqueIDForDisplay() const override;

	//inherited from ccDrawableObject
	bool hasColors() const override;
	bool hasNormals() const override;
	bool hasScalarFields() const override;
	bool hasDisplayedScalarField() const override;
	//void removeFromDisplay(const ccGenericGLDisplay* win) override; //for proper VBO release
	//void setDisplay(ccGenericGLDisplay* win) override;

	//inherited from CVLib::GenericCloud
	unsigned char testVisibility(const CCVector3& P) const override;

	//inherited from ccGenericPointCloud
	const ecvColor::Rgb* geScalarValueColor(ScalarType d) const override;
	const ecvColor::Rgb* getPointScalarValueColor(unsigned pointIndex) const override;
	ScalarType getPointDisplayedDistance(unsigned pointIndex) const override;
	
	const ecvColor::Rgb& getPointColor(unsigned pointIndex) const override;
	const ColorsTableType& getPointColors() const { return *rgbColors(); }
	ecvColor::Rgb& getPointColorPtr(size_t pointIndex);
	Eigen::Vector3d getEigenColor(size_t index) const;
	std::vector<Eigen::Vector3d> getEigenColors() const;
	void setEigenColors(const std::vector<Eigen::Vector3d>& colors);

	const CompressedNormType& getPointNormalIndex(unsigned pointIndex) const override;
	const CCVector3& getPointNormal(unsigned pointIndex) const override;
	CCVector3& getPointNormalPtr(size_t pointIndex);
	std::vector<CCVector3> getPointNormals() const;
	void setPointNormals(const std::vector<CCVector3>& normals);
	Eigen::Vector3d getEigenNormal(size_t index) const;
	std::vector<Eigen::Vector3d> getEigenNormals() const;
	void setEigenNormals(const std::vector<Eigen::Vector3d>& normals);

	CVLib::ReferenceCloud* crop(const ccBBox& box, bool inside = true) override;
	CVLib::ReferenceCloud* crop(const ecvOrientedBBox& bbox) override;

	virtual void applyRigidTransformation(const ccGLMatrix& trans) override;
	virtual void scale(PointCoordinateType fx, PointCoordinateType fy, PointCoordinateType fz, CCVector3 center = CCVector3(0,0,0)) override;
	inline void refreshBB() override { invalidateBoundingBox(); }
	/** \warning if removeSelectedPoints is true, any attached octree will be deleted. **/
	ccGenericPointCloud* createNewCloudFromVisibilitySelection(bool removeSelectedPoints = false, VisibilityTableType* visTable = nullptr, bool silent = false) override;

	//! Sets whether visibility check is enabled or not (e.g. during distances computation)
	/** See ccPointCloud::testVisibility.
	**/
	inline void enableVisibilityCheck(bool state) { m_visibilityCheckEnabled = state; }

	//! Returns whether the mesh as an associated sensor or not
	bool hasSensor() const;

	//! Comptes the closest point of this cloud relatively to another cloud
	/** The output (reference) clouds will have as many points as this cloud
		(with the indexes pointing on the closest point in the other cloud)
	**/
	QSharedPointer<CVLib::ReferenceCloud> computeCPSet(	ccGenericPointCloud& otherCloud,
														CVLib::GenericProgressCallback* progressCb = nullptr,
														unsigned char octreeLevel = 0);

	//! Interpolate colors from another cloud (nearest neighbor only)
	bool interpolateColorsFrom(	ccGenericPointCloud* cloud,
								CVLib::GenericProgressCallback* progressCb = nullptr,
								unsigned char octreeLevel = 0);

	//! Sets a particular point color
	/** WARNING: colors must be enabled.
	**/
	void setPointColor(size_t pointIndex, const ecvColor::Rgb& col);
	void setPointColor(size_t pointIndex, const Eigen::Vector3d& col);
	void setEigenColor(size_t index, const Eigen::Vector3d& color);

	//! Sets a particular point compressed normal
	/** WARNING: normals must be enabled.
	**/
	void setPointNormalIndex(size_t pointIndex, CompressedNormType norm);

	//! Sets a particular point normal (shortcut)
	/** WARNING: normals must be enabled.
		Normal is automatically compressed before storage.
	**/
	void setPointNormal(size_t pointIndex, const CCVector3& N);
	void setEigenNormal(size_t index, const Eigen::Vector3d& normal);

	//! Pushes a compressed normal vector
	/** \param index compressed normal vector
	**/
	void addNormIndex(CompressedNormType index);

	//! Pushes a normal vector on stack (shortcut)
	/** \param N normal vector
	**/
	void addNorm(const CCVector3& N);
	void addEigenNorm(const Eigen::Vector3d& N);
	void addEigenNorms(const std::vector<Eigen::Vector3d>& normals);

	void addNorms(const std::vector<CCVector3> &Ns);
	void addNorms(const std::vector<CompressedNormType> &idxes);
	std::vector<CompressedNormType> getNorms() const;
	void getNorms(std::vector<CompressedNormType>& idxes) const;

	//! Adds a normal vector to the one at a specific index
	/** The resulting sum is automatically normalized and compressed.
		\param N normal vector to add (size: 3)
		\param index normal index to modify
	**/
	void addNormAtIndex(const PointCoordinateType* N, unsigned index);

	//! Sets the (compressed) normals table
	void setNormsTable(NormsIndexesTableType* norms);

	//! Converts normals to RGB colors
	/** See ccNormalVectors::ConvertNormalToRGB
		\return success
	**/
	bool convertNormalToRGB();

	//! Converts normals to two scalar fields: 'dip' and 'dip direction'
	/**	One input scalar field may be empty if the corresponding value is not required
		\param[out] dipSF dip values
		\param[out] dipDirSF dip direction values
		\return success
	**/
	bool convertNormalToDipDirSFs(ccScalarField* dipSF, ccScalarField* dipDirSF);

	//! Pushes an RGB color on stack
	/** \param C RGB color
	**/
	void addRGBColor(const ecvColor::Rgb& C);
	void addRGBColors(const std::vector<ecvColor::Rgb>& colors);
	void addEigenColor(const Eigen::Vector3d& color);
	void addEigenColors(const std::vector<Eigen::Vector3d>& colors);

	//! Pushes an RGB color on stack (shortcut)
	/** \param r red component
		\param g green component
		\param b blue component
	**/
	inline void addRGBColor(ColorCompType r, ColorCompType g, ColorCompType b) { addRGBColor(ecvColor::Rgb(r, g, b)); }

	//! Pushes a grey color on stack (shortcut)
	/** Shortcut: color is converted to RGB(g, g, g)
		\param g grey component
	**/
	inline void addGreyColor(ColorCompType g) { addRGBColor(ecvColor::Rgb(g, g, g)); }

	//! Converts RGB to grey scale colors
	/** \return success
	**/
	bool convertRGBToGreyScale();

	//! Multiplies all color components of all points by coefficients
	/** If the cloud has no color, all points are considered white and
		the color array is automatically allocated.
		\param r red component
		\param g green component
		\param b blue component
		\return success
	**/
	bool colorize(float r, float g, float b);

	//! Assigns color to points proportionnaly to their 'height'
	/** Height is defined wrt to the specified dimension (heightDim).
		Color array is automatically allocated if necessary.
		\param heightDim ramp dimension (0:X, 1:Y, 2:Z)
		\param colorScale color scale to use
		\return success
	**/
	bool setRGBColorByHeight(unsigned char heightDim, ccColorScale::Shared colorScale);

	//! Assigns color to points by 'banding'
	/** Banding is performed along the specified dimension
		Color array is automatically allocated if necessary.
		\param dim banding dimension (0:X, 1:Y, 2:Z)
		\param freq banding frequency
		\return success
	**/
	bool setRGBColorByBanding(unsigned char dim, double freq);

	//! Sets RGB colors with current scalar field (values & parameters)
	/** \return success
	**/
	bool setRGBColorWithCurrentScalarField(bool mixWithExistingColor = false);

	//! Set a unique color for the whole cloud (shortcut)
	/** Color array is automatically allocated if necessary.
		\param r red component
		\param g green component
		\param b blue component
		\return success
	**/
	inline bool setRGBColor(ColorCompType r, ColorCompType g, ColorCompType b) { return setRGBColor(ecvColor::Rgb(r, g, b)); }

	//! Set a unique color for the whole cloud
	/** Color array is automatically allocated if necessary.
		\param col RGB color (size: 3)
		\return success
	**/
	bool setRGBColor(const ecvColor::Rgb& col);

	//! Inverts normals (if any)
	void invertNormals();

	//! Filters out points whose scalar values falls into an interval
	/** Threshold values should be expressed relatively to the current displayed scalar field.
		\param minVal minimum value
		\param maxVal maximum value
		\param outside whether to select the points inside or outside of the specified interval
		\return resulting cloud (remaining points)
	**/
	ccPointCloud* filterPointsByScalarValue(ScalarType minVal, ScalarType maxVal, bool outside = false);

	//! Hides points whose scalar values falls into an interval
	/** Values are taken from the current OUTPUT scalar field.
		\param minVal minimum value (below, points are hidden)
		\param maxVal maximum value (above, points are hidden)
	**/
	void hidePointsByScalarValue(ScalarType minVal, ScalarType maxVal);

	//! Unrolls the cloud and its normals on a cylinder
	/** This method is redundant with the "developCloudOnCylinder" method of CVLib,
		apart that it can also handle the cloud normals.
		\param radius unrolling cylinder radius
		\param coneAxisDim dimension along which the cylinder axis is aligned (X=0, Y=1, Z=2)
		\param center a point belonging to the cylinder axis (automatically computed if not specified)
		\param exportDeviationSF to export the deviation fro the ideal cone as a scalar field
		\param progressCb for progress notification
		\return the unrolled point cloud
		**/
	ccPointCloud* unrollOnCylinder(	PointCoordinateType radius,
									unsigned char coneAxisDim,
									CCVector3* center = nullptr,
									bool exportDeviationSF = false,
									double startAngle_deg = 0.0,
									double stopAngle_deg = 360.0,
									CVLib::GenericProgressCallback* progressCb = nullptr) const;

	//! Unrolls the cloud (and its normals) on a cone
	/** \param coneAngle_deg cone apex angle (between 0 and 180 degrees)
		\param coneApex cone apex 3D position
		\param coneAxisDim dimension along which the cone axis is aligned (X=0, Y=1, Z=2)
		\param developStraightenedCone if true, this method will unroll a straightened version of the cone (as a cylinder)
		\param baseRadius unrolling straightened cone base radius (necessary if developStraightenedCone is true)
		\param exportDeviationSF to export the deviation fro the ideal cone as a scalar field
		\param progressCb for progress notification
		\return the unrolled point cloud
	**/
	ccPointCloud* unrollOnCone(	double coneAngle_deg,
								const CCVector3& coneApex,
								unsigned char coneAxisDim,
								bool developStraightenedCone,
								PointCoordinateType baseRadius,
								bool exportDeviationSF = false,
								CVLib::GenericProgressCallback* progressCb = nullptr) const;

	//! Adds associated SF color ramp info to current GL context
	void addColorRampInfo(CC_DRAW_CONTEXT& context);

	//! Adds an existing scalar field to this cloud
	/** Warning: the cloud takes ownership of it!
		\param sf existing scalar field
		\return index of added scalar field (or -1 if an error occurred)
	**/
	int addScalarField(ccScalarField* sf);

	//! Returns pointer on RGB colors table
	ColorsTableType* rgbColors() const { return m_rgbColors; }

	//! Returns pointer on compressed normals indexes table
	NormsIndexesTableType* normals() const { return m_normals; }

	//! Crops the cloud inside (or outside) a 2D polyline
	/** \warning Always returns a selection (potentially empty) if successful.
		\param poly croping polyline
		\param orthoDim dimension orthogonal to the plane in which the segmentation should occur (X=0, Y=1, Z=2)
		\param inside whether selected points are inside or outside the polyline
		\return points falling inside (or outside) as a selection
	**/
	CVLib::ReferenceCloud* crop2D(const ccPolyline* poly, unsigned char orthoDim, bool inside = true);

	//! Appends a cloud to this one
	/** Same as the += operator with pointCountBefore == size()
		\param cloud cloud to be added
		\param pointCountBefore the number of points previously contained in this cloud
		\param ignoreChildren whether to copy input cloud's children or not
		\return the resulting point cloud
	**/
	const ccPointCloud& append(ccPointCloud* cloud, unsigned pointCountBefore, bool ignoreChildren = false);

	//! Enhances the RGB colors with the current scalar field (assuming it's intensities)
	bool enhanceRGBWithIntensitySF(int sfIdx, bool useCustomIntensityRange = false, double minI = 0.0, double maxI = 1.0);

	//! Exports the specified coordinate dimension(s) to scalar field(s)
	bool exportCoordToSF(bool exportDims[3]);

public: // for python interface
	inline virtual bool isEmpty() const override { return !hasPoints(); }

	inline virtual Eigen::Vector3d getMinBound() const override {
		return ComputeMinBound(CCVector3::fromArrayContainer(m_points));
	}

	inline virtual Eigen::Vector3d getMaxBound() const override {
		return ComputeMaxBound(CCVector3::fromArrayContainer(m_points));
	}
	inline virtual Eigen::Vector3d getGeometryCenter() const override {
		return ComputeCenter(CCVector3::fromArrayContainer(m_points));
	}

	virtual ccBBox getAxisAlignedBoundingBox() const override;
	virtual ecvOrientedBBox getOrientedBoundingBox() const override;
	virtual ccPointCloud& transform(const Eigen::Matrix4d& trans) override;
	virtual ccPointCloud& translate(const Eigen::Vector3d &translation,
		bool relative = true) override;
	virtual ccPointCloud& scale(const double s, const Eigen::Vector3d &center) override;
	virtual ccPointCloud& rotate(const Eigen::Matrix3d &R, const Eigen::Vector3d &center) override;

	/// Normalize point normals to length 1.
	ccPointCloud &normalizeNormals();

	/// \brief Function to compute the point to point distances between point
	/// clouds.
	///
	/// For each point in the \p source point cloud, compute the distance to the
	/// \p target point cloud.
	///
	/// \param target The target point cloud.
	std::vector<double> computePointCloudDistance(const ccPointCloud &target);

	/// \brief Function to select points from \p input ccPointCloud into
	/// \p output ccPointCloud.
	///
	/// Points with indices in \param indices are selected.
	///
	/// \param indices Indices of points to be selected.
	/// \param invert Set to `True` to invert the selection of indices.
	std::shared_ptr<ccPointCloud> selectByIndex(
		const std::vector<size_t> &indices, bool invert = false) const;

	/// \brief Remove all points fromt he point cloud that have a nan entry, or
	/// infinite entries.
	///
	/// Also removes the corresponding normals and color entries.
	///
	/// \param remove_nan Remove NaN values from the ccPointCloud.
	/// \param remove_infinite Remove infinite values from the ccPointCloud.
	ccPointCloud &removeNonFinitePoints(bool remove_nan = true, bool remove_infinite = true);


	/// \brief Function to downsample input ccPointCloud into output ccPointCloud
   /// with a voxel.
   ///
   /// Normals and colors are averaged if they exist.
   ///
   /// \param voxel_size Defines the resolution of the voxel grid,
   /// smaller value leads to denser output point cloud.
	std::shared_ptr<ccPointCloud> voxelDownSample(double voxel_size);

	/// \brief Function to downsample using geometry.ccPointCloud.VoxelDownSample
	///
	/// Also records point cloud index before downsampling.
	///
	/// \param voxel_size Voxel size to downsample into.
	/// \param min_bound Minimum coordinate of voxel boundaries
	/// \param max_bound Maximum coordinate of voxel boundaries
	/// \param approximate_class Whether to approximate.
	std::tuple<std::shared_ptr<ccPointCloud>, Eigen::MatrixXi, std::vector<std::vector<int>>>
	voxelDownSampleAndTrace(double voxel_size,
			const Eigen::Vector3d &min_bound,
			const Eigen::Vector3d &max_bound,
			bool approximate_class = false) const;

	/// \brief Function to downsample input ccPointCloud into output ccPointCloud
	/// uniformly.
	///
	/// The sample is performed in the order of the points with the 0-th point
	/// always chosen, not at random.
	///
	/// \param every_k_points Sample rate, the selected point indices are [0, k,
	/// 2k, ��].
	std::shared_ptr<ccPointCloud> uniformDownSample(size_t every_k_points) const;

	/// \brief Function to crop ccPointCloud into output ccPointCloud
	///
	/// All points with coordinates outside the bounding box \p bbox are
	/// clipped.
	///
	/// \param bbox AxisAlignedBoundingBox to crop points.
	std::shared_ptr<ccPointCloud> Crop(const ccBBox &bbox) const;

	/// \brief Function to crop ccPointCloud into output ccPointCloud
	///
	/// All points with coordinates outside the bounding box \p bbox are
	/// clipped.
	///
	/// \param bbox OrientedBoundingBox to crop points.
	std::shared_ptr<ccPointCloud> Crop(const ecvOrientedBBox &bbox) const;

	/// \brief Function to remove points that have less than \p nb_points in a
	/// sphere of a given radius.
	///
	/// \param nb_points Number of points within the radius.
	/// \param search_radius Radius of the sphere.
	std::tuple<std::shared_ptr<ccPointCloud>, std::vector<size_t>>
		removeRadiusOutliers(size_t nb_points, double search_radius) const;

	/// \brief Function to remove points that are further away from their
	/// \p nb_neighbor neighbors in average.
	///
	/// \param nb_neighbors Number of neighbors around the target point.
	/// \param std_ratio Standard deviation ratio.
	std::tuple<std::shared_ptr<ccPointCloud>, std::vector<size_t>>
		removeStatisticalOutliers(size_t nb_neighbors, double std_ratio) const;

	/// \brief Function to compute the normals of a point cloud.
	///
	/// Normals are oriented with respect to the input point cloud if normals
	/// exist.
	///
	/// \param search_param The KDTree search parameters for neighborhood
	/// search. \param fast_normal_computation If true, the normal estiamtion
	/// uses a non-iterative method to extract the eigenvector from the
	/// covariance matrix. This is faster, but is not as numerical stable.
	bool estimateNormals(
		const cloudViewer::geometry::KDTreeSearchParam &search_param = 
		cloudViewer::geometry::KDTreeSearchParamKNN(),
		bool fast_normal_computation = true);

	/// \brief Function to orient the normals of a point cloud.
	///
	/// \param orientation_reference Normals are oriented with respect to
	/// orientation_reference.
	bool orientNormalsToAlignWithDirection(
		const Eigen::Vector3d &orientation_reference =
		Eigen::Vector3d(0.0, 0.0, 1.0));

	/// \brief Function to orient the normals of a point cloud.
	///
	/// \param camera_location Normals are oriented with towards the
	/// camera_location.
	bool orientNormalsTowardsCameraLocation(
		const Eigen::Vector3d &camera_location = Eigen::Vector3d::Zero());

	/// \brief Function to compute the Mahalanobis distance for points
	/// in an input point cloud.
	///
	/// See: https://en.wikipedia.org/wiki/Mahalanobis_distance
	std::vector<double> computeMahalanobisDistance() const;

	/// Function to compute the distance from a point to its nearest neighbor in
	/// the input point cloud
	std::vector<double> computeNearestNeighborDistance() const;

	double computeResolution() const;

	/// Function that computes the convex hull of the point cloud using qhull
	std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>> computeConvexHull() const;

	/// \brief This is an implementation of the Hidden Point Removal operator
	/// described in Katz et. al. 'Direct Visibility of Point Sets', 2007.
	///
	/// Additional information about the choice of radius
	/// for noisy point clouds can be found in Mehra et. al. 'Visibility of
	/// Noisy Point Cloud Data', 2010.
	///
	/// \param camera_location All points not visible from that location will be
	/// removed. \param radius The radius of the sperical projection.
	std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
		hiddenPointRemoval(const Eigen::Vector3d &camera_location, const double radius) const;

	/// \brief Cluster ccPointCloud using the DBSCAN algorithm
	/// Ester et al., "A Density-Based Algorithm for Discovering Clusters
	/// in Large Spatial Databases with Noise", 1996
	///
	/// Returns a list of point labels, -1 indicates noise according to
	/// the algorithm.
	///
	/// \param eps Density parameter that is used to find neighbouring points.
	/// \param min_points Minimum number of points to form a cluster.
	/// \param print_progress If `true` the progress is visualized in the
	/// console.
	std::vector<int> clusterDBSCAN(double eps,
								   size_t min_points,
								   bool print_progress = false) const;

	/// \brief Segment ccPointCloud plane using the RANSAC algorithm.
	///
	/// \param distance_threshold Max distance a point can be from the plane
	/// model, and still be considered an inlier.
	/// \param ransac_n Number of initial points to be considered inliers in
	/// each iteration.
	/// \param num_iterations Number of iterations.
	/// \return Returns the plane model ax + by + cz + d = 0 and the indices of
	/// the plane inliers.
	std::tuple<Eigen::Vector4d, std::vector<size_t>> segmentPlane(
		const double distance_threshold = 0.01,
		const int ransac_n = 3,
		const int num_iterations = 100) const;

	/// \brief Factory function to create a pointcloud from a depth image and a
	/// camera model.
	///
	/// Given depth value d at (u, v) image coordinate, the corresponding 3d
	/// point is: z = d / depth_scale\n x = (u - cx) * z / fx\n y = (v - cy) * z
	/// / fy\n
	///
	/// \param depth The input depth image can be either a float image, or a
	/// uint16_t image. \param intrinsic Intrinsic parameters of the camera.
	/// \param extrinsic Extrinsic parameters of the camera.
	/// \param depth_scale The depth is scaled by 1 / \p depth_scale.
	/// \param depth_trunc Truncated at \p depth_trunc distance.
	/// \param stride Sampling factor to support coarse point cloud extraction.
	///
	/// \Return An empty pointcloud if the conversion fails.
	/// If \param project_valid_depth_only is true, return point cloud, which
	/// doesn't
	/// have nan point. If the value is false, return point cloud, which has
	/// a point for each pixel, whereas invalid depth results in NaN points.
	static std::shared_ptr<ccPointCloud> CreateFromDepthImage(
		const cloudViewer::geometry::Image &depth,
		const cloudViewer::camera::PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity(),
		double depth_scale = 1000.0,
		double depth_trunc = 1000.0,
		int stride = 1,
		bool project_valid_depth_only = true);

	/// \brief Factory function to create a pointcloud from an RGB-D image and a
	/// camera model.
	///
	/// Given depth value d at (u, v) image coordinate, the corresponding 3d
	/// point is: z = d / depth_scale\n x = (u - cx) * z / fx\n y = (v - cy) * z
	/// / fy\n
	///
	/// \param image The input image.
	/// \param intrinsic Intrinsic parameters of the camera.
	/// \param extrinsic Extrinsic parameters of the camera.
	///
	/// \Return An empty pointcloud if the conversion fails.
	/// If \param project_valid_depth_only is true, return point cloud, which
	/// doesn't
	/// have nan point. If the value is false, return point cloud, which has
	/// a point for each pixel, whereas invalid depth results in NaN points.
	static std::shared_ptr<ccPointCloud> CreateFromRGBDImage(
		const cloudViewer::geometry::RGBDImage &image,
		const cloudViewer::camera::PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity(),
		bool project_valid_depth_only = true);

	/// \brief Function to create a PointCloud from a VoxelGrid.
	///
	/// It transforms the voxel centers to 3D points using the original point
	/// cloud coordinate (with respect to the center of the voxel grid).
	///
	/// \param voxel_grid The input VoxelGrid.
	std::shared_ptr<ccPointCloud> createFromVoxelGrid(
		const cloudViewer::geometry::VoxelGrid &voxel_grid);

	/// \brief Assigns each vertex in the ccMesh the same color
	///
	/// \param color RGB colors of vertices.
	ccPointCloud &paintUniformColor(const Eigen::Vector3d &color);

protected:

	//inherited from ccHObject
	void drawMeOnly(CC_DRAW_CONTEXT& context) override;
	void applyGLTransformation(const ccGLMatrix& trans) override;
	bool toFile_MeOnly(QFile& out) const override;
	bool fromFile_MeOnly(QFile& in, short dataVersion, int flags) override;
	void notifyGeometryUpdate() override;

	//inherited from ccPointCloud
	/** \warning Doesn't handle scan grids!
	**/
	void swapPoints(unsigned firstIndex, unsigned secondIndex) override;

	virtual void removePoints(size_t index) override;

	//! Colors
	ColorsTableType* m_rgbColors;

	//! Normals (compressed)
	NormsIndexesTableType* m_normals;

	//! Specifies whether current scalar field color scale should be displayed or not
	bool m_sfColorScaleDisplayed;

	//! Currently displayed scalar field
	ccScalarField* m_currentDisplayedScalarField;
	//! Currently displayed scalar field index
	int m_currentDisplayedScalarFieldIndex;

	//! Associated grid structure
	std::vector<Grid::Shared> m_grids;

	//! Whether visibility check is available or not (during comparison)
	/** See ccPointCloud::testVisibility
	**/
	bool m_visibilityCheckEnabled;

protected: // VBO
	//! Release VBOs
	void releaseVBOs();

	class VBO : public QGLBuffer
	{
	public:
		int rgbShift;
		int normalShift;

		//! Inits the VBO
		/** \return the number of allocated bytes (or -1 if an error occurred)
		**/
		int init(int count, bool withColors, bool withNormals, bool* reallocated = nullptr);

		VBO()
			: QGLBuffer(QGLBuffer::VertexBuffer)
			, rgbShift(0)
			, normalShift(0)
		{}
	};

	//! VBO set
	struct vboSet
	{
		//! States of the VBO(s)
		enum STATES { NEW, INITIALIZED, FAILED };

		//! Update flags
		enum UPDATE_FLAGS {
			UPDATE_POINTS = 1,
			UPDATE_COLORS = 2,
			UPDATE_NORMALS = 4,
			UPDATE_ALL = UPDATE_POINTS | UPDATE_COLORS | UPDATE_NORMALS
		};

		vboSet()
			: hasColors(false)
			, colorIsSF(false)
			, sourceSF(nullptr)
			, hasNormals(false)
			, totalMemSizeBytes(0)
			, updateFlags(0)
			, state(NEW)
		{}

		std::vector<VBO*> vbos;
		bool hasColors;
		bool colorIsSF;
		ccScalarField* sourceSF;
		bool hasNormals;
		int totalMemSizeBytes;
		int updateFlags;

		//! Current state
		STATES state;
	};

	//! Set of VBOs attached to this cloud
	vboSet m_vboManager;

public: //Level of Detail (LOD)

	//! Intializes the LOD structure
	/** \return success
	**/
	bool initLOD();

	//! Clears the LOD structure
	void clearLOD();

protected: //Level of Detail (LOD)

	//! L.O.D. structure
	ccPointCloudLOD* m_lod;

protected: //waveform (e.g. from airborne scanners)

	//! General waveform descriptors
	FWFDescriptorSet m_fwfDescriptors;

	//! Per-point waveform accessors
	std::vector<ccWaveform> m_fwfWaveforms;

	//! Waveforms raw data storage
	SharedFWFDataContainer m_fwfData;

};

#endif // ECV_POINT_CLOUD_HEADER
