// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CV_CORE_LIB
#include <Polyline.h>

// Local
#include "eCV_db.h"
#include "ecvShiftedObject.h"

class ccPointCloud;
class ccGenericPointCloud;

//! Colored polyline
/** Extends the cloudViewer::Polyline class
 **/
class ECV_DB_LIB_API ccPolyline : public cloudViewer::Polyline,
                                  public ccShiftedObject {
public:
    //! Default constructor
    /** \param associatedCloud the associated point cloud (i.e. the vertices)
     **/
    explicit ccPolyline(GenericIndexedCloudPersist* associatedCloud);
    explicit ccPolyline(ccPointCloud& associatedCloud);

    //! Copy constructor
    /** \param poly polyline to clone
     **/
    ccPolyline(const ccPolyline& poly);

    //! Destructor
    virtual ~ccPolyline() override = default;

    //! Returns class ID
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::POLY_LINE;
    }

    // inherited methods (ccHObject)
    virtual bool isSerializable() const override { return true; }
    virtual void applyGLTransformation(const ccGLMatrix& trans) override;
    virtual unsigned getUniqueIDForDisplay() const override;

    // inherited methods (ccDrawableObject)
    virtual bool hasColors() const override;

    // inherited methods (ccShiftedObject)
    virtual void setGlobalShift(const CCVector3d& shift) override;
    virtual void setGlobalScale(double scale) override;

    //! Defines if the polyline is considered as 2D or 3D
    /** \param state if true, the polyline is 2D
     **/
    void set2DMode(bool state);

    //! Returns whether the polyline is considered as 2D or 3D
    inline bool is2DMode() const { return m_mode2D; }

    //! Defines if the polyline is considered as processed polyline
    /** \param state if true, the polyline is 2D
     **/
    void setTransformFlag(bool state) { m_needTransform = state; };

    //! Returns whether the polyline is considered as 2D or 3D
    inline bool needTransform() const { return m_needTransform; }

    //! Defines if the polyline is drawn in background or foreground
    /** \param state if true, the polyline is drawn in foreground
     **/
    void setForeground(bool state);

    //! Sets the polyline color
    /** \param col RGB color
     **/
    inline void setColor(const ecvColor::Rgb& col) {
        enableTempColor(false);
        m_rgbColor = col;
    }

    //! Sets the width of the line
    /**  \param width the desired width
     **/
    void setWidth(PointCoordinateType width);

    //! Returns the width of the line
    /** \return the width of the line in pixels
     **/
    inline PointCoordinateType getWidth() const { return m_width; }

    //! Returns the polyline color
    /** \return a pointer to the polyline RGB color
     **/
    inline const ecvColor::Rgb& getColor() const { return m_rgbColor; }

    // inherited methods (ccHObject)
    virtual ccBBox getOwnBB(bool withGLFeatures = false) override;
    inline virtual void drawBB(CC_DRAW_CONTEXT& context,
                               const ecvColor::Rgb& col) override {
        // DGM: only for 3D polylines!
        if (!is2DMode()) ccShiftedObject::drawBB(context, col);
    }

    //! Splits the polyline into several parts based on a maximum edge length
    /** \warning output polylines set (parts) may be empty if all the vertices
    are too far from each other! \param maxEdgeLength maximum edge length
            \param[out] parts output polyline parts
            \return success
    **/
    bool split(PointCoordinateType maxEdgeLength,
               std::vector<ccPolyline*>& parts);

    //! Add another reference cloud
    /** \warning Both clouds should have the same reference cloud!
            \warning No verification for duplicates!
            Thread safe.
    **/
    bool add(const ccPointCloud& cloud);

    //! Computes the polyline length
    PointCoordinateType computeLength() const;

    //! Sets whether to display or hide the polyline vertices
    void showVertices(bool state) { m_showVertices = state; }
    //! Whether the polyline vertices should be displayed or not
    bool verticesShown() const { return m_showVertices; }
    bool arrowShown() const { return m_showArrow; }
    unsigned getArrowIndex() const { return m_arrowIndex; }
    PointCoordinateType getArrowLength() const { return m_arrowLength; }

    //! Sets the width of vertex markers
    void setVertexMarkerWidth(int width) { m_vertMarkWidth = width; }
    //! Returns the width of vertex markers
    int getVertexMarkerWidth() const { return m_vertMarkWidth; }

    //! Initializes the polyline with a given set of vertices and the parameters
    //! of another polyline
    /** \warning Even the 'closed' state is copied as is!
            \param vertices set of vertices (can be null, in which case the
    polyline vertices will be cloned) \param poly polyline \return success
    **/
    bool initWith(ccPointCloud* vertices, const ccPolyline& poly);

    //! Copy the parameters from another polyline
    void importParametersFrom(const ccPolyline& poly);

    //! Shows an arrow in place of a given vertex
    void showArrow(bool state, unsigned vertIndex, PointCoordinateType length);

    //! Returns the number of segments
    unsigned segmentCount() const;

    //! Samples points on the polyline
    ccPointCloud* samplePoints(bool densityBased,
                               double samplingParameter,
                               bool withRGB);

    //! Smoothes the polyline (Chaikin algorithm)
    /** \param ratio between 0 and 0.5 (excluded)
            \param iterationCount of iteration
            \return smoothed polyline
    **/
    ccPolyline* smoothChaikin(PointCoordinateType ratio,
                              unsigned iterationCount) const;

    //! Creates a polyline mesh with the selected vertices only
    /** This method is called after a graphical segmentation.
            It creates one or several new polylines with the segments having
    both vertices tagged as "visible" (see
    ccGenericPointCloud::visibilityArray).
    **/
    bool createNewPolylinesFromSelection(std::vector<ccPolyline*>& output);

    //! Helper to determine if the input cloud acts as vertices of a polyline
    static bool IsCloudVerticesOfPolyline(ccGenericPointCloud* cloud,
                                          ccPolyline** polyline = nullptr);

    //! Creates a circle as a polyline
    static ccPolyline* Circle(const CCVector3& center,
                              PointCoordinateType radius,
                              unsigned resolution = 48);

    inline virtual bool IsEmpty() const override { return !hasPoints(); }
    virtual Eigen::Vector3d GetMinBound() const override;
    virtual Eigen::Vector3d GetMaxBound() const override;
    virtual Eigen::Vector3d GetCenter() const override;
    virtual ccBBox GetAxisAlignedBoundingBox() const override;
    virtual ecvOrientedBBox GetOrientedBoundingBox() const override;
    virtual ccPolyline& Transform(
            const Eigen::Matrix4d& transformation) override;
    virtual ccPolyline& Translate(const Eigen::Vector3d& translation,
                                  bool relative = true) override;
    virtual ccPolyline& Scale(const double s,
                              const Eigen::Vector3d& center) override;
    virtual ccPolyline& Rotate(const Eigen::Matrix3d& R,
                               const Eigen::Vector3d& center) override;

    ccPolyline& operator+=(const ccPolyline& polyline);
    ccPolyline& operator=(const ccPolyline& polyline);
    ccPolyline operator+(const ccPolyline& polyline) const;

    /// \brief Assigns each line in the LineSet the same color.
    ///
    /// \param color Specifies the color to be applied.
    ccPolyline& PaintUniformColor(const Eigen::Vector3d& color) {
        setColor(ecvColor::Rgb::FromEigen(color));
        return *this;
    }

public:  // meta-data keys
    //! Meta data key: vertical direction (for 2D polylines, contour plots,
    //! etc.)
    /** Expected value: 0(=X), 1(=Y) or 2(=Z) as int
     **/
    static QString MetaKeyUpDir() { return "up.dir"; }
    //! Meta data key: contour plot constant altitude (for contour plots, etc.)
    /** Expected value: altitude as double
     **/
    static QString MetaKeyConstAltitude() { return "contour.altitude"; }
    //! Meta data key: profile abscissa along generatrix
    static QString MetaKeyAbscissa() { return "profile.abscissa"; }
    //! Meta data key (prefix): intersection point between profile and its
    //! generatrix
    /** Expected value: 3D vector
            \warning: must be followed by '.x', '.y' or '.z'
    **/
    static QString MetaKeyPrefixCenter() { return "profile.center"; }
    //! Meta data key (prefix): generatrix orientation at the point of
    //! intersection with the profile
    /** Expected value: 3D vector
            \warning: must be followed by '.x', '.y' or '.z'
    **/
    static QString MetaKeyPrefixDirection() { return "profile.direction"; }

protected:
    // inherited from ccHObject
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;

    // inherited methods (ccHObject)
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;

    //! Unique RGB color
    ecvColor::Rgb m_rgbColor;

    //! Width of the line
    PointCoordinateType m_width;

    //! Whether polyline should be considered as 2D (true) or 3D (false)
    bool m_mode2D;

    //! Whether polyline should be considered as processed polyline
    bool m_needTransform;

    //! Whether polyline should draws itself in background (false) or foreground
    //! (true)
    bool m_foreground;

    //! Whether vertices should be displayed or not
    bool m_showVertices;

    //! Vertex marker width
    int m_vertMarkWidth;

    //! Whether to show an arrow or not
    bool m_showArrow;
    //! Arrow length
    PointCoordinateType m_arrowLength;
    //! Arrow index
    unsigned m_arrowIndex;
};
