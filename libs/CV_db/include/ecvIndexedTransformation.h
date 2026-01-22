// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvGLMatrix.h"

//! A 4x4 'transformation' matrix (column major order) associated to an index
//! (typically a timestamp)
class CV_DB_LIB_API ccIndexedTransformation : public ccGLMatrix {
public:
    //! Default constructor
    /** Matrix is set to identity (see ccGLMatrix::toIdentity) by default.
            Index is set to zero by default.
    **/
    ccIndexedTransformation();

    //! Constructor from a transformation matrix
    /** Index is set to zero by default.
            \param matrix transformation matrix
    **/
    ccIndexedTransformation(const ccGLMatrix& matrix);

    //! Constructor from a transformation matrix and an index
    /** \param matrix transformation matrix
            \param index associated index (e.g. timestamp)
    **/
    ccIndexedTransformation(const ccGLMatrix& matrix, double index);

    //! Copy constructor
    ccIndexedTransformation(const ccIndexedTransformation& trans);

    //! Returns associated index (e.g. timestamp)
    inline double getIndex() const { return m_index; }

    //! Sets associated index (e.g. timestamp)
    inline void setIndex(double index) { m_index = index; }

    //! Interpolates two transformations at an absolute position (index)
    /** Warning: interpolation index must lie between the two input matrices
    indexes! \param interpIndex interpolation position (should be between trans1
    and trans2 indexes). \param trans1 first transformation \param trans2 second
    transformation
    **/
    static ccIndexedTransformation Interpolate(
            double interpIndex,
            const ccIndexedTransformation& trans1,
            const ccIndexedTransformation& trans2);

    //! Multiplication by a ccGLMatrix operator
    ccIndexedTransformation operator*(const ccGLMatrix& mat) const;

    //! (in place) Multiplication by a ccGLMatrix operator
    /** Warning: index is not modified by this operation.
     **/
    ccIndexedTransformation& operator*=(const ccGLMatrix& mat);

    //! (in place) Translation operator
    /** Warning: index is not modified by this operation.
     **/
    ccIndexedTransformation& operator+=(const CCVector3& T);
    //! (in place) Translation operator
    /** Warning: index is not modified by this operation.
     **/
    ccIndexedTransformation& operator-=(const CCVector3& T);

    //! Returns transposed transformation
    /** Warning: index is not modified by this operation.
     **/
    ccIndexedTransformation transposed() const;

    //! Returns inverse transformation
    /** Warning: index is not modified by this operation.
     **/
    ccIndexedTransformation inverse() const;

    // inherited from ccGLMatrix
    virtual bool toAsciiFile(QString filename, int precision = 12) const;
    virtual bool fromAsciiFile(QString filename);

    // inherited from ccSerializableObject
    bool isSerializable() const override { return true; }
    bool toFile(QFile& out, short dataVersion) const override;
    short minimumFileVersion() const override;
    bool fromFile(QFile& in,
                  short dataVersion,
                  int flags,
                  LoadedIDMap& oldToNewIDMap) override;

protected:
    //! Associated index (e.g. timestamp)
    double m_index;
};
