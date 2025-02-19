//##########################################################################
//#                                                                        #
//#                               cloudViewer                              #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU Library General Public License as       #
//#  published by the Free Software Foundation; version 2 or later of the  #
//#  License.                                                              #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef CV_SCALAR_FIELD_HEADER
#define CV_SCALAR_FIELD_HEADER

// Local
#include "CVConst.h"
#include "CVShareable.h"

// System
#include <vector>

namespace cloudViewer {

//! A simple scalar field (to be associated to a point cloud)
/** A mono-dimensionnal array of scalar values. It has also specific
        parameters for display purposes.

        Invalid values can be represented by NAN_VALUE.
**/
class ScalarField : public std::vector<ScalarType>, public CCShareable {
public:
    //! Default constructor
    /** [SHAREABLE] Call 'link' when associating this structure to an object.
            \param name scalar field name
    **/
    CV_CORE_LIB_API explicit ScalarField(const char* name = nullptr);

    //! Copy constructor
    /** \param sf scalar field to copy
            \warning May throw a std::bad_alloc exception
    **/
    CV_CORE_LIB_API ScalarField(const ScalarField& sf);

    //! Sets scalar field name
    CV_CORE_LIB_API void setName(const char* name);

    //! Returns scalar field name
    inline const char* getName() const { return m_name; }

    //! Returns the specific NaN value
    static inline ScalarType NaN() { return NAN_VALUE; }

    //! Computes the mean value (and optionally the variance value) of the
    //! scalar field
    /** \param mean a field to store the mean value
            \param variance if not void, the variance will be computed and
    stored here
    **/
    CV_CORE_LIB_API void computeMeanAndVariance(
            ScalarType& mean, ScalarType* variance = nullptr) const;

    //! Determines the min and max values
    CV_CORE_LIB_API virtual void computeMinAndMax();

    //! Returns whether a scalar value is valid or not
    static inline bool ValidValue(ScalarType value) {
        return std::isfinite(value);
    }

    //! Sets the value as 'invalid' (i.e. NAN_VALUE)
    inline void flagValueAsInvalid(std::size_t index) { at(index) = NaN(); }

    //! Returns the number of valid values in this scalar field
    CV_CORE_LIB_API std::size_t countValidValues() const;

    //! Returns the minimum value
    inline ScalarType getMin() const { return m_minVal; }
    //! Returns the maximum value
    inline ScalarType getMax() const { return m_maxVal; }

    //! Fills the array with a particular value
    inline void fill(ScalarType fillValue = 0) {
        if (empty())
            resize(capacity(), fillValue);
        else
            std::fill(begin(), end(), fillValue);
    }

    //! Reserves memory (no exception thrown)
    CV_CORE_LIB_API bool reserveSafe(std::size_t count);
    //! Resizes memory (no exception thrown)
    CV_CORE_LIB_API bool resizeSafe(std::size_t count,
                                    bool initNewElements = false,
                                    ScalarType valueForNewElements = 0);

    // Shortcuts (for backward compatibility)
    inline ScalarType& getValue(std::size_t index) { return at(index); }
    inline const ScalarType& getValue(std::size_t index) const {
        return at(index);
    }
    inline void setValue(std::size_t index, ScalarType value) {
        at(index) = value;
    }
    inline void addElement(ScalarType value) { push_back(value); }
    inline unsigned currentSize() const {
        return static_cast<unsigned>(size());
    }
    inline void swap(std::size_t i1, std::size_t i2) {
        std::swap(at(i1), at(i2));
    }

public:  // methods
    //! Default destructor
    /** Call release instead.
     **/
    CV_CORE_LIB_API ~ScalarField() override = default;

protected:  // members
    //! Scalar field name
    char m_name[256];

    //! Minimum value
    ScalarType m_minVal;
    //! Maximum value
    ScalarType m_maxVal;
};

inline void ScalarField::computeMinAndMax() {
    ScalarType minVal, maxVal;

    bool minMaxInitialized = false;
    for (std::size_t i = 0; i < size(); ++i) {
        const ScalarType& val = at(i);
        if (ValidValue(val)) {
            if (minMaxInitialized) {
                if (val < minVal)
                    minVal = val;
                else if (val > maxVal)
                    maxVal = val;
            } else {
                // first valid value is used to init min and max
                minVal = maxVal = val;
                minMaxInitialized = true;
            }
        }
    }

    if (minMaxInitialized) {
        m_minVal = minVal;
        m_maxVal = maxVal;
    } else  // particular case: zero valid values
    {
        m_minVal = m_maxVal = 0;
    }
}

}  // namespace cloudViewer

#endif  // CV_SCALAR_FIELD_HEADER
