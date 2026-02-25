// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "CVConst.h"
#include "CVShareable.h"

// System
#include <vector>

namespace cloudViewer {

/**
 * @class ScalarField
 * @brief Mono-dimensional array of scalar values
 *
 * A scalar field is a one-dimensional array of scalar values that can be
 * associated with a point cloud for storing per-point attributes such as
 * intensity, distance, elevation, etc. Invalid values are represented by NaN.
 *
 * This class extends std::vector<ScalarType> and is shareable (reference
 * counted).
 */
class ScalarField : public std::vector<ScalarType>, public CCShareable {
public:
    /**
     * @brief Default constructor
     *
     * Creates a new scalar field with an optional name.
     * @param name Scalar field name (optional)
     * @note [SHAREABLE] Call 'link' when associating to an object
     */
    CV_CORE_LIB_API explicit ScalarField(const char* name = nullptr);

    /**
     * @brief Copy constructor
     * @param sf Scalar field to copy
     * @warning May throw std::bad_alloc exception
     */
    CV_CORE_LIB_API ScalarField(const ScalarField& sf);

    /**
     * @brief Set the scalar field name
     * @param name New name for the scalar field
     */
    CV_CORE_LIB_API void setName(const char* name);

    /**
     * @brief Get the scalar field name
     * @return Scalar field name
     */
    inline const char* getName() const { return m_name; }

    /**
     * @brief Get the NaN value constant
     * @return NaN value used for invalid entries
     */
    static inline ScalarType NaN() { return NAN_VALUE; }

    /**
     * @brief Compute mean and variance
     *
     * Calculates the mean value and optionally the variance of the scalar
     * field.
     * @param mean Output: mean value
     * @param variance Output: variance (optional, nullptr to skip)
     */
    CV_CORE_LIB_API void computeMeanAndVariance(
            ScalarType& mean, ScalarType* variance = nullptr) const;

    /**
     * @brief Compute min and max values
     *
     * Determines and caches the minimum and maximum values in the field.
     */
    CV_CORE_LIB_API virtual void computeMinAndMax();

    /**
     * @brief Check if a scalar value is valid
     * @param value Value to check
     * @return true if value is finite (not NaN or infinite)
     */
    static inline bool ValidValue(ScalarType value) {
        return std::isfinite(value);
    }

    /**
     * @brief Mark value as invalid
     *
     * Sets the value at the specified index to NaN.
     * @param index Index of value to invalidate
     */
    inline void flagValueAsInvalid(std::size_t index) { at(index) = NaN(); }

    /**
     * @brief Count valid values
     * @return Number of valid (non-NaN) values
     */
    CV_CORE_LIB_API std::size_t countValidValues() const;

    /**
     * @brief Get minimum value
     * @return Cached minimum value
     */
    inline ScalarType getMin() const { return m_minVal; }

    /**
     * @brief Get maximum value
     * @return Cached maximum value
     */
    inline ScalarType getMax() const { return m_maxVal; }

    /**
     * @brief Fill array with a value
     * @param fillValue Value to fill with (default: 0)
     */
    inline void fill(ScalarType fillValue = 0) {
        if (empty())
            resize(capacity(), fillValue);
        else
            std::fill(begin(), end(), fillValue);
    }

    /**
     * @brief Reserve memory (exception-safe)
     * @param count Number of elements to reserve space for
     * @return true if successful
     */
    CV_CORE_LIB_API bool reserveSafe(std::size_t count);

    /**
     * @brief Resize memory (exception-safe)
     * @param count New size
     * @param initNewElements Whether to initialize new elements (default:
     * false)
     * @param valueForNewElements Value for initialization (default: 0)
     * @return true if successful
     */
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
    ScalarType minVal = 0.0f;
    ScalarType maxVal = 0.0f;

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
