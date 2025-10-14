// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_ARRAY_HEADER
#define ECV_ARRAY_HEADER

// Local
#include "ecvHObject.h"

// CV_CORE_LIB
#include <CVShareable.h>

// System
#include <vector>

//! Shareable array that can be properly inserted in the DB tree
template <class Type, int N, class ComponentType>
class ccArray : public std::vector<Type>, public CCShareable, public ccHObject {
public:
    //! Base type
    typedef ccArray<Type, N, ComponentType> Base;

    //! Default constructor
    ccArray(QString name = QString()) : ccHObject(name) {
        setFlagState(CC_LOCKED, true);
    }

    //! Duplicates array
    virtual Base* clone() {
        Base* cloneArray = new Base(getName());
        if (!copy(*cloneArray)) {
            // error message already issued
            cloneArray->release();
            cloneArray = nullptr;
        }
        return cloneArray;
    }

    //! Copies the content of this array in another one
    bool copy(Base& dest) const {
        try {
            // copy only the data
            static_cast<std::vector<Type>&>(dest) =
                    static_cast<const std::vector<Type>&>(*this);
        } catch (const std::bad_alloc&) {
            CVLog::Warning("[ccArray::copy] Not enough memory");
            return false;
        }
        return true;
    }

    //! Reserves memory (no exception thrown)
    bool reserveSafe(size_t count) {
        try {
            this->reserve(count);
        } catch (const std::bad_alloc&) {
            // not enough memory
            return false;
        }
        return true;
    }

    //! Returns whether some memory has been allocated or not
    inline bool isAllocated() const { return this->capacity() != 0; }

    //! Resizes memory (no exception thrown)
    bool resizeSafe(size_t count,
                    bool initNewElements = false,
                    const Type* valueForNewElements = nullptr) {
        try {
            if (initNewElements) {
                if (!valueForNewElements) {
                    CVLog::Warning(
                            "[ccArray::resizeSafe] Internal error: no new "
                            "element specified");
                    return false;
                }
                this->resize(count, *valueForNewElements);
            } else {
                this->resize(count);
            }
        } catch (const std::bad_alloc&) {
            // not enough memory
            return false;
        }
        return true;
    }

    // inherited from ccHObject
    inline virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::ARRAY;
    }
    inline virtual bool isShareable() const override { return true; }
    inline virtual bool isSerializable() const override { return true; }

    // Shortcuts (for backward compatibility)
    inline Type& getValue(size_t index) { return this->at(index); }
    inline const Type& getValue(size_t index) const { return this->at(index); }
    inline void setValue(size_t index, const Type& value) {
        this->at(index) = value;
    }
    inline void addElement(const Type& value) { this->emplace_back(value); }
    inline void fill(const Type& value) {
        if (this->empty())
            this->resize(this->capacity(), value);
        else
            std::fill(this->begin(), this->end(), value);
    }
    inline unsigned currentSize() const {
        return static_cast<unsigned>(this->size());
    }
    inline void clear(bool releaseMemory = false) {
        if (releaseMemory)
            this->resize(0);
        else
            this->std::vector<Type>::clear();
    }
    inline void swap(size_t i1, size_t i2) {
        std::swap(this->at(i1), this->at(i2));
    }

protected:
    //! Destructor (protected)
    /** Use release instead.
     **/
    virtual ~ccArray() {}

    // inherited from ccHObject
    inline virtual bool toFile_MeOnly(QFile& out) const override {
        return ccSerializationHelper::GenericArrayToFile<Type, N,
                                                         ComponentType>(*this,
                                                                        out);
    }
    inline virtual bool fromFile_MeOnly(QFile& in,
                                        short dataVersion,
                                        int flags,
                                        LoadedIDMap& oldToNewIDMap) override {
        return ccSerializationHelper::GenericArrayFromFile<Type, N,
                                                           ComponentType>(
                *this, in, dataVersion);
    }
};

#endif  // ECV_ARRAY_HEADER
