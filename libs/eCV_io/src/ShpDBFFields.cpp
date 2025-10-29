// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifdef CV_SHP_SUPPORT

#include "ShpDBFFields.h"

// system
#include <assert.h>

bool IntegerDBFField::save(DBFHandle handle, int fieldIndex) const {
    if (!handle || fieldIndex < 0) {
        assert(false);
        return false;
    }

    for (size_t i = 0; i < values.size(); ++i)
        DBFWriteIntegerAttribute(handle, static_cast<int>(i), fieldIndex,
                                 values[i]);

    return true;
}

bool DoubleDBFField::save(DBFHandle handle, int fieldIndex) const {
    if (!handle || fieldIndex < 0) {
        assert(false);
        return false;
    }

    for (size_t i = 0; i < values.size(); ++i)
        DBFWriteDoubleAttribute(handle, static_cast<int>(i), fieldIndex,
                                values[i]);

    return true;
}

bool DoubleDBFField3D::save(DBFHandle handle,
                            int xFieldIndex,
                            int yFieldIndex,
                            int zFieldIndex) const {
    if (!handle || xFieldIndex < 0 || yFieldIndex < 0 || zFieldIndex < 0) {
        assert(false);
        return false;
    }

    for (size_t i = 0; i < values.size(); ++i) {
        DBFWriteDoubleAttribute(handle, static_cast<int>(i), xFieldIndex,
                                values[i].x);
        DBFWriteDoubleAttribute(handle, static_cast<int>(i), yFieldIndex,
                                values[i].y);
        DBFWriteDoubleAttribute(handle, static_cast<int>(i), zFieldIndex,
                                values[i].z);
    }

    return true;
}

#endif  // CV_SHP_SUPPORT
