// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// System
#include <cstring>

//! Flags
class ccFlags {
public:
    //! Sets all bits to 0
    void reset() { memset(table, 0, sizeof(bool) * 8); }

    //! Converts a byte to this structure
    void fromByte(unsigned char byte) {
        unsigned char i, mask = 1;
        for (i = 0; i < 8; ++i) {
            table[i] = ((byte & mask) == mask);
            mask <<= 1;
        }
    }

    //! Converts this structure to a byte
    unsigned char toByte() const {
        unsigned char i, byte = 0, mask = 1;
        for (i = 0; i < 8; ++i) {
            if (table[i]) byte |= mask;
            mask <<= 1;
        }

        return byte;
    }

    //! Table of 8 booleans (one per bit)
    bool table[8];
};
