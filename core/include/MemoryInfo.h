// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "CVCoreLib.h"

#include <cstddef>
#include <ostream>

namespace cloudViewer {
namespace system {

struct CV_CORE_LIB_API MemoryInfo
{
    std::size_t totalRam{0};
    std::size_t freeRam{0};
    std::size_t availableRam{0};
    //	std::size_t sharedRam{0};
    //	std::size_t bufferRam{0};
    std::size_t totalSwap{0};
    std::size_t freeSwap{0};
    std::string toString();
};

MemoryInfo CV_CORE_LIB_API getMemoryInfo();

std::ostream& operator<<(std::ostream& os, const MemoryInfo& infos);

}
}

