// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/utility/ISAInfo.h"

#include "cloudViewer/utility/Logging.h"
#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

TEST(ISAInfo, GetSelectedISATarget) {
    EXPECT_NE(utility::ISAInfo::GetInstance().SelectedTarget(),
              utility::ISATarget::UNKNOWN);
}

}  // namespace tests
}  // namespace cloudViewer
