// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <QRegularExpression>

#include "IoAbstractLoader.h"

class AssimpIfc final : public IoAbstractLoader {
public:
    AssimpIfc();

private:
    void _postProcess(ccHObject &ioContainer) override;

    void _recursiveRename(ccHObject *ioContainer);

    QRegularExpression mNameMatcher;
};
