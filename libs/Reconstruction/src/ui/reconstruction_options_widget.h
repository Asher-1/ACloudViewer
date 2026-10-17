// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>
#include <functional>

#include "controllers/option_manager.h"
#include "ui/options_widget.h"

namespace colmap {

// Selects which mapper the GUI runs when starting a reconstruction.
enum class MapperType { INCREMENTAL, HIERARCHICAL, GLOBAL };

class ReconstructionOptionsWidget : public QWidget {
public:
    ReconstructionOptionsWidget(QWidget* parent,
                                OptionManager* options,
                                MapperType* mapper_type,
                                std::function<void()> on_mapper_type_changed);
};

}  // namespace colmap
