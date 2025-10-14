// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UI_RECONSTRUCTION_OPTIONS_WIDGET_H_
#define COLMAP_SRC_UI_RECONSTRUCTION_OPTIONS_WIDGET_H_

#include <QtCore>
#include <QtWidgets>

#include "ui/options_widget.h"
#include "util/option_manager.h"

namespace colmap {

class MapperGeneralOptionsWidget : public OptionsWidget {
public:
    MapperGeneralOptionsWidget(QWidget* parent, OptionManager* options);
};

class MapperTriangulationOptionsWidget : public OptionsWidget {
public:
    MapperTriangulationOptionsWidget(QWidget* parent, OptionManager* options);
};

class MapperRegistrationOptionsWidget : public OptionsWidget {
public:
    MapperRegistrationOptionsWidget(QWidget* parent, OptionManager* options);
};

class MapperInitializationOptionsWidget : public OptionsWidget {
public:
    MapperInitializationOptionsWidget(QWidget* parent, OptionManager* options);
};

class MapperBundleAdjustmentOptionsWidget : public OptionsWidget {
public:
    MapperBundleAdjustmentOptionsWidget(QWidget* parent,
                                        OptionManager* options);
};

class MapperFilteringOptionsWidget : public OptionsWidget {
public:
    MapperFilteringOptionsWidget(QWidget* parent, OptionManager* options);
};

class ReconstructionOptionsWidget : public QWidget {
public:
    ReconstructionOptionsWidget(QWidget* parent, OptionManager* options);
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_RECONSTRUCTION_OPTIONS_WIDGET_H_
