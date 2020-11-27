#pragma once
// Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "ecvIOPluginInterface.h"


class MeshIO final : public QObject, public ccIOPluginInterface
{
   Q_OBJECT
   Q_INTERFACES( ccPluginInterface ccIOPluginInterface )

   Q_PLUGIN_METADATA( IID "ecvcorp.cloudviewer.plugin.MeshIO" FILE "../info.json" )

 public:
   explicit MeshIO( QObject *parent = nullptr );

 private:
   void registerCommands( ccCommandLineInterface *inCmdLine ) override;

   FilterList getFilters() override;
};
