//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef ECV_PERSISTENT_SETTINGS_HEADER
#define ECV_PERSISTENT_SETTINGS_HEADER

#include "CVPluginAPI.h"

//Qt
#include <QString>

//! Persistent settings key (to be used with QSettings)
class CVPLUGIN_LIB_API ecvPS
{
public:
	
    static inline const QString LoadFile                        () { return "LoadFile"; }
    static inline const QString SaveFile                        () { return "SaveFile"; }
    static inline const QString ThemeSettings                   () { return "ThemeSettings"; }
    static inline const QString CurrentTheme                    () { return "CurrentTheme"; }
    static inline const QString MainWindowSettings              () { return "MainWindowSettings"; }
    static inline const QString MainWinGeom                     () { return "mainWindowGeometry"; }
    static inline const QString MainWinState                    () { return "mainWindowState"; }
    static inline const QString CurrentPath                     () { return "currentPath"; }
    static inline const QString TextureFilePath                 () { return "TextureFilePath"; }
    static inline const QString SelectedImageInputFilter        () { return "selectedImageInputFilter"; }
    static inline const QString SelectedInputFilter             () { return "selectedInputFilter"; }
    static inline const QString SelectedOutputFilterCloud       () { return "selectedOutputFilterCloud"; }
    static inline const QString SelectedOutputFilterMesh        () { return "selectedOutputFilterMesh"; }
    static inline const QString SelectedOutputFilterImage       () { return "selectedOutputFilterImage"; }
    static inline const QString SelectedOutputFilterPoly        () { return "selectedOutputFilterPoly"; }
    static inline const QString DuplicatePointsGroup            () { return "duplicatePoints"; }
    static inline const QString DuplicatePointsMinDist          () { return "minDist"; }
    static inline const QString HeightGridGeneration            () { return "HeightGridGeneration"; }
    static inline const QString VolumeCalculation               () { return "VolumeCalculation"; }
    static inline const QString Console                         () { return "Console"; }
    static inline const QString GlobalShift                     () { return "GlobalShift"; }
    static inline const QString MaxAbsCoord                     () { return "MaxAbsCoord"; }
    static inline const QString MaxAbsDiag                      () { return "MaxAbsDiag"; }
    static inline const QString AutoPickRotationCenter          () { return "AutoPickRotationCenter"; }
    static inline const QString AutoShowCenter                  () { return "AutoShowCenter"; }
    static inline const QString AutoShowReconstructionToolBar   () { return "ReconstructionToolBar"; }
};

#endif //ECV_PERSISTENT_SETTINGS_HEADER
