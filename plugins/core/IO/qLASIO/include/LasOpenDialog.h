#pragma once

//##########################################################################
//#                                                                        #
//#                CloudViewer PLUGIN: LAS-IO Plugin                      #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                   COPYRIGHT: Thomas Montaigu                           #
//#                                                                        #
//##########################################################################

#include "LasDetails.h"
#include "LasExtraScalarField.h"
#include "LasScalarField.h"
#include "LasTiler.h"

// GUI generated by Qt Designer
#include <ui_lasopendialog.h>

// system
#include <string>
#include <vector>

// CORE
#include <CVGeom.h>
#include <CVLog.h>

/// Dialog shown to the user when opening a LAS file
class LasOpenDialog : public QDialog
    , public Ui::LASOpenDialog
{
	Q_OBJECT

  public:
	enum class Action
	{
		/// The user wants to load the file in ACloudViewer
		Load,
		/// The user wants to tile the file into multiple smaller ones
		Tile,
	};

	/// Default constructor
	explicit LasOpenDialog(QWidget* parent = nullptr);

	/// Set some informations about the file
	/// to be displayed to the user.
	void setInfo(int versionMinor, int pointFormatId, qulonglong numPoints);

	/// Sets the list of standard LAS scalar fields as well as
	/// user defined extra LAS scalar fields that are available in the file
	/// that the user is able to check which one should be loaded.
	void setAvailableScalarFields(const std::vector<LasScalarField>&      scalarFields,
	                              const std::vector<LasExtraScalarField>& extraScalarFields);

	/// Removes from the lists scalar fields and extra scalar fields
	/// which the user unchecked from the list of fields to load.
	void filterOutNotChecked(std::vector<LasScalarField>&      scalarFields,
	                         std::vector<LasExtraScalarField>& extraScalarFields);

	/// Returns whether the user wants to ignore (not load)
	/// fields for which values are all default values.
	bool shouldIgnoreFieldsWithDefaultValues() const;

	/// Returns whether the user wants to treat the
	/// rgb from the file as 8-bit components.
	bool shouldForce8bitColors() const;

	/// Returns quiet_NaN if the time shift value should be
	/// automatically found.
	///
	/// Otherwise, returns the value manually specified by the user.
	double timeShiftValue() const;

	/// Returns the action the user wants to do.
	///
	/// The action is based on the active tab when the
	/// user accepted the dialog.
	Action action() const;

	/// Returns the tiling options.
	///
	/// Only valid when the action is Tiling
	LasTilingOptions tilingOptions() const;

	void resetShouldSkipDialog();

	bool shouldSkipDialog() const;

  private:
	bool isChecked(const LasScalarField& lasScalarField) const;

	bool isChecked(const LasExtraScalarField& lasExtraScalarField) const;

	void doSelectAll(bool doSelect);
	void doSelectAllESF(bool doSelect);

	/// Connected to the "automatic time shift" check box.
	///
	/// Depending on if the user checks or un-checks the automatic time shift,
	/// we need to enable / disable the double spin box that
	/// is used to get the manually entered time shift.
	void onAutomaticTimeShiftToggle(bool checked);

	void onApplyAll();

	void onBrowseTilingOutputDir();

	void onCurrentTabChanged(int index);

  private:
	bool m_shouldSkipDialog{false};
};