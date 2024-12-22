#pragma once

//##########################################################################
//#                                                                        #
//#                     ACLOUDVIEWER PLUGIN: q3DMASC                       #
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
//#                 COPYRIGHT: Dimitri Lague / CNRS / UEB                  #
//#                                                                        #
//##########################################################################

//Qt
#include <QDialog>
#include <QFile>
#include <QTextStream>

#include <ui_Train3DMASCDialog.h>

#include "confusionmatrix.h"

//! 3DMASC plugin 'train' dialog
class Train3DMASCDialog : public QDialog, public Ui::Train3DMASCDialog
{
	Q_OBJECT

public:

	//! Default constructor
	Train3DMASCDialog(QWidget* parent = nullptr);
	~Train3DMASCDialog();

	void readSettings();
	void writeSettings();

	void clearResults();

	//! Adds a feature (entry) to the results table
	/** \return the row index
	**/
	int addFeature(QString name, float importance, bool isChecked = true);
	int addScale(double scale, bool isChecked = true);
	void scaleStateChanged(QTableWidgetItem* item);
	void connectScaleSelectionToFeatureSelection();
	void setResultText(QString text);
	void setFirstRunDone();
	inline void setClassifierSaved() { classifierSaved = true; saveRequested = false; }

	bool isFeatureSelected(QString featureName) const;
	void setFeatureImportance(QString featureName, float importance);
	void sortByFeatureImportance();
	
	inline bool shouldSaveClassifier() const { return saveRequested; }

	void addConfusionMatrixAndSaveTraces(ConfusionMatrix* ptr);
	void setInputFilePath(QString filename);
	void setCheckBoxSaveTrace(bool state);
	bool openTraceFile();
	bool closeTraceFile();
	void saveTraces(ConfusionMatrix *confusionMatrix);
	bool getSaveTrace();
	QString getTracePath();
	int getRun();

protected slots:

	void onClose();
	void onSave();
	void onExportResults(QString filePath = "");

protected: //members

	bool classifierSaved;
	bool saveRequested;
	std::vector<ConfusionMatrix*> toDeleteLater;
	bool traceFileConfigured;
	QFile *m_traceFile;
	QString m_tracePath;
	QTextStream m_traceStream;
	QString m_parameterFilePath;
	QString m_baseName;
	uint run;
};
