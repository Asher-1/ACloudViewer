// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ##########################################################################
// #                                                                        #
// #                     ACLOUDVIEWER PLUGIN: qCANUPO                       #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #      COPYRIGHT: UEB (UNIVERSITE EUROPEENNE DE BRETAGNE) / CNRS         #
// #                                                                        #
// ##########################################################################

// ACloudViewer
#include <ecvCommandLineInterface.h>

// Local
#include "q3DMASCTools.h"

// qCC_db
#include <ecvProgressDialog.h>

// Qt
#include <QCoreApplication>
#include <QDialog>
#include <QFileInfo>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

static const char COMMAND_3DMASC_CLASSIFY[] = "3DMASC_CLASSIFY";
static const char COMMAND_3DMASC_KEEP_ATTRIBS[] = "KEEP_ATTRIBUTES";
static const char COMMAND_3DMASC_ONLY_FEATURES[] = "ONLY_FEATURES";
static const char COMMAND_3DMASC_SKIP_FEATURES[] = "SKIP_FEATURES";

struct Command3DMASCClassif : public ccCommandLineInterface::Command {
    Command3DMASCClassif()
        : ccCommandLineInterface::Command("3DMASC Classify",
                                          COMMAND_3DMASC_CLASSIFY) {}

    virtual bool process(ccCommandLineInterface& cmd) override {
        cmd.print("[3DMASC]");

        if (cmd.clouds().empty()) {
            return cmd.error("No cloud loaded");
        }

        int minArgumentCount = 2;
        if (cmd.arguments().size() < minArgumentCount) {
            return cmd.error(
                    QString("Missing parameter(s): options, classifier "
                            "filename (.txt) and cloud roles after \"-%1\"")
                            .arg(COMMAND_3DMASC_CLASSIFY));
        }

        bool keepAttributes = false;
        bool onlyFeatures = false;
        bool skipFeatures = false;
        QString featureSourceFilename;
        while (true) {
            QString argument = cmd.arguments().front();
            if (ccCommandLineInterface::IsCommand(
                        argument, COMMAND_3DMASC_KEEP_ATTRIBS)) {
                keepAttributes = true;
                cmd.print("Will keep attributes");
                // local option confirmed, we can move on
                cmd.arguments().pop_front();
            } else if (ccCommandLineInterface::IsCommand(
                               argument, COMMAND_3DMASC_ONLY_FEATURES)) {
                onlyFeatures = true;
                cmd.print("Will compute only the features");
                // local option confirmed, we can move on
                cmd.arguments().pop_front();

                // we are bound to to keep the features!
                keepAttributes = true;
            } else if (ccCommandLineInterface::IsCommand(
                               argument, COMMAND_3DMASC_SKIP_FEATURES)) {
                skipFeatures = true;
                cmd.print("Will skip the computation of features");
                // local option confirmed, we can move on
                cmd.arguments().pop_front();

                featureSourceFilename = cmd.arguments().front();
                if (featureSourceFilename.isEmpty()) {
                    return cmd.error(
                            QString("Missing parameter(s): feature sources "
                                    "filename after \"-%1\"")
                                    .arg(COMMAND_3DMASC_SKIP_FEATURES));
                }
                cmd.arguments().pop_front();

                // we only expect the classifier filename now
                --minArgumentCount;
            } else {
                // urecognized option
                break;
            }
        }

        if (onlyFeatures && skipFeatures) {
            return cmd.error(
                    "Can't compute only the features and skip them at the same "
                    "time :p");
        }

        if (cmd.arguments().size() < minArgumentCount) {
            return cmd.error(
                    QString("Missing parameter(s): classifier filename (.txt) "
                            "and/or cloud roles after \"-%1\"")
                            .arg(COMMAND_3DMASC_CLASSIFY));
        }

        QString classifierFilename = cmd.arguments().front();
        cmd.print("Classifier filename: " + classifierFilename);
        QCoreApplication::processEvents();
        cmd.arguments().pop_front();

        ccPointCloud* classifiedCloud = nullptr;
        SFCollector generatedScalarFields;
        masc::Feature::Source::Set featureSources;

        if (!skipFeatures) {
            // we need to load the cloud roles and match them with the already
            // loaded clouds
            QString cloudRolesStr = cmd.arguments().front();
            cmd.print("Cloud roles: " + cloudRolesStr);
            QCoreApplication::processEvents();
            cmd.arguments().pop_front();

            // process the cloud roles description
            QStringList tokens = cloudRolesStr.simplified().split(
                    QChar(' '), QtCompat::SkipEmptyParts);

            masc::Tools::NamedClouds cloudPerRole;
            QString mainCloudRole;
            for (const QString& token : tokens) {
                QStringList subTokens = token.split("=");
                int subTokenCount = subTokens.size();
                if (subTokenCount != 2) {
                    return cmd.error(
                            "Malformed cloud roles description (expecting: "
                            "\"PC1=1 PC2=3 CTX=2\" for instance)");
                }
                QString role = subTokens[0].toUpper();
                bool ok = false;
                unsigned cloudIndex = subTokens[1].toUInt(&ok);
                if (!ok || cloudIndex == 0) {
                    return cmd.error(
                            "Malformed cloud roles description (expecting the "
                            "cloud index corresponding to each role - starting "
                            "from 1)");
                }
                if (cloudIndex > cmd.clouds().size()) {
                    return cmd.error(QString("Cloud index %1 exceeds the "
                                             "number of loaded clouds (=%2)")
                                             .arg(cloudIndex)
                                             .arg(cmd.clouds().size()));
                }
                cloudPerRole.insert(role, cmd.clouds()[cloudIndex - 1].pc);

                if (mainCloudRole.isEmpty()) {
                    mainCloudRole = role;
                }
            }

            // try to load the cloud roles from the classifier file
            QList<QString> cloudLabels;
            QString corePointsLabel;
            bool filenamesSpecified = false;
            QMap<QString, QString> labelsAndNames;
            if (!masc::Tools::LoadClassifierCloudLabels(
                        classifierFilename, cloudLabels, corePointsLabel,
                        filenamesSpecified, labelsAndNames)) {
                return cmd.error("Failed to read classifier file");
            }

            if (!corePointsLabel.isEmpty()) {
                // we use the core points source as 'main role' by default
                mainCloudRole = corePointsLabel;
                cmd.print("Core points source: " + corePointsLabel +
                          " (will be used as the classified cloud)");
            }
            cmd.print("The classified cloud role will be " + mainCloudRole);

            // if (!filenamesSpecified)
            //{
            //	return cmd.error("Filenames were not specified for at least one
            // role");
            // }

            for (QString label : cloudLabels) {
                if (!cloudPerRole.contains(label.toUpper())) {
                    return cmd.error(
                            QString("Role %1 has not been defined").arg(label));
                }
            }

            // load features
            masc::Feature::Set features;
            std::vector<double> scales;
            if (!masc::Tools::LoadFile(classifierFilename, &cloudPerRole, true,
                                       &features, &scales, nullptr, nullptr,
                                       nullptr, cmd.widgetParent())) {
                return cmd.error("Failed to load the classifier");
            }

            // internal consistency check
            if (!cloudPerRole.contains(mainCloudRole)) {
                return cmd.error("Classified cloud not loaded/defined?!");
            }

            // remove the test cloud (if any)
            if (cloudPerRole.contains("TEST")) {
                delete cloudPerRole["TEST"];
                cloudPerRole.remove("TEST");
            }

            // the 'main cloud' is the cloud that should be classified
            masc::CorePoints corePoints;
            corePoints.origin = corePoints.cloud = classifiedCloud =
                    cloudPerRole[mainCloudRole];
            corePoints.role = mainCloudRole;

            // prepare the main cloud
            QScopedPointer<ecvProgressDialog> pDlg;
            if (!cmd.silentMode()) {
                pDlg.reset(new ecvProgressDialog(true, cmd.widgetParent()));
                pDlg->setAutoClose(false);  // we don't want the progress dialog
                                            // to 'pop' for each feature
            }

            QString errorMessage;
            if (!masc::Tools::PrepareFeatures(corePoints, features,
                                              errorMessage, pDlg.data(),
                                              &generatedScalarFields)) {
                generatedScalarFields.releaseSFs(false);
                return cmd.error(errorMessage);
            }

            if (pDlg) {
                pDlg->setAutoClose(true);  // restore the default behavior of
                                           // the progress dialog
                pDlg->close();
                QCoreApplication::processEvents();
            }

            // don't forget to extract the sources before finishing this step
            masc::Feature::ExtractSources(features, featureSources);

            if (onlyFeatures) {
                QFileInfo fi(classifierFilename);
                featureSourceFilename = fi.absolutePath() + "/" +
                                        fi.completeBaseName() +
                                        "_feature_sources.txt";
                if (masc::Feature::SaveSources(featureSources,
                                               featureSourceFilename)) {
                    cmd.print("Feature sources file saved: " +
                              featureSourceFilename);
                    // return true;
                } else {
                    return cmd.error(
                            "Faild to write feature sources to file: " +
                            featureSourceFilename);
                }
            }
        } else {
            // we use the first loaded cloud by default
            classifiedCloud = cmd.clouds().front().pc;

            // load the feature 'sources'
            if (!masc::Feature::LoadSources(featureSources,
                                            featureSourceFilename)) {
                return cmd.error("Failed to load feature sources from: " +
                                 featureSourceFilename);
            }
        }

        // apply classifier
        if (!onlyFeatures) {
            masc::Classifier classifier;
            if (!masc::Tools::LoadFile(classifierFilename, nullptr, false,
                                       nullptr, nullptr, nullptr, &classifier,
                                       nullptr, cmd.widgetParent())) {
                return cmd.error("Failed to load the classifier");
            }

            QString errorMessage;
            if (!classifier.classify(featureSources, classifiedCloud,
                                     errorMessage, cmd.widgetParent())) {
                generatedScalarFields.releaseSFs(false);
                return cmd.error(errorMessage);
            }

            generatedScalarFields.releaseSFs(keepAttributes);
        }

        if (cmd.autoSaveMode() || onlyFeatures) {
            for (CLCloudDesc& desc : cmd.clouds()) {
                if (desc.pc == classifiedCloud) {
                    QString errorStr = cmd.exportEntity(
                            desc,
                            onlyFeatures ? "WITH_FEATURES" : "CLASSIFIED");
                    if (!errorStr.isEmpty()) {
                        return cmd.error(errorStr);
                    }
                    break;
                }
            }
        }

        return true;
    }
};
