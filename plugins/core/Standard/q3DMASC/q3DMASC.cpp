// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "q3DMASC.h"

// local
#include "q3DMASCClassifier.h"
#include "q3DMASCCommands.h"
#include "q3DMASCDisclaimerDialog.h"
#include "q3DMASCTools.h"
#include "qClassify3DMASCDialog.h"
#include "qTrain3DMASCDialog.h"

// qCC_db
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>

// Qt
#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>
#include <QtCore>
#include <QtGui>

q3DMASCPlugin::q3DMASCPlugin(QObject* parent /*=0*/)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/q3DMASCPlugin/info.json"),
      m_classifyAction(0),
      m_trainAction(0) {}

void q3DMASCPlugin::onNewSelection(
        const ccHObject::Container& selectedEntities) {
    if (m_classifyAction) {
        // classification: only one point cloud
        // m_classifyAction->setEnabled(selectedEntities.size() == 1 &&
        // selectedEntities[0]->isA(CV_TYPES::POINT_CLOUD));
        m_classifyAction->setEnabled(
                m_app->dbRootObject()->getChildrenNumber() != 0);
    }

    if (m_trainAction) {
        // m_trainAction->setEnabled(m_app && m_app->dbRootObject() &&
        // m_app->dbRootObject()->getChildrenNumber() != 0); //need some loaded
        // entities to train the classifier!
        m_trainAction->setEnabled(true);
    }

    m_selectedEntities = selectedEntities;
}

QList<QAction*> q3DMASCPlugin::getActions() {
    QList<QAction*> group;

    if (!m_trainAction) {
        m_trainAction = new QAction("Train classifier", this);
        m_trainAction->setToolTip("Train classifier");
        m_trainAction->setIcon(QIcon(QString::fromUtf8(
                ":/CC/plugin/q3DMASCPlugin/3DMASC TRAIN.png")));
        connect(m_trainAction, SIGNAL(triggered()), this,
                SLOT(doTrainAction()));
    }
    group.push_back(m_trainAction);

    if (!m_classifyAction) {
        m_classifyAction = new QAction("Classify", this);
        m_classifyAction->setToolTip("Classify cloud");
        m_classifyAction->setIcon(QIcon(QString::fromUtf8(
                ":/CC/plugin/q3DMASCPlugin/3DMASC CLASSIFY.png")));
        connect(m_classifyAction, SIGNAL(triggered()), this,
                SLOT(doClassifyAction()));
    }
    group.push_back(m_classifyAction);

    return group;
}

void q3DMASCPlugin::doClassifyAction() {
    if (!m_app) {
        assert(false);
        return;
    }

    // disclaimer accepted?
    if (!ShowClassifyDisclaimer(m_app)) {
        return;
    }

    QString inputFilename;
    {
        QSettings settings;
        settings.beginGroup("3DMASC");
        QString inputPath =
                settings.value("FilePath",
                               QCoreApplication::applicationDirPath())
                        .toString();
        inputFilename = QFileDialog::getOpenFileName(
                m_app->getMainWindow(), "Load 3DMASC classifier file",
                inputPath, "*.txt");
        if (inputFilename.isNull()) {
            // process cancelled by the user
            return;
        }
        settings.setValue("FilePath", QFileInfo(inputFilename).absolutePath());
        settings.endGroup();
    }

    QList<QString> cloudLabels;
    QString corePointsLabel;
    bool filenamesSpecified = false;
    QMap<QString, QString> rolesAndNames;
    if (!masc::Tools::LoadClassifierCloudLabels(
                inputFilename, cloudLabels, corePointsLabel, filenamesSpecified,
                rolesAndNames)) {
        m_app->dispToConsole("Failed to read classifier file (see Console)",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }
    if (cloudLabels.empty()) {
        m_app->dispToConsole("Invalid classifier file (no cloud label defined)",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    } else if (cloudLabels.size() >
               4 + (cloudLabels.contains("TEST") ? 1 : 0)) {
        m_app->dispToConsole(
                "This classifier uses more than 4 clouds (the GUI version "
                "cannot handle it)",
                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // now show a dialog where the user will be able to set the cloud roles
    Classify3DMASCDialog classifDlg(m_app);
    classifDlg.setCloudRoles(cloudLabels, corePointsLabel, rolesAndNames);
    classifDlg.label_trainOrClassify->setText(corePointsLabel +
                                              " will be classified");
    classifDlg.classifierFileLineEdit->setText(inputFilename);
    classifDlg.testCloudComboBox->hide();
    classifDlg.testLabel->hide();
    if (!classifDlg.exec()) {
        // process cancelled by the user
        return;
    }
    static bool s_keepAttributes =
            classifDlg.keepAttributesCheckBox->isChecked();

    masc::Tools::NamedClouds clouds;
    QString mainCloudLabel = corePointsLabel;
    classifDlg.getClouds(clouds);

    masc::Feature::Set features;
    masc::Classifier classifier;
    if (!masc::Tools::LoadClassifier(inputFilename, clouds, features,
                                     classifier, m_app->getMainWindow())) {
        return;
    }
    if (!classifier.isValid()) {
        m_app->dispToConsole("No classifier or invalid classifier",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    if (clouds.contains("TEST")) {
        // remove the test cloud (if any)
        clouds.remove("TEST");
    }

    // the 'main cloud' is the cloud that should be classified
    masc::CorePoints corePoints;
    corePoints.origin = corePoints.cloud = clouds[mainCloudLabel];
    corePoints.role = mainCloudLabel;

    // prepare the main cloud
    ecvProgressDialog progressDlg(true, m_app->getMainWindow());
    progressDlg.show();
    progressDlg.setAutoClose(false);  // we don't want the progress dialog to
                                      // 'pop' for each feature
    QString error;
    SFCollector generatedScalarFields;
    if (!masc::Tools::PrepareFeatures(corePoints, features, error, &progressDlg,
                                      &generatedScalarFields)) {
        m_app->dispToConsole(error, ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        generatedScalarFields.releaseSFs(false);
        return;
    }
    progressDlg.hide();
    QCoreApplication::processEvents();

    // apply classifier
    {
        QString errorMessage;
        masc::Feature::Source::Set featureSources;
        masc::Feature::ExtractSources(features, featureSources);
        if (!classifier.classify(featureSources, corePoints.cloud, errorMessage,
                                 m_app->getMainWindow())) {
            m_app->dispToConsole(errorMessage,
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            generatedScalarFields.releaseSFs(false);
            return;
        }

        generatedScalarFields.releaseSFs(s_keepAttributes);
    }
}

struct FeatureSelection {
    FeatureSelection(masc::Feature::Shared f = masc::Feature::Shared(nullptr))
        : feature(f) {}
    masc::Feature::Shared feature;
    bool selected = true;
    bool prepared = false;
    float importance = std::numeric_limits<float>::quiet_NaN();
};

void q3DMASCPlugin::saveTrainParameters(const masc::TrainParameters& params) {
    QSettings settings("OSUR", "q3DMASC");
    settings.setValue("TrainParameters/maxDepth", params.rt.maxDepth);
    settings.setValue("TrainParameters/minSampleCount",
                      params.rt.minSampleCount);
    settings.setValue("TrainParameters/activeVarCount",
                      params.rt.activeVarCount);
    settings.setValue("TrainParameters/maxTreeCount", params.rt.maxTreeCount);
}

void q3DMASCPlugin::loadTrainParameters(masc::TrainParameters& params) {
    QSettings settings("OSUR", "q3DMASC");
    params.rt.maxDepth = settings.value("TrainParameters/maxDepth", 25).toInt();
    params.rt.minSampleCount =
            settings.value("TrainParameters/minSampleCount", 10).toInt();
    params.rt.activeVarCount =
            settings.value("TrainParameters/activeVarCount", 0).toInt();
    params.rt.maxTreeCount =
            settings.value("TrainParameters/maxTreeCount", 100).toInt();
}

void q3DMASCPlugin::doTrainAction() {
    // disclaimer accepted?
    if (!ShowTrainDisclaimer(m_app)) return;

    QString inputFilename;
    {
        QSettings settings;
        settings.beginGroup("3DMASC");
        QString inputPath =
                settings.value("FilePath",
                               QCoreApplication::applicationDirPath())
                        .toString();
        inputFilename = QFileDialog::getOpenFileName(
                m_app->getMainWindow(), "Load 3DMASC training file", inputPath,
                "*.txt");
        if (inputFilename.isNull()) {
            // process cancelled by the user
            return;
        }
        settings.setValue("FilePath", QFileInfo(inputFilename).absolutePath());
        settings.endGroup();
    }

    // load the cloud labels (PC1, PC2, CTX, etc.)
    QList<QString> cloudLabels;
    QString corePointsLabel;
    bool filenamesSpecified = false;
    QMap<QString, QString> rolesAndNames;
    if (!masc::Tools::LoadClassifierCloudLabels(
                inputFilename, cloudLabels, corePointsLabel, filenamesSpecified,
                rolesAndNames)) {
        m_app->dispToConsole("Failed to read classifier file (see Console)",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }
    if (cloudLabels.empty()) {
        m_app->dispToConsole("Invalid classifier file (no cloud label defined)",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    masc::Tools::NamedClouds loadedClouds;
    masc::CorePoints corePoints;

    // if no filename is specified in the training file, we are bound to ask the
    // user to specify them
    bool useCloudsFromDB =
            (!filenamesSpecified ||
             QMessageBox::question(m_app->getMainWindow(), "Use clouds in DB",
                                   "Use clouds in db (yes) or clouds specified "
                                   "in the file(no)?",
                                   QMessageBox::Yes,
                                   QMessageBox::No) == QMessageBox::Yes);
    QString mainCloudLabel = corePointsLabel;
    if (useCloudsFromDB) {
        if (cloudLabels.size() > 4 + (cloudLabels.contains("TEST") ? 1 : 0)) {
            m_app->dispToConsole(
                    "This classifier uses more than 4 different clouds (the "
                    "GUI version cannot handle it)",
                    ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
            return;
        }

        // now show a dialog where the user will be able to set the cloud roles
        Classify3DMASCDialog classifDlg(m_app, true);
        classifDlg.setWindowTitle("3DMASC Train");
        classifDlg.setCloudRoles(cloudLabels, corePointsLabel, rolesAndNames);
        classifDlg.label_trainOrClassify->setText(
                "The classifier will be trained on " + corePointsLabel);
        classifDlg.classifierFileLineEdit->setText(inputFilename);
        classifDlg.keepAttributesCheckBox
                ->hide();  // this parameter is set in the trainDlg dialog
        if (!classifDlg.exec()) {
            // process cancelled by the user
            return;
        }

        classifDlg.getClouds(loadedClouds);
        m_app->dispToConsole("Training cloud: " + mainCloudLabel,
                             ecvMainAppInterface::STD_CONSOLE_MESSAGE);
        corePoints.origin = loadedClouds[mainCloudLabel];
        corePoints.role = mainCloudLabel;
    }

    static masc::TrainParameters s_params;
    loadTrainParameters(
            s_params);  // load the saved parameters or the default values
    masc::Feature::Set features;
    std::vector<double> scales;
    if (!masc::Tools::LoadTrainingFile(inputFilename, features, scales,
                                       loadedClouds, s_params, &corePoints,
                                       m_app->getMainWindow())) {
        m_app->dispToConsole("Failed to load the training file",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    if (!corePoints.origin) {
        m_app->dispToConsole("Core points not defined",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    if (mainCloudLabel.isEmpty()) {
        mainCloudLabel = corePoints.role;
    }

    if (!masc::Tools::GetClassificationSF(corePoints.origin)) {
        m_app->dispToConsole(
                "Missing 'Classification' field on core points cloud",
                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    ccHObject* group = new ccHObject("3DMASC");
    if (!useCloudsFromDB) {
        // add the loaded clouds to the main DB (so that we don't need to handle
        // them anymore)
        for (masc::Tools::NamedClouds::const_iterator it = loadedClouds.begin();
             it != loadedClouds.end(); ++it) {
            group->addChild(it.value());
        }
    }

    for (masc::Tools::NamedClouds::iterator it = loadedClouds.begin();
         it != loadedClouds.end(); ++it) {
        if (it.value())
            m_app->dispToConsole(it.key() + " = " + it.value()->getName(),
                                 ecvMainAppInterface::STD_CONSOLE_MESSAGE);
        else
            CVLog::Warning(it.key() + " is not associated to a point cloud");
    }

    // test role
    ccPointCloud* testCloud = nullptr;
    bool needTestSuite = false;
    masc::Feature::Set featuresTest;
    std::vector<double> scalesTest;
    if (loadedClouds.contains("TEST")) {
        testCloud = loadedClouds["TEST"];
        loadedClouds.remove("TEST");

        if (testCloud != corePoints.origin && testCloud != corePoints.cloud) {
            // we need a duplicated test suite!!!
            needTestSuite = true;
            // replace the main cloud by the test cloud
            masc::Tools::NamedClouds loadedCloudsTest;
            loadedCloudsTest = loadedClouds;
            loadedCloudsTest[mainCloudLabel] = testCloud;

            // simply reload the classification file to create duplicated
            // features
            masc::TrainParameters tempParams;
            if (!masc::Tools::LoadTrainingFile(inputFilename, featuresTest,
                                               scalesTest, loadedCloudsTest,
                                               tempParams)) {
                m_app->dispToConsole(
                        "Failed to load the training file (for TEST)",
                        ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                return;
            }
        }
    }

    // show the training dialog for the first time
    Train3DMASCDialog trainDlg(m_app->getMainWindow());
    trainDlg.setWindowModality(
            Qt::WindowModal);  // to be able to move the confusion matrix window
    trainDlg.maxDepthSpinBox->setValue(s_params.rt.maxDepth);
    trainDlg.maxTreeCountSpinBox->setValue(s_params.rt.maxTreeCount);
    trainDlg.activeVarCountSpinBox->setValue(s_params.rt.activeVarCount);
    trainDlg.minSampleCountSpinBox->setValue(s_params.rt.minSampleCount);
    trainDlg.testDataRatioSpinBox->setValue(
            static_cast<int>(s_params.testDataRatio * 100));
    trainDlg.testDataRatioSpinBox->setEnabled(testCloud == nullptr);
    trainDlg.setInputFilePath(inputFilename);

    // display the loaded features and let the user select the ones to use
    trainDlg.setResultText("Select features and press 'Run'");

    std::vector<FeatureSelection> originalFeatures;
    originalFeatures.reserve(features.size());
    for (const masc::Feature::Shared& f : features) {
        originalFeatures.push_back(FeatureSelection(f));
        trainDlg.addFeature(f->toString(), originalFeatures.back().importance,
                            originalFeatures.back().selected);
    }

    for (double scale : scales) trainDlg.addScale(scale, true);
    trainDlg.connectScaleSelectionToFeatureSelection();

    std::vector<FeatureSelection> originalFeaturesTest;
    if (testCloud && needTestSuite) {
        originalFeaturesTest.reserve(featuresTest.size());
        for (const masc::Feature::Shared& f : featuresTest) {
            originalFeaturesTest.push_back(FeatureSelection(f));
        }
    }

    static bool s_keepAttributes = trainDlg.keepAttributesCheckBox->isChecked();
    if (!trainDlg.exec()) {
        delete group;
        return;
    }
    assert(!trainDlg.shouldSaveClassifier());  // the save button should be
                                               // disabled at this point

    // compute the core points (if necessary)
    ecvProgressDialog progressDlg(true, m_app->getMainWindow());
    progressDlg.setAutoClose(false);
    if (!corePoints.prepare(&progressDlg)) {
        m_app->dispToConsole("Failed to compute/prepare the core points!",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        delete group;
        return;
    }

    if (corePoints.cloud != corePoints.origin) {
        // auto-hide the other clouds
        for (ccPointCloud* pc : loadedClouds) {
            pc->setEnabled(false);
        }
        // set an explicit name for the core points
        QString corePointsName = corePoints.origin->getName();
        switch (corePoints.selectionMethod) {
            case masc::CorePoints::NONE:
                break;
            case masc::CorePoints::RANDOM:
                corePointsName += "_SS_Random@" +
                                  QString::number(corePoints.selectionParam);
                break;
            case masc::CorePoints::SPATIAL:
                corePointsName += "_SS_Spatial@" +
                                  QString::number(corePoints.selectionParam);
                break;
            default:
                assert(false);
        }
        corePoints.cloud->setName(
                QString("Core points (%1)").arg(corePointsName));
        group->addChild(corePoints.cloud);
    }

    if (group->getChildrenNumber() != 0) {
        m_app->addToDB(group);
        QCoreApplication::processEvents();
    } else {
        delete group;
        group = nullptr;
    }

    // train / test subsets
    QSharedPointer<cloudViewer::ReferenceCloud> trainSubset, testSubset;
    float previousTestSubsetRatio = -1.0f;
    SFCollector generatedScalarFields;
    SFCollector generatedScalarFieldsTest;

    // we will train + evaluate the classifier, then display the results
    // then let the user change parameters and (potentially) start again
    for (int iteration = 0;; ++iteration) {
        // look for selected features
        features.clear();
        masc::Feature::Set toPrepare;
        for (size_t i = 0; i < originalFeatures.size(); ++i) {
            originalFeatures[i].selected = trainDlg.isFeatureSelected(
                    originalFeatures[i].feature->toString());

            // if the feature is selected
            if (originalFeatures[i].selected) {
                if (!originalFeatures[i].prepared) {
                    // we should prepare it first!
                    toPrepare.push_back(originalFeatures[i].feature);
                }
                features.push_back(originalFeatures[i].feature);
            }
        }

        masc::Classifier classifier;
        if (features.empty()) {
            m_app->dispToConsole("No feature selected!",
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        } else {
            // prepare the features (should be done once)
            if (!toPrepare.empty()) {
                progressDlg.show();
                QString error;
                if (!masc::Tools::PrepareFeatures(corePoints, toPrepare, error,
                                                  &progressDlg,
                                                  &generatedScalarFields)) {
                    m_app->dispToConsole(
                            error, ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                    generatedScalarFields.releaseSFs(false);
                    generatedScalarFieldsTest.releaseSFs(false);
                    return;
                }
                progressDlg.hide();
                QCoreApplication::processEvents();
                m_app->redrawAll();

                // flag the prepared features as 'prepared' ;)
                for (FeatureSelection& fs : originalFeatures) {
                    if (fs.selected && !fs.prepared) fs.prepared = true;
                }
            }

            // retrieve parameters
            s_params.rt.maxDepth = trainDlg.maxDepthSpinBox->value();
            s_params.rt.maxTreeCount = trainDlg.maxTreeCountSpinBox->value();
            s_params.rt.activeVarCount =
                    trainDlg.activeVarCountSpinBox->value();
            s_params.rt.minSampleCount =
                    trainDlg.minSampleCountSpinBox->value();
            float testDataRatio = 0.0f;

            if (!testCloud) {
                // we need to generate test subsets
                testDataRatio = s_params.testDataRatio =
                        trainDlg.testDataRatioSpinBox->value() / 100.0f;
                if (testDataRatio < 0.0f || testDataRatio > 0.99f) {
                    assert(false);
                    m_app->dispToConsole(
                            "Invalid test data ratio",
                            ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                    trainSubset.clear();
                    testSubset.clear();
                } else if (previousTestSubsetRatio != testDataRatio) {
                    if (!trainSubset)
                        trainSubset.reset(new cloudViewer::ReferenceCloud(
                                corePoints.cloud));
                    trainSubset->clear();

                    if (!testSubset)
                        testSubset.reset(new cloudViewer::ReferenceCloud(
                                corePoints.cloud));
                    testSubset->clear();

                    // randomly select the training points
                    if (!masc::Tools::RandomSubset(
                                corePoints.cloud, testDataRatio,
                                testSubset.data(), trainSubset.data())) {
                        m_app->dispToConsole(
                                "Not enough memory to generate the test "
                                "subsets",
                                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                        generatedScalarFields.releaseSFs(false);
                        generatedScalarFieldsTest.releaseSFs(false);
                        return;
                    }
                    previousTestSubsetRatio = testDataRatio;
                }
            }

            // extract the sources (after having prepared the features!)
            masc::Feature::Source::Set featureSources;
            masc::Feature::ExtractSources(features, featureSources);

            // train the classifier
            {
                QString errorMessage;
                if (!classifier.train(corePoints.cloud, s_params.rt,
                                      featureSources, errorMessage,
                                      trainSubset.data(), m_app,
                                      m_app->getMainWindow())) {
                    m_app->dispToConsole(
                            errorMessage,
                            ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                    generatedScalarFields.releaseSFs(false);
                    generatedScalarFieldsTest.releaseSFs(false);
                    return;
                }
                trainDlg.setFirstRunDone();
                //				trainDlg.shouldSaveClassifier();
                //// useless?
            }

            // test the trained classifier
            {
                if (testCloud) {
                    // look for selected features
                    if (needTestSuite) {
                        featuresTest.clear();
                        masc::Feature::Set toPrepareTest;
                        for (size_t i = 0; i < originalFeaturesTest.size();
                             ++i) {
                            originalFeaturesTest[i]
                                    .selected = trainDlg.isFeatureSelected(
                                    originalFeatures[i].feature->toString());
                            // if the feature is selected
                            if (originalFeaturesTest[i].selected) {
                                if (!originalFeaturesTest[i].prepared) {
                                    // we should prepare it first!
                                    toPrepareTest.push_back(
                                            originalFeaturesTest[i].feature);
                                }
                                featuresTest.push_back(
                                        originalFeaturesTest[i].feature);
                            }
                        }

                        // prepare the features and the test cloud
                        if (!toPrepareTest.empty()) {
                            masc::CorePoints corePointsTest;
                            corePointsTest.cloud = testCloud;
                            corePointsTest.origin = testCloud;
                            corePointsTest.role = mainCloudLabel;
                            progressDlg.show();
                            QCoreApplication::processEvents();
                            QString error;
                            if (!masc::Tools::PrepareFeatures(
                                        corePointsTest, toPrepareTest, error,
                                        &progressDlg,
                                        &generatedScalarFieldsTest)) {
                                m_app->dispToConsole(
                                        error, ecvMainAppInterface::
                                                       ERR_CONSOLE_MESSAGE);
                                generatedScalarFields.releaseSFs(false);
                                generatedScalarFieldsTest.releaseSFs(false);
                                return;
                            }
                            progressDlg.hide();
                            QCoreApplication::processEvents();
                            m_app->redrawAll();

                            // flag the prepared features as 'prepared' ;)
                            for (FeatureSelection& fs : originalFeaturesTest) {
                                if (fs.selected && !fs.prepared)
                                    fs.prepared = true;
                            }
                        }
                    }
                }

                masc::Classifier::AccuracyMetrics metrics;
                QString errorMessage;
                if (!classifier.evaluate(
                            featureSources,
                            testCloud ? testCloud : corePoints.cloud, metrics,
                            errorMessage, trainDlg,
                            testCloud ? nullptr : testSubset.data(),
                            testCloud ? "Classification_prediction"
                                      : "",  // outputSFName, empty is the test
                                             // cloud is not a separate cloud
                            m_app->getMainWindow(), m_app)) {
                    m_app->dispToConsole(
                            errorMessage,
                            ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                    generatedScalarFields.releaseSFs(false);
                    generatedScalarFieldsTest.releaseSFs(false);
                    return;
                }

                QString resultText =
                        QString("Correct guess = %1 / %2 --> accuracy = %3")
                                .arg(metrics.goodGuess)
                                .arg(metrics.sampleCount)
                                .arg(metrics.ratio);
                m_app->dispToConsole(resultText,
                                     ecvMainAppInterface::STD_CONSOLE_MESSAGE);
                trainDlg.setResultText(resultText);

                cv::Mat importanceMat = classifier.getVarImportance();
                // m_app->dispToConsole(QString("Var importance size = %1 x
                // %2").arg(importanceMat.rows).arg(importanceMat.cols));
                assert(static_cast<int>(features.size()) == importanceMat.rows);
                int selectedFeatureIndex = 0;
                for (size_t i = 0; i < originalFeatures.size(); ++i) {
                    if (originalFeatures[i].selected) {
                        // m_app->dispToConsole(QString("Feature #%1 importance
                        // = %2").arg(i + 1).arg(importanceMat.at<float>(i,
                        // 0)));
                        assert(selectedFeatureIndex < importanceMat.rows);
                        originalFeatures[i].importance =
                                importanceMat.at<float>(selectedFeatureIndex,
                                                        0);
                        ++selectedFeatureIndex;
                    } else {
                        originalFeatures[i].importance =
                                std::numeric_limits<float>::quiet_NaN();
                    }
                    trainDlg.setFeatureImportance(
                            originalFeatures[i].feature->toString(),
                            originalFeatures[i].importance);
                }

                trainDlg.sortByFeatureImportance();

                // if the checkbox "Save traces" is checked
                if (trainDlg.getSaveTrace()) {
                    // save the classifier in the trace directory with a generic
                    // name depending on the run
                    QString tracePath = trainDlg.getTracePath();
                    if (!tracePath.isEmpty()) {
                        QString outputFilePath =
                                tracePath + "/run_" +
                                QString::number(trainDlg.getRun()) + ".txt";
                        if (masc::Tools::SaveClassifier(
                                    outputFilePath, features, mainCloudLabel,
                                    classifier, m_app->getMainWindow())) {
                            m_app->dispToConsole(
                                    "Classifier succesfully saved to " +
                                            outputFilePath,
                                    ecvMainAppInterface::STD_CONSOLE_MESSAGE);
                            trainDlg.setClassifierSaved();
                        } else {
                            m_app->dispToConsole(
                                    "Failed to save classifier file");
                        }

                        // save feature importance
                        QString filename = tracePath + "/run_" +
                                           QString::number(trainDlg.getRun()) +
                                           "_feature_importance.txt";
                        QFile file(filename);
                        if (!file.open(QFile::Text | QFile::WriteOnly)) {
                            CVLog::Warning(
                                    QString("Can't open file '%1' for writing")
                                            .arg(filename));
                        }
                        QTextStream stream(&file);
                        stream << "# feature importance" << endl;
                        for (size_t i = 0; i < originalFeatures.size(); ++i) {
                            if (originalFeatures[i].selected) {
                                stream << originalFeatures[i]
                                                  .feature->toString()
                                       << " " << originalFeatures[i].importance
                                       << endl;
                            }
                        }
                    }
                }
            }
        }

        // now wait for the user input
        while (true)  // ew!
        {
            if (!trainDlg.exec()) {
                saveTrainParameters(s_params);
                // the dialog can be closed
                if (trainDlg.keepAttributesCheckBox->isChecked())
                    s_keepAttributes = true;
                else
                    s_keepAttributes = false;
                generatedScalarFields.releaseSFs(s_keepAttributes);
                generatedScalarFieldsTest.releaseSFs(s_keepAttributes);
                return;
            }

            // if the save button has been clicked
            if (trainDlg.shouldSaveClassifier()) {
                // ask for the output filename
                QString outputFilename;
                {
                    QSettings settings;
                    settings.beginGroup("3DMASC");
                    QString outputPath =
                            settings.value("FilePath",
                                           QCoreApplication::
                                                   applicationDirPath())
                                    .toString();
                    outputFilename = QFileDialog::getSaveFileName(
                            m_app->getMainWindow(), "Save 3DMASC classifier",
                            outputPath, "*.txt");
                    if (outputFilename.isNull()) {
                        // process cancelled by the user
                        continue;
                    }
                    settings.setValue("FilePath",
                                      QFileInfo(outputFilename).absolutePath());
                    settings.endGroup();
                }

                // save the classifier
                if (masc::Tools::SaveClassifier(outputFilename, features,
                                                mainCloudLabel, classifier,
                                                m_app->getMainWindow())) {
                    m_app->dispToConsole(
                            "Classifier succesfully saved to " + outputFilename,
                            ecvMainAppInterface::STD_CONSOLE_MESSAGE);
                    trainDlg.setClassifierSaved();
                } else {
                    m_app->dispToConsole("Failed to save classifier file");
                }
            } else  // we will run the classifier another time
            {
                // stop the local loop
                break;
            }
        }
    }
}

void q3DMASCPlugin::registerCommands(ccCommandLineInterface* cmd) {
    if (!cmd) {
        assert(false);
        return;
    }

    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new Command3DMASCClassif));
}
