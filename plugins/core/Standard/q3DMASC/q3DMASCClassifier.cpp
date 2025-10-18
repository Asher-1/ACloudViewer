// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "q3DMASCClassifier.h"

// Local
#include "ScalarFieldWrappers.h"
#include "q3DMASCTools.h"

// qCC_db
#include <CVLog.h>
#include <ecvDisplayTools.h>
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>
#include <ecvScalarField.h>

// qPDALIO
#include "../../../core/IO/qPDALIO/include/LASFields.h"

// CCPluginAPI
#include <ecvMainAppInterface.h>

// Qt
#include <QCoreApplication>
#include <QMessageBox>
#include <QProgressDialog>
#include <QtConcurrent>

#include "confusionmatrix.h"
#include "qTrain3DMASCDialog.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(CV_MAC_OS) || defined(CV_LINUX)
#include <unistd.h>
#endif

using namespace masc;

Classifier::Classifier() {}

bool Classifier::isValid() const {
    return (m_rtrees && m_rtrees->isClassifier() && m_rtrees->isTrained());
}

static IScalarFieldWrapper::Shared GetSource(const Feature::Source& fs,
                                             const ccPointCloud* cloud) {
    IScalarFieldWrapper::Shared source(nullptr);

    switch (fs.type) {
        case Feature::Source::ScalarField: {
            assert(!fs.name.isEmpty());
            int sfIdx = cloud->getScalarFieldIndexByName(qPrintable(fs.name));
            if (sfIdx >= 0) {
                source.reset(
                        new ScalarFieldWrapper(cloud->getScalarField(sfIdx)));
            } else {
                CVLog::Warning(
                        QObject::tr("Internal error: unknown scalar field '%1'")
                                .arg(fs.name));
                return IScalarFieldWrapper::Shared(nullptr);
            }
        } break;

        case Feature::Source::DimX:
            source.reset(new DimScalarFieldWrapper(
                    cloud, DimScalarFieldWrapper::DimX));
            break;
        case Feature::Source::DimY:
            source.reset(new DimScalarFieldWrapper(
                    cloud, DimScalarFieldWrapper::DimY));
            break;
        case Feature::Source::DimZ:
            source.reset(new DimScalarFieldWrapper(
                    cloud, DimScalarFieldWrapper::DimZ));
            break;

        case Feature::Source::Red:
            source.reset(new ColorScalarFieldWrapper(
                    cloud, ColorScalarFieldWrapper::Red));
            break;
        case Feature::Source::Green:
            source.reset(new ColorScalarFieldWrapper(
                    cloud, ColorScalarFieldWrapper::Green));
            break;
        case Feature::Source::Blue:
            source.reset(new ColorScalarFieldWrapper(
                    cloud, ColorScalarFieldWrapper::Blue));
            break;
    }

    return source;
}

bool Classifier::classify(const Feature::Source::Set& featureSources,
                          ccPointCloud* cloud,
                          QString& errorMessage,
                          QWidget* parentWidget /*=nullptr*/,
                          ecvMainAppInterface* app /*nullptr*/
) {
    if (!cloud) {
        assert(false);
        errorMessage = QObject::tr("Invalid input");
        return false;
    }

    if (!isValid()) {
        errorMessage = QObject::tr("Invalid classifier");
        return false;
    }

    if (featureSources.empty()) {
        errorMessage = QObject::tr(
                "Training method called without any feature (source)?!");
        return false;
    }

    // add a ccConfidence value if needed
    int cvConfidenceIdx =
            cloud->getScalarFieldIndexByName("Classification_confidence");
    if (cvConfidenceIdx >= 0)  // if the scalar field exists, delete it
        cloud->deleteScalarField(cvConfidenceIdx);
    cvConfidenceIdx = cloud->addScalarField("Classification_confidence");
    ccScalarField* cvConfidenceSF =
            static_cast<ccScalarField*>(cloud->getScalarField(cvConfidenceIdx));

    // look for the classification field
    cloudViewer::ScalarField* classificationSF =
            Tools::GetClassificationSF(cloud);
    ccScalarField* classifSFBackup = nullptr;

    if (classificationSF)  // save classification field (if any) by renaming it
                           // "Classification_backup"
    {
        CVLog::Warning(
                "Classification SF found: copy it in Classification_backup, a "
                "confusion matrix will be generated");
        // delete Classification_backup field (if any)
        int sfIdx = cloud->getScalarFieldIndexByName("Classification_backup");
        if (sfIdx >= 0) cloud->deleteScalarField(sfIdx);

        classificationSF->setName(
                "Classification_backup");  // rename the classification field
        classifSFBackup = static_cast<ccScalarField*>(classificationSF);
    }

    // create the classification SF
    ccScalarField* _classificationSF =
            new ccScalarField(LAS_FIELD_NAMES[LAS_CLASSIFICATION]);
    if (!_classificationSF->resizeSafe(cloud->size())) {
        _classificationSF->release();
        errorMessage = QObject::tr("Not enough memory");
        return false;
    }
    cloud->addScalarField(_classificationSF);
    classificationSF = _classificationSF;

    assert(classificationSF);
    classificationSF->fill(0);  // 0 = no classification?

    int sampleCount = static_cast<int>(cloud->size());
    int attributesPerSample = static_cast<int>(featureSources.size());

    CVLog::Print(
            QObject::tr("[3DMASC] Classifying %1 points with %2 feature(s)")
                    .arg(sampleCount)
                    .arg(attributesPerSample));

    // create the field wrappers
    std::vector<IScalarFieldWrapper::Shared> wrappers;
    {
        wrappers.reserve(attributesPerSample);
        for (int fIndex = 0; fIndex < attributesPerSample; ++fIndex) {
            const Feature::Source& fs = featureSources[fIndex];

            IScalarFieldWrapper::Shared source = GetSource(fs, cloud);
            if (!source || !source->isValid()) {
                assert(false);
                errorMessage =
                        QObject::tr("Internal error: invalid source '%1'")
                                .arg(fs.name);
                return false;
            }

            wrappers.push_back(source);
        }
    }

    QScopedPointer<ecvProgressDialog> pDlg;
    if (parentWidget) {
        pDlg.reset(new ecvProgressDialog(parentWidget));
        pDlg->setLabelText(QString("Classify (%1 points)").arg(sampleCount));
        pDlg->show();
        QCoreApplication::processEvents();
    }
    cloudViewer::NormalizedProgress nProgress(pDlg.data(), cloud->size());

    bool success = true;
    int numberOfTrees = static_cast<int>(m_rtrees->getRoots().size());
    bool cancelled = false;

#ifndef _DEBUG
#if defined(_OPENMP)
#pragma omp parallel for num_threads(omp_get_max_threads() - 2)
#endif
#endif
    for (int i = 0; i < static_cast<int>(cloud->size()); ++i) {
        {
            // allocate the data matrix
            cv::Mat test_data;
            try {
                test_data.create(1, attributesPerSample, CV_32FC1);
            } catch (const cv::Exception& cvex) {
                errorMessage = cvex.msg.c_str();
                success = false;
                cancelled = true;
            }

            if (!cancelled) {
                for (int fIndex = 0; fIndex < attributesPerSample; ++fIndex) {
                    double value = wrappers[fIndex]->pointValue(i);
                    test_data.at<float>(0, fIndex) = static_cast<float>(value);
                }

                float predictedClass =
                        m_rtrees->predict(test_data.row(0), cv::noArray(),
                                          cv::ml::DTrees::PREDICT_MAX_VOTE);
                classificationSF->setValue(i, static_cast<int>(predictedClass));
                // compute the confidence
                cv::Mat result;
                m_rtrees->getVotes(test_data, result,
                                   cv::ml::DTrees::PREDICT_MAX_VOTE);
                int classIndex = -1;
                for (int col = 0; col < result.cols;
                     col++)  // look for the index of the predicted class
                    if (predictedClass == result.at<int>(0, col)) {
                        classIndex = col;
                        break;
                    }
                if (classIndex != -1) {
                    float nbVotes = result.at<int>(
                            1, classIndex);  // get the number of votes
                    cvConfidenceSF->setValue(
                            i,
                            static_cast<ScalarType>(
                                    nbVotes /
                                    numberOfTrees));  // compute the confidence
                } else
                    cvConfidenceSF->setValue(i, NAN_VALUE);

                if (pDlg && !nProgress.oneStep()) {
                    // process cancelled by the user
                    success = false;
                    cancelled = true;
                }
            }
        }
    }

    classificationSF->computeMinAndMax();
    cvConfidenceSF->computeMinAndMax();

    // show the classification field by default
    {
        int classifSFIdx =
                cloud->getScalarFieldIndexByName(classificationSF->getName());
        cloud->setCurrentDisplayedScalarField(classifSFIdx);
        cloud->showSF(true);
    }

    if (parentWidget) {
        ecvDisplayTools::SetRedrawRecursive(false);
        cloud->setRedrawFlagRecursive(true);
        ecvDisplayTools::RedrawDisplay(false, true);
        QCoreApplication::processEvents();
    }

    if (classifSFBackup != nullptr) {
        if (app) {
            ConfusionMatrix* confusionMatrix =
                    new ConfusionMatrix(*classifSFBackup, *classificationSF);
        }
    }

    return success;
}

bool Classifier::evaluate(const Feature::Source::Set& featureSources,
                          ccPointCloud* testCloud,
                          AccuracyMetrics& metrics,
                          QString& errorMessage,
                          Train3DMASCDialog& train3DMASCDialog,
                          cloudViewer::ReferenceCloud* testSubset /*=nullptr=*/,
                          QString outputSFName /*=QString()*/,
                          QWidget* parentWidget /*=nullptr*/,
                          ecvMainAppInterface* app /*=nullptr*/) {
    if (!testCloud) {
        // invalid input
        assert(false);
        errorMessage = QObject::tr("Invalid input cloud");
        return false;
    }
    metrics.sampleCount = metrics.goodGuess = 0;
    metrics.ratio = 0.0f;

    if (!m_rtrees || !m_rtrees->isTrained()) {
        errorMessage = QObject::tr("Classifier hasn't been trained yet");
        return false;
    }

    if (featureSources.empty()) {
        errorMessage = QObject::tr(
                "Training method called without any feature (source)?!");
        return false;
    }
    if (testSubset && testSubset->getAssociatedCloud() != testCloud) {
        errorMessage = QObject::tr(
                "Invalid test subset (associated point cloud is different)");
        return false;
    }

    // look for the classification field
    cloudViewer::ScalarField* classifSF = Tools::GetClassificationSF(testCloud);
    if (!classifSF || classifSF->size() < testCloud->size()) {
        assert(false);
        errorMessage = QObject::tr(
                "Missing/invalid 'Classification' field on input cloud");
        return false;
    }

    ccScalarField* outSF = nullptr;
    ccScalarField* cvConfidenceSF = nullptr;

    if (!outputSFName.isEmpty()) {
        int outIdx =
                testCloud->getScalarFieldIndexByName(qPrintable(outputSFName));
        if (outIdx >= 0)
            testCloud->deleteScalarField(outIdx);
        else
            CVLog::Print("add " + outputSFName + " to the TEST cloud");
        outIdx = testCloud->addScalarField(qPrintable(outputSFName));
        outSF = static_cast<ccScalarField*>(testCloud->getScalarField(outIdx));
    }

    if (outSF)  // add a Classification_confidence value to the test cloud if
                // needed
    {
        int cvConfidenceIdx = testCloud->getScalarFieldIndexByName(
                "Classification_confidence");
        if (cvConfidenceIdx >= 0)  // if the scalar field exists, delete it
            testCloud->deleteScalarField(cvConfidenceIdx);
        else
            CVLog::Print("add Classification_confidence to the TEST cloud");
        cvConfidenceIdx =
                testCloud->addScalarField("Classification_confidence");
        cvConfidenceSF = static_cast<ccScalarField*>(
                testCloud->getScalarField(cvConfidenceIdx));
    }

    unsigned testSampleCount =
            (testSubset ? testSubset->size() : testCloud->size());
    int attributesPerSample = static_cast<int>(featureSources.size());

    CVLog::Print(
            QObject::tr("[3DMASC] Testing data: %1 samples with %2 feature(s)")
                    .arg(testSampleCount)
                    .arg(attributesPerSample));

    // allocate the data matrix
    cv::Mat test_data;
    try {
        test_data.create(static_cast<int>(testSampleCount), attributesPerSample,
                         CV_32FC1);
    } catch (const cv::Exception& cvex) {
        errorMessage = cvex.msg.c_str();
        return false;
    }

    QScopedPointer<ecvProgressDialog> pDlg;
    if (parentWidget) {
        pDlg.reset(new ecvProgressDialog(parentWidget));
        pDlg->setLabelText(QString("Evaluating the classifier on %1 points")
                                   .arg(testSampleCount));
        pDlg->show();
        QCoreApplication::processEvents();
    }
    cloudViewer::NormalizedProgress nProgress(pDlg.data(), testSampleCount);

    // fill the data matrix
    for (int fIndex = 0; fIndex < attributesPerSample; ++fIndex) {
        const Feature::Source& fs = featureSources[fIndex];
        IScalarFieldWrapper::Shared source = GetSource(fs, testCloud);
        if (!source || !source->isValid()) {
            assert(false);
            errorMessage = QObject::tr("Internal error: invalid source '%1'")
                                   .arg(fs.name);
            return false;
        }

        for (unsigned i = 0; i < testSampleCount; ++i) {
            unsigned pointIndex =
                    (testSubset ? testSubset->getPointGlobalIndex(i) : i);
            double value = source->pointValue(pointIndex);
            test_data.at<float>(i, fIndex) = static_cast<float>(value);
        }
    }

    int numberOfTrees = static_cast<int>(m_rtrees->getRoots().size());

    // estimate the efficiency of the classifier
    std::vector<ScalarType> actualClass(testSampleCount);
    std::vector<ScalarType> predictectedClass(testSampleCount);
    {
        metrics.sampleCount = testSampleCount;
        metrics.goodGuess = 0;

        for (unsigned i = 0; i < testSampleCount; ++i) {
            unsigned pointIndex =
                    (testSubset ? testSubset->getPointGlobalIndex(i) : i);
            ScalarType pointClass = classifSF->getValue(pointIndex);
            int iClass = static_cast<int>(pointClass);
            // if (iClass < 0 || iClass > 255)
            //{
            //	errorMessage = QObject::tr("Classification values out of range
            //(0-255)"); 	return false;
            // }

            float fPredictedClass =
                    m_rtrees->predict(test_data.row(i), cv::noArray(),
                                      cv::ml::DTrees::PREDICT_MAX_VOTE);
            int iPredictedClass = static_cast<int>(fPredictedClass);
            actualClass.at(i) = iClass;
            predictectedClass.at(i) = iPredictedClass;
            if (iPredictedClass == iClass) {
                ++metrics.goodGuess;
            }
            if (outSF) {
                outSF->setValue(pointIndex,
                                static_cast<ScalarType>(iPredictedClass));
                if (cvConfidenceSF) {
                    // compute the confidence
                    cv::Mat result;
                    m_rtrees->getVotes(test_data.row(i), result,
                                       cv::ml::DTrees::PREDICT_MAX_VOTE);
                    int classIndex = -1;
                    for (int col = 0; col < result.cols;
                         col++)  // look for the index of the predicted class
                        if (iPredictedClass == result.at<int>(0, col)) {
                            classIndex = col;
                            break;
                        }
                    if (classIndex != -1) {
                        float nbVotes = result.at<int>(
                                1, classIndex);  // get the number of votes
                        cvConfidenceSF->setValue(
                                i, static_cast<ScalarType>(
                                           nbVotes /
                                           numberOfTrees));  // compute the
                                                             // confidence
                    } else
                        cvConfidenceSF->setValue(i, NAN_VALUE);
                }
            }

            if (pDlg && !nProgress.oneStep()) {
                // process cancelled by the user
                return false;
            }
        }

        if (outSF) outSF->computeMinAndMax();
        if (cvConfidenceSF) cvConfidenceSF->computeMinAndMax();

        metrics.ratio =
                static_cast<float>(metrics.goodGuess) / metrics.sampleCount;
    }

    ConfusionMatrix* confusionMatrix =
            new ConfusionMatrix(actualClass, predictectedClass);
    train3DMASCDialog.addConfusionMatrixAndSaveTraces(confusionMatrix);
    if (app) {
        confusionMatrix->show();
    }

    // show the Classification_prediction field by default
    if (outSF) {
        int classifSFIdx =
                testCloud->getScalarFieldIndexByName(outSF->getName());
        testCloud->setCurrentDisplayedScalarField(classifSFIdx);
        testCloud->showSF(true);
    }

    if (parentWidget) {
        ecvDisplayTools::SetRedrawRecursive(false);
        testCloud->setRedraw(true);
        ecvDisplayTools::RedrawDisplay(false, true);
        QCoreApplication::processEvents();
    }

    return true;
}

bool Classifier::train(const ccPointCloud* cloud,
                       const RandomTreesParams& params,
                       const Feature::Source::Set& featureSources,
                       QString& errorMessage,
                       cloudViewer::ReferenceCloud* trainSubset /*=nullptr*/,
                       ecvMainAppInterface* app /*=nullptr*/,
                       QWidget* parentWidget /*=nullptr*/) {
    if (featureSources.empty()) {
        errorMessage = QObject::tr(
                "Training method called without any feature (source)?!");
        return false;
    }
    if (!cloud) {
        errorMessage = QObject::tr("Invalid input cloud");
        return false;
    }

    if (trainSubset && trainSubset->getAssociatedCloud() != cloud) {
        errorMessage = QObject::tr(
                "Invalid train subset (associated point cloud is different)");
        return false;
    }

    // look for the classification field
    cloudViewer::ScalarField* classifSF = Tools::GetClassificationSF(cloud);
    if (!classifSF || classifSF->size() < cloud->size()) {
        assert(false);
        errorMessage = QObject::tr(
                "Missing/invalid 'Classification' field on input cloud");
        return false;
    }

    int sampleCount =
            static_cast<int>(trainSubset ? trainSubset->size() : cloud->size());
    int attributesPerSample = static_cast<int>(featureSources.size());

    if (app) {
        app->dispToConsole(
                QString("[3DMASC] Training data: %1 samples with %2 feature(s)")
                        .arg(sampleCount)
                        .arg(attributesPerSample));
    }

    cv::Mat training_data, train_labels;
    try {
        training_data.create(sampleCount, attributesPerSample, CV_32FC1);
        train_labels.create(sampleCount, 1, CV_32FC1);
    } catch (const cv::Exception& cvex) {
        errorMessage = cvex.msg.c_str();
        return false;
    }

    // fill the classification labels vector
    {
        for (int i = 0; i < sampleCount; ++i) {
            int pointIndex =
                    (trainSubset ? static_cast<int>(
                                           trainSubset->getPointGlobalIndex(i))
                                 : i);
            ScalarType pointClass = classifSF->getValue(pointIndex);
            int iClass = static_cast<int>(pointClass);
            // if (iClass < 0 || iClass > 255)
            //{
            //	errorMessage = QObject::tr("Classification values out of range
            //(0-255)"); 	return false;
            // }

            train_labels.at<float>(i) = static_cast<unsigned char>(iClass);
        }
    }

    // fill the training data matrix
    for (int fIndex = 0; fIndex < attributesPerSample; ++fIndex) {
        const Feature::Source& fs = featureSources[fIndex];

        IScalarFieldWrapper::Shared source = GetSource(fs, cloud);
        if (!source || !source->isValid()) {
            assert(false);
            errorMessage = QObject::tr("Internal error: invalid source '%1'")
                                   .arg(fs.name);
            return false;
        }

        for (int i = 0; i < sampleCount; ++i) {
            int pointIndex =
                    (trainSubset ? static_cast<int>(
                                           trainSubset->getPointGlobalIndex(i))
                                 : i);
            double value = source->pointValue(pointIndex);
            training_data.at<float>(i, fIndex) = static_cast<float>(value);
        }
    }

    QScopedPointer<QProgressDialog> pDlg;
    if (parentWidget) {
        pDlg.reset(new QProgressDialog(parentWidget));
        pDlg->setRange(0, 0);  // infinite loop
        pDlg->setLabelText("Training classifier");
        pDlg->show();
        QCoreApplication::processEvents();
    }

    m_rtrees = cv::ml::RTrees::create();
    m_rtrees->setMaxDepth(params.maxDepth);
    m_rtrees->setMinSampleCount(params.minSampleCount);
    m_rtrees->setRegressionAccuracy(0);
    // If true then surrogate splits will be built. These splits allow to work
    // with missing data and compute variable importance correctly. Default
    // value is false.
    m_rtrees->setUseSurrogates(false);
    m_rtrees->setPriors(cv::Mat());
    // m_rtrees->setMaxCategories(params.maxCategories); //not important?
    m_rtrees->setCalculateVarImportance(true);
    m_rtrees->setActiveVarCount(params.activeVarCount);
    cv::TermCriteria terminationCriteria(
            cv::TermCriteria::MAX_ITER, params.maxTreeCount,
            std::numeric_limits<double>::epsilon());
    m_rtrees->setTermCriteria(terminationCriteria);

    QFuture<bool> future = QtConcurrent::run([&]() {
        // Code in this block will run in another thread
        try {
            cv::Mat sampleIndexes =
                    cv::Mat::zeros(1, training_data.rows, CV_8U);
            //			cv::Mat trainSamples = sampleIndexes.colRange(0,
            // sampleCount);
            // trainSamples.setTo(cv::Scalar::all(1));

            cv::Mat varTypes(training_data.cols + 1, 1, CV_8U);
            varTypes.setTo(cv::Scalar::all(cv::ml::VAR_ORDERED));
            varTypes.at<uchar>(training_data.cols) = cv::ml::VAR_CATEGORICAL;

            cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(
                    training_data, cv::ml::ROW_SAMPLE,
                    train_labels,                 /* samples layout responses */
                    cv::noArray(), sampleIndexes, /* varIdx sampleIdx */
                    cv::noArray(), varTypes);     // sampleWeights varType

            bool success = m_rtrees->train(trainData);
            if (!success || !m_rtrees->isClassifier()) {
                errorMessage = "Training failed";
                return false;
            }
        } catch (const cv::Exception& cvex) {
            m_rtrees.release();
            errorMessage = cvex.msg.c_str();
            return false;
        } catch (const std::exception& stdex) {
            errorMessage = stdex.what();
            return false;
        } catch (...) {
            errorMessage = QObject::tr("Unknown error");
            return false;
        }

        return true;
    });

    while (!future.isFinished()) {
#if defined(CV_WINDOWS)
        ::Sleep(500);
#else
        usleep(500 * 1000);
#endif
        if (pDlg) {
            if (pDlg->wasCanceled()) {
                //				future.cancel();
                QMessageBox msgBox;
                msgBox.setText(
                        "The training is still in progress, not possible to "
                        "cancel.");
                msgBox.exec();
                //				break;
                pDlg->reset();
                pDlg->show();
            }
            pDlg->setValue(pDlg->value() + 1);
        }
        QCoreApplication::processEvents();
    }

    if (pDlg) {
        pDlg->close();
        QCoreApplication::processEvents();
    }

    if (future.isCanceled() || !future.result() || !m_rtrees->isTrained()) {
        errorMessage = QObject::tr("Training failed for an unknown reason...");
        m_rtrees.release();
        return false;
    }

    return true;
}

bool Classifier::toFile(QString filename,
                        QWidget* parentWidget /*=nullptr*/) const {
    if (!m_rtrees) {
        CVLog::Warning(
                QObject::tr("Classifier hasn't been trained, can't save it"));
        return false;
    }

    // save the classifier
    QProgressDialog pDlg(parentWidget);
    pDlg.setRange(0, 0);  // infinite loop
    pDlg.setLabelText(QObject::tr("Saving classifier"));
    pDlg.show();
    QCoreApplication::processEvents();

    cv::String cvFilename = filename.toStdString();
    m_rtrees->save(cvFilename);

    pDlg.close();
    QCoreApplication::processEvents();

    CVLog::Print("Classifier file saved to: " +
                 QString::fromStdString(cvFilename));
    return true;
}

bool Classifier::fromFile(QString filename,
                          QWidget* parentWidget /*=nullptr*/) {
    // load the classifier
    QScopedPointer<QProgressDialog> pDlg;
    if (parentWidget) {
        pDlg.reset(new QProgressDialog(parentWidget));
        pDlg->setRange(0, 0);  // infinite loop
        pDlg->setLabelText(QObject::tr("Loading classifier"));
        pDlg->show();
        QCoreApplication::processEvents();
    }

    try {
        m_rtrees = cv::ml::RTrees::load(filename.toStdString());
    } catch (const cv::Exception& cvex) {
        CVLog::Warning(cvex.msg.c_str());
        CVLog::Error("Failed to load file: " + filename);
        return false;
    }

    if (pDlg) {
        pDlg->close();
        QCoreApplication::processEvents();
    }

    if (m_rtrees->empty() || !m_rtrees->isClassifier()) {
        CVLog::Error(QObject::tr("Loaded classifier is invalid"));
        return false;
    } else if (!m_rtrees->isTrained()) {
        CVLog::Warning(
                QObject::tr("Loaded classifier doesn't seem to be trained"));
    }

    return true;
}
