// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "q3DMASCTools.h"

// Local
#include "ContextBasedFeature.h"
#include "DualCloudFeature.h"
#include "NeighborhoodFeature.h"
#include "PointFeature.h"
#include "ecvMainAppInterface.h"

// qCC_io
#include <FileIOFilter.h>
// qCC_db
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

// qPDALIO
#include "../../../core/IO/qPDALIO/include/LASFields.h"

// Qt
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMutex>
#include <QTextStream>

// system
#include <assert.h>

#if defined(_OPENMP)
#include <omp.h>
#endif
using namespace masc;

bool Tools::SaveClassifier(QString filename,
                           const Feature::Set& features,
                           const QString corePointsRole,
                           const masc::Classifier& classifier,
                           QWidget* parent /*=nullptr*/) {
    // first save the classifier data (same base filename but with the yaml
    // extension)
    QFileInfo fi(filename);
    QString yamlFilename = fi.baseName() + ".yaml";
    QString yamlAbsoluteFilename =
            fi.absoluteDir().absoluteFilePath(yamlFilename);
    if (!classifier.toFile(yamlAbsoluteFilename, parent)) {
        CVLog::Error("Failed to save the classifier data");
        return false;
    }

    QFile file(filename);
    if (!file.open(QFile::Text | QFile::WriteOnly)) {
        CVLog::Warning(
                QString("Can't open file '%1' for writing").arg(filename));
        return false;
    }

    QTextStream stream(&file);

    stream << "# 3DMASC classifier file" << endl;
    stream << "classifier: " << yamlFilename << endl;

    // look for all clouds (labels)
    QList<QString> cloudLabels;
    for (Feature::Shared f : features) {
        if (f->cloud1 && !cloudLabels.contains(f->cloud1Label))
            cloudLabels.push_back(f->cloud1Label);
        if (f->cloud2 && !cloudLabels.contains(f->cloud2Label))
            cloudLabels.push_back(f->cloud2Label);
    }
    if (!corePointsRole.isEmpty() && !cloudLabels.contains(corePointsRole)) {
        cloudLabels.push_back(corePointsRole);
    }

    stream << "# Clouds (roles)" << endl;
    for (const QString& label : cloudLabels) {
        stream << "cloud: " << label << endl;
    }

    if (!corePointsRole.isEmpty()) {
        stream << "# Core points (classified role)" << endl;
        stream << "core_points: " << corePointsRole << endl;
    }

    stream << "# Features" << endl;
    for (Feature::Shared f : features) {
        stream << "feature: " << f->toString() << endl;
    }

    return true;
}

bool Tools::LoadClassifierCloudLabels(QString filename,
                                      QList<QString>& labels,
                                      QString& corePointsLabel,
                                      bool& filenamesSpecified,
                                      QMap<QString, QString>& rolesAndNames) {
    // just in case
    corePointsLabel.clear();
    labels.clear();
    rolesAndNames.clear();

    QFile file(filename);
    if (!file.open(QFile::Text | QFile::ReadOnly)) {
        CVLog::Warning(QString("Can't open file '%1'").arg(filename));
        return false;
    }

    QTextStream stream(&file);
    int filenameCount = 0;
    for (int lineNumber = 0;; ++lineNumber) {
        QString line = stream.readLine();
        if (line.isNull()) {
            // eof
            break;
        }
        ++lineNumber;

        line = line.toUpper();
        if (line.startsWith("CLOUD:")) {
            QString command = line.mid(6).trimmed();
            QStringList tokens = command.split('=');
            if (tokens.size() == 0) {
                CVLog::Warning(
                        "Malformed file: expecting some tokens after 'cloud:' "
                        "on line #" +
                        QString::number(lineNumber));
                return false;
            }

            QString label = tokens.front();
            if (labels.contains(label)) {
                CVLog::Warning(QString("Malformed file: role '%1:' is already "
                                       "defined/used on line #%2")
                                       .arg(label)
                                       .arg(lineNumber));
                return false;
            }
            labels.push_back(label);
            rolesAndNames[label] = tokens.back();

            if (tokens.size() > 1) ++filenameCount;
        } else if (line.startsWith("CORE_POINTS:")) {
            if (!corePointsLabel.isEmpty()) {
                // core points defined multiple times?!
                continue;
            }
            QString command = line.mid(12);
            QStringList tokens = command.split('_');
            if (tokens.empty()) {
                CVLog::Warning(
                        "Malformed file: expecting tokens after 'core_points:' "
                        "on line #" +
                        QString::number(lineNumber));
                return false;
            }
            corePointsLabel = tokens[0].trimmed();
        }
    }

    filenamesSpecified = (filenameCount > 0 && filenameCount == labels.size());

    return true;
}

bool CheckFeatureUnicity(std::vector<Feature::Shared>& rawFeatures,
                         Feature::Shared feature) {
    if (!feature) return false;

    // check that the feature does not exists already!
    for (const auto& feat : rawFeatures) {
        if (feat->toString() == feature->toString()) {
            return false;
        }
    }
    return true;
}

static bool CreateFeaturesFromCommand(const QString& command,
                                      QString corePointsRole,
                                      int lineNumber,
                                      const Tools::NamedClouds& clouds,
                                      std::vector<Feature::Shared>& rawFeatures,
                                      std::vector<double>& scales) {
    QStringList tokens = command.split('_');
    if (tokens.empty()) {
        CVLog::Warning(
                "Malformed file: expecting at least one token after 'feature:' "
                "on line #" +
                QString::number(lineNumber));
        return false;
    }

    Feature::Shared feature;

    // read the type
    QString typeStr = tokens[0].trimmed().toUpper();
    {
        for (int iteration = 0; iteration < 1;
             ++iteration)  // fake loop for easy break
        {
            PointFeature::PointFeatureType pointFeatureType =
                    PointFeature::FromUpperString(typeStr);
            if (pointFeatureType != PointFeature::Invalid) {
                // we have a point feature
                PointFeature* pointFeature = new PointFeature(pointFeatureType);

                // specific case: 'SF'
                if (pointFeatureType == PointFeature::SF) {
                    QString sfIndexStr = typeStr.mid(2);
                    bool ok = true;
                    int sfIndex = sfIndexStr.toInt(&ok);
                    if (!ok) {
                        CVLog::Warning(
                                QString("Malformed file: expecting a valid "
                                        "integer value after 'SF' on line #%1")
                                        .arg(lineNumber));
                        delete pointFeature;
                        return false;
                    }
                    pointFeature->sourceSFIndex = sfIndex;
                }

                feature.reset(pointFeature);
                break;
            }

            NeighborhoodFeature::NeighborhoodFeatureType
                    neighborhoodFeatureType =
                            NeighborhoodFeature::FromUpperString(typeStr);
            if (neighborhoodFeatureType != NeighborhoodFeature::Invalid) {
                // we have a neighborhood feature
                feature = NeighborhoodFeature::Shared(
                        new NeighborhoodFeature(neighborhoodFeatureType));
                break;
            }

            ContextBasedFeature::ContextBasedFeatureType
                    contextBasedFeatureType =
                            ContextBasedFeature::FromUpperString(typeStr);
            if (contextBasedFeatureType != ContextBasedFeature::Invalid) {
                // we have a context-based feature
                QString featureStr =
                        ContextBasedFeature::ToString(contextBasedFeatureType);
                int kNN = 1;
                if (featureStr.length() < typeStr.length()) {
                    bool ok = false;
                    kNN = typeStr.mid(featureStr.length()).toInt(&ok);
                    if (!ok || kNN <= 0) {
                        CVLog::Warning(QString("Malformed file: expecting a "
                                               "valid and positive number "
                                               "after '%1' on line #%2")
                                               .arg(featureStr)
                                               .arg(lineNumber));
                        return false;
                    }
                }
                feature = ContextBasedFeature::Shared(
                        new ContextBasedFeature(contextBasedFeatureType, kNN));
                break;
            }

            DualCloudFeature::DualCloudFeatureType dualCloudFeatureType =
                    DualCloudFeature::FromUpperString(typeStr);
            if (dualCloudFeatureType != DualCloudFeature::Invalid) {
                // we have a dual cloud feature
                feature = DualCloudFeature::Shared(
                        new DualCloudFeature(dualCloudFeatureType));
                break;
            }

            if (!feature) {
                CVLog::Warning(QString("Malformed file: unrecognized token "
                                       "'%1' after 'feature:' on line #%2")
                                       .arg(typeStr)
                                       .arg(lineNumber));
                return false;
            }
        }
    }
    assert(feature);

    // read the scales
    bool useAllScales = false;
    {
        QString scaleStr = tokens[1].toUpper();
        if (!scaleStr.startsWith("SC")) {
            CVLog::Warning(QString("Malformed file: unrecognized token '%1' "
                                   "(expecting the scale descriptor 'SC...' on "
                                   "line #%2")
                                   .arg(typeStr)
                                   .arg(lineNumber));
            return false;
        }

        if (scaleStr == "SC0") {
            // no scale
        } else if (scaleStr == "SCX") {
            // all scales
            useAllScales = true;
        } else {
            // read the specific scale value
            bool ok = true;
            feature->scale = scaleStr.mid(2).toDouble(&ok);
            if (!ok) {
                CVLog::Warning(QString("Malformed file: expecting a valid "
                                       "number after 'SC:' on line #%1")
                                       .arg(lineNumber));
                return false;
            }
        }
    }

    // process the next tokens (may not be ordered)
    int cloudCount = 0;
    bool statDefined = false;
    bool mathDefined = false;
    bool contextBasedFeatureDeprecatedSyntax = false;
    for (int i = 2; i < tokens.size(); ++i) {
        QString token = tokens[i].trimmed().toUpper();

        // is the token a 'stat' one?
        if (!statDefined) {
            if (token == "MEAN") {
                feature->stat = Feature::MEAN;
                statDefined = true;
            } else if (token == "MODE") {
                feature->stat = Feature::MODE;
                statDefined = true;
            } else if (token == "MEDIAN") {
                feature->stat = Feature::MEDIAN;
                statDefined = true;
            } else if (token == "STD") {
                feature->stat = Feature::STD;
                statDefined = true;
            } else if (token == "RANGE") {
                feature->stat = Feature::RANGE;
                statDefined = true;
            } else if (token == "SKEW") {
                feature->stat = Feature::SKEW;
                statDefined = true;
            }

            if (statDefined) {
                continue;
            }
        }

        // is the token a cloud name?
        if (cloudCount < 2) {
            bool cloudNameMatches = false;
            for (QMap<QString, ccPointCloud*>::const_iterator it =
                         clouds.begin();
                 it != clouds.end(); ++it) {
                QString key = it.key().toUpper();
                if (key == token) {
                    if (cloudCount == 0) {
                        feature->cloud1 = it.value();
                        feature->cloud1Label = key;

                        if (feature &&
                            feature->getType() ==
                                    Feature::Type::ContextBasedFeature) {
                            // only one cloud is necessary for context based
                            // features the class should be just after the cloud
                            // name in the regular syntax
                            if (i + 1 < tokens.size()) {
                                bool ok = false;
                                int classLabel = tokens[i + 1].toInt(&ok);
                                if (!ok) {
                                    contextBasedFeatureDeprecatedSyntax =
                                            true;  // let's try the deprecated
                                                   // syntax
                                } else {
                                    qSharedPointerCast<ContextBasedFeature>(
                                            feature)
                                            ->ctxClassLabel = classLabel;
                                    ++i;
                                }
                            } else {
                                CVLog::Warning(
                                        QString("Malformed context based "
                                                "features at line %1")
                                                .arg(lineNumber));
                                return false;
                            }
                        }

                    } else if (cloudCount == 1) {
                        if (contextBasedFeatureDeprecatedSyntax) {
                            feature->cloud1 = it.value();
                            feature->cloud1Label = key;
                        } else {
                            feature->cloud2 = it.value();
                            feature->cloud2Label = key;
                        }

                        if (feature &&
                            feature->getType() ==
                                    Feature::Type::ContextBasedFeature) {
                            // this is the DEPRECATED syntax for the context
                            // based feature the class is just after the cloud
                            // name
                            if (i + 1 < tokens.size()) {
                                bool ok = false;
                                int classLabel = tokens[i + 1].toInt(&ok);
                                if (!ok) {
                                    CVLog::Warning(
                                            QString("ContextBasedFeature: "
                                                    "expecting a class number "
                                                    "after the context cloud "
                                                    "'%1' on line #%2")
                                                    .arg(token)
                                                    .arg(lineNumber));
                                    return false;
                                } else {
                                    CVLog::Warning(
                                            "ContextBasedFeature: you are "
                                            "using the DEPRECATED syntax, the "
                                            "feature should contain only one "
                                            "cloud, as in DZ1_SC0_CTX_10)");
                                    qSharedPointerCast<ContextBasedFeature>(
                                            feature)
                                            ->ctxClassLabel = classLabel;
                                    ++i;
                                }
                            } else {
                                CVLog::Warning(
                                        QString("Malformed context based "
                                                "features at line %1")
                                                .arg(lineNumber));
                                return false;
                            }
                        }
                    } else {
                        // we can't fall here
                        assert(false);
                    }
                    ++cloudCount;
                    cloudNameMatches = true;
                    break;
                }
            }

            if (cloudNameMatches) {
                continue;
            }
        }

        // is the token a 'math' one?
        if (!mathDefined) {
            if (token == "MINUS") {
                feature->op = Feature::MINUS;
                mathDefined = true;
            } else if (token == "PLUS") {
                feature->op = Feature::PLUS;
                mathDefined = true;
            } else if (token == "DIVIDE") {
                feature->op = Feature::DIVIDE;
                mathDefined = true;
            } else if (token == "MULTIPLY") {
                feature->op = Feature::MULTIPLY;
                mathDefined = true;
            }

            if (mathDefined) {
                continue;
            }
        }

        // is the token a 'context' descriptor?
        // if (feature->getType() == Feature::Type::ContextBasedFeature &&
        // token.startsWith("CTX"))
        //{
        //	//read the context label
        //	QString ctxLabelStr = token.mid(3);
        //	bool ok = true;
        //	int ctxLabel = ctxLabelStr.toInt(&ok);
        //	if (!ok)
        //	{
        //		CVLog::Warning(QString("Malformed file: expecting a
        // valid integer value after 'CTX' on line #%1").arg(lineNumber));
        // return false;
        //	}
        //	static_cast<ContextBasedFeature*>(feature.data())->ctxClassLabel
        //= ctxLabel; 	continue;
        // }

        // if we are here, it means we couldn't find a correspondance for the
        // current token
        CVLog::Warning(QString("Malformed file: unrecognized or unexpected "
                               "token '%1' on line #%2")
                               .arg(token)
                               .arg(lineNumber));
        return false;
    }

    // now create the various versions of rules (if any)
    if (useAllScales) {
        if (scales.empty()) {
            CVLog::Warning(
                    "Malformed file: 'SCx' token used while no scale is "
                    "defined" +
                    QString(" (line %1)").arg(lineNumber));
            return false;
        }
        feature->scale = scales.front();

        // we will duplicate the original feature AFTER having checked its
        // consistency!
    }

    // now check the consistency of the rule
    QString errorMessage;
    if (!feature->checkValidity(corePointsRole, errorMessage)) {
        CVLog::Warning("Malformed feature: " + errorMessage +
                       QString(" (line %1)").arg(lineNumber));
        return false;
    }

    if (!CheckFeatureUnicity(
                rawFeatures,
                feature))  // check that the feature does not exists already!
    {
        CVLog::Warning("[3DMASC] duplicated feature " + feature->toString() +
                       ", check your parameter file");
        return false;
    } else {
        // save the feature
        rawFeatures.push_back(feature);
    }

    if (useAllScales) {
        for (size_t i = 1; i < scales.size(); ++i) {
            // copy the original rule
            Feature::Shared newFeature = feature->clone();
            newFeature->scale = scales.at(i);

            // as we only change the scale value, all the duplicated features
            // should be valid
            assert(newFeature->checkValidity(corePointsRole, errorMessage));

            if (!CheckFeatureUnicity(rawFeatures,
                                     newFeature))  // check that the feature
                                                   // does not exists already!
            {
                CVLog::Warning("[3DMASC] duplicated feature " +
                               newFeature->toString() +
                               ", check your parameter file");
                return false;
            } else {
                // save the feature
                rawFeatures.push_back(newFeature);
            }
        }
    }

    return true;
}

static bool ReadScales(const QString& command,
                       std::vector<double>& scales,
                       int lineNumber) {
    assert(scales.empty());

    QStringList tokens = command.split(';');
    if (tokens.empty()) {
        CVLog::Warning(
                "Malformed file: expecting at least one token after 'scales:' "
                "on line #" +
                QString::number(lineNumber));
        return false;
    }

    for (const QString& token : tokens) {
        if (token.contains(':')) {
            // it's probably a range
            QStringList subTokens = token.trimmed().split(':');
            if (subTokens.size() != 3) {
                CVLog::Warning(QString("Malformed file: expecting 3 tokens for "
                                       "a range of scales (%1)")
                                       .arg(token));
                return false;
            }
            bool ok[3] = {true, true, true};
            double start = subTokens[0].trimmed().toDouble(ok);
            double step = subTokens[1].toDouble(ok + 1);
            double stop = subTokens[2].toDouble(ok + 2);
            if (!ok[0] || !ok[1] || !ok[2]) {
                CVLog::Warning(QString("Malformed file: invalid values in "
                                       "scales range (%1) on line #%2")
                                       .arg(token)
                                       .arg(lineNumber));
                return false;
            }
            if (stop < start || step <= 1.0 - 6) {
                CVLog::Warning(QString("Malformed file: invalid range (%1) on "
                                       "line #%2")
                                       .arg(token)
                                       .arg(lineNumber));
                return false;
            }

            for (double v = start; v <= stop + 1.0e-6; v += step) {
                scales.push_back(v);
            }
        } else {
            bool ok = true;
            double v = token.trimmed().toDouble(&ok);
            if (!ok) {
                CVLog::Warning(QString("Malformed file: invalid scale value "
                                       "(%1) on line #%2")
                                       .arg(token)
                                       .arg(lineNumber));
                return false;
            }
            scales.push_back(v);
        }
    }

    scales.shrink_to_fit();
    return true;
}

static bool ReadCorePoints(const QString& command,
                           const Tools::NamedClouds& clouds,
                           masc::CorePoints& corePoints,
                           int lineNumber) {
    QStringList tokens = command.split('_');
    if (tokens.empty()) {
        CVLog::Warning(
                "Malformed file: expecting tokens after 'core_points:' on line "
                "#" +
                QString::number(lineNumber));
        return false;
    }
    QString pcName = tokens[0].trimmed();
    if (!clouds.contains(pcName)) {
        CVLog::Warning(
                QString("Malformed file: unknown cloud '%1' on line #%2 (make "
                        "sure it is declared before the core points)")
                        .arg(pcName)
                        .arg(lineNumber));
        return false;
    }
    corePoints.origin = clouds[pcName];
    corePoints.role = pcName;

    // should we sub-sample the origin cloud?
    if (tokens.size() > 1) {
        if (tokens[1].toUpper() == "SS") {
            if (tokens.size() < 3) {
                CVLog::Warning(
                        "Malformed file: missing token after 'SS' on line #" +
                        QString::number(lineNumber));
                return false;
            }
            QString options = tokens[2];
            if (options.startsWith('R')) {
                corePoints.selectionMethod = CorePoints::RANDOM;
            } else if (options.startsWith('S')) {
                corePoints.selectionMethod = CorePoints::SPATIAL;
            } else {
                CVLog::Warning(
                        "Malformed file: unknown option after 'SS' on line #" +
                        QString::number(lineNumber));
                return false;
            }

            // read the subsampling parameter (ignore the first character)
            bool ok = false;
            corePoints.selectionParam = options.mid(1).toDouble(&ok);
            if (!ok) {
                CVLog::Warning(
                        "Malformed file: expecting a number after 'SS_X' on "
                        "line #" +
                        QString::number(lineNumber));
                return false;
            }

        }  // end of subsampling options
    }

    return true;
}

static bool ReadCloud(const QString& command,
                      Tools::NamedClouds& clouds,
                      const QDir& defaultDir,
                      int lineNumber,
                      FileIOFilter::LoadParameters& loadParameters) {
    QStringList tokens = command.split('=');
    if (tokens.size() != 2) {
        CVLog::Warning(
                "Malformed file: expecting 2 tokens after 'cloud:' on line #" +
                QString::number(lineNumber));
        return false;
    }

    QString pcName = tokens[0].trimmed();
    QString pcFilename = defaultDir.absoluteFilePath(tokens[1].trimmed());
    // try to open the cloud
    {
        CC_FILE_ERROR error = CC_FERR_NO_ERROR;
        ccHObject* object =
                FileIOFilter::LoadFromFile(pcFilename, loadParameters, error);
        if (error != CC_FERR_NO_ERROR || !object) {
            // error message already issued
            if (object) delete object;
            return false;
        }
        ccHObject::Container cloudsInFile;
        object->filterChildren(cloudsInFile, false, CV_TYPES::POINT_CLOUD,
                               true);
        if (cloudsInFile.empty()) {
            CVLog::Warning("File doesn't contain a single cloud");
            delete object;
            return false;
        } else if (cloudsInFile.size() > 1) {
            CVLog::Warning(
                    "File contains more than one cloud, only the first one "
                    "will be kept");
        }
        ccPointCloud* pc = static_cast<ccPointCloud*>(cloudsInFile.front());
        for (size_t i = 1; i < cloudsInFile.size(); ++i) {
            delete cloudsInFile[i];
        }
        if (pc->getParent()) pc->getParent()->detachChild(pc);
        pc->setName(pcName);  // DGM: warning, may not be acceptable in the GUI
                              // version?
        clouds.insert(pcName, pc);
    }

    return true;
}

bool Tools::LoadFile(
        const QString& filename,
        Tools::NamedClouds* clouds,
        bool cloudsAreProvided,
        std::vector<Feature::Shared>*
                rawFeatures /*=nullptr*/,  // requires 'clouds'
        std::vector<double>* rawScales /*=nullptr*/,
        masc::CorePoints* corePoints /*=nullptr*/,  // requires 'clouds'
        masc::Classifier* classifier /*=nullptr*/,
        TrainParameters* parameters /*=nullptr*/,
        QWidget* parent /*=nullptr*/) {
    QFileInfo fi(filename);
    if (!fi.exists()) {
        CVLog::Warning(QString("Can't find file '%1'").arg(filename));
        return false;
    }

    QFile file(filename);
    if (!file.open(QFile::Text | QFile::ReadOnly)) {
        CVLog::Warning(QString("Can't open file '%1'").arg(filename));
        return false;
    }

    // to use the same 'global shift' for multiple files
    CCVector3d loadCoordinatesShift(0, 0, 0);
    bool loadCoordinatesTransEnabled = false;
    FileIOFilter::LoadParameters loadParameters;
    if (!cloudsAreProvided) {
        loadParameters.alwaysDisplayLoadDialog = true;
        loadParameters.shiftHandlingMode =
                ecvGlobalShiftManager::DIALOG_IF_NECESSARY;
        loadParameters.coordinatesShift = &loadCoordinatesShift;
        loadParameters.coordinatesShiftEnabled = &loadCoordinatesTransEnabled;
        loadParameters.parentWidget = parent;
        FileIOFilter::ResetSesionCounter();
    }

    try {
        assert(!rawFeatures || rawFeatures->empty());
        std::vector<double> scales;

        QTextStream stream(&file);
        bool badFeatures = false;
        for (int lineNumber = 1;; ++lineNumber) {
            if (stream.atEnd()) break;
            QString line = stream.readLine();
            if (line.isEmpty()) {
                continue;
            }

            if (line.startsWith("#")) {
                // comment
                continue;
            }

            // strip out the potential comment at the end of the line as well
            int commentIndex = line.indexOf('#');
            if (commentIndex >= 0) line = line.left(commentIndex);

            QString upperLine = line.toUpper();
            if (upperLine.startsWith("CLASSIFIER:"))  // classifier
            {
                if (!classifier) {
                    // no need to load the classifier
                    continue;
                }
                if (classifier->isValid()) {
                    CVLog::Warning(
                            "Malformed file: can't declare the classifier file "
                            "twice! (line #" +
                            QString::number(lineNumber) + ")");
                    return false;
                }
                QString yamlFilename = line.mid(11).trimmed();
                QString yamlAbsoluteFilename =
                        fi.absoluteDir().absoluteFilePath(yamlFilename);
                if (!classifier->fromFile(yamlAbsoluteFilename, parent)) {
                    CVLog::Warning("Failed to load the classifier file from " +
                                   yamlAbsoluteFilename);
                    return false;
                }
                CVLog::Print("[3DMASC] Classifier data loaded from " +
                             yamlAbsoluteFilename);
            } else if (upperLine.startsWith("CLOUD:"))  // clouds
            {
                if (!clouds || cloudsAreProvided) {
                    // no need to load the clouds in this case
                    continue;
                }
                QString command = line.mid(6);
                if (!ReadCloud(command, *clouds, fi.absoluteDir(), lineNumber,
                               loadParameters)) {
                    return false;
                }
            } else if (upperLine.startsWith("TEST:"))  // test cloud
            {
                if (!clouds || cloudsAreProvided) {
                    // no need to load the clouds in this case
                    continue;
                }
                QString command = line.mid(5);
                if (!ReadCloud("TEST=" + command, *clouds, fi.absoluteDir(),
                               lineNumber,
                               loadParameters))  // add the TEST keyword so that
                                                 // the cloud will be loaded as
                                                 // the TEST cloud
                {
                    return false;
                }
            } else if (upperLine.startsWith("CORE_POINTS:"))  // core points
            {
                if (!corePoints) {
                    // no need to load the core points
                    continue;
                }
                if (corePoints->origin) {
                    CVLog::Warning(
                            "Core points already defined (those declared on "
                            "line #" +
                            QString::number(lineNumber) + " will be ignored)");
                } else {
                    QString command = line.mid(12);
                    if (clouds && !ReadCorePoints(command, *clouds, *corePoints,
                                                  lineNumber)) {
                        return false;
                    }
                }
            } else if (upperLine.startsWith("SCALES:"))  // scales
            {
                if (!scales.empty()) {
                    CVLog::Warning(
                            "Malformed file: scales defined twice (line #" +
                            QString::number(lineNumber) + ")");
                    return false;
                }

                QString command = line.mid(7);
                if (!ReadScales(command, scales, lineNumber)) {
                    return false;
                } else {
                    if (rawScales)
                        for (auto scale : scales) rawScales->push_back(scale);
                }
            } else if (upperLine.startsWith("FEATURE:"))  // feature
            {
                QString command = line.mid(8);

                if (rawFeatures && clouds) {
                    if (!CreateFeaturesFromCommand(
                                command,
                                corePoints ? corePoints->role : QString(),
                                lineNumber, *clouds, *rawFeatures, scales)) {
                        // error message already issued
                        // return false;
                        badFeatures = true;  // we continue as we want to get
                                             // ALL the errors
                    }
                }
            } else if (upperLine.startsWith("PARAM_"))  // parameter
            {
                if (parameters)  // no need to actually read the parameters if
                                 // the caller didn't requested them
                {
                    QStringList tokens = upperLine.split("=");
                    if (tokens.size() != 2) {
                        CVLog::Warning(
                                QString("Line #%1: malformed parameter command "
                                        "(expecting param_XXX=Y)")
                                        .arg(lineNumber));
                        return false;
                    }
                    bool ok = false;
                    if (tokens[0] == "PARAM_MAX_DEPTH") {
                        parameters->rt.maxDepth = tokens[1].toInt(&ok);
                    } else if (tokens[0] == "PARAM_MAX_TREE_COUNT") {
                        parameters->rt.maxTreeCount = tokens[1].toInt(&ok);
                    } else if (tokens[0] == "PARAM_ACTIVE_VAR_COUNT") {
                        parameters->rt.activeVarCount = tokens[1].toInt(&ok);
                    } else if (tokens[0] == "PARAM_MIN_SAMPLE_COUNT") {
                        parameters->rt.minSampleCount = tokens[1].toInt(&ok);
                    } else if (tokens[0] == "PARAM_TEST_DATA_RATIO") {
                        parameters->testDataRatio = tokens[1].toFloat(&ok);
                    } else {
                        CVLog::Warning(
                                QString("Line #%1: unrecognized parameter: ")
                                        .arg(lineNumber) +
                                tokens[0]);
                    }
                    if (!ok) {
                        CVLog::Warning(QString("Line #%1: invalid value for "
                                               "parameter ")
                                               .arg(lineNumber) +
                                       tokens[0]);
                    }
                }
            } else {
                CVLog::Warning(
                        QString("Line #%1: unrecognized token/command: ")
                                .arg(lineNumber) +
                        (line.length() < 10 ? line : line.left(10) + "..."));
                return false;
            }
        }

        if (badFeatures) {
            return false;
        }
    } catch (const std::bad_alloc&) {
        CVLog::Warning("Not enough memory");
        return false;
    }

    if (rawFeatures) rawFeatures->shrink_to_fit();

    return true;
}

bool Tools::LoadClassifier(QString filename,
                           NamedClouds& clouds,
                           Feature::Set& rawFeatures,
                           masc::Classifier& classifier,
                           QWidget* parent /*=nullptr*/) {
    return LoadFile(filename, &clouds, true, &rawFeatures, nullptr, nullptr,
                    &classifier, nullptr, parent);
}

bool Tools::LoadTrainingFile(QString filename,
                             Feature::Set& rawFeatures,
                             std::vector<double>& rawScales,
                             NamedClouds& loadedClouds,
                             TrainParameters& parameters,
                             CorePoints* corePoints /*=nullptr*/,
                             QWidget* parentWidget /*=nullptr*/) {
    bool cloudsWereProvided = !loadedClouds.empty();
    if (LoadFile(filename, &loadedClouds, cloudsWereProvided, &rawFeatures,
                 &rawScales, corePoints, nullptr, &parameters, parentWidget)) {
        return true;
    } else {
        if (!cloudsWereProvided) {
            // delete the already loaded clouds (if any)
            for (NamedClouds::iterator it = loadedClouds.begin();
                 it != loadedClouds.end(); ++it)
                delete it.value();
        }
        return false;
    }
}

cloudViewer::ScalarField* Tools::RetrieveSF(const ccPointCloud* cloud,
                                            const QString& sfName,
                                            bool caseSensitive /*=true*/) {
    if (!cloud) {
        assert(false);
        return nullptr;
    }
    int sfIdx = -1;
    if (caseSensitive) {
        sfIdx = cloud->getScalarFieldIndexByName(qPrintable(sfName));
    } else {
        QString sfNameUpper = sfName.toUpper();
        for (unsigned i = 0; i < cloud->getNumberOfScalarFields(); ++i) {
            if (QString(cloud->getScalarField(i)->getName()).toUpper() ==
                sfNameUpper) {
                sfIdx = static_cast<int>(i);
                break;
            }
        }
    }

    if (sfIdx >= 0) {
        return cloud->getScalarField(sfIdx);
    } else {
        return nullptr;
    }
}

struct FeaturesAndScales {
    std::vector<double> scales;
    size_t featureCount = 0;
    QMap<double, std::vector<PointFeature::Shared>> pointFeaturesPerScale;
    QMap<double, std::vector<NeighborhoodFeature::Shared>>
            neighborhoodFeaturesPerScale;
    QMap<double, std::vector<ContextBasedFeature::Shared>>
            contextBasedFeaturesPerScale;
};

bool Tools::PrepareFeatures(
        const CorePoints& corePoints,
        Feature::Set& features,
        QString& errorStr,
        cloudViewer::GenericProgressCallback* progressCb /*=nullptr*/,
        SFCollector* generatedScalarFields /*=nullptr*/) {
    if (features.empty() || !corePoints.origin) {
        // invalid input parameters
        assert(false);
        return false;
    }

    // gather all the scales that need to be extracted
    QMap<ccPointCloud*, FeaturesAndScales> cloudsWithScaledFeatures;
    // and prepare the features (scalar fields, etc.) at the same time
    for (const Feature::Shared& feature : features) {
        QString errorMessage("invalid pointer");
        assert(!corePoints.role.isEmpty());
        if (!feature ||
            !feature->checkValidity(corePoints.role, errorMessage)) {
            errorStr = "Invalid rule/feature: " + errorMessage;
            return false;
        }

        // prepare the feature
        if (!feature->prepare(corePoints, errorStr, progressCb,
                              generatedScalarFields)) {
            // something failed (error should be up to date)
            return false;
        }

        if (feature->scaled()) {
            try {
                switch (feature->getType()) {
                    // Point features
                    case Feature::Type::PointFeature: {
                        // build the scaled feature list attached to the first
                        // cloud
                        if (feature->cloud1 &&
                            !feature->sf1WasAlreadyExisting)  // nothing to
                                                              // compute if the
                                                              // scalar field
                                                              // was already
                                                              // there
                        {
                            FeaturesAndScales& fas =
                                    cloudsWithScaledFeatures[feature->cloud1];
                            fas.pointFeaturesPerScale[feature->scale].push_back(
                                    qSharedPointerCast<PointFeature>(feature));
                            ++fas.featureCount;
                            if (std::find(fas.scales.begin(), fas.scales.end(),
                                          feature->scale) == fas.scales.end()) {
                                fas.scales.push_back(feature->scale);
                            }
                        }
                        // build the scaled feature list attached to the second
                        // cloud (if any)
                        if (feature->cloud2 &&
                            feature->cloud2 != feature->cloud1 &&
                            feature->op != Feature::NO_OPERATION) {
                            if (!feature->sf1WasAlreadyExisting)  // nothing to
                                                                  // compute if
                                                                  // the scalar
                                                                  // field was
                                                                  // already
                                                                  // there
                            {
                                if (!feature->sf2WasAlreadyExisting) {
                                    FeaturesAndScales& fas =
                                            cloudsWithScaledFeatures
                                                    [feature->cloud2];
                                    ++fas.featureCount;
                                    fas.pointFeaturesPerScale[feature->scale]
                                            .push_back(qSharedPointerCast<
                                                       PointFeature>(feature));
                                    if (std::find(fas.scales.begin(),
                                                  fas.scales.end(),
                                                  feature->scale) ==
                                        fas.scales.end()) {
                                        fas.scales.push_back(feature->scale);
                                    }
                                }
                            }
                        }
                    } break;

                    // Neighborhood features
                    case Feature::Type::NeighborhoodFeature: {
                        // build the scaled feature list attached to the first
                        // cloud
                        if (feature->cloud1 &&
                            !feature->sf1WasAlreadyExisting)  // nothing to
                                                              // compute if the
                                                              // scalar field
                                                              // was already
                                                              // there
                        {
                            FeaturesAndScales& fas =
                                    cloudsWithScaledFeatures[feature->cloud1];
                            fas.neighborhoodFeaturesPerScale[feature->scale]
                                    .push_back(qSharedPointerCast<
                                               NeighborhoodFeature>(feature));
                            ++fas.featureCount;
                            if (std::find(fas.scales.begin(), fas.scales.end(),
                                          feature->scale) == fas.scales.end()) {
                                fas.scales.push_back(feature->scale);
                            }
                        }

                        // build the scaled feature list attached to the second
                        // cloud (if any)
                        if (feature->cloud2 &&
                            feature->cloud2 != feature->cloud1 &&
                            feature->op != Feature::NO_OPERATION) {
                            if (!feature->sf1WasAlreadyExisting)  // nothing to
                                                                  // compute if
                                                                  // the scalar
                                                                  // field was
                                                                  // already
                                                                  // there
                            {
                                if (!feature->sf2WasAlreadyExisting) {
                                    FeaturesAndScales& fas =
                                            cloudsWithScaledFeatures
                                                    [feature->cloud2];
                                    fas
                                            .neighborhoodFeaturesPerScale
                                                    [feature->scale]
                                            .push_back(qSharedPointerCast<
                                                       NeighborhoodFeature>(
                                                    feature));
                                    ++fas.featureCount;
                                    if (std::find(fas.scales.begin(),
                                                  fas.scales.end(),
                                                  feature->scale) ==
                                        fas.scales.end()) {
                                        fas.scales.push_back(feature->scale);
                                    }
                                }
                            }
                        }
                    } break;

                    // Context-based features
                    case Feature::Type::ContextBasedFeature: {
                        // build the scaled feature list attached to the context
                        // cloud
                        if (feature->cloud1 &&
                            !feature->sf1WasAlreadyExisting)  // nothing to
                                                              // compute if the
                                                              // scalar field
                                                              // was already
                                                              // there
                        {
                            FeaturesAndScales& fas =
                                    cloudsWithScaledFeatures[feature->cloud1];
                            fas.contextBasedFeaturesPerScale[feature->scale]
                                    .push_back(qSharedPointerCast<
                                               ContextBasedFeature>(feature));
                            ++fas.featureCount;
                            if (std::find(fas.scales.begin(), fas.scales.end(),
                                          feature->scale) == fas.scales.end()) {
                                fas.scales.push_back(feature->scale);
                            }
                        }
                    } break;

                    default:
                        assert(false);
                        break;
                }
            } catch (const std::bad_alloc&) {
                errorStr = "Not enough memory";
                return false;
            }
        }
    }

    bool success = true;

    // if we have scaled features
    if (!cloudsWithScaledFeatures.empty()) {
        // for each cloud
        for (QMap<ccPointCloud*, FeaturesAndScales>::iterator it =
                     cloudsWithScaledFeatures.begin();
             success && it != cloudsWithScaledFeatures.end(); ++it) {
            FeaturesAndScales& fas = it.value();
            ccPointCloud* sourceCloud = it.key();

            // sort the scales
            std::sort(fas.scales.begin(), fas.scales.end());

            // get the octree
            ccOctree::Shared octree = sourceCloud->getOctree();
            if (!octree) {
                CVLog::Print(QString("Computing octree of cloud %1 (%2 points)")
                                     .arg(sourceCloud->getName())
                                     .arg(sourceCloud->size()));
                if (progressCb) progressCb->start();
                QCoreApplication::processEvents();
                octree = sourceCloud->computeOctree(progressCb);
                if (!octree) {
                    errorStr =
                            "[Tools::PrepareFeatures] Failed to compute octree "
                            "(not enough memory?)";
                    return false;
                }
            }

            // now extract the neighborhoods from the biggest to the smallest
            // scale
            double largetScale = fas.scales.back();
            PointCoordinateType largestRadius =
                    static_cast<PointCoordinateType>(
                            largetScale / 2);  // scale is the diameter!
            unsigned char octreeLevel =
                    octree->findBestLevelForAGivenNeighbourhoodSizeExtraction(
                            largestRadius);

            unsigned pointCount = corePoints.size();
            QString logMessage = QString("Computing %1 features on cloud %2 at "
                                         "%3 core points")
                                         .arg(fas.featureCount)
                                         .arg(sourceCloud->getName())
                                         .arg(pointCount);
            if (progressCb) {
                progressCb->setMethodTitle("Compute features");
                progressCb->setInfo(qPrintable(logMessage));
            }
            CVLog::Print(logMessage);
            cloudViewer::NormalizedProgress nProgress(progressCb, pointCount);

            bool cancelled = false;

#ifndef _DEBUG
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::max(1, omp_get_max_threads() - 2))
#endif
#endif
            for (int i = 0; i < static_cast<int>(pointCount); ++i) {
                if (!cancelled) {
                    QString localErrorStr;
                    bool localSuccess = true;

                    // spherical neighborhood extraction structure
                    cloudViewer::DgmOctree::
                            NearestNeighboursSphericalSearchStruct nNSS;
                    {
                        nNSS.level = octreeLevel;
                        nNSS.queryPoint = *corePoints.cloud->getPoint(i);
                        octree->getTheCellPosWhichIncludesThePoint(
                                &nNSS.queryPoint, nNSS.cellPos, nNSS.level);
                        octree->computeCellCenter(nNSS.cellPos, nNSS.level,
                                                  nNSS.cellCenter);
                    }

                    // we extract the point's neighbors
                    unsigned kNN =
                            octree->findNeighborsInASphereStartingFromCell(
                                    nNSS, largestRadius, true);
                    if (kNN != 0) {
                        nNSS.pointsInNeighbourhood.resize(kNN);

                        // for each scale (from the largest to the smallest)
                        for (size_t scaleIndex = 0;
                             scaleIndex < fas.scales.size(); ++scaleIndex) {
                            double currentScale =
                                    fas.scales[fas.scales.size() - 1 -
                                               scaleIndex];  // from the biggest
                                                             // to the smallest!

                            if (scaleIndex != 0) {
                                double radius = currentScale /
                                                2;  // scale is the diameter!
                                double sqRadius = radius * radius;
                                // remove the farthest points
                                for (; kNN > 0; --kNN) {
                                    if (nNSS.pointsInNeighbourhood[kNN - 1]
                                                .squareDistd <= sqRadius) {
                                        break;
                                    }
                                }

                                if (kNN == 0) {
                                    // no need to go further
                                    break;
                                }
                                nNSS.pointsInNeighbourhood.resize(kNN);
                            }

                            // Point features
                            for (PointFeature::Shared& feature :
                                 fas.pointFeaturesPerScale[currentScale]) {
                                if (feature->cloud1 == sourceCloud &&
                                    feature->statSF1 && feature->field1 &&
                                    localSuccess) {
                                    double outputValue = 0;
                                    if (!feature->computeStat(
                                                nNSS.pointsInNeighbourhood,
                                                feature->field1, outputValue)) {
                                        // an error occurred
                                        localErrorStr =
                                                "An error occurred during the "
                                                "computation of feature " +
                                                feature->toString() +
                                                " on cloud " +
                                                feature->cloud1->getName();
                                        localSuccess = false;
                                        break;
                                    }

                                    ScalarType v1 = static_cast<ScalarType>(
                                            outputValue);
                                    feature->statSF1->setValue(i, v1);
                                }

                                if (feature->cloud2 == sourceCloud &&
                                    feature->statSF2 && feature->field2 &&
                                    localSuccess) {
                                    assert(feature->op !=
                                           Feature::NO_OPERATION);
                                    double outputValue = 0;
                                    if (!feature->computeStat(
                                                nNSS.pointsInNeighbourhood,
                                                feature->field2, outputValue)) {
                                        // an error occurred
                                        localErrorStr =
                                                "An error occurred during the "
                                                "computation of feature " +
                                                feature->toString() +
                                                " on cloud " +
                                                feature->cloud2->getName();
                                        localSuccess = false;
                                        break;
                                    }

                                    ScalarType v2 = static_cast<ScalarType>(
                                            outputValue);
                                    feature->statSF2->setValue(i, v2);
                                }
                            }

                            // Neighborhood features
                            for (NeighborhoodFeature::Shared& feature :
                                 fas.neighborhoodFeaturesPerScale
                                         [currentScale]) {
                                if (feature->cloud1 == sourceCloud &&
                                    feature->sf1 && localSuccess) {
                                    double outputValue = 0;
                                    if (!feature->computeValue(
                                                nNSS.pointsInNeighbourhood,
                                                nNSS.queryPoint, outputValue)) {
                                        // an error occurred
                                        localErrorStr =
                                                "An error occurred during the "
                                                "computation of feature " +
                                                feature->toString() +
                                                " on cloud " +
                                                feature->cloud1->getName();
                                        localSuccess = false;
                                        break;
                                    }

                                    ScalarType v1 = static_cast<ScalarType>(
                                            outputValue);
                                    feature->sf1->setValue(i, v1);
                                }

                                if (feature->cloud2 == sourceCloud &&
                                    feature->sf2 && localSuccess) {
                                    assert(feature->op !=
                                           Feature::NO_OPERATION);
                                    double outputValue = 0;
                                    if (!feature->computeValue(
                                                nNSS.pointsInNeighbourhood,
                                                nNSS.queryPoint, outputValue)) {
                                        // an error occurred
                                        localErrorStr =
                                                "An error occurred during the "
                                                "computation of feature " +
                                                feature->toString() +
                                                " on cloud " +
                                                feature->cloud2->getName();
                                        localSuccess = false;
                                        break;
                                    }

                                    ScalarType v2 = static_cast<ScalarType>(
                                            outputValue);
                                    feature->sf2->setValue(i, v2);
                                }
                            }

                            // Context-based features
                            for (ContextBasedFeature::Shared& feature :
                                 fas.contextBasedFeaturesPerScale
                                         [currentScale]) {
                                if (feature->cloud1 == sourceCloud &&
                                    feature->sf && localSuccess) {
                                    ScalarType outputValue = 0;
                                    if (!feature->computeValue(
                                                nNSS.pointsInNeighbourhood,
                                                nNSS.queryPoint, outputValue)) {
                                        // an error occurred
                                        localErrorStr =
                                                "An error occurred during the "
                                                "computation of feature " +
                                                feature->toString() +
                                                " on cloud " +
                                                feature->cloud1->getName();
                                        localSuccess = false;
                                        break;
                                    }

                                    feature->sf->setValue(i, outputValue);
                                }
                            }

                            if (!localSuccess) {
                                localErrorStr = localErrorStr + " at scale  " +
                                                QString::number(currentScale) +
                                                " on point " +
                                                QString::number(i);
                                break;
                            }

                        }  // for each scale
                    }

                    if (!localSuccess) {
                        cancelled = true;
                        success = false;
#if defined(_OPENMP)
                        errorStr = "Feature computation failed for point " +
                                   QString::number(i) + " (using OpenMP with " +
                                   QString::number(omp_get_num_threads()) +
                                   " threads)";
#else
                        errorStr = "Feature computation failed for point " +
                                   QString::number(i);
#endif
                        CVLog::Error(localErrorStr);
                    }

                    if (progressCb) {
                        if (!cancelled) {
                            cancelled = !nProgress.oneStep();
                            if (cancelled) {
                                // process cancelled by the user
#if defined(_OPENMP)
                                errorStr =
                                        "Process cancelled at point " +
                                        QString::number(i) +
                                        " (using OpenMP with " +
                                        QString::number(omp_get_num_threads()) +
                                        " threads)";
#else
                                errorStr = "Process cancelled at point " +
                                           QString::number(i);
#endif
                                CVLog::Warning(errorStr);
                                success = false;
                            }
                        }
                    }
                }
            }  // for each point

        }  // for each cloud
    }

    for (const Feature::Shared& feature : features) {
        // we have to 'finish' the process for scaled features
        if (feature->scaled() && !feature->finish(corePoints, errorStr)) {
            return false;
        }
    }

    return success;
}

bool Tools::RandomSubset(ccPointCloud* cloud,
                         float ratio,
                         cloudViewer::ReferenceCloud* inRatioSubset,
                         cloudViewer::ReferenceCloud* outRatioSubset) {
    if (!cloud) {
        CVLog::Warning("Invalid input cloud");
        return false;
    }
    if (!inRatioSubset || !outRatioSubset) {
        CVLog::Warning("Invalid input refence clouds");
        return false;
    }
    if (inRatioSubset->getAssociatedCloud() != cloud ||
        outRatioSubset->getAssociatedCloud() != cloud) {
        CVLog::Warning(
                "Invalid input reference clouds (associated cloud is wrong)");
        return false;
    }
    if (ratio < 0.0f || ratio > 1.0f) {
        CVLog::Warning(QString("Invalid parameter (ratio: %1)").arg(ratio));
        return false;
    }

    unsigned inSampleCount =
            static_cast<unsigned>(floor(cloud->size() * ratio));
    assert(inSampleCount <= cloud->size());
    unsigned outSampleCount = cloud->size() - inSampleCount;

    // we draw the smallest population (faster)
    unsigned targetCount = inSampleCount;
    bool defaultState = false;
    if (outSampleCount < inSampleCount) {
        targetCount = outSampleCount;
        defaultState = true;
    }

    // reserve memory
    std::vector<bool> pointInsideRatio;
    try {
        pointInsideRatio.resize(cloud->size(), defaultState);
    } catch (const std::bad_alloc&) {
        CVLog::Warning("Not enough memory");
        return false;
    }

    if (!inRatioSubset->reserve(inSampleCount) ||
        !outRatioSubset->reserve(outSampleCount)) {
        CVLog::Warning("Not enough memory");
        inRatioSubset->clear();
        outRatioSubset->clear();
        return false;
    }

    // randomly choose the 'in' or 'out' indexes
    int randIndex = 0;
    unsigned randomCount = 0;
    while (randomCount < targetCount) {
        randIndex = ((randIndex + std::rand()) % cloud->size());
        if (pointInsideRatio[randIndex] == defaultState) {
            pointInsideRatio[randIndex] = !defaultState;
            ++randomCount;
        }
    }

    // now dispatch the points
    {
        for (unsigned i = 0; i < cloud->size(); ++i) {
            if (pointInsideRatio[i])
                inRatioSubset->addPointIndex(i);
            else
                outRatioSubset->addPointIndex(i);
        }
        assert(inRatioSubset->size() == inSampleCount);
        assert(outRatioSubset->size() == outSampleCount);
    }

    return true;
}

cloudViewer::ScalarField* Tools::GetClassificationSF(
        const ccPointCloud* cloud) {
    if (!cloud) {
        // invalid input cloud
        assert(false);
        return nullptr;
    }
    // look for the classification field
    int classifSFIdx = cloud->getScalarFieldIndexByName(
            LAS_FIELD_NAMES
                    [LAS_CLASSIFICATION]);  // LAS_FIELD_NAMES[LAS_CLASSIFICATION]
                                            // = "Classification"
    if (classifSFIdx < 0) {
        return nullptr;
    }
    return cloud->getScalarField(classifSFIdx);
}
