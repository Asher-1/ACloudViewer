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
//#          COPYRIGHT: CLOUDVIEWER  project                               #
//#                                                                        #
//##########################################################################

#include "ecvLibAlgorithms.h"

// cloudViewer
#include <ScalarFieldTools.h>

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvOctree.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

// Local
#include "ecvCommon.h"
#include "ecvConsole.h"
#include "ecvProgressDialog.h"
#include "ecvRegistrationTools.h"
#include "ecvUtils.h"

// Qt
#include <QApplication>
#include <QElapsedTimer>
#include <QInputDialog>
#include <QMessageBox>

// This is included only for temporarily removing an object from the tree.
//	TODO figure out a cleaner way to do this without having to include all
// of MainWindow.h
#include "MainWindow.h"

namespace ccLibAlgorithms {
static QString GetDensitySFName(
        cloudViewer::GeometricalAnalysisTools::Density densityType,
        bool approx,
        double densityKernelSize = 0.0) {
    QString sfName;

    // update the name with the density type
    switch (densityType) {
        case cloudViewer::GeometricalAnalysisTools::DENSITY_KNN:
            sfName = CC_LOCAL_KNN_DENSITY_FIELD_NAME;
            break;
        case cloudViewer::GeometricalAnalysisTools::DENSITY_2D:
            sfName = CC_LOCAL_SURF_DENSITY_FIELD_NAME;
            break;
        case cloudViewer::GeometricalAnalysisTools::DENSITY_3D:
            sfName = CC_LOCAL_VOL_DENSITY_FIELD_NAME;
            break;
        default:
            assert(false);
            break;
    }

    sfName += QString(" (r=%2)").arg(densityKernelSize);

    if (approx) sfName += " [approx]";

    return sfName;
}

PointCoordinateType GetDefaultCloudKernelSize(ccGenericPointCloud* cloud,
                                              unsigned knn /*=12*/) {
    assert(cloud);
    if (cloud && cloud->size() != 0) {
        // we get 1% of the cloud bounding box
        // and we divide by the number of points / 10e6 (so that the kernel for
        // a 20 M. points cloud is half the one of a 10 M. cloud)
        ccBBox box = cloud->getOwnBB();

        // old way
        // PointCoordinateType radius = box.getDiagNorm() *
        // static_cast<PointCoordinateType>(0.01/std::max(1.0,1.0e-7*static_cast<double>(cloud->size())));

        // new way
        CCVector3 d = box.getDiagVec();
        PointCoordinateType volume = d[0] * d[1] * d[2];
        PointCoordinateType surface =
                pow(volume, static_cast<PointCoordinateType>(2.0 / 3.0));
        PointCoordinateType surfacePerPoint = surface / cloud->size();
        return sqrt(surfacePerPoint * knn);
    }

    return -PC_ONE;
}

PointCoordinateType GetDefaultCloudKernelSize(
        const ccHObject::Container& entities, unsigned knn /*=12*/) {
    PointCoordinateType sigma = -PC_ONE;

    size_t selNum = entities.size();
    // computation of a first sigma guess
    for (size_t i = 0; i < selNum; ++i) {
        ccPointCloud* pc = ccHObjectCaster::ToPointCloud(entities[i]);
        PointCoordinateType sigmaCloud = GetDefaultCloudKernelSize(pc);

        // we keep the smallest value
        if (sigma < 0 || sigmaCloud < sigma) sigma = sigmaCloud;
    }

    return sigma;
}

bool ComputeGeomCharacteristics(const GeomCharacteristicSet& characteristics,
                                PointCoordinateType radius,
                                ccHObject::Container& entities,
                                const CCVector3* roughnessUpDir /*=nullptr*/,
                                QWidget* parent /*=nullptr*/) {
    // no feature case
    if (characteristics.empty()) {
        // nothing to do
        assert(false);
        return true;
    }

    // single features case
    if (characteristics.size() == 1) {
        return ComputeGeomCharacteristic(characteristics.front().charac,
                                         characteristics.front().subOption,
                                         radius, entities, roughnessUpDir,
                                         parent);
    }

    // multiple features case
    QScopedPointer<ecvProgressDialog> pDlg;
    if (parent) {
        pDlg.reset(new ecvProgressDialog(true, parent));
        pDlg->setAutoClose(false);
    }

    for (const GeomCharacteristic& g : characteristics) {
        if (!ComputeGeomCharacteristic(g.charac, g.subOption, radius, entities,
                                       roughnessUpDir, parent, pDlg.data())) {
            return false;
        }
    }

    return true;
}

bool ComputeGeomCharacteristic(
        cloudViewer::GeometricalAnalysisTools::GeomCharacteristic c,
        int subOption,
        PointCoordinateType radius,
        ccHObject::Container& entities,
        const CCVector3* roughnessUpDir /*=nullptr*/,
        QWidget* parent /*= nullptr*/,
        ecvProgressDialog* progressDialog /*=nullptr*/) {
    size_t selNum = entities.size();
    if (selNum < 1) return false;

    // generate the right SF name
    QString sfName;

    switch (c) {
        case cloudViewer::GeometricalAnalysisTools::Feature: {
            switch (subOption) {
                case cloudViewer::Neighbourhood::EigenValuesSum:
                    sfName = "Eigenvalues sum";
                    break;
                case cloudViewer::Neighbourhood::Omnivariance:
                    sfName = "Omnivariance";
                    break;
                case cloudViewer::Neighbourhood::EigenEntropy:
                    sfName = "Eigenentropy";
                    break;
                case cloudViewer::Neighbourhood::Anisotropy:
                    sfName = "Anisotropy";
                    break;
                case cloudViewer::Neighbourhood::Planarity:
                    sfName = "Planarity";
                    break;
                case cloudViewer::Neighbourhood::Linearity:
                    sfName = "Linearity";
                    break;
                case cloudViewer::Neighbourhood::PCA1:
                    sfName = "PCA1";
                    break;
                case cloudViewer::Neighbourhood::PCA2:
                    sfName = "PCA2";
                    break;
                case cloudViewer::Neighbourhood::SurfaceVariation:
                    sfName = "Surface variation";
                    break;
                case cloudViewer::Neighbourhood::Sphericity:
                    sfName = "Sphericity";
                    break;
                case cloudViewer::Neighbourhood::Verticality:
                    sfName = "Verticality";
                    break;
                case cloudViewer::Neighbourhood::EigenValue1:
                    sfName = "1st eigenvalue";
                    break;
                case cloudViewer::Neighbourhood::EigenValue2:
                    sfName = "2nd eigenvalue";
                    break;
                case cloudViewer::Neighbourhood::EigenValue3:
                    sfName = "3rd eigenvalue";
                    break;
                default:
                    assert(false);
                    CVLog::Error(
                            "Internal error: invalid sub option for Feature "
                            "computation");
                    return false;
            }

            sfName += QString(" (%1)").arg(radius);
        } break;

        case cloudViewer::GeometricalAnalysisTools::Curvature: {
            switch (subOption) {
                case cloudViewer::Neighbourhood::GAUSSIAN_CURV:
                    sfName = CC_CURVATURE_GAUSSIAN_FIELD_NAME;
                    break;
                case cloudViewer::Neighbourhood::MEAN_CURV:
                    sfName = CC_CURVATURE_MEAN_FIELD_NAME;
                    break;
                case cloudViewer::Neighbourhood::NORMAL_CHANGE_RATE:
                    sfName = CC_CURVATURE_NORM_CHANGE_RATE_FIELD_NAME;
                    break;
                default:
                    assert(false);
                    CVLog::Error(
                            "Internal error: invalid sub option for Curvature "
                            "computation");
                    return false;
            }
            sfName += QString(" (%1)").arg(radius);
        } break;

        case cloudViewer::GeometricalAnalysisTools::LocalDensity:
            sfName = GetDensitySFName(
                    static_cast<cloudViewer::GeometricalAnalysisTools::Density>(
                            subOption),
                    false, radius);
            break;

        case cloudViewer::GeometricalAnalysisTools::ApproxLocalDensity:
            sfName = GetDensitySFName(
                    static_cast<cloudViewer::GeometricalAnalysisTools::Density>(
                            subOption),
                    true);
            break;

        case cloudViewer::GeometricalAnalysisTools::Roughness:
            sfName = CC_ROUGHNESS_FIELD_NAME + QString(" (%1)").arg(radius);
            break;

        case cloudViewer::GeometricalAnalysisTools::MomentOrder1:
            sfName = CC_MOMENT_ORDER1_FIELD_NAME + QString(" (%1)").arg(radius);
            break;

        default:
            assert(false);
            return false;
    }

    ecvProgressDialog* pDlg = progressDialog;
    if (!pDlg && parent) {
        pDlg = new ecvProgressDialog(true, parent);
        pDlg->setAutoClose(false);
    }

    for (size_t i = 0; i < selNum; ++i) {
        // is the ith selected data is eligible for processing?
        if (entities[i]->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ccGenericPointCloud* cloud =
                    ccHObjectCaster::ToGenericPointCloud(entities[i]);

            ccPointCloud* pc = 0;
            int sfIdx = -1;
            if (cloud->isA(CV_TYPES::POINT_CLOUD)) {
                pc = static_cast<ccPointCloud*>(cloud);

                sfIdx = pc->getScalarFieldIndexByName(qPrintable(sfName));
                if (sfIdx < 0) sfIdx = pc->addScalarField(qPrintable(sfName));
                if (sfIdx >= 0)
                    pc->setCurrentScalarField(sfIdx);
                else {
                    ecvConsole::Error(
                            QString("Failed to create scalar field on cloud "
                                    "'%1' (not enough memory?)")
                                    .arg(pc->getName()));
                    continue;
                }
            }

            ccOctree::Shared octree = cloud->getOctree();
            if (!octree) {
                if (pDlg) {
                    pDlg->show();
                }
                octree = cloud->computeOctree(pDlg);
                if (!octree) {
                    ecvConsole::Error(
                            QString("Couldn't compute octree for cloud '%1'!")
                                    .arg(cloud->getName()));
                    break;
                }
            }

            cloudViewer::GeometricalAnalysisTools::ErrorCode result =
                    cloudViewer::GeometricalAnalysisTools::
                            ComputeCharactersitic(c, subOption, cloud, radius,
                                                  roughnessUpDir, pDlg,
                                                  octree.data());

            if (result == cloudViewer::GeometricalAnalysisTools::NoError) {
                if (pc && sfIdx >= 0) {
                    pc->setCurrentDisplayedScalarField(sfIdx);
                    pc->showSF(sfIdx >= 0);
                    pc->getCurrentInScalarField()->computeMinAndMax();
                    if (c == cloudViewer::GeometricalAnalysisTools::Roughness &&
                        roughnessUpDir != nullptr) {
                        // signed roughness should be displayed with a
                        // symmetrical color scale
                        ccScalarField* sf = dynamic_cast<ccScalarField*>(
                                pc->getCurrentInScalarField());
                        if (sf) {
                            sf->setSymmetricalScale(true);
                        } else {
                            assert(false);
                        }
                    }
                }
            } else {
                QString errorMessage;
                switch (result) {
                    case cloudViewer::GeometricalAnalysisTools::InvalidInput:
                        errorMessage = "Internal error (invalid input)";
                        break;
                    case cloudViewer::GeometricalAnalysisTools::NotEnoughPoints:
                        errorMessage = "Not enough points";
                        break;
                    case cloudViewer::GeometricalAnalysisTools::
                            OctreeComputationFailed:
                        errorMessage =
                                "Failed to compute octree (not enough memory?)";
                        break;
                    case cloudViewer::GeometricalAnalysisTools::ProcessFailed:
                        errorMessage = "Process failed";
                        break;
                    case cloudViewer::GeometricalAnalysisTools::
                            UnhandledCharacteristic:
                        errorMessage =
                                "Internal error (unhandled characteristic)";
                        break;
                    case cloudViewer::GeometricalAnalysisTools::NotEnoughMemory:
                        errorMessage = "Not enough memory";
                        break;
                    case cloudViewer::GeometricalAnalysisTools::
                            ProcessCancelledByUser:
                        errorMessage = "Process cancelled by user";
                        break;
                    default:
                        assert(false);
                        errorMessage = "Unknown error";
                        break;
                }

                ecvConsole::Warning(
                        QString("Failed to apply processing to cloud '%1'")
                                .arg(cloud->getName()));
                ecvConsole::Warning(errorMessage);

                if (pc && sfIdx >= 0) {
                    pc->deleteScalarField(sfIdx);
                    sfIdx = -1;
                }

                if (pDlg != progressDialog) {
                    delete pDlg;
                    pDlg = nullptr;
                }

                return false;
            }
        }
    }

    if (pDlg != progressDialog) {
        delete pDlg;
        pDlg = nullptr;
    }

    return true;
}

bool ApplyCCLibAlgorithm(CC_LIB_ALGORITHM algo,
                         ccHObject::Container& entities,
                         QWidget* parent /*=0*/,
                         void** additionalParameters /*=0*/) {
    size_t selNum = entities.size();
    if (selNum < 1) return false;

    // generic parameters
    QString sfName;

    // computeScalarFieldGradient parameters
    bool euclidean = false;

    switch (algo) {
        case CCLIB_ALGO_SF_GRADIENT: {
            sfName = CC_GRADIENT_NORMS_FIELD_NAME;
            // parameters already provided?
            if (additionalParameters) {
                euclidean = *static_cast<bool*>(additionalParameters[0]);
            } else  // ask the user!
            {
                euclidean = (QMessageBox::question(
                                     parent, "Gradient",
                                     "Is the scalar field composed of "
                                     "(euclidean) distances?",
                                     QMessageBox::Yes | QMessageBox::No,
                                     QMessageBox::No) == QMessageBox::Yes);
            }
        } break;

        default:
            assert(false);
            return false;
    }

    for (size_t i = 0; i < selNum; ++i) {
        // is the ith selected data is eligible for processing?
        ccGenericPointCloud* cloud = nullptr;
        switch (algo) {
            case CCLIB_ALGO_SF_GRADIENT:
                // for scalar field gradient, we can apply it directly on meshes
                bool lockedVertices;
                cloud = ccHObjectCaster::ToGenericPointCloud(entities[i],
                                                             &lockedVertices);
                if (lockedVertices) {
                    ecvUtils::DisplayLockedVerticesWarning(
                            entities[i]->getName(), selNum == 1);
                    cloud = nullptr;
                }
                if (cloud) {
                    // but we need an already displayed SF!
                    if (cloud->isA(CV_TYPES::POINT_CLOUD)) {
                        ccPointCloud* pc = static_cast<ccPointCloud*>(cloud);
                        int outSfIdx =
                                pc->getCurrentDisplayedScalarFieldIndex();
                        if (outSfIdx < 0) {
                            cloud = nullptr;
                        } else {
                            // we set as 'output' SF the currently displayed
                            // scalar field
                            pc->setCurrentOutScalarField(outSfIdx);
                            sfName = QString("%1(%2)").arg(
                                    CC_GRADIENT_NORMS_FIELD_NAME,
                                    pc->getScalarFieldName(outSfIdx));
                        }
                    } else  // if (!cloud->hasDisplayedScalarField()) //TODO:
                            // displayed but not necessarily set as OUTPUT!
                    {
                        cloud = nullptr;
                    }
                }
                break;

            // by default, we apply processings on clouds only
            default:
                if (entities[i]->isKindOf(CV_TYPES::POINT_CLOUD))
                    cloud = ccHObjectCaster::ToGenericPointCloud(entities[i]);
                break;
        }

        if (cloud) {
            ccPointCloud* pc = nullptr;
            int sfIdx = -1;
            if (cloud->isA(CV_TYPES::POINT_CLOUD)) {
                pc = static_cast<ccPointCloud*>(cloud);

                sfIdx = pc->getScalarFieldIndexByName(qPrintable(sfName));
                if (sfIdx < 0) sfIdx = pc->addScalarField(qPrintable(sfName));
                if (sfIdx >= 0)
                    pc->setCurrentInScalarField(sfIdx);
                else {
                    ecvConsole::Error(
                            QString("Failed to create scalar field on cloud "
                                    "'%1' (not enough memory?)")
                                    .arg(pc->getName()));
                    continue;
                }
            }

            QScopedPointer<ecvProgressDialog> pDlg;
            if (parent) {
                pDlg.reset(new ecvProgressDialog(true, parent));
            }

            ccOctree::Shared octree = cloud->getOctree();
            if (!octree) {
                if (pDlg) {
                    pDlg->show();
                }
                octree = cloud->computeOctree(pDlg.data());
                if (!octree) {
                    ecvConsole::Error(
                            QString("Couldn't compute octree for cloud '%1'!")
                                    .arg(cloud->getName()));
                    break;
                }
            }

            int result = 0;
            QElapsedTimer eTimer;
            eTimer.start();
            switch (algo) {
                case CCLIB_ALGO_SF_GRADIENT:
                    result = cloudViewer::ScalarFieldTools::
                            computeScalarFieldGradient(
                                    cloud,
                                    0,  // auto --> FIXME: should be properly
                                        // set by the user!
                                    euclidean, false, pDlg.data(),
                                    octree.data());
                    break;

                default:
                    // missed something?
                    assert(false);
            }
            qint64 elapsedTime_ms = eTimer.elapsed();

            if (result == 0) {
                if (pc && sfIdx >= 0) {
                    pc->setCurrentDisplayedScalarField(sfIdx);
                    pc->showSF(sfIdx >= 0);
                    pc->getCurrentInScalarField()->computeMinAndMax();
                }
                // cloud->prepareDisplayForRefresh();
                ecvConsole::Print("[Algortihm] Timing: %3.2f s.",
                                  static_cast<double>(elapsedTime_ms) / 1000.0);
            } else {
                ecvConsole::Warning(
                        QString("Failed to apply processing to cloud '%1'")
                                .arg(cloud->getName()));
                if (pc && sfIdx >= 0) {
                    pc->deleteScalarField(sfIdx);
                    sfIdx = -1;
                }
            }
        }
    }

    return true;
}

bool ApplyScaleMatchingAlgorithm(ScaleMatchingAlgorithm algo,
                                 ccHObject::Container& entities,
                                 double icpRmsDiff,
                                 int icpFinalOverlap,
                                 unsigned refEntityIndex /*=0*/,
                                 QWidget* parent /*=0*/) {
    if (entities.size() < 2 || refEntityIndex >= entities.size()) {
        CVLog::Error(
                "[ApplyScaleMatchingAlgorithm] Invalid input parameter(s)");
        return false;
    }

    std::vector<double> scales;
    try {
        scales.resize(entities.size(), -1.0);
    } catch (const std::bad_alloc&) {
        CVLog::Error("Not enough memory!");
        return false;
    }

    // check the reference entity
    ccHObject* refEntity = entities[refEntityIndex];
    if (!refEntity->isKindOf(CV_TYPES::POINT_CLOUD) &&
        !refEntity->isKindOf(CV_TYPES::MESH)) {
        CVLog::Warning(
                "[Scale Matching] The reference entity must be a cloud or a "
                "mesh!");
        return false;
    }

    unsigned count = static_cast<unsigned>(entities.size());

    // now compute the scales
    ecvProgressDialog pDlg(true, parent);
    pDlg.setMethodTitle(QObject::tr("Computing entities scales"));
    pDlg.setInfo(QObject::tr("Entities: %1").arg(count));
    cloudViewer::NormalizedProgress nProgress(&pDlg, 2 * count - 1);
    pDlg.start();
    QApplication::processEvents();

    for (unsigned i = 0; i < count; ++i) {
        ccHObject* ent = entities[i];
        // try to get the underlying cloud (or the vertices set for a mesh)
        bool lockedVertices;
        ccGenericPointCloud* cloud =
                ccHObjectCaster::ToGenericPointCloud(ent, &lockedVertices);
        if (cloud && !lockedVertices) {
            switch (algo) {
                case BB_MAX_DIM:
                case BB_VOLUME: {
                    ccBBox box = ent->getOwnBB();
                    if (box.isValid())
                        scales[i] = algo == BB_MAX_DIM ? box.getMaxBoxDim()
                                                       : box.computeVolume();
                    else
                        CVLog::Warning(QString("[Scale Matching] Entity '%1' "
                                               "has an invalid bounding-box!")
                                               .arg(ent->getName()));
                } break;

                case PCA_MAX_DIM: {
                    cloudViewer::Neighbourhood Yk(cloud);
                    if (!Yk.getLSPlane()) {
                        CVLog::Warning(QString("[Scale Matching] Failed to "
                                               "perform PCA on entity '%1'!")
                                               .arg(ent->getName()));
                        break;
                    }
                    // deduce the scale
                    {
                        const CCVector3* X = Yk.getLSPlaneX();
                        const CCVector3* O = Yk.getGravityCenter();
                        double minX = 0;
                        double maxX = 0;
                        for (unsigned j = 0; j < cloud->size(); ++j) {
                            double x = (*cloud->getPoint(j) - *O).dot(*X);
                            if (j != 0) {
                                minX = std::min(x, minX);
                                maxX = std::max(x, maxX);
                            } else {
                                minX = maxX = x;
                            }
                        }
                        scales[i] = maxX - minX;
                    }
                } break;

                case ICP_SCALE: {
                    ccGLMatrix transMat;
                    double finalError = 0.0;
                    double finalScale = 1.0;
                    unsigned finalPointCount = 0;

                    cloudViewer::ICPRegistrationTools::Parameters parameters;
                    {
                        parameters.convType = cloudViewer::
                                ICPRegistrationTools::MAX_ERROR_CONVERGENCE;
                        parameters.minRMSDecrease = icpRmsDiff;
                        parameters.nbMaxIterations = 0;
                        parameters.adjustScale = true;
                        parameters.filterOutFarthestPoints = false;
                        parameters.samplingLimit = 50000;
                        parameters.finalOverlapRatio = icpFinalOverlap / 100.0;
                        parameters.transformationFilters =
                                0;  // cloudViewer::RegistrationTools::SKIP_ROTATION
                        parameters.maxThreadCount = 0;
                        parameters.useC2MSignedDistances = false;
                        parameters.normalsMatching =
                                cloudViewer::ICPRegistrationTools::NO_NORMAL;
                    }

                    if (ccRegistrationTools::ICP(ent, refEntity, transMat,
                                                 finalScale, finalError,
                                                 finalPointCount, parameters,
                                                 false, false, parent)) {
                        scales[i] = finalScale;
                    } else {
                        CVLog::Warning(QString("[Scale Matching] Failed to "
                                               "register entity '%1'!")
                                               .arg(ent->getName()));
                    }

                } break;

                default:
                    assert(false);
                    break;
            }
        } else if (cloud && lockedVertices) {
            // locked entities
            ecvUtils::DisplayLockedVerticesWarning(ent->getName(), false);
        } else {
            // we need a cloud or a mesh
            CVLog::Warning(QString("[Scale Matching] Entity '%1' can't be "
                                   "rescaled this way!")
                                   .arg(ent->getName()));
        }

        // if the reference entity is invalid!
        if (scales[i] <= 0 && i == refEntityIndex) {
            CVLog::Error(
                    "Reference entity has an invalid scale! Can't proceed.");
            return false;
        }

        if (!nProgress.oneStep()) {
            // process cancelled by user
            return false;
        }
    }

    CVLog::Print(QString("[Scale Matching] Reference entity scale: %1")
                         .arg(scales[refEntityIndex]));

    // now we can rescale
    pDlg.setMethodTitle(QObject::tr("Rescaling entities"));
    {
        ccHObject::Container toBeRescaleEntities;
        for (unsigned i = 0; i < count; ++i) {
            if (i == refEntityIndex) continue;
            if (scales[i] < 0) continue;

            CVLog::Print(QString("[Scale Matching] Entity '%1' scale: %2")
                                 .arg(entities[i]->getName())
                                 .arg(scales[i]));
            if (scales[i] <= ZERO_TOLERANCE_D) {
                CVLog::Warning("[Scale Matching] Entity scale is too small!");
                continue;
            }

            ccHObject* ent = entities[i];

            bool lockedVertices;
            ccGenericPointCloud* cloud =
                    ccHObjectCaster::ToGenericPointCloud(ent, &lockedVertices);
            if (!cloud || lockedVertices) continue;

            double scaled = 1.0;
            if (algo == ICP_SCALE)
                scaled = scales[i];
            else
                scaled = scales[refEntityIndex] / scales[i];

            PointCoordinateType scale_pc =
                    static_cast<PointCoordinateType>(scaled);

            // we temporarily detach entity, as it may undergo
            //"severe" modifications (octree deletion, etc.) --> see
            // ccPointCloud::scale
            MainWindow* instance = dynamic_cast<MainWindow*>(parent);
            MainWindow::ccHObjectContext objContext;
            if (instance) {
                objContext = instance->removeObjectTemporarilyFromDBTree(cloud);
            }

            CCVector3 C = cloud->getOwnBB().getCenter();

            cloud->scale(scale_pc, scale_pc, scale_pc, C);

            if (instance) instance->putObjectBackIntoDBTree(cloud, objContext);
            // cloud->prepareDisplayForRefresh_recursive();

            // don't forget the 'global shift'!
            const CCVector3d& shift = cloud->getGlobalShift();
            cloud->setGlobalShift(shift * scaled);
            toBeRescaleEntities.push_back(cloud);
            // DGM: nope! Not the global scale!
        }

        // only refresh rescaled
        ecvDisplayTools::SetRedrawRecursive(false);
        for (unsigned i = 0; i < toBeRescaleEntities.size(); ++i) {
            ccHObject* obj = toBeRescaleEntities[i];
            if (obj) {
                ecvDisplayTools::RemoveBB(obj->getViewId());
                obj->setRedraw(true);
            }
        }
        ecvDisplayTools::RedrawDisplay();

        if (!nProgress.oneStep()) {
            // process cancelled by user
            return false;
        }
    }

    return true;
}
}  // namespace ccLibAlgorithms
