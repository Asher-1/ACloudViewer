// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "G3PointAction.h"

// ECV_PLUGINS
#include <ecvMainAppInterface.h>
#include <ecvQtHelpers.h>

// CV_CORE_LIB
#include <CVGeom.h>
#include <DgmOctree.h>
#include <DgmOctreeReferenceCloud.h>
#include <Neighbourhood.h>

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvHObject.h>
#include <ecvNormalVectors.h>
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>

// Qt
#include <QApplication>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QMainWindow>
#include <QMutex>
#include <QOpenGLShaderProgram>
#include <QPushButton>
#include <QThreadPool>
#include <QtConcurrent>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

// Eigen
#include <G3PointDialog.h>
#include <G3PointDisclaimer.h>
#include <WolmanCustomPlot.h>
#include <math.h>

#include <Eigen/Geometry>
#include <set>

namespace G3Point {
std::shared_ptr<G3PointAction> G3PointAction::s_g3PointAction;

QPointer<G3PointPlots> G3PointAction::s_g3PointPlots;

G3PointAction::G3PointAction(ccPointCloud* cloud, ecvMainAppInterface* app)
    : m_app(app), m_dlg(nullptr), m_grainsAsEllipsoids(nullptr) {
    assert(cloud);
    setCloud(cloud);
}

G3PointAction::~G3PointAction() {
    if (s_g3PointAction) {
        if (m_dlg) {
            delete m_dlg;
        }
    }
}

void G3PointAction::GetG3PointAction(ccPointCloud* cloud,
                                     ecvMainAppInterface* app) {
    // disclaimer accepted?
    if (!G3PointDisclaimer::show(app)) return;

    s_g3PointPlots = new G3PointPlots(cloud->getName());

    if (!s_g3PointAction)  // create the singleton if needed
    {
        s_g3PointAction.reset(new G3PointAction(cloud, app));
    } else {
        int sfIdx = cloud->getScalarFieldIndexByName("g3point_label");
        if (sfIdx == -1) {
            CVLog::Warning(
                    "[G3PointAction::GetG3PointAction] no g3point_label scalar "
                    "field, reset action");
            s_g3PointAction.reset(new G3PointAction(cloud, app));
        } else {
            CVLog::Print(
                    "[G3PointAction::GetG3PointAction] use existing "
                    "g3point_label scalar field");
            s_g3PointAction->setCloud(cloud);
        }
    }
    s_g3PointAction->showDlg();
}

RGBAColorsTableType getRandomColors(size_t randomColorsNumber) {
    Q_ASSERT(randomColorsNumber > 1);

    RGBAColorsTableType randomColors;
    if (!randomColors.reserveSafe(static_cast<unsigned>(randomColorsNumber))) {
        CVLog::Error(QObject::tr("Not enough memory!"));
    }

    // generate random colors
    int max = 255;
    std::mt19937 gen(42);  // to seed mersenne twister.
    std::uniform_int_distribution<uint16_t> dist(
            0, max);  // 1-byte types are not allowed

    for (int i = 0; i < randomColorsNumber; ++i) {
        ecvColor::Rgb col;
        col.r = static_cast<unsigned char>(dist(gen));
        col.g = static_cast<unsigned char>(dist(gen));
        //		col.b = static_cast<unsigned char>(dist(gen));
        col.b = max - static_cast<ColorCompType>(
                              (static_cast<double>(col.r) +
                               static_cast<double>(col.g)) /
                              2);  // cast to double to avoid overflow (whatever
                                   // the type of ColorCompType!!!)
        randomColors.addElement(ecvColor::Rgba(col, max));
    }

    return randomColors;
}

bool G3PointAction::sfConvertToRandomRGB(
        const ccHObject::Container& selectedEntities,
        QWidget* parent)  // take as is from ccEntityAction.cpp
{
    int s_randomColorsNumber = 256;

    Q_ASSERT(s_randomColorsNumber > 1);

    RGBAColorsTableType* randomColors = new RGBAColorsTableType;
    if (!randomColors->reserveSafe(
                static_cast<unsigned>(s_randomColorsNumber))) {
        CVLog::Error(QObject::tr("Not enough memory!"));
        return false;
    }

    // generate random colors
    int max = 255;

    std::mt19937 gen(42);  // to seed mersenne twister.
    std::uniform_int_distribution<uint16_t> dist(
            0, max);  // 1-byte types are not allowed

    for (int i = 0; i < s_randomColorsNumber; ++i) {
        ecvColor::Rgb col;
        col.r = static_cast<unsigned char>(dist(gen));
        col.g = static_cast<unsigned char>(dist(gen));
        col.b = static_cast<unsigned char>(dist(gen));
        //		col.b = max -
        // static_cast<ColorCompType>((static_cast<double>(col.r) +
        // static_cast<double>(col.g)) / 2); //cast to double to avoid overflow
        //(whatever the type of ColorCompType!!!)
        randomColors->addElement(ecvColor::Rgba(col, max));
    }

    // apply random colors
    for (ccHObject* ent : selectedEntities) {
        ccGenericPointCloud* cloud = nullptr;

        bool lockedVertices = false;
        cloud = ccHObjectCaster::ToPointCloud(ent, &lockedVertices);
        if (lockedVertices) {
            CVLog::Warning(
                    "[G3Point::sfConvertToRandomRGB] "
                    "DisplayLockedVerticesWarning");
            continue;
        }
        if (cloud != nullptr)  // TODO
        {
            ccPointCloud* pc = static_cast<ccPointCloud*>(cloud);
            ccScalarField* sf = pc->getCurrentDisplayedScalarField();
            // if there is no displayed SF --> nothing to do!
            if (sf && sf->currentSize() >= pc->size()) {
                if (!pc->resizeTheRGBTable(false)) {
                    CVLog::Error(QObject::tr("Not enough memory!"));
                    break;
                } else {
                    ScalarType minSF = sf->getMin();
                    ScalarType maxSF = sf->getMax();

                    ScalarType step =
                            (maxSF - minSF) / (s_randomColorsNumber - 1);
                    if (step == 0) step = static_cast<ScalarType>(1.0);

                    for (unsigned i = 0; i < pc->size(); ++i) {
                        ScalarType val = sf->getValue(i);
                        unsigned colIndex =
                                static_cast<unsigned>((val - minSF) / step);
                        if (colIndex == s_randomColorsNumber) --colIndex;

                        pc->setPointColor(i, randomColors->getValue(colIndex));
                    }

                    pc->showColors(true);
                    pc->showSF(false);  // just in case
                }
            }

            m_cloud->redrawDisplay();
        }
    }

    return true;
}

void G3PointAction::addToStack(int index,
                               const Eigen::ArrayXi& n_donors,
                               const Eigen::ArrayXXi& donors,
                               std::vector<int>& stack) {
    stack.push_back(index);

    for (int k = 0; k < n_donors(index); k++) {
        addToStack(donors(index, k), n_donors, donors, stack);
    }
}

int G3PointAction::segmentLabels(bool useParallelStrategy) {
    CVLog::Print("[segment_labels]");
    // for each point, find in the neighborhood the point with the minimum slope
    // (the receiver)
    Eigen::ArrayXd min_slopes(m_neighborsSlopes.rowwise().minCoeff());
    Eigen::ArrayXi index_of_min_slope = Eigen::ArrayXi::Zero(m_cloud->size());
    Eigen::ArrayXi receivers(m_cloud->size());

    for (unsigned index = 0; index < m_cloud->size(); index++) {
        double min_slope = min_slopes(index);
        int count = 0;
        for (int k = 0; k < m_kNN; k++) {
            if (m_neighborsSlopes(index, k) == min_slope) {
                index_of_min_slope(index) = k;
                count++;
                if (count > 1)
                    std::cout << "[segment_labels] slope already seen, index "
                              << index << ", k " << k << std::endl;
            }
        }
        receivers(index) = m_neighborsIndexes(index, index_of_min_slope(index));
    }

    // if the minimum slope is positive, the receiver is a local maximum
    int nb_maxima = (min_slopes > 0).count();
    Eigen::ArrayXi localMaximumIndexes = Eigen::ArrayXi::Zero(nb_maxima);
    int l = 0;
    for (unsigned int k = 0; k < m_cloud->size(); k++) {
        if (min_slopes(k) > 0) {
            localMaximumIndexes(l) = k;
            receivers(k) = k;
            l++;
        }
    }

    // identify the donors for each receiver
    CVLog::Print("[segment_labels] identify the donors for each receiver");
    Eigen::ArrayXi nDonors = Eigen::ArrayXi::Zero(m_cloud->size());
    Eigen::ArrayXXi donors = Eigen::ArrayXXi::Zero(m_cloud->size(), m_kNN);

    CVLog::Print("[segment_labels] loop");
    for (unsigned int k = 0; k < m_cloud->size(); k++) {
        int receiver = receivers(k);
        if (receiver != k)  // this receiver is not a local maximum
        {
            if (nDonors(receiver) < m_kNN - 1) {
                nDonors(receiver) = nDonors(receiver) + 1;
                donors(receiver, nDonors(receiver) - 1) = k;
            } else {
                CVLog::Warning(QString("maximum number of donors reached, k "
                                       "%1, receiver(k) %2, nDonors %3")
                                       .arg(k)
                                       .arg(receiver)
                                       .arg(nDonors(receiver)));
            }
        }
    }

    // build the stacks
    CVLog::Print("[segment_labels] build the stacks");
    Eigen::ArrayXi labels = Eigen::ArrayXi::Zero(m_cloud->size());
    Eigen::ArrayXi labelsnpoint = Eigen::ArrayXi::Zero(m_cloud->size());
    std::vector<std::vector<int>> stacks;

    int sfIdx = m_cloud->getScalarFieldIndexByName("g3point_label");
    if (sfIdx == -1) {
        sfIdx = m_cloud->addScalarField("g3point_label");
        if (sfIdx == -1) {
            CVLog::Error(
                    "[G3Point::segment_labels] impossible to create scalar "
                    "field g3point_label");
        }
    }

    cloudViewer::ScalarField* g3point_label = m_cloud->getScalarField(sfIdx);
    RGBAColorsTableType randomColors =
            getRandomColors(localMaximumIndexes.size());

    if (!m_cloud->resizeTheRGBTable(false)) {
        CVLog::Error(QObject::tr("Not enough memory!"));
        return -1;
    }

    for (int k = 0; k < localMaximumIndexes.size(); k++) {
        int localMaximumIndex = localMaximumIndexes(k);
        std::vector<int> stack;
        addToStack(localMaximumIndex, nDonors, donors, stack);
        // labels
        for (auto i : stack) {
            labels(i) = k;
            labelsnpoint(i) = static_cast<int>(stack.size());
            if (g3point_label) {
                g3point_label->setValue(i, k);
                m_cloud->setPointColor(i, randomColors.getValue(k));
            }
        }
        stacks.push_back(stack);
    }

    if (g3point_label) {
        g3point_label->computeMinAndMax();
    }

    m_cloud->setCurrentDisplayedScalarField(sfIdx);
    m_cloud->showColors(true);
    m_cloud->showSF(false);

    if (m_app) {
        m_app->refreshAll();
        m_app->updateUI();
    }

    int nLabels = localMaximumIndexes.size();

    return nLabels;
}

double G3PointAction::angleRot2VecMat(const Eigen::Vector3d& a,
                                      const Eigen::Vector3d& b) {
    double angle;
    Eigen::Vector3d c;
    double d;

    c = a.cross(b);

    d = a.dot(b);

    angle = atan2(c.norm(), d) * 180 / M_PI;

    return angle;
}

Eigen::ArrayXXd G3PointAction::computeMeanAngleBetweenNormalsAtBorders() {
    // Find the indexborder nodes (no donor and many other labels in the
    // neighbourhood)
    Eigen::ArrayXXi duplicated_labels(m_cloud->size(), m_kNN);
    for (int n = 0; n < m_kNN; n++) {
        duplicated_labels(Eigen::all, n) = m_labels;
    }
    Eigen::ArrayXXi labels_of_neighbors(m_cloud->size(), m_kNN);
    for (int index = 0; index < static_cast<int>(m_cloud->size()); index++) {
        for (int n = 0; n < m_kNN; n++) {
            labels_of_neighbors(index, n) =
                    m_labels(m_neighborsIndexes(index, n));
        }
    }

    Eigen::ArrayXi temp = m_kNN - (labels_of_neighbors == duplicated_labels)
                                          .cast<int>()
                                          .rowwise()
                                          .sum();
    auto condition = ((temp >= m_kNN / 4) && (m_ndon == 0));
    Eigen::ArrayXi indborder(condition.count());

    int l = 0;
    for (int c = 0; c < condition.size(); c++) {
        if (condition(c)) {
            indborder(l) = c;
            l++;
        }
    }

    // Compute the angle of the normal vector between the neighbours of each
    // grain / label
    size_t nlabels = m_stacks.size();
    Eigen::ArrayXXd A = Eigen::ArrayXXd::Zero(nlabels, nlabels);
    Eigen::ArrayXXd Aangle(nlabels, nlabels);
    Aangle.fill(NAN);
    Eigen::ArrayXXi N = Eigen::ArrayXXi::Zero(nlabels, nlabels);

    for (auto i : indborder) {
        auto neighbors = m_neighborsIndexes(
                i, Eigen::all);  // indexes of the neighbors of i
        Eigen::Vector3d N1(m_normals(i, Eigen::all));  // normal at i
        for (auto j : neighbors) {
            // Take the normals vector for i and j
            Eigen::Vector3d N2(m_normals(j, Eigen::all));  // normal at j
            double angle = angleRot2VecMat(N1, N2);
            if ((m_labels(i) != -1) &&
                (m_labels(j) != -1))  // points which belong to the discarded
                                      // grains have the -1 label
            {
                if ((m_labels(i) > A.rows()) || (m_labels(j) > A.rows())) {
                    std::cout << "ERROR " << m_labels(i) << " " << m_labels(j)
                              << "(nlabels " << nlabels << ")" << std::endl;
                } else {
                    A(m_labels(i), m_labels(j)) =
                            A(m_labels(i), m_labels(j)) + angle;
                    N(m_labels(i), m_labels(j)) =
                            N(m_labels(i), m_labels(j)) + 1;
                }
            }
        }
    }

    /// compute the means
    for (int r = 0; r < nlabels; r++) {
        for (int c = 0; c < nlabels; c++) {
            if (N(r, c) != 0) {
                Aangle(r, c) = A(r, c) / N(r, c);
            }
        }
    }

    return Aangle;
}

bool G3PointAction::checkStacks(const std::vector<std::vector<int>>& stacks,
                                int count) {
    std::set<int> indexes;
    bool ret = true;
    int errorCount = 0;

    // the stacks shall contain each point, only one time
    for (auto& stack : stacks) {
        for (int index : stack) {
            if (indexes.count(index)) {
                CVLog::Warning(
                        "[G3PointAction::check_stacks] index already in the "
                        "set " +
                        QString::number(index));
                ret = false;
                errorCount++;
            }
            indexes.insert(index);
        }
    }

    if (errorCount) {
        CVLog::Warning("[G3PointAction::check_stacks] number of duplicates " +
                       QString::number(errorCount));
        ret = false;
    }

    // the number of points in the stacks shall be equal to count
    if (indexes.size() != count) {
        CVLog::Warning("[G3PointAction::check_stacks] point in stacks " +
                       QString::number(indexes.size()) + ", expected " +
                       QString::number(m_cloud->size()) +
                       " (may be due to duplicates in the point cloud)");
        ret = false;
    }

    return ret;
}

bool G3PointAction::updateLocalMaximumIndexes() {
    size_t nlabels = m_stacks.size();

    Eigen::ArrayXi localMaximumIndexes = Eigen::ArrayXi::Ones(nlabels) * (-1);

    for (int k = 0; k < nlabels; k++) {
        auto& stack = m_stacks[k];
        size_t nPoints = stack.size();
        Eigen::ArrayXd elevations(nPoints);
        for (int index = 0; index < nPoints; index++) {
            const CCVector3* point = m_cloud->getPoint(stack[index]);
            elevations(index) = point->z;
        }
        Eigen::Index maxIndex;
        elevations.maxCoeff(&maxIndex);
        localMaximumIndexes(k) = stack[maxIndex];
    }

    if ((localMaximumIndexes == -1).any()) {
        CVLog::Error(
                "[G3PointAction::updateLocalMaximumIndexes] CANCEL value error "
                "in the indexes of the local maximima");
        return false;
    }

    m_localMaximumIndexes = localMaximumIndexes;

    return true;
}

bool G3PointAction::updateLabelsAndColors() {
    CVLog::Print(QString("[G3PointAction::update_labels_and_colors]"));

    // g3point_label scalar field
    int sfIdx = m_cloud->getScalarFieldIndexByName("g3point_label");
    if (sfIdx == -1) {
        sfIdx = m_cloud->addScalarField("g3point_label");
        if (sfIdx == -1) {
            CVLog::Error(
                    "[G3PointAction::update_labels_and_colors] impossible to "
                    "create scalar field g3point_label");
            return false;
        }
    }
    cloudViewer::ScalarField* g3point_label = m_cloud->getScalarField(sfIdx);

    RGBAColorsTableType randomColors = getRandomColors(m_stacks.size());

    if (!m_cloud->resizeTheRGBTable(false)) {
        CVLog::Error(
                QObject::tr("[G3PointAction::update_labels_and_colors] Not "
                            "enough memory!"));
        return false;
    }

    for (unsigned index = 0; index < m_cloud->size();
         index++)  // points which are not in the stacks will have the label -1
    {
        g3point_label->setValue(index, -1);
    }

    m_labels = -1;

    for (int k = 0; k < m_stacks.size(); k++) {
        const std::vector<int>& stack = m_stacks[k];
        // labels
        for (auto i : stack) {
            m_labels(i) = k;
            m_labelsnpoint(i) = static_cast<int>(stack.size());
            if (g3point_label) {
                g3point_label->setValue(i, k);
                m_cloud->setPointColor(i, randomColors.getValue(k));
            }
        }
    }

    if (g3point_label) {
        g3point_label->computeMinAndMax();
    }

    m_cloud->setCurrentDisplayedScalarField(sfIdx);
    m_cloud->showColors(true);
    m_cloud->showSF(false);

    m_cloud->redrawDisplay();
    // m_cloud->prepareDisplayForRefresh();

    if (m_app) {
        m_app->refreshAll();
        m_app->updateUI();
    }

    return true;
}

bool G3PointAction::exportLocalMaximaAsCloud(
        const Eigen::ArrayXi& localMaximumIndexes) {
    // create cloud
    // QString cloudName = m_cloud->getName() + "_g3point";
    QString cloudName = "g3point_summits";
    ccPointCloud* cloud = new ccPointCloud(cloudName);

    RGBAColorsTableType randomColors =
            getRandomColors(localMaximumIndexes.size());

    for (auto index : localMaximumIndexes) {
        cloud->addPoint(*m_cloud->getPoint(index));
    }

    // allocate colors if necessary
    if (cloud->resizeTheRGBTable()) {
        for (int index = 0; index < static_cast<int>(cloud->size()); index++) {
            cloud->setPointColor(index, randomColors.getValue(index));
        }
    }

    cloud->showColors(true);

    int nbChildren = m_cloud->getChildrenNumber();
    std::vector<ccHObject*> toDelete;
    for (int k = 0; k < nbChildren; k++) {
        auto child = m_cloud->getChild(k);

        if (child->getName() == cloudName) {
            toDelete.push_back(child);
        }
    }
    for (auto& child : toDelete) {
        m_app->removeFromDB(child, true);
    }

    // parent->addChild(cloud, ccHObject::DP_PARENT_OF_OTHER, 0);
    // m_app->addToDB(cloud);
    m_cloud->addChild(cloud);
    m_app->addToDB(cloud);

    cloud->setEnabled(false);

    return true;
}

bool G3PointAction::processNewStacks(std::vector<std::vector<int>>& newStacks,
                                     int pointCount) {
    if (!checkStacks(newStacks, pointCount)) {
        CVLog::Error(
                "[G3PointAction::processNewStacks] newStacks is not valid");
        return false;
    }

    // new stacks are valid, set the class attribute
    m_stacks = newStacks;

    updateLocalMaximumIndexes();

    updateLabelsAndColors();

    exportLocalMaximaAsCloud(m_localMaximumIndexes);

    return true;
}

bool G3PointAction::buildStacksFromG3PointLabelSF(
        cloudViewer::ScalarField* g3PointLabel) {
    m_stacks.clear();

    // get all the different labels
    std::set<float> labels;
    for (int idx = 0; idx < g3PointLabel->size(); idx++) {
        labels.insert(g3PointLabel->getValue(idx));
    }

    // rebuild the stacks
    for (auto label : labels) {
        std::vector<int> stack;
        for (int idx = 0; idx < static_cast<int>(m_cloud->size()); idx++) {
            if (g3PointLabel->getValue(idx) == label) {
                stack.push_back(idx);
            }
        }
        m_stacks.push_back(stack);
    }

    if (processNewStacks(m_stacks, m_cloud->size())) {
        return true;
    } else {
        return false;
    }
}

bool G3PointAction::merge(XXb& condition) {
    std::vector<std::vector<int>> newStacks;
    Eigen::ArrayXi newLabels = Eigen::ArrayXi::Ones(m_labels.size()) * (-1);
    int countNewLabels = 0;
    size_t nlabels = m_stacks.size();

    if (condition.rows() != m_stacks.size())  // check that condition is valid
    {
        CVLog::Error("[G3PointAction::merge] the shape of the condition (" +
                     QString::number(condition.rows()) + ", " +
                     QString::number(condition.cols()) +
                     ") is not coherent with the stacks size " +
                     QString::number(m_stacks.size()));
        return false;
    }

    for (int label = 0; label < nlabels; label++) {
        if (newLabels(label) == -1)  // the label has not already been merged
        {
            newLabels(label) = countNewLabels;
            newStacks.push_back(
                    m_stacks[label]);  // initialize a newStack with the stack
                                       // of the current label
            countNewLabels++;
        }

        for (int otherLabel = 0; otherLabel < nlabels; otherLabel++) {
            if (otherLabel == label) {
                continue;  // do not try to merge a label with itself
            }

            // shall we merge otherLabel with label?
            if (!condition(label, otherLabel)) {
                std::vector<int>& labelStack = newStacks[newLabels(label)];

                if (newLabels(otherLabel) !=
                    -1)  // the other label has already been merged
                {
                    std::vector<int>& otherLabelStack =
                            newStacks[newLabels(otherLabel)];
                    if (newLabels(label) >
                        newLabels(otherLabel))  // merge label in otherLabel
                    {
                        // add the label stack to the otherLabel stack
                        otherLabelStack.insert(
                                otherLabelStack.end(), labelStack.begin(),
                                labelStack.end());  // add the stack to the
                                                    // label stack
                        // empty the label stack
                        labelStack.clear();
                        // update the label
                        newLabels(label) = newLabels(otherLabel);
                    }
                    if (newLabels(label) <
                        newLabels(otherLabel))  // merge otherLabel in label
                    {
                        // add the otherLabel stack to the label stack
                        labelStack.insert(
                                labelStack.end(), otherLabelStack.begin(),
                                otherLabelStack.end());  // add the stack to the
                                                         // label stack
                        // empty the otherLabel stack
                        otherLabelStack.clear();
                        // update the otherLabel
                        newLabels(otherLabel) = newLabels(label);
                    }
                } else  // merge otherLabel and label
                {
                    std::vector<int>& otherLabelStack = m_stacks[otherLabel];
                    // add the otherLabel stack to the label stack
                    labelStack.insert(
                            labelStack.end(), otherLabelStack.begin(),
                            otherLabelStack.end());  // add the stack to the
                                                     // label stack
                    // update the otherLabel
                    newLabels(otherLabel) = newLabels(label);
                }
            }
        }
    }

    // remove empty stacks
    std::vector<std::vector<int>> newStacksWithoutEmpty;
    for (auto& stack : newStacks) {
        if (!stack.empty()) {
            newStacksWithoutEmpty.push_back(stack);
        }
    }

    newStacks = newStacksWithoutEmpty;

    CVLog::Print(
            "[G3PointAction::merge] keep " + QString::number(newStacks.size()) +
            "/" + QString::number(m_stacks.size()) + " labels (" +
            QString::number(m_stacks.size() - newStacks.size()) + " removed)");

    if (!processNewStacks(newStacks, m_cloud->size())) {
        CVLog::Error("[G3PointAction::merge] processing newStacks failed");
        return false;
    }

    return true;
}

bool G3PointAction::keep(Xb& condition) {
    std::vector<std::vector<int>> newStacks;
    size_t pointCount = 0;

    for (int index = 0; index < condition.size(); index++) {
        std::vector<int>& stack = m_stacks[index];
        if (condition(index))  // if the condition is met, we keep the stack
        {
            newStacks.push_back(stack);
            pointCount = pointCount + stack.size();
        }
    }

    CVLog::Print(
            "[G3PointAction::keep] keep " + QString::number(newStacks.size()) +
            "/" + QString::number(m_stacks.size()) + " gains (" +
            QString::number(m_stacks.size() - newStacks.size()) + " removed)");

    if (!processNewStacks(newStacks, static_cast<int>(pointCount))) {
        CVLog::Error("[G3PointAction::keep] processing newStacks failed");
    }

    return true;
}

template <typename T>
bool G3PointAction::EigenArrayToFile(QString name, T array) {
    const Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                    Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file(name.toLatin1());
    file << array.format(CSVFormat);
    return true;
}

bool G3PointAction::cluster() {
    CVLog::Print("[cluster labels]");
    size_t nlabels = m_stacks.size();

    m_maxAngle1 = m_dlg->getMaxAngle1();
    m_radiusFactor = m_dlg->getRadiusFactor();

    // Compute the distances between the sinks associated to each label
    Eigen::ArrayXXd D1(nlabels, nlabels);
    if (m_localMaximumIndexes.size() != nlabels) {
        CVLog::Error("[G3PointAction::cluster] m_localMaximumIndexes size (" +
                     QString::number(m_localMaximumIndexes.size()) +
                     ") different from nlabels " + QString::number(nlabels));
        return false;
    }
    for (int i = 0; i < nlabels; i++) {
        for (int j = 0; j < nlabels; j++) {
            D1(i, j) = (*m_cloud->getPoint(m_localMaximumIndexes(i)) -
                        *m_cloud->getPoint(m_localMaximumIndexes(j)))
                               .norm();
        }
    }

    // Estimate the distances between labels using the areas
    int k = 0;
    Eigen::ArrayXXd D2 = Eigen::ArrayXXd::Zero(nlabels, nlabels);
    Eigen::ArrayXd radius = Eigen::ArrayXd::Zero(nlabels);
    for (auto& stack : m_stacks)  // Radius of each label (assuming the surface
                                  // corresponds to a disk)
    {
        radius(k) = sqrt(m_area(stack).sum() / M_PI);
        k++;
    }
    for (int i = 0; i < nlabels;
         i++)  // Compute inter-distances by summing radius
    {
        for (int j = 0; j < nlabels; j++) {
            D2(i, j) = radius(i) + radius(j);
        }
    }

    // If the radius of the sink is above the distance to the other sink (by a
    // factor of rad_factor), set Dist to 1
    Eigen::ArrayXXi Dist = Eigen::ArrayXXi::Zero(nlabels, nlabels);
    Dist = (m_radiusFactor * D2 > D1).select(1, Dist);
    for (int i = 0; i < nlabels; i++)  // set the values of the diagonal to 0
    {
        Dist(i, i) = 0;
    }

    // If labels are neighbours, set Nneigh to 1
    Eigen::ArrayXXi Nneigh = Eigen::ArrayXXi::Zero(nlabels, nlabels);
    k = 0;
    for (auto& stack : m_stacks) {
        Eigen::ArrayXXi labels(stack.size(), m_kNN);
        for (int index = 0; index < stack.size(); index++) {
            for (int n = 0; n < m_kNN; n++) {
                labels(index, n) =
                        m_labels(m_neighborsIndexes(stack[index], n));
            }
        }
        auto reshaped = labels.reshaped();
        std::set<int> unique_elements(reshaped.begin(), reshaped.end());
        for (auto unique : unique_elements) {
            Nneigh(k, unique) = 1;
        }
        k++;
    }

    Eigen::ArrayXXd A = computeMeanAngleBetweenNormalsAtBorders();

    // merge labels if sinks are
    // => close to each other (Dist == 1)
    // => neighbours (Nneigh == 1)
    // => normals are similar (A < m_maxAngle1)
    // => A is not NaN

    if (!checkStacks(m_stacks, m_cloud->size())) {
        CVLog::Error("m_stacks is not valid");
        return false;
    }

    // create the condition matrix and force the symmetry of the matrix
    XXb condition = (Dist < 1) || (Nneigh < 1) || (A > m_maxAngle1) || (A != A);
    XXb symmetrical_condition =
            (condition == condition.transpose()).select(condition, true);
    symmetrical_condition.count();
    condition = symmetrical_condition;

    std::vector<std::vector<int>> newStacks;
    Eigen::ArrayXi newLabels = Eigen::ArrayXi::Ones(m_labels.size()) * (-1);
    int countNewLabels = 0;

    for (int label = 0; label < nlabels; label++) {
        if (newLabels(label) == -1)  // the label has not already been merged
        {
            newLabels(label) = countNewLabels;
            newStacks.push_back(
                    m_stacks[label]);  // initialize a newStack with the stack
                                       // of the current label
            countNewLabels++;
        }

        for (int otherLabel = 0; otherLabel < nlabels; otherLabel++) {
            if (otherLabel == label) {
                continue;  // do not try to merge a label with itself
            }

            // shall we merge otherLabel with label?
            if (!condition(label, otherLabel)) {
                std::vector<int>& labelStack = newStacks[newLabels(label)];

                if (newLabels(otherLabel) !=
                    -1)  // the other label has already been merged
                {
                    std::vector<int>& otherLabelStack =
                            newStacks[newLabels(otherLabel)];
                    if (newLabels(label) >
                        newLabels(otherLabel))  // merge label in otherLabel
                    {
                        // add the label stack to the otherLabel stack
                        otherLabelStack.insert(
                                otherLabelStack.end(), labelStack.begin(),
                                labelStack.end());  // add the stack to the
                                                    // label stack
                        // empty the label stack
                        labelStack.clear();
                        // update the label
                        newLabels(label) = newLabels(otherLabel);
                    }
                    if (newLabels(label) <
                        newLabels(otherLabel))  // merge otherLabel in label
                    {
                        // add the otherLabel stack to the label stack
                        labelStack.insert(
                                labelStack.end(), otherLabelStack.begin(),
                                otherLabelStack.end());  // add the stack to the
                                                         // label stack
                        // empty the otherLabel stack
                        otherLabelStack.clear();
                        // update the otherLabel
                        newLabels(otherLabel) = newLabels(label);
                    }
                } else  // merge otherLabel and label
                {
                    std::vector<int>& otherLabelStack = m_stacks[otherLabel];
                    // add the otherLabel stack to the label stack
                    labelStack.insert(
                            labelStack.end(), otherLabelStack.begin(),
                            otherLabelStack.end());  // add the stack to the
                                                     // label stack
                    // update the otherLabel
                    newLabels(otherLabel) = newLabels(label);
                }
            }
        }
    }

    // remove empty stacks
    std::vector<std::vector<int>> newStacksWithoutEmpty;
    for (auto& stack : newStacks) {
        if (!stack.empty()) {
            newStacksWithoutEmpty.push_back(stack);
        }
    }

    newStacks = newStacksWithoutEmpty;

    CVLog::Print(
            "[G3PointAction::merge] keep " + QString::number(newStacks.size()) +
            "/" + QString::number(m_stacks.size()) + " labels (" +
            QString::number(m_stacks.size() - newStacks.size()) + " removed)");

    if (!processNewStacks(newStacks, m_cloud->size())) {
        CVLog::Error("[G3PointAction::cluster] new stacks are not valid");
        return false;
    }

    return true;
}

void G3PointAction::fit() {
    if (m_stacks.empty()) {
        CVLog::Warning(
                "[G3PointAction::fit] stacks are empty, try to rebuild them "
                "using g3point_label scalar field");
        int idx = m_cloud->getScalarFieldIndexByName("g3point_label");
        if (idx == -1) {
            CVLog::Warning(
                    "[G3PointAction::fit] no existing g3point_label scalar "
                    "field");
            return;
        }
        cloudViewer::ScalarField* g3PointLabel = m_cloud->getScalarField(idx);
        if (!buildStacksFromG3PointLabelSF(g3PointLabel)) {
            CVLog::Warning(
                    "[G3PointAction::fit] not possible to build stacks from "
                    "existing g3point_scalar field");
            return;
        }
    }

    // plot display grains as ellipsoids
    m_grainColors.reset(new RGBAColorsTableType(
            getRandomColors(m_localMaximumIndexes.size())));
    m_grainsAsEllipsoids =
            new GrainsAsEllipsoids(m_cloud, m_app, m_stacks, *m_grainColors);
    m_grainsAsEllipsoids->setName("g3point_ellipsoids");

    // add connections with the dialog
    connect(m_dlg, &G3PointDialog::onlyOneClicked, m_grainsAsEllipsoids,
            &GrainsAsEllipsoids::showOnlyOne);
    connect(m_dlg, &G3PointDialog::allClicked, m_grainsAsEllipsoids,
            &GrainsAsEllipsoids::showAll);
    connect(m_dlg, &G3PointDialog::onlyOneChanged, m_grainsAsEllipsoids,
            &GrainsAsEllipsoids::setOnlyOne);
    connect(m_dlg, &G3PointDialog::transparencyChanged, m_grainsAsEllipsoids,
            &GrainsAsEllipsoids::setTransparency);
    connect(m_dlg, &G3PointDialog::drawSurfaces, m_grainsAsEllipsoids,
            &GrainsAsEllipsoids::drawSurfaces);
    connect(m_dlg, &G3PointDialog::drawLines, m_grainsAsEllipsoids,
            &GrainsAsEllipsoids::drawLines);
    connect(m_dlg, &G3PointDialog::drawPoints, m_grainsAsEllipsoids,
            &GrainsAsEllipsoids::drawPoints);
    connect(m_dlg, &G3PointDialog::glPointSize, m_grainsAsEllipsoids,
            &GrainsAsEllipsoids::setGLPointSize);

    m_dlg->setOnlyOneMax(static_cast<int>(m_stacks.size()));
    m_dlg->emitSignals();  // force to send parameters to m_grainsAsEllipsoids

    // m_cloud->getParent()->addChild(m_grainsAsEllipsoids.get());
    // m_app->addToDB(m_cloud);
    m_cloud->addChild(m_grainsAsEllipsoids);
    m_app->addToDB(m_grainsAsEllipsoids);

    m_app->updateUI();
}

void G3PointAction::exportResults() {
    if (m_grainsAsEllipsoids) {
        m_grainsAsEllipsoids->exportResultsAsCloud();
    }
}

Eigen::VectorXf arange(double start, double stop, double step) {
    // Values are generated within the half-open interval [0, stop)
    double range = (stop - start);
    int n_steps = floor(range / step) + 1;
    Eigen::VectorXf vector(n_steps);
    for (int k = 0; k < n_steps; k++) {
        vector(k) = start + k * step;
    }
    return vector;
}

template <typename T>
void myPrint(QString name, T array) {
    std::cout << name.toStdString() << "\n" << array << std::endl;
}

// from https://stackoverflow.com/questions/11964552/finding-quartiles
// this gives the same values as python with the default method
// in Python, use hazen as a method to get the same results as in Matlab
// Hyndman and Fan, 1996) R. J. Hyndman and Y. Fan, “Sample quantiles in
// statistical packages”, The American Statistician, 50(4), pp. 361-365, 1996
template <typename T1, typename T2>
typename T1::value_type quant(T1& x, const T2& q) {
    assert(q >= 0.0 && q <= 1.0);

    std::sort(x.begin(), x.end());
    const auto n = x.size();
    const auto id = (n - 1) * q;
    const auto lo = floor(id);
    const auto hi = ceil(id);
    const auto qs = x[lo];
    const auto h = (id - lo);

    return (1.0 - h) * qs + h * x[hi];
}

template <typename T>
double std_dev(const T& vec) {
    // un-biased estimation of the standard deviation => divide by vec.size() -
    // 1
    return std::sqrt((vec - vec.mean()).square().sum() / (vec.size() - 1));
}

void G3PointAction::plots() { angles(); }

void G3PointAction::showWolman(const Eigen::ArrayXf& d_sample,
                               const Eigen::Array3d& dq_final,
                               const Eigen::Array3d& edq) {
    // QCustomPlot
    if (!s_g3PointPlots) s_g3PointPlots = new G3PointPlots(m_cloud->getName());
    s_g3PointPlots->addToTabWidget(
            new WolmanCustomPlot(d_sample, dq_final, edq));
    s_g3PointPlots->show();
}

bool G3PointAction::wolman() {
    int n_iter = m_dlg->getWolmanNbIter();

    if (!m_grainsAsEllipsoids) {
        CVLog::Error(
                "[G3PointAction::wolman] no ellipsoid, not possible to do "
                "Wolman analysis");
        return false;
    }

    // get the ellipsoid y axes andd labels
    int nEllipsoids = static_cast<int>(m_grainsAsEllipsoids->m_center.size());
    Eigen::VectorXf b_axis(nEllipsoids);
    Eigen::VectorXi ellipsoidLabels(nEllipsoids);
    for (int i = 0; i < nEllipsoids; i++) {
        b_axis(i) = 2 * m_grainsAsEllipsoids->m_radii[i]
                                .y();  // files are in radius, we need diameters
        ellipsoidLabels(i) = i;
    }

    // rebuild vectors with the coordinates and the labels of the original
    // points
    int n_points = m_grainsAsEllipsoids->m_cloud->size();
    Eigen::ArrayXf x(n_points);
    Eigen::ArrayXf y(n_points);
    Eigen::ArrayXf z(n_points);
    Eigen::ArrayXi pointsLabels(n_points);
    int sfIdx = m_grainsAsEllipsoids->m_cloud->getScalarFieldIndexByName(
            "g3point_label");
    if (sfIdx == -1) {
        CVLog::Error("[G3PointAction::wolman] no g3point_label");
        return false;
    }
    cloudViewer::ScalarField* g3point_label = m_cloud->getScalarField(sfIdx);
    for (int i = 0; i < n_points; i++) {
        const CCVector3* P = m_grainsAsEllipsoids->m_cloud->getPoint(i);
        x(i) = P->x;
        y(i) = P->y;
        z(i) = P->z;
        pointsLabels(i) = g3point_label->getValue(i);
    }

    float dx = 1.1 * b_axis.maxCoeff();

    std::vector<Eigen::ArrayXf> d;

    QMutex mutex;  // To protect d.push_back
    std::vector<int> iter_indices(n_iter);
    std::iota(iter_indices.begin(), iter_indices.end(), 0);

    ecvProgressDialog pDlg(true, m_app->getMainWindow());
    pDlg.setMethodTitle("Wolman Analysis");
    pDlg.setMaximum(n_iter);
    pDlg.show();

    // atomic counter for progress
    std::atomic<int> progress_count(0);

    QtConcurrent::blockingMap(iter_indices, [&](int k) {
        if (pDlg.wasCanceled()) return;

        // local variables
        Eigen::ArrayXf x_grid;
        Eigen::ArrayXf y_grid;
        Eigen::ArrayXf distances;
        Eigen::Index minLoc;

        // Thread-local RNG
        static thread_local std::mt19937 urbg(std::random_device{}());
        std::uniform_real_distribution<float> dist{0, 1};

        float r0 = dist(urbg);
        float r1 = dist(urbg);

        x_grid = arange(x.minCoeff() - r0 * dx, x.maxCoeff(), dx);
        y_grid = arange(y.minCoeff() - r1 * dx, y.maxCoeff(), dx);
        int nx = x_grid.size();
        int ny = y_grid.size();
        Eigen::ArrayXXf dist_grid(
                nx, ny);  // Rename 'dist' to avoid conflict with RNG
        Eigen::ArrayXXf iWolman(nx, ny);
        for (int ix = 0; ix < nx; ix++) {
            for (int iy = 0; iy < ny; iy++) {
                distances = ((x - x_grid(ix)).pow(2) + (y - y_grid(iy)).pow(2))
                                    .sqrt();
                dist_grid(ix, iy) = distances.minCoeff(&minLoc);
                iWolman(ix, iy) = minLoc;
            }
        }

        XXb condition = (dist_grid < dx / 10);
        Eigen::ArrayXf iWolmanSelection(condition.count());
        int indexInWolmanSelection = 0;

        for (int ix = 0; ix < nx; ix++) {
            for (int iy = 0; iy < ny; iy++) {
                if (condition(ix, iy)) {
                    iWolmanSelection(indexInWolmanSelection) = iWolman(ix, iy);
                    indexInWolmanSelection++;
                }
            }
        }
        Eigen::ArrayXi wolmanSelection;
        wolmanSelection = pointsLabels(iWolmanSelection);

        std::unordered_set<int> setOfA(wolmanSelection.begin(),
                                       wolmanSelection.end());
        std::vector<int> y_ind;
        for (auto label : wolmanSelection) {
            if (m_grainsAsEllipsoids->m_fitNotOK.count(label)) {
                continue;
            } else {
                y_ind.push_back(label);
            }
        }
        Eigen::ArrayXf d_item(y_ind.size());
        for (int i = 0; i < y_ind.size(); i++) {
            d_item(i) = b_axis(y_ind[i]) * 1000;  // conversion to mm
        }

        {
            QMutexLocker locker(&mutex);
            d.push_back(d_item);
        }
    });

    Eigen::ArrayXXf dq(n_iter, 3);
    Eigen::ArrayXf d_sample = d[0];
    dq(0, Eigen::all) << quant(d[0], 0.1), quant(d[0], 0.5), quant(d[0], 0.9);
    for (int i = 1; i < n_iter; i++) {
        Eigen::ArrayXf tmp(d_sample.size() + d[i].size());
        tmp << d_sample, d[i];
        d_sample = tmp;
        dq(i, Eigen::all) << quant(d[i], 0.1), quant(d[i], 0.5),
                quant(d[i], 0.9);
    }

    // std::cout << "d_sample " << d_sample << std::endl;

    // compute standard deviation
    Eigen::Array3d edq{std_dev(dq(Eigen::all, 0)), std_dev(dq(Eigen::all, 1)),
                       std_dev(dq(Eigen::all, 2))};
    Eigen::Array3d dq_final{quant(d_sample, 0.1), quant(d_sample, 0.5),
                            quant(d_sample, 0.9)};

    // std::cout << "d_sample\n" << d_sample << std::endl;
    CVLog::Print(
            QString("[G3PointAction::wolman] quantl 0.1 0.5 0.9 \n%1 %2 %3")
                    .arg(dq_final(0))
                    .arg(dq_final(1))
                    .arg(dq_final(2)));

    CVLog::Print(
            QString("[G3PointAction::wolman] std_dev 0.1 0.5 0.9 \n%1 %2 %3")
                    .arg(edq(0))
                    .arg(edq(1))
                    .arg(edq(2)));

    showWolman(d_sample, dq_final, edq);

    return true;
}

bool G3PointAction::angles() {
    if (!m_grainsAsEllipsoids) {
        CVLog::Error(
                "[G3PointAction::angles] no ellipsoid, not possible to do "
                "angles analysis");
        return false;
    }

    float delta = static_cast<float>(1e32);
    int n_ellipsoids =
            static_cast<float>(m_grainsAsEllipsoids->m_rotationMatrix.size());
    QVector<double> granuloAngleMView(n_ellipsoids);
    QVector<double> granuloAngleXView(n_ellipsoids);

    for (int i = 0; i < n_ellipsoids; i++) {
        float u, v, w;

        Eigen::Vector3f p2{m_grainsAsEllipsoids->m_rotationMatrix[i](0, 0),
                           m_grainsAsEllipsoids->m_rotationMatrix[i](1, 0),
                           m_grainsAsEllipsoids->m_rotationMatrix[i](2, 0)};

        // x-y plot - mapview (angle with y axis)
        Eigen::Vector3f p1{m_grainsAsEllipsoids->m_center[i].x(),
                           m_grainsAsEllipsoids->m_center[i].y() + delta,
                           m_grainsAsEllipsoids->m_center[i].z()};
        float angle = atan2(p1.cross(p2).norm(), p1.dot(p2));
        u = p2(0);
        v = p2(1);
        if ((angle > M_PI / 2) || (angle < -M_PI / 2)) {
            u = -u;
            v = -v;
        }
        granuloAngleMView[i] = (atan(v / u) + M_PI / 2) * 180 / M_PI;

        // x-z plot
        p1 << m_grainsAsEllipsoids->m_center[i].x(),
                m_grainsAsEllipsoids->m_center[i].y(),
                m_grainsAsEllipsoids->m_center[i].z() + delta;
        angle = atan2(p1.cross(p2).norm(), p1.dot(p2));
        v = p2(0);
        w = p2(1);
        if ((angle > M_PI / 2) || (angle < -M_PI / 2)) {
            v = -v;
            w = -w;
        }
        granuloAngleXView[i] = (atan(v / w) + M_PI / 2) * 180 / M_PI;
    }

    // QCustomPlot
    if (!s_g3PointPlots) s_g3PointPlots = new G3PointPlots(m_cloud->getName());
    int nbBins = m_dlg->getAnglesNbBins();
    s_g3PointPlots->addToTabWidget(
            new AnglesCustomPlot(granuloAngleMView, "Azimut", nbBins));
    s_g3PointPlots->addToTabWidget(
            new AnglesCustomPlot(granuloAngleXView, "Dip", nbBins));
    s_g3PointPlots->show();

    return true;
}

bool G3PointAction::cleanLabels() {
    CVLog::Print("[cleanLabels]");

    m_maxAngle2 = m_dlg->getMaxAngle2();
    m_nMin = m_dlg->getNMin();
    m_minFlatness = m_dlg->getMinFlatness();

    // merge points considering the normals at the border
    {
        CVLog::Print(
                "[cleanLabels] merge points considering the normals at the "
                "border");
        Eigen::ArrayXXd A = computeMeanAngleBetweenNormalsAtBorders();
        XXb condition = (A > m_maxAngle2) ||
                        (A != A);  // add true on the diagonal (important for
                                   // the if hereafter)
        XXb symmetrical_condition =
                (condition == condition.transpose()).select(condition, true);

        merge(symmetrical_condition);

        QApplication::processEvents();
    }

    // remove small labels
    {
        CVLog::Print("[cleanLabels] remove small labels");
        Eigen::ArrayXi stackSize(m_stacks.size());
        for (size_t k = 0; k < m_stacks.size(); k++) {
            stackSize(k) = static_cast<int>(m_stacks[k].size());
        }
        Xb condition = (stackSize > m_nMin);
        size_t numberOfGrainsToKeep = condition.count();
        if (numberOfGrainsToKeep == m_stacks.size()) {
            CVLog::Print("[cleanLabels] all grains are larger than " +
                         QString::number(m_nMin) +
                         " points, nothing to remove");
        } else if (numberOfGrainsToKeep) {
            keep(condition);
        } else {
            CVLog::Error(
                    "[cleanLabels] no remaining grain after removing those "
                    "with less than " +
                    QString::number(m_nMin) + " points");
            return false;
        }
        QApplication::processEvents();
    }

    // clean labels
    {
        CVLog::Print("[cleanLabels] remove flattish labels");
        Eigen::ArrayX3d s(m_stacks.size(), 3);
        for (size_t k = 0; k < m_stacks.size(); k++) {
            std::vector<int>& stack = m_stacks[k];
            // get the points of the label
            size_t nPoints = stack.size();
            Eigen::MatrixX3d points(nPoints, 3);
            for (int index = 0; index < nPoints; index++) {
                const CCVector3* point = m_cloud->getPoint(stack[index]);
                points(index, 0) = point->x;
                points(index, 1) = point->y;
                points(index, 2) = point->z;
            }
            // compute the centroid of the label
            Eigen::RowVector3d centroid = points.colwise().mean();
            points.rowwise() -= centroid;
            // SVD decomposition A = U S V∗
            s(k, Eigen::all) = points.jacobiSvd().singularValues();
        }
        // filtering condition: (l2 / l0 > min_flatness) or (l1 / l0 > 2 *
        // min_flatness)
        Xb condition =
                (s(Eigen::all, 2) / s(Eigen::all, 0) > m_minFlatness) ||
                (s(Eigen::all, 1) / s(Eigen::all, 0) > 2. * m_minFlatness);
        size_t numberOfGrainsToKeep = condition.count();
        if (numberOfGrainsToKeep == m_stacks.size()) {
            CVLog::Print("[cleanLabels] no flattish grain, nothing to remove");
        } else if (numberOfGrainsToKeep) {
            keep(condition);
        } else {
            CVLog::Error(
                    "[cleanLabels] no remaining grain after removing the "
                    "flattish ones");
            return false;
        }
    }

    return true;
}

void G3PointAction::addToStackBraunWillett(int index,
                                           const Eigen::ArrayXi& delta,
                                           const Eigen::ArrayXi& Di,
                                           std::vector<int>& stack,
                                           int local_maximum) {
    stack.push_back(index);

    for (int k = delta[index]; k < delta[index + 1]; k++) {
        if (Di[k] != local_maximum)  // avoid infinite loop
        {
            addToStackBraunWillett(Di[k], delta, Di, stack, local_maximum);
        }
    }
}

int G3PointAction::segmentLabelsBraunWillett() {
    CVLog::Print("[segment_labels_braun_willett]");

    bool steepestSlope = m_dlg->isSteepestSlope();

    // for each point, find in the neighborhood the point with the extreme
    // slope, depending on the mode (the receiver)
    Eigen::ArrayXd extreme_slopes;
    if (steepestSlope) {
        CVLog::Print(
                "[segment_labels] classical steepest slope algorithm [Braun, "
                "Willett 2013]");
        extreme_slopes = m_neighborsSlopes.rowwise().maxCoeff();
    } else {
        CVLog::Print(
                "[segment_labels] reversed version of the steepest slope "
                "algorithm [Braun, Willett 2013]");
        extreme_slopes = m_neighborsSlopes.rowwise().minCoeff();
    }
    Eigen::ArrayXi index_of_extreme_slope =
            Eigen::ArrayXi::Zero(m_cloud->size());
    Eigen::ArrayXi receivers(m_cloud->size());

    for (unsigned index = 0; index < m_cloud->size(); index++) {
        double extreme_slope = extreme_slopes(index);
        for (int k = 0; k < m_kNN; k++) {
            if (m_neighborsSlopes(index, k) == extreme_slope) {
                index_of_extreme_slope(index) = k;
                break;
            }
        }
        receivers(index) =
                m_neighborsIndexes(index, index_of_extreme_slope(index));
    }

    // if the minimum slope is positive, the receiver is a local maximum
    int nb_maxima;
    if (steepestSlope) {
        nb_maxima = (extreme_slopes <= 0).count();
    } else {
        nb_maxima = (extreme_slopes >= 0).count();
    }
    m_initial_localMaximumIndexes = Eigen::ArrayXi::Zero(
            nb_maxima);  // we need nb_maxima for the initialization
    int l = 0;
    for (unsigned int k = 0; k < m_cloud->size(); k++) {
        if (steepestSlope) {
            if (extreme_slopes(k) <= 0) {
                m_initial_localMaximumIndexes(l) = k;
                receivers(k) = k;
                l++;
            }
        } else {
            if (extreme_slopes(k) >= 0) {
                m_initial_localMaximumIndexes(l) = k;
                receivers(k) = k;
                l++;
            }
        }
    }

    // get the number of donors per receiver (di) and build the list of donors
    // per receiver (Dij)
    CVLog::Print(
            "[segment_labels_braun_willett] identify the donors for each "
            "receiver");
    Eigen::ArrayXi di =
            Eigen::ArrayXi::Zero(m_cloud->size());  // number of donors
    std::vector<std::vector<int>> Dij;  // lists of donors (one list per point)
    for (unsigned int k = 0; k < m_cloud->size();
         k++)  // initialize the lists of donors
    {
        std::vector<int> list_of_donors;
        Dij.push_back(list_of_donors);
    }
    CVLog::Print("[segment_labels_braun_willett] create di and Dij");
    for (unsigned int k = 0; k < m_cloud->size(); k++) {
        int receiver = receivers(k);
        di[receiver] = di[receiver] +
                       1;  // increment the number of donors of the receiver
        Dij[receiver].push_back(
                k);  // add the donor to the list of donors of the receiver
    }

    m_ndon = di;

    // build Di, the list of donors
    Eigen::ArrayXi Di =
            Eigen::ArrayXi::Zero(m_cloud->size());  // list of donors
    int idx = 0;
    for (auto& list_ : Dij)  // build the list of donors
    {
        for (int point : list_) {
            Di[idx] = point;
            idx++;
        }
    }

    // build delta, the index array
    Eigen::ArrayXi delta = Eigen::ArrayXi::Zero(m_cloud->size() +
                                                1);  // index of the first donor
    delta[m_cloud->size()] = m_cloud->size();
    std::vector<int> delta_vec(m_cloud->size());
    for (int i = m_cloud->size() - 1; i >= 0; i--) {
        delta[i] = delta[i + 1] - di[i];
        delta_vec[i] = delta[i];
    }

    // build the stacks
    CVLog::Print("[segment_labels_braun_willett] build the stacks");

    int sfIdx =
            m_cloud->getScalarFieldIndexByName("g3point_initial_segmentation");
    if (sfIdx == -1) {
        sfIdx = m_cloud->addScalarField("g3point_initial_segmentation");
        if (sfIdx == -1) {
            CVLog::Error(
                    "[G3Point::segment_labels] impossible to create scalar "
                    "field g3point_initial_segmentation");
        }
    }
    cloudViewer::ScalarField* g3point_label = m_cloud->getScalarField(sfIdx);

    RGBAColorsTableType randomColors =
            getRandomColors(m_initial_localMaximumIndexes.size());

    if (!m_cloud->resizeTheRGBTable(false)) {
        CVLog::Error(QObject::tr("Not enough memory!"));
        return -1;
    }

    for (int k = 0; k < m_initial_localMaximumIndexes.size(); k++) {
        int localMaximumIndex = m_initial_localMaximumIndexes(k);
        std::vector<int> stack;
        addToStackBraunWillett(localMaximumIndex, delta, Di, stack,
                               localMaximumIndex);
        // labels
        for (auto i : stack) {
            m_initial_labels(i) = k;
            m_initial_labelsnpoint(i) = static_cast<float>(stack.size());
            if (g3point_label) {
                g3point_label->setValue(i, k);
                m_cloud->setPointColor(i, randomColors.getValue(k));
            }
        }
        m_initialStacks.push_back(stack);
    }

    if (!checkStacks(m_initialStacks,
                     m_cloud->size()))  // check the stacks coherency
    {
        m_cloud->deleteScalarField(sfIdx);
        return -1;
    }

    if (g3point_label) {
        g3point_label->computeMinAndMax();
    }

    m_cloud->setCurrentDisplayedScalarField(sfIdx);
    m_cloud->showColors(true);
    m_cloud->showSF(false);

    m_cloud->redrawDisplay();
    // m_cloud->prepareDisplayForRefresh();

    if (m_app) {
        m_app->refreshAll();
        m_app->updateUI();
    }

    int nLabels = m_initial_localMaximumIndexes.size();

    return nLabels;
}

void G3PointAction::getNeighborsDistancesSlopes(unsigned index,
                                                std::vector<char>& duplicates) {
    const CCVector3* P = m_cloud->getPoint(index);

    m_nNSS.level = m_bestOctreeLevel;
    double maxSquareDist = 0;
    int neighborhoodSize = 0;
    cloudViewer::ReferenceCloud Yk(m_cloud);

    // get the nearest neighbors
    if (m_octree->findPointNeighbourhood(
                P, &Yk, static_cast<unsigned>(m_kNN + 1), m_bestOctreeLevel,
                maxSquareDist, 0,
                &neighborhoodSize) >= static_cast<unsigned>(m_kNN)) {
        assert(m_kNN == m_neighborsIndexes.cols());
        for (int k = 0; k < m_kNN; k++) {
            // store the index of the neighbor
            m_neighborsIndexes(index, k) = Yk.getPointGlobalIndex(k + 1);
            // compute the distance to the neighbor
            const CCVector3* neighbor = Yk.getPoint(k + 1);
            float distance = (*P - *neighbor).norm();
            m_neighborsDistances(index, k) = distance;
            // compute the slope to the neighbor
            if (distance != 0) {
                m_neighborsSlopes(index, k) = (P->z - neighbor->z) / distance;
            } else {
                m_neighborsSlopes(index, k) =
                        0.;  // it is possible to have duplicates in the cloud
                duplicates[index] = 1;
            }
        }
    }
}

void G3PointAction::computeNodeSurfaces() {
    m_area = M_PI * m_neighborsDistances.rowwise().minCoeff().square();
}

bool G3PointAction::computeNormalsAndOrientThem() {
    double radius = 1.;  // one should set the radius value

    CV_LOCAL_MODEL_TYPES model = CV_LOCAL_MODEL_TYPES::QUADRIC;
    ccNormalVectors::Orientation orientation =
            ccNormalVectors::Orientation::UNDEFINED;

    orientation = ccNormalVectors::Orientation::PLUS_Z;
    model = CV_LOCAL_MODEL_TYPES::LS;

    QScopedPointer<ecvProgressDialog> progressDialog(nullptr);

    if (!m_cloud->getOctree()) {
        if (!m_cloud->computeOctree(progressDialog.data())) {
            CVLog::Error("Failed to compute octree for cloud " +
                         m_cloud->getName());
            return false;
        }
    }

    float thisCloudRadius = radius;
    if (std::isnan(thisCloudRadius)) {
        ccOctree::BestRadiusParams params;
        {
            params.aimedPopulationPerCell = 16;
            params.aimedPopulationRange = 4;
            params.minCellPopulation = 6;
            params.minAboveMinRatio = 0.97;
        }
        thisCloudRadius = ccOctree::GuessBestRadius(
                m_cloud, params, m_cloud->getOctree().data());
        if (thisCloudRadius == 0) {
            return CVLog::Error(
                    "Failed to determine best normal radius for cloud " +
                    m_cloud->getName());
        }
        CVLog::Print("Cloud " + m_cloud->getName() +
                     " radius = " + QString::number(thisCloudRadius));
    }

    CVLog::Print("computeNormalsWithOctree started...");
    bool success = m_cloud->computeNormalsWithOctree(
            model, orientation, thisCloudRadius, progressDialog.data());
    if (success) {
        CVLog::Print("computeNormalsWithOctree success");
    } else {
        CVLog::Error("computeNormalsWithOctree failed");
        return false;
    }

    return true;
}

void G3PointAction::orientNormals(const Eigen::Vector3d& sensorCenter) {
    int pointCloud = m_cloud->size();

    for (int i = 0; i < pointCloud; i++) {
        const CCVector3* point = m_cloud->getPoint(i);
        Eigen::Vector3d P1 =
                sensorCenter - Eigen::Vector3d(point->x, point->y, point->z);
        Eigen::Vector3d P2 = m_normals(i, Eigen::all);
        double angle = atan2(P1.cross(P2).norm(), P1.dot(P2));
        if ((angle < -M_PI / 2) || (angle > M_PI / 2)) {
            m_normals(i, 0) = -m_normals(i, 0);
            m_normals(i, 1) = -m_normals(i, 1);
            m_normals(i, 2) = -m_normals(i, 2);
        }
    }
}

bool G3PointAction::findNearestNeighborsNanoFlann(
        const unsigned globalIndex,
        cloudViewer::ReferenceCloud* points,
        const KDTree* kdTree) {
    // Prepare query
    const CCVector3* Q = m_cloud->getPoint(globalIndex);
    float query[3] = {Q->x, Q->y, Q->z};

    std::vector<size_t> retIndexes(m_kNN);
    std::vector<float> outDistsSqr(m_kNN);

    // Perform search
    nanoflann::KNNResultSet<float> resultSet(m_kNN);
    resultSet.init(&retIndexes[0], &outDistsSqr[0]);
    if (kdTree->findNeighbors(resultSet, &query[0])) {
        points->resize(m_kNN);
        for (int i = 0; i < m_kNN; ++i) {
            points->setPointIndex(i, retIndexes[i]);
        }
        return true;
    } else {
        return false;
    }
}

bool G3PointAction::computeNormWithFlann(unsigned index,
                                         NormsTableType* theNorms,
                                         const G3PointAction::KDTree* kdTree) {
    CCVector3 N;

    QScopedPointer<cloudViewer::ReferenceCloud> points(
            new cloudViewer::ReferenceCloud(m_cloud));
    if (findNearestNeighborsNanoFlann(index, points.data(), kdTree)) {
        cloudViewer::Neighbourhood neighbourhood(points.data());
        N = *neighbourhood.getLSPlaneNormal();
    } else {
        return false;
    }

    theNorms->setValue(index, N);

    return true;
}

bool G3PointAction::computeNormals() {
    // if there are normals, already, propose to keep them
    if (m_cloud->hasNormals()) {
        QMessageBox msgBox;
        msgBox.setInformativeText("Recompute normals?");
        msgBox.setText("There are existing normals, keep them or recompute.");
        QPushButton* keepButton =
                msgBox.addButton(tr("Keep"), QMessageBox::ActionRole);
        QPushButton* recomputeButton =
                msgBox.addButton(tr("Recompute"), QMessageBox::AcceptRole);
        QPushButton* cancelButton =
                msgBox.addButton(tr("Cancel"), QMessageBox::AcceptRole);

        msgBox.exec();

        if (msgBox.clickedButton() == keepButton) {
            return true;
        } else if (msgBox.clickedButton() == cancelButton) {
            return false;
        }
    }

    unsigned pointCount = m_cloud->size();

    if (!m_cloud || m_cloud->size() == 0) {
        CVLog::Error("Invalid cloud.");
        return false;
    }

    CloudAdaptor adaptor(m_cloud);

    // Build KD-tree (parameter: number of leaf nodes to inspect per query)

    size_t leaf_max_size = 10;
    nanoflann::KDTreeSingleIndexAdaptorFlags flags =
            nanoflann::KDTreeSingleIndexAdaptorFlags::None;
    unsigned int n_thread_build = 0;  // 0 => nanoflann automatically determines
                                      // the number of threads to use

    nanoflann::KDTreeSingleIndexAdaptorParams params(leaf_max_size, flags,
                                                     n_thread_build);
    QSharedPointer<KDTree> m_kdTree(new KDTree(3, adaptor, params));
    m_kdTree->buildIndex();

    // we instantiate 3D normal vectors
    QSharedPointer<NormsTableType> theNorms(new NormsTableType);
    QScopedPointer<NormsIndexesTableType> normsIndexes(
            new NormsIndexesTableType);
    static const CCVector3 blankN(0, 0, 0);
    if (!theNorms->resizeSafe(pointCount, true, &blankN)) {
        normsIndexes->resize(0);
        return false;
    }

    CVLog::Print("[computeNormals]");
#ifdef QT_DEBUG
    // manually call the static per-point method!
    for (unsigned index = 0; index < pointCount; ++index) {
        computeNormWithFlann(index, theNorms.data(), m_kNN, m_kdTree.data(),
                             m_cloud);
    }
#else
    std::vector<unsigned> pointsIndexes;
    pointsIndexes.resize(pointCount);
    for (unsigned i = 0; i < pointCount; ++i) {
        pointsIndexes[i] = i;
    }
    int threadCount = std::max(1, ccQtHelpers::GetMaxThreadCount() - 2);
    CVLog::Print(
            "[computeNormals] parallel strategy, thread "
            "count " +
            QString::number(threadCount));
    QThreadPool::globalInstance()->setMaxThreadCount(threadCount);
    QtConcurrent::blockingMap(pointsIndexes, [=](int index) {
        computeNormWithFlann(index, theNorms.data(), m_kdTree.data());
    });
#endif

    if (!m_cloud->hasNormals()) {
        if (!m_cloud->resizeTheNormsTable()) {
            CVLog::Error(QString("Not enough memory to compute normals on "
                                 "cloud '%1'")
                                 .arg(m_cloud->getName()));
            return false;
        }
    }

    // we hide normals during process
    m_cloud->showNormals(false);

    // compress the normals
    for (unsigned i = 0; i < theNorms->currentSize(); i++) {
        const CCVector3& N = theNorms->at(i);
        const CompressedNormType nCode = ccNormalVectors::GetNormIndex(N);
        m_cloud->setPointNormalIndex(i, nCode);
    }

    // preferred orientation
    CVLog::Print("[computeNormals] orient normals, PLUS_Z ");
    ccNormalVectors::UpdateNormalOrientations(m_cloud, *m_cloud->normals(),
                                              ccNormalVectors::PLUS_Z);

    return true;
}

bool G3PointAction::queryNeighbors(ccPointCloud* cloud,
                                   ecvMainAppInterface* appInterface,
                                   bool useParallelStrategy) {
    QString errorStr;

    ecvProgressDialog progressDlg(true, appInterface->getMainWindow());
    progressDlg.show();
    progressDlg.setAutoClose(false);  // we don't want the progress dialog to
                                      // 'pop' for each feature
    cloudViewer::GenericProgressCallback* progressCb = &progressDlg;
    m_octree = m_cloud->getOctree();

    // get the octree
    if (!m_octree) {
        CVLog::Print(QString("Computing octree of cloud %1 (%2 points)")
                             .arg(m_cloud->getName())
                             .arg(m_cloud->size()));
        if (progressCb) progressCb->start();
        QCoreApplication::processEvents();
        m_octree = m_cloud->computeOctree(progressCb);
        if (!m_octree) {
            errorStr = "Failed to compute octree (not enough memory?)";
            return false;
        }
    }

    m_bestOctreeLevel = m_octree->findBestLevelForAGivenPopulationPerCell(
            static_cast<unsigned>(std::max(3, m_kNN)));

    size_t nPoints = m_cloud->size();
    cloudViewer::DgmOctree::NearestNeighboursSearchStruct nNSS;
    std::vector<unsigned> pointsIndexes(nPoints);
    std::vector<char> duplicates(nPoints, 0);

    if (useParallelStrategy) {
        for (unsigned i = 0; i < nPoints; ++i) {
            pointsIndexes[i] = i;
        }
        int threadCount = std::max(1, ccQtHelpers::GetMaxThreadCount() - 2);
        CVLog::Print(
                QString("[query_neighbor] parallel strategy, thread count %1")
                        .arg(threadCount));
        QThreadPool::globalInstance()->setMaxThreadCount(threadCount);
        QtConcurrent::blockingMap(pointsIndexes, [&](int index) {
            getNeighborsDistancesSlopes(index, duplicates);
        });
    } else {
        // manually call the static per-point method!
        for (unsigned i = 0; i < nPoints; ++i) {
            getNeighborsDistancesSlopes(i, duplicates);
        }
    }

    auto trueCount = std::count(duplicates.begin(), duplicates.end(), 1);
    if (trueCount) {
        CVLog::Warning("[G3Point] You have duplicates (" +
                       QString::number(trueCount) +
                       "), the algorithm will continue but you may think of "
                       "cleaning your cloud.");
    }

    return true;
}

void G3PointAction::segment() {
    init();

    // Find neighbors of each point of the cloud
    bool useParallelStrategy;
#ifdef NDEBUG
    useParallelStrategy = true;
#else
    useParallelStrategy = false;
#endif
    queryNeighbors(m_cloud, m_app, useParallelStrategy);

    computeNodeSurfaces();

    computeNormals();

    // compute the centroid
    unsigned pointCount = m_cloud->size();
    CCVector3d G(0, 0, 0);
    {
        for (unsigned i = 0; i < pointCount; ++i) {
            const CCVector3* P = m_cloud->getPoint(i);
            G += *P;
        }
        G /= pointCount;
    }

    Eigen::Vector3d sensorCenter(G.x, G.y, 1000);
    orientNormals(sensorCenter);

    // Perform initial segmentation
    int nLabels = segmentLabelsBraunWillett();
    if (nLabels == -1) {
        CVLog::Error(
                "[G3Point::segment] initial segmentation failed, abort "
                "segmentation");
        return;
    }

    exportLocalMaximaAsCloud(m_initial_localMaximumIndexes);

    m_app->dispToConsole("[G3Point] initial segmentation: " +
                                 QString::number(nLabels) + " labels",
                         ecvMainAppInterface::STD_CONSOLE_MESSAGE);

    m_dlg->enableClusterAndOrClean(true);

    // in case we want to test the fitting of ellipsoids now, without clustering
    // nor cleaning
    m_labels = m_initial_labels;
    m_labelsnpoint = m_initial_labelsnpoint;
    m_localMaximumIndexes = m_initial_localMaximumIndexes;
    m_stacks = m_initialStacks;
}

void G3PointAction::clusterAndOrClean() {
    // initialize variables for clustering and / or cleaning  to the initial
    // segmentation
    m_labels = m_initial_labels;
    m_labelsnpoint = m_initial_labelsnpoint;
    m_localMaximumIndexes = m_initial_localMaximumIndexes;
    m_stacks = m_initialStacks;

    if (m_dlg->clusterIsChecked()) {
        if (!cluster()) {
            CVLog::Error(
                    "[G3PointAction::clusterAndOrClean] clustering failed");
            return;
        }
    }

    if (m_dlg->cleanIsChecked()) {
        if (!cleanLabels()) {
            CVLog::Error("[G3PointAction::clusterAndOrClean] cleaning failed");
            return;
        }
    }
}

void G3PointAction::getBorders() {
    // Find the indexborder nodes (no donor and many other labels in the
    // neighbourhood)
    Eigen::ArrayXXi duplicatedLabelsInColumns(m_cloud->size(), m_kNN);
    for (int n = 0; n < m_kNN; n++) {
        duplicatedLabelsInColumns(Eigen::all, n) = m_labels;
    }
    Eigen::ArrayXXi labelsOfNeighbors(m_cloud->size(), m_kNN);
    for (int index = 0; index < static_cast<float>(m_cloud->size()); index++) {
        for (int n = 0; n < m_kNN; n++) {
            labelsOfNeighbors(index, n) =
                    m_labels(m_neighborsIndexes(index, n));
        }
    }

    Eigen::ArrayXi temp =
            m_kNN - (labelsOfNeighbors == duplicatedLabelsInColumns)
                            .cast<int>()
                            .rowwise()
                            .sum();
    auto condition = ((temp >= m_kNN / 4) && (m_ndon == 0));

    // create cloud
    cloudViewer::ReferenceCloud referenceCloud(m_cloud);
    for (int index = 0; index < condition.size(); index++) {
        if (condition(index)) {
            referenceCloud.addPointIndex(index);
        }
    }

    ccPointCloud* borderCloud = m_cloud->partialClone(&referenceCloud);

    // add cloud to the database
    borderCloud->setName(m_cloud->getName() + "_borders");
    m_cloud->getParent()->addChild(borderCloud, ccHObject::DP_PARENT_OF_OTHER,
                                   0);
    m_app->addToDB(borderCloud);
}

void G3PointAction::init() {
    // initialize the matrices which will contain the results
    m_neighborsIndexes = Eigen::ArrayXXi::Zero(m_cloud->size(), m_kNN);
    m_neighborsDistances = Eigen::ArrayXXd::Zero(m_cloud->size(), m_kNN);
    m_neighborsSlopes = Eigen::ArrayXXd::Zero(m_cloud->size(), m_kNN);
    m_normals = Eigen::ArrayXXd::Zero(m_cloud->size(), 3);

    m_labels = Eigen::ArrayXi::Zero(m_cloud->size());
    m_labelsnpoint = Eigen::ArrayXi::Zero(m_cloud->size());
    m_initial_labels = Eigen::ArrayXi::Zero(m_cloud->size());
    m_initial_labelsnpoint = Eigen::ArrayXi::Zero(m_cloud->size());

    m_stacks.clear();         // needed in case of several runs
    m_initialStacks.clear();  // needed in case of several runs
}

void G3PointAction::showDlg() {
    if (!m_dlg) {
        m_dlg = new G3PointDialog(m_cloud->getName());

        connect(m_dlg, &G3PointDialog::kNNEditingFinished,
                s_g3PointAction.get(), &G3PointAction::setKNN);
        connect(m_dlg, &G3PointDialog::segment, s_g3PointAction.get(),
                &G3Point::G3PointAction::segment);
        connect(m_dlg, &G3PointDialog::clusterAndOrClean, s_g3PointAction.get(),
                &G3Point::G3PointAction::clusterAndOrClean);
        connect(m_dlg, &G3PointDialog::getBorders, s_g3PointAction.get(),
                &G3Point::G3PointAction::getBorders);

        connect(m_dlg, &G3PointDialog::fit, s_g3PointAction.get(),
                &G3Point::G3PointAction::fit);
        connect(m_dlg, &G3PointDialog::exportResults, s_g3PointAction.get(),
                &G3Point::G3PointAction::exportResults);
        connect(m_dlg, &G3PointDialog::wolman, s_g3PointAction.get(),
                &G3Point::G3PointAction::wolman);
        connect(m_dlg, &G3PointDialog::angles, s_g3PointAction.get(),
                &G3Point::G3PointAction::plots);

        connect(m_dlg, &QDialog::finished, s_g3PointAction.get(),
                &G3Point::G3PointAction::clean);
        connect(m_dlg, &QDialog::finished, s_g3PointAction.get(),
                &G3Point::G3PointAction::resetDlg);  // dialog is defined with
                                                     // Qt::WA_DeleteOnClose
    } else {
        m_dlg->enableClusterAndOrClean(false);
    }

    m_dlg->show();
}

void G3PointAction::resetDlg() {
    // dialog is defined with Qt::WA_DeleteOnClose, you have to reset the
    // pointer when it's closed by the user
    m_dlg = nullptr;
}

void G3PointAction::clean() {
    m_neighborsIndexes.resize(0, 0);
    m_neighborsDistances.resize(0, 0);
    m_neighborsSlopes.resize(0, 0);
    m_normals.resize(0, 0);

    m_labels.resize(0);
    m_labelsnpoint.resize(0);
    m_localMaximumIndexes.resize(0);
    m_ndon.resize(0);
    m_area.resize(0);

    m_stacks.clear();

    m_octree.clear();
}

bool G3PointAction::setCloud(ccPointCloud* cloud) {
    m_cloud = cloud;

    // at this step, it is possible to initialize the stacks if the cloud has a
    // g3point_label scalar field

    int sfIdx = m_cloud->getScalarFieldIndexByName("g3point_label");
    if (sfIdx != -1) {
        init();

        // copy the g3point_label scalar field
        int sfIdxBackup =
                m_cloud->getScalarFieldIndexByName("g3point_label_backup");
        if (sfIdxBackup == -1)  // the scalar field g3point_label_backup does
                                // not exist, create it
        {
            sfIdxBackup = m_cloud->addScalarField("g3point_label_backup");
            if (sfIdxBackup == -1) {
                CVLog::Error(
                        "[G3PointAction::setCloud] impossible to create scalar "
                        "field g3point_label_backup");
                return false;
            } else {
                CVLog::Warning(
                        "[G3PointAction::setCloud] duplicate existing scalar "
                        "field g3point_label => g3point_label_backup");
            }
        }

        cloudViewer::ScalarField* g3PointLabelSF =
                m_cloud->getScalarField(sfIdx);
        cloudViewer::ScalarField* g3PointLabelBackupSF =
                m_cloud->getScalarField(sfIdxBackup);

        // get the set of labels
        std::set<ScalarType> labelsSet;
        for (unsigned int index = 0; index < m_cloud->size(); index++) {
            labelsSet.insert(g3PointLabelSF->getValue(index));
        }
        // remove -1 from the set
        labelsSet.erase(-1);
        // build a map to handle cases were g3point_label is not a regular range
        // with step 1 keys = labels values = index
        std::map<ScalarType, int> labelsMap;
        int index = 0;
        for (auto label : labelsSet) {
            labelsMap[label] = index;
            index++;
        }

        // build stacks from g3point_index
        int nPointsInGrains = 0;

        ecvProgressDialog pDlg(true, m_app->getMainWindow());
        pDlg.setMethodTitle("Processing g3point_label");
        pDlg.setInfo("Classifying points...");
        pDlg.setMaximum(m_cloud->size());
        pDlg.show();
        QApplication::processEvents();

        std::vector<std::vector<int>> newStacks(labelsSet.size());

        // Optimize: reduce processEvents calls
        // Update progress every 1% or at least every 10000 points
        unsigned updateStep = std::max<unsigned>(10000, m_cloud->size() / 100);

        for (unsigned int index = 0; index < m_cloud->size(); index++) {
            ScalarType label = g3PointLabelSF->getValue(index);
            g3PointLabelBackupSF->setValue(
                    index, label);  // save the originale G3Point label
            if (label != -1)  // -1 is a generic index corresponding to all
                              // points which do not belong to valid grains
            {
                newStacks[labelsMap[label]].push_back(index);
                nPointsInGrains++;
            }

            if (index % updateStep == 0) {
                if (pDlg.wasCanceled()) {
                    return false;
                }
                pDlg.setValue(index);
            }
        }

        CVLog::Print(
                "[G3PointAction::setCloud] g3point_label available, found " +
                QString::number(nPointsInGrains) + "/" +
                QString::number(cloud->size()) + " points belonging to " +
                QString::number(newStacks.size()) + " grains");

        processNewStacks(newStacks, nPointsInGrains);
    }

    return true;
}

void G3PointAction::setKNN() { m_kNN = m_dlg->getkNN(); }

void G3PointAction::createAction(ecvMainAppInterface* appInterface) {
    if (appInterface == nullptr) {
        // The application interface should have already been initialized when
        // the plugin is loaded
        Q_ASSERT(false);

        return;
    }

    // we need one point cloud
    if (!appInterface->haveOneSelection()) {
        appInterface->dispToConsole("Select only one cloud!",
                                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // a real point cloud
    const ccHObject::Container& selectedEntities =
            appInterface->getSelectedEntities();

    ccHObject* ent = selectedEntities[0];
    if (!ent->isA(CV_TYPES::POINT_CLOUD)) {
        appInterface->dispToConsole("Select a cloud!",
                                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    GetG3PointAction(ccHObjectCaster::ToPointCloud(ent), appInterface);
}

}  // namespace G3Point
